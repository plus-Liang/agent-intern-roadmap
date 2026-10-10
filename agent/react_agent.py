import json5

"""
ReAct Agent。
LLM 在循环里自主决策：思考 → 调工具 → 观察 → 再思考。
"""
import hashlib
import json
import os
import re
import time
import uuid
from shared.llm_client import chat
from shared.logger import log_event
from shared import limits
from agent.tools_registry import (
    list_tools_description,
    call_tool,
    get_current_resume,
    TOOLS,
)
from agent import user_profile
from agent import reminder
from agent import chat_history


MAX_TURNS = 6

# ========== 第 2 道闸门：单次请求的 token 预算熔断 ==========
#
# 上限 RUN_TOKEN_BUDGET（默认 30k）。**降级，不拒绝** —— 触顶时停止 ReAct 循环，
# 用已经跑出来的 steps 收尾，而不是把用户的这一条消息整体拒掉。
# 累加在 shared/llm_client._record_usage（唯一用量入口）里完成，
# 这里只负责每轮 chat() 之前判断一次，并记一条 budget_stop 事件。
BUDGET_STOP_ANSWER = (
    "本轮消耗已达上限，先给到这里。"
    "如果需要更完整的结果，可以把问题拆成几步、或换个更具体的问法再问。"
)

# ========== 第 1 道闸门的「长输出轮」判据（见 _long_output_turn） ==========
#
# 为什么需要：`glm-5.3-flash` 是思考模型，`max_tokens` 同时卡住思考
# （`reasoning_content`）与正文。默认 1024 下，工具调用轮实测 176 token 就收尾
# （够用），但「查看我的完整简历」这类要复述长内容的轮次会被思考吃光额度
# → finish_reason=length、正文为空。命中判据的轮次换用更大的额度。
#
# 工具返回超过这个长度就算「模型可能要把读进去的东西再写出来」。实测一份
# 结构化简历 dict 约 1.5k~5k 字符，故取 800（描述/项目多的简历都会命中）。
LONG_OBSERVATION_CHARS = 800

# 问题里出现这些词，说明用户明确在要一份**长文本**（而不是一句话结论）。
# 命中就预先进长输出档：`max_tokens` 只是上限、模型说完就停，猜错不花 token，
# 但猜漏要多付一次「被截断的 1024 轮」。
LONG_OUTPUT_HINTS = ("完整", "全部", "列全", "逐条", "全文", "详细", "原样", "JD")

# check_reminders 的 observation 里最多列几条超期记录。
# 全列出来会把上下文撑满（用户可能投了几十家），反正最久的排最前，
# 截断后另外给一句「共 N 条」的说明，需要完整列表时用户可以再问。
MAX_REMINDER_ITEMS = 20

# 只影响**日志打印**的截断长度：steps 里存的是完整 observation（判定/归档要用原文），
# 控制台只打个开头，免得刷屏。想看全文直接看 steps / 评估结果 JSON。
OBSERVATION_LOG_CHARS = 500

# 消息历史软上限：超过就把中间部分压成摘要，防止长对话把 token 撑爆。
# 可用环境变量 MAX_HISTORY 覆盖。
MAX_HISTORY = 10

SUMMARY_PROMPT_TEMPLATE = (
    "以下是之前的对话历史，请用 2-3 句话概括关键信息"
    "（用户问了什么、调用了哪些工具、得到什么结果）：\n\n{history}\n\n摘要："
)


# ========== 稳定前缀（KV Cache 友好） ==========
#
# 上下文决定能力上限，前缀越稳定缓存命中越高。所以 system prompt 拆成两半：
#   - STATIC_PREFIX：角色 + 工具定义 + 规则。只要代码和工具表不变，它逐字节不变，
#     每次 LLM 调用的开头都是同一段，前缀缓存（KV Cache）可以直接命中；
#   - DYNAMIC_CONTEXT：用户偏好 / 当前简历 / 当前时间，每次调用都在变，
#     以 **user** 消息（不是第二条 system）的形式追加在静态前缀之后。
#
# 动态部分为什么用 user 而不是 system：部分 OpenAI 兼容端点对多条 system
# 消息处理不一致（有的只认第一条、有的直接报错），用 user 消息最稳。
STATIC_PREFIX = """你是一个求职助手 Agent。你可以调用工具帮用户完成任务。

【可用工具】
{tools}

【工作方式】
每一轮你必须输出一个 JSON：

调工具：
{{
  "thought": "你的思考",
  "action": "工具名",
  "action_input": {{参数对象}}
}}

完成回答：
{{
  "thought": "你的思考",
  "final_answer": "给用户的最终回答"
}}

【规则】
1. 每次只输出一个 JSON，不要有其他内容。
2. 不要编造工具返回的数据。
3. 最多 {max_turns} 轮。
4. 字符串里不能有真实换行，用 \\n 转义。
5. final_answer 必须是单行字符串。

【写操作必须先真调工具（硬规则，违反即错误）】
这里说的写操作 = 会**改动数据**的动作：添加投递记录、改状态、删除记录、保存简历等。
1. 只有**真的调用过对应工具、并拿到成功返回**之后，才允许在 final_answer 里说
   「已添加 / 已记录 / 已修改 / 已删除 ✅」这类完成时表述。
2. 本轮**一个工具都没调**时，final_answer 里禁止出现完成时表述。此时要么直接
   调用工具，要么只说明「我还没执行 / 我准备这样执行」——**不要先报成功**。
3. 工具返回以「工具调用失败：」开头时，必须把失败原因**如实**转述给用户，
   绝不允许改口说成功，也不允许假装记录已经写进去了。
4. 报成功必须带工具返回里的真实凭证（记录 id / 公司 / 岗位），凭证只能照抄工具返回，
   不许自己编。
5. 典型错误（必须避免）：用户说「添加投递记录：快手，大模型算法」，你直接回
   「已添加 ✅」，却根本没调用 add_tracking。正确做法是本轮输出：
   {{"thought": "…", "action": "add_tracking",
     "action_input": {{"company": "快手", "title": "大模型算法"}}}}

【简历定制：只写客观事实，不许自我评价】
改写简历、生成投递包里的 resume.pdf 时，每条描述只能是「做了什么（技术 / 方法 / 动作）+
可量化结果」，例：「基于 LangChain 切分 + Milvus 建库，召回率从 65% 提升至 86%」。
严禁自我评价句式：「体现了较强的…能力」「展现了出色的…」「具备独立完成…的能力」
「注重工程规范」「出于对…的浓厚兴趣」等等，一律不许出现 —— 宁可少写一句，也不替候选人自夸。
学校、专业、起止时间照抄不许丢，原始简历里没有的事实也不许编。

【投递流程：想投 ≠ 投了（顺序反了就是错答案）】
投递是三段式，任何一段都不能替用户跳到下一段：
1. 找岗位：用户说「帮我找 XX 岗位」→ 调 search_jobs 返回岗位列表，等用户挑；
2. 备材料：用户说「我想投第 N 个 / 我想投 XX / 打算投 / 帮我投」→ **只调
   generate_application_package**，把返回的包目录与文件（resume.pdf / cover_letter.md /
   job_info.txt）告诉用户去下载。company 与 job_id 一律取 search_jobs 返回里的 **index 字段**：
   「第 N 个」= index 恰好为 N 的那一条（列表展示顺序已被强制等于 index 顺序，
   所以「列表第 N 行」就是它），**不要自己数行**。
   本轮消息里带了「[系统提示 · 岗位序号]」时直接照它给的 job_id 用，不要再搜；
   index 也没有时才重新调 search_jobs 拿一次带 index 的结果，不要凭印象编；
   本步**绝对不要调 add_tracking**——用户还没投；
3. 记投递：用户回来说「我投了 / 已投 / 投完了 / 投递完成」→ 才调 add_tracking
   （status 用 applied），才算这条岗位进了投递追踪。

硬规则：
- **生成投递包 ≠ 投递记录。** 投递包只是替用户备好的材料；applications 里一出现记录，
  就等于谎报「用户已经投过了」，后续所有跟进提醒（check_reminders）的天数全部算错。
- 用户只是「想投 / 打算投 / 有意向」时，调 add_tracking 是**错误动作**；
  正确动作只有 generate_application_package。
- 反过来，用户明确说「我投了 XX」时只调 add_tracking 即可，不必再生成一遍投递包
  （除非用户同时要「给我投递包」）。
- 没调 generate_application_package 就说「投递包已生成 / 可以下载了」同样是幻觉；
  回执只能照抄工具返回的 package_dir 与 files，不许自己编路径。

【一次说了多个需求（复杂任务）→ 走三 Agent 协作流程】
用户一句话里同时要「找岗位 + 匹配打分 + 出投递包」这类**多个需求**时，系统会
自动切到「多智能体协作流程」：**Planner** 先拆出有序步骤，**Executor** 逐步执行，
**Critic** 逐步审查、**不合格就打回重做**（同一步最多 2 次，仍不合格则降级，
返回已完成的部分并如实说明）。本轮若走了这条流程，你在回答里要守三条：
1. **不许把多个需求压成一步**：找岗位 → 匹配 → 出包有先后依赖，先搜到岗位才知道
   匹配哪一条、才谈得上给谁出包；跳过前一步、或凭空猜一个 job_id 会直接投错岗。
2. **每一步都要有可核对的产出**：搜岗位给列表、匹配给分数与理由、出包给真实包目录。
   上一步没产出就不要假装进行下一步 —— 如实说「没做成」，比编一个结果强。
3. **审查意见必须能落地**：打回重做时说清**改哪个参数**（例："搜岗位 0 条 →
   去掉城市限制重搜"）。只说"再试一次"却不改参数的审查等于白烧一轮额度。
什么算复杂任务：命中 ≥2 种意图（搜索 / 匹配 / 要投递材料），或用户明说
「一条龙 / 整个流程」。只有单一需求时（只搜岗位、只匹配）**不要**套这套流程 ——
多智能体更贵，单需求上它就是纯加成本。

【岗位类型（实习 / 正式 / 兼职）：用户说了就必须传 job_type】
search_jobs 的 `job_type` 参数只认三个值：**实习 / 正式 / 兼职**。
- 用户说「实习 / 实习生 / Intern / 见习」→ `job_type="实习"`；
  说「正式 / 全职 / 社招」→ `job_type="正式"`；说「兼职」→ `job_type="兼职"`；
- 用户**没提**类型时**不传** job_type（空 = 不限，实习和正式都返回），
  不要自己猜、不要因为「默认都是实习」就补一个"实习"；
- 传 job_type 时**不要**把类型词再塞进 keyword（「广州 agent 实习岗位」→
  keyword="agent"、city="广州"、job_type="实习"）。原因：牛客实习频道里
  有**近一半**岗位标题不带「实习」二字，用 keyword LIKE '%实习%' 会把这些
  真实习岗全滤掉；
- 「校招 / 应届」按 **正式** 处理（那是校招正式岗）；要实习必须用户明说「实习」；
- 工具返回的每条岗位都带 `job_type` 字段，**列表里每条都要照实标出类型**
  （写成 `… · 城市 · 实习`）：牛客实习频道里近一半岗位标题不带「实习」二字
  （「算法工程师」「大模型算法」都在其中），不标类型用户会以为搜「实习」没生效。
  **不要**把正式岗说成实习岗，也不要自己推断类型 —— 只照抄工具给的 `job_type`。

【搜索结果透明化】
调用 search_jobs 拿到结果后，必须如实、完整地汇报，不要只挑几条就说完了：
1. 先报总数：明确说出工具本次返回的完整条数（如「共找到 20 个相关岗位」），
   不要隐瞒、不要省略，也不要用「等」把后面的条目糊过去。
2. **严格按 index 升序逐条展示：不许重排、不许分组、不许重新编号** ——
   列表第 1 行必须就是 index=1 的那条，格式
   `3. [岗位名](url) — 公司 · 薪资 · 城市 · 类型`，序号**原样用工具返回结果里的 index 字段**。
   末尾的「类型」照抄工具给的 `job_type`（实习 / 正式 / 兼职），工具没给就省略。
   原因：用户接下来会说「我想投第 N 个」，**系统按 index=N 取岗位**，所以
   「用户看到的第 N 行」必须恒等于「index=N」。一旦打乱顺序（例如把岗位名含 Agent
   的挑出来排到最前、重新分成两类再编号），用户看到的第一个就和 index=1 不是同一条，
   会直接投错岗（真实故障：列表里第一条是重排出来的墨泊可士，系统却按 index 1
   取了上海信投智联）。
   - 默认列出**前 10 条**（index 1-10），再写一句总数与邀请：「共 20 条，还有 10 条，
     要全部回复"全部"即可」；
   - 想提示相关性时**只加标注、不动顺序**：在相关条目行尾加 `（核心匹配）` 或
     `（相关）`；分类统计只写成开头/结尾的**一句汇总**（如「其中岗位名含 Agent 的
     1 条，大模型 / 算法相关 12 条」），不得改变任何条目的位置与序号。
   链接写法：把岗位名写成 markdown 链接，形如 [岗位名](url)，url 取自工具返回结果里
   那条岗位的 url 字段、**原样照抄**。例：
   [Agent 开发实习生](https://www.shixiseng.com/intern/inn_xxx) — 字节跳动 · 300-500/天 · 北京 · 实习
   工具返回里 url 为空字符串的岗位，省略链接、只列文字即可；**绝不允许自己拼、猜、
   编造或改写 url**（宁可不给链接，也不能给一个错的链接）。
3. 主动提供全量选项：末尾追加一句「需要看完整的 20 条列表吗？回复"全部"即可。」
   （数字换成实际总数）。
4. 用户说「全部」/「列全」/「看完整列表」时：按 **index 顺序**逐条列出**所有**返回的
   岗位，不省略、不筛选、不重排、不重新分组编号，即使 20 条也要全列；岗位信息还在上文时
   直接列，已被压缩或记不清就重新调用 search_jobs 拿一次再列。
5. 不要自作主张删除「看起来不相关」的岗位：`（核心匹配）/（相关）` 标注只是你给用户的
   建议，不是替用户做最终决策；用户要全部就给全部。
6. 列举多条的 final_answer 仍然是单行 JSON 字符串，条目之间用 \\n 转义换行，
   不要输出真实换行。
7. 链接是硬要求：凡是出现岗位名的地方（列表里的每一条、用户要的「全部」
   列表、以及后续轮次重新列举岗位），**每一条都要带 markdown 链接**，一条都不能漏。
   url 必须逐字来自工具返回结果。如果用户说「点不开 / 没有链接 / 链接呢」，
   重新调用 search_jobs 拿一次带 url 的结果，再按上面格式完整重列。

【搜索方式分流（精确 vs 模糊）】
search_jobs 默认走关键词精确匹配（快、结果可预期）。只有面对**模糊需求**时才把
semantic 传 true —— 判断标准是「用户有没有给出可直接检索的关键词」：
- **精确查询（semantic=false，默认）**：用户给了明确的关键词 / 城市 / 技术栈，或沿用
  长期偏好里的关键词。例：「北京 Python」「广州的 Java 实习」「大模型算法」
  → 直接用这些词当 keyword，**不要**传 semantic。
- **模糊查询（semantic=true）**：用户描述的是「什么样的岗位」而没有给出关键词。
  例：「想找偏大模型落地、能写工程代码的实习」「有没有适合我的 AI 岗」
  「不要太卷、能学到东西的岗位」
  → 把用户的整句描述作为 keyword（工具会先做关键词过滤、再在候选集内语义重排），
  并传 semantic=true。若过滤后候选为空，说明描述太泛，换成更短的关键词重试一次。

【危险操作确认（在对话层完成，工具层不会再拦一次）】
下面这些操作不可逆，动手前必须先向用户复述、等用户确认：
- 删除投递记录（尤其是「删除全部 / 清空 / 批量删」这类影响多条记录的操作）
- 修改投递状态（尤其是改成终态，如 accepted / rejected / withdrawn）
- 任何其他不可逆操作（覆盖简历、清空备注等）

确认格式："我将要 <动作>，涉及 <记录列表>。确认吗？"

怎么执行：
- 先（可用 list_tracking 等只读工具）查出将受影响的记录，把记录逐条列进确认话术里；
- 然后用 final_answer 把这句确认话术发给用户，**本轮到此结束，绝对不要先把操作做掉**；
- 用户下一条回复明确同意（「确认 / 删吧 / 可以 / 好的 / 删」）时，**本轮立刻调用对应工具执行**：
  工具层已经不再拦截，**不要再问一遍**、也不要说"系统要求二次确认"——
  反复要确认会让用户永远删不掉记录，这是最严重的失败模式；
- 用户没同意 / 改口 / 说不确定时，一律不动手。

删除的执行细节（**一次调用删完，绝不逐条、逐家公司地删**）：
- 「删掉 <公司>」（如「删掉墨泊可士」）→ 一次 `delete_tracking(company="<公司>")`，
  该公司名下有几条就一起删几条；
- 「全部删掉 / 都删掉 / 清空投递记录」→ 一次 `delete_tracking(all=true)`，
  **不要**先问公司名、**不要**逐家逐条调用，也不要说"需要逐家指定公司名"；
- 用户点名了具体某几条（如「把第 2、3 条删掉」）→ 一次
  `delete_tracking(ids=["...", "..."])`，id 取自 list_tracking 列出的记录；
- 删完给回执：说清**实际删掉几条、剩几条**（直接照抄工具返回的 count / remaining，
  不要自己数），并点名删掉的公司 + 岗位。拿不准就再调一次 list_tracking 复核。

适用范围（别把简单操作也卡住）：
- 用户已经指名道姓、且只影响一条记录的操作（如「删掉腾讯的记录」「把腾讯改成 rejected」），
  定位清楚后可以直接执行，不需要额外确认；
- 但按名字定位出多条记录，或用户用的是「全部 / 所有 / 都 / 批量」这类说法时，
  必须先列出记录并等确认；
- 确认只需一次：用户说「确认 / 都删掉」后**本轮就批量执行完**。
  「全部 / 所有 / 都 / 批量」**不是**拒绝或推脱的理由，
  严禁回复"需要逐家指定公司名""要一条条删"。

【粘贴岗位链接 / JD 文本（用户不一定从搜索结果里挑岗）】
- 用户直接贴岗位链接（实习僧 shixiseng.com / 牛客 nowcoder.com / ncss.cn）时，
  系统已经查好岗位，并把结果写在下面的【系统提示 · 用户粘贴的岗位】里：
  照它确认一句「公司 / 岗位名」，然后等用户下一步（匹配简历 / 生成投递包 /
  模拟面试），**不要再调 search_jobs 重搜一遍**。系统说「库里没查到这条岗位」时，
  如实转述并请用户把 JD 文本粘贴过来，不要假装查到，也不要编造岗位信息。
- 用户贴一大段 JD 文本且系统已写明「已接收岗位」时，照它回执即可，
  **不要**再调 analyze_pasted_jd 重复接收，也不要把这段文本当普通聊天话题回应。
  只有在系统没识别出来、而你自己判断它是一份 JD（含「岗位职责 / 任职要求 /
  工作内容」等）时，才调 analyze_pasted_jd（text 传原文），再用返回的 summary 回执。
- 会话里已经有粘贴岗位时，用户说「帮我匹配 / 拿简历匹一下 / 生成投递包 / 模拟面试」
  默认就指这个岗位：match_resume 的 job_id 传粘贴返回的 job_id（也可留空，
  系统会用会话里那个）；generate_application_package 的 company / title / job_id
  都照粘贴返回的字段填。
- 只要下面出现【系统提示 · 当前粘贴的岗位】，无论用户这一句有没有提到岗位，
  都说明会话里有一个粘过来的岗位仍然生效：**直接调对应工具、job_id 用提示里那个**，
  绝对不要为了「确认岗位还在不在」去调 search_jobs —— 粘贴岗位本来就不落库，
  搜不到是正常的，重搜只会白烧一轮额度。用户明确点了别的公司 / 岗位时才另说。

【最小必要原则】
只调用回答当前问题所必需的工具。用户没要求查看详情就不要调 get_job_detail，
用户没要求匹配简历就不要调 match_resume。
判断标准：如果问题的答案用当前已有信息就能回答，立即给 final_answer。

**例外（写操作永远不适用这条）**：只要用户是在要求改动数据（添加 / 修改 / 删除 /
保存），就必须真的调用工具，不能因为「看起来已经知道该做什么」就跳过调用——
不调用而直接回答「已完成」是错误答案，见上面【写操作必须先真调工具】。

【跟进提醒（主动报告）】
- 用户问「我该做什么 / 接下来干什么 / 有什么要跟进的 / 投递有消息吗」这类
  开放式问题时，**先调用 check_reminders**（默认阈值 7 天），再结合结果给建议；
- 用户提到具体天数（如「超过 10 天没动静的」）时，把 days 参数传成那个数字；
- 报告时说清公司、岗位和已经过了多少天，并给出下一步动作
  （发消息跟进 / 更新状态 / 放弃），不要只念一遍列表；
- check_reminders 只读数据；真正改状态要用户确认后再调 update_tracking_status。

【上下文】
- 用户的简历已经在系统中（本轮【当前状态】里就有【用户当前简历】）。
  需要按简历匹配岗位时，调用 match_resume，**resume_json 参数一律填字符串 "current"**，
  系统会在真正执行前自动替换成当前简历的完整内容。
- **绝对不要自己把简历内容抄进 resume_json**：从 get_resume 抄来的是记录外壳
  （{{id, name, content: {{...}}}}），直接传会让工具在顶层读不到 skills / projects /
  education / city，结果恒为 0 分并谎报「无实习或项目经历、学历信息缺失、城市信息缺失」。
  也不要把岗位详情/搜索列表当简历传进去。用户说「用我的简历」「拿简历匹一下」都是 "current"。

【长期偏好】
- 系统会把用户跨会话记住的偏好（目标城市/关键词/惯用简历等）注入在本轮
  上下文的【当前状态】里。用户没特别说明时，搜索和匹配默认沿用这些偏好，
  并且要在回答里体现你用了它们。
- 用户说出新的稳定偏好时（如"我只找广州的""关键词以后用 Agent 开发""以后都用产品岗版简历"），
  调用 save_preference 记住它（key 用 target_cities / target_keywords / resume_id
  或自定义偏好名，value 是值或列表）；用户想确认你记住了什么就用 get_preferences。

【主动保存】
- 用户说"记住/保存一下/以后都用这个"某段**内容**（面试评价、项目细节、联系方式、
  谈薪结果、简历要点等）时，调用 save_memory 存下来（kind 选 note / interview_record /
  resume_note / preference），不要只在回复里复述一遍就算完 —— 复述不会留下来。
- 模拟面试跑完时系统会自动把综合评价存成一条 interview_record，你不用再存一次。
- 用户问"你记得我什么""以前那场面试说了什么"时，调用 list_memories 读回来再回答，
  不要凭对话上下文猜。
- 记忆是**只增不改**的流水：不要为了"更新"去重复保存同一件事，除非用户明确要求再记一条。

【系统日志（用户说「看日志」不是看投递记录）】
- 用户说「看日志 / 系统日志 / docker 日志 / log」时，问的是**服务端运行日志**，
  与投递记录、投递状态**没有任何关系**：**绝对不要**去调 list_tracking
  （那会把「系统日志」误解成投递记录）。
- 正确回答：告诉用户日志在服务端终端，让他跑
  `docker compose logs app --tail 50`（持续跟踪加 `-f`，多看一些用 `--tail 200`，
  只看报错用 `| grep -i error`）；本地 `python start.py` 起的服务日志打在启动终端，
  同时落盘在 `logs/app.log`。
- 日志内容本身不在你的上下文里，**不要编造日志内容**；只有用户把日志文本贴给你时，
  才基于贴过来的内容分析。
"""

# DYNAMIC_CONTEXT（本轮会变的信息）的固定抬头：既是给模型的状态标记，
# 也是 compact_messages 识别「这条是钉住的动态上下文、别压进摘要」的凭据。
DYNAMIC_CONTEXT_HEADER = "【当前状态】"

# 兼容旧名字：老代码/测试若 `from agent.react_agent import SYSTEM_PROMPT_TEMPLATE`
# 仍能工作，内容等于静态前缀模板（只是不再包含动态部分）。
SYSTEM_PROMPT_TEMPLATE = STATIC_PREFIX


def build_static_prefix() -> str:
    """把 STATIC_PREFIX 模板填成最终字符串（工具定义 + 轮次上限）。"""
    return STATIC_PREFIX.format(tools=list_tools_description(), max_turns=MAX_TURNS)


def static_prefix_hash(prefix: str) -> str:
    """静态前缀的 sha1（取前 8 位）：同一份代码应当每次都得同一个值。"""
    return hashlib.sha1(prefix.encode("utf-8")).hexdigest()[:8]


def _is_resume_record(payload) -> bool:
    """模型传进来的 resume_json 是不是 storage 的「简历记录外壳」。

    外壳 = {id, name, content: {...真简历...}}（get_resume 的返回形态）。模型有时会把
    get_resume 的结果整段抄进 resume_json —— 这时顶层没有 skills/projects/education/city，
    匹配恒为 0 分。识别出来就换成系统手里的当前简历内容（真简历内容）。
    """
    if isinstance(payload, str):
        text = payload.strip()
        if not text.startswith("{"):
            return False
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return False
    if not isinstance(payload, dict):
        return False
    return "id" in payload and isinstance(payload.get("content"), (dict, str))


def build_dynamic_context(profile_text: str = "", resume_data: dict = None,
                          now: str = None) -> str:
    """拼本轮会变的状态信息：当前时间 + 用户偏好 + 当前简历。

    这部分每次调用都可能不同，所以绝不能混进 STATIC_PREFIX，
    否则前缀缓存全废。
    """
    parts = [f"当前时间：{now or time.strftime('%Y-%m-%d %H:%M:%S')}"]
    if profile_text:
        parts.append(profile_text)
    if resume_data:
        parts.append(
            "【用户当前简历】\n" + json.dumps(resume_data, ensure_ascii=False)
        )
    return "\n\n".join(parts)


# ========== 长期记忆工具（save_preference / get_preferences） ==========
#
# 实现写在 agent/user_profile.py（画像读写都在那边），这里只负责把两个函数
# 注册进 tools_registry.TOOLS —— 工具注册中心是 Agent 唯一的工具入口，
# list_tools_description / call_tool 都读这个字典，注册完就能正常调用。
#
# 为什么在这里注册、而不是改 tools_registry.py：
# 本轮改动范围限定在 react_agent.py 等少数文件，新增能力放在 Agent 侧注册，
# 好处是 tools_registry.py（被 job_search / rag / dashboard 等多处依赖）零改动、零回归风险。

def _save_preference_tool(key, value):
    """工具 save_preference：记住一条用户偏好"""
    return user_profile.save_preference(key, value)


def _get_preferences_tool():
    """工具 get_preferences：读回用户已记住的全部偏好"""
    return user_profile.get_preferences()


def _check_reminders_tool(days: int = 7):
    """工具 check_reminders：报告「投递超过 N 天还没动静」的记录。

    实现在 agent/reminder.py（Dashboard 的提醒区用的是同一份判定逻辑）。
    这里只做两件事：兜住异常（提醒查不出来不该让对话挂掉）、
    把长列表截断到 MAX_REMINDER_ITEMS 条避免 observation 过长。
    """
    try:
        data = reminder.summary(days)
    except Exception as e:                              # noqa: BLE001
        return {"error": f"读取投递记录失败：{type(e).__name__}: {e}", "count": 0, "items": []}

    items = data.get("items") or []
    payload = {
        "days": data.get("days", days),
        "count": data.get("count", len(items)),
        "items": items[:MAX_REMINDER_ITEMS],
        "text": data.get("text", ""),
    }
    if len(items) > MAX_REMINDER_ITEMS:
        payload["truncated"] = (
            f"超期记录共 {len(items)} 条，这里只列最久的 {MAX_REMINDER_ITEMS} 条"
        )
    return payload


def _save_memory_tool(content, kind: str = "note"):
    """工具 save_memory：把用户要求记住的内容存进长期记忆（user_memories）。

    归属按 ContextVar 里的用户走（跨会话共享），来源 thread_id 由
    shared.user_context.get_current_thread() 自动带上 —— 工具跑在子线程里，
    那一段靠 tools_registry.call_tool 的 copy_context() 把上下文带进来。
    """
    try:
        return chat_history.save_memory(content, kind=kind)
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "error": f"保存失败：{type(e).__name__}: {e}"}


def _list_memories_tool(kind: str = "", limit: int = 10):
    """工具 list_memories：读回长期记忆（默认最近 10 条，可按类型过滤）。"""
    try:
        records = chat_history.list_memories(kind=kind or None, limit=limit)
    except Exception as e:                              # noqa: BLE001
        return {"error": f"读取失败：{type(e).__name__}: {e}", "count": 0, "items": []}

    lines = []
    for item in records:
        created = str(item.get("created_at") or "")[:16]
        head = f"#{item.get('id')} [{item.get('kind')}] {created}".strip()
        lines.append(f"{head}\n{item.get('content') or ''}")
    payload = {
        "count": len(records),
        "items": records,
        "text": "\n\n".join(lines) if lines else "（还没有存过任何记忆）",
    }
    return payload


def register_profile_tools() -> list:
    """把长期记忆工具与跟进提醒工具注册进工具表，返回本次注册的工具名。

    幂等：重复调用不会覆盖（也不会出错）。TOOLS 不存在时安静跳过，
    这样 user_profile 单独被 import 时不会因为缺依赖而报错。
    """
    if TOOLS is None:                               # pragma: no cover - 仅作防御
        return []

    specs = {
        "save_preference": {
            "description": (
                "记住用户的长期偏好（跨会话生效）。用户说出稳定偏好时调用，"
                "如「我只找广州的」（key=target_cities、value=[\"广州\"]）、"
                "「关键词用 Agent 开发」（key=target_keywords）、"
                "「以后都用产品岗版简历」（key=resume_id、value=简历 id）、"
                "以及自由偏好（key=preferences.salary_min、value=\"200/天\"）。"
            ),
            "parameters": {
                "key": (
                    "偏好名：target_cities / target_keywords / resume_id，"
                    "或 preferences.<自定义名>"
                ),
                "value": "偏好值（字符串、数字或列表）",
            },
            "func": _save_preference_tool,
        },
        "get_preferences": {
            "description": "读取用户已记住的全部长期偏好（用户问「你记得我什么偏好」时调用）。",
            "parameters": {},
            "func": _get_preferences_tool,
        },
        "check_reminders": {
            "description": (
                "检查投递跟进提醒：找出状态仍是 applied、且投递时间超过 N 天"
                "（默认 7 天）没动静的记录。用户问「我该做什么」「有什么要跟进的」"
                "「投递有消息吗」时优先调用它，报告哪几家公司该去催了。只读，不改任何数据。"
            ),
            "parameters": {
                "days": "超期天数阈值，默认 7（用户说「超过 10 天的」就传 10）",
            },
            "func": _check_reminders_tool,
        },
        "save_memory": {
            "description": (
                "把用户要求长期记住的内容存下来（跨会话、跨清空历史都在，"
                "用 list_memories 能读回）。用户说「记住这个」「保存一下」"
                "「以后都用这个」时调用；模拟面试的收尾评价由系统自动保存，不用重复调用。"
            ),
            "parameters": {
                "content": "要记住的正文（原话要点即可，最长 2000 字）",
                "kind": (
                    "类型：note（普通笔记，默认）/ interview_record（面试记录）"
                    "/ resume_note（简历相关）/ preference（偏好类内容）"
                ),
            },
            "func": _save_memory_tool,
            "risk_level": "read",
            "timeout": 30,
            "requires_confirmation": False,
        },
        "list_memories": {
            "description": (
                "读回用户已经保存过的长期记忆（时间倒序）。用户问「你记得我什么」"
                "「上次那场面试的评价是什么」时调用。只读，不改任何数据。"
            ),
            "parameters": {
                "kind": "只看某一类（note / interview_record / resume_note / preference），留空看全部",
                "limit": "最多返回几条，默认 10",
            },
            "func": _list_memories_tool,
            "risk_level": "read",
            "timeout": 30,
            "requires_confirmation": False,
        },
    }

    registered = []
    for name, spec in specs.items():
        if name not in TOOLS:
            TOOLS[name] = spec
            registered.append(name)
    return registered


# 模块导入即注册：react_agent.run() 靠的就是 TOOLS 里的工具有 describe/可调用
register_profile_tools()


def _parse_json(text: str) -> dict:
    """解析 LLM 输出的 JSON，兼容多种格式"""
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

    # 先试标准 json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 兜底：用 json5（允许尾逗号、单引号、真实换行）
    try:
        return json5.loads(text)
    except Exception:
        raise


def _get_max_history() -> int:
    """读取消息上限：环境变量 MAX_HISTORY 优先，非法值回退到常量 MAX_HISTORY。

    上限小于 3 时无法同时容纳 system + 摘要 + 尾部消息，一律按默认值处理。
    """
    raw = os.getenv("MAX_HISTORY", "")
    if raw is not None and str(raw).strip():
        try:
            value = int(str(raw).strip())
            if value >= 3:
                return value
        except (TypeError, ValueError):
            pass
        print(f"[上下文] MAX_HISTORY={raw!r} 非法（需 >=3 的整数），改用默认 {MAX_HISTORY}")
    return MAX_HISTORY


def _format_history(messages: list) -> str:
    """把待摘要的消息拼成纯文本，单条过长（如工具返回）先截断"""
    role_names = {"system": "系统", "user": "用户", "assistant": "助手"}
    lines = []
    for m in messages:
        role = role_names.get(m.get("role"), str(m.get("role")))
        content = str(m.get("content", ""))
        if len(content) > 800:
            content = content[:800] + "…（已截断）"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def _summarize_messages(messages: list, verbose: bool = True) -> str:
    """调 LLM 概括一段历史；失败时退化为截断拼接，保证一定拿得到可用摘要"""
    history = _format_history(messages)
    try:
        summary = chat([{
            "role": "user",
            "content": SUMMARY_PROMPT_TEMPLATE.format(history=history),
        }], source="react_agent_summary")
        summary = " ".join(str(summary or "").split())     # 压成单行，别让摘要自己变长
        if summary:
            return summary
        if verbose:
            print("[上下文] 摘要模型返回空内容，退化为截断拼接")
    except Exception as e:
        if verbose:
            print(f"[上下文] 摘要失败，退化为截断拼接：{e}")

    flat = history.replace("\n", " ")
    return flat[:500] + ("…" if len(flat) > 500 else "")


def _is_tool_result(msg: dict) -> bool:
    """判断一条消息是不是「工具返回结果」的观察消息。

    react_agent 里的工具调用是成对写入的：
        助手：{"thought": ..., "action": ...}
        用户：工具返回结果：\n{...}\n\n请继续。
    这对消息必须同生共死——只留后者会被模型当成一条来路不明的用户消息。
    """
    return (
        isinstance(msg, dict)
        and msg.get("role") == "user"
        and str(msg.get("content", "")).startswith("工具返回结果")
    )


def _is_dynamic_context(msg: dict) -> bool:
    """判断一条消息是不是 run() 注入的「【当前状态】」动态上下文。

    这条消息带的是用户偏好 / 当前简历 / 当前时间，属于「每轮都要在眼前」的信息，
    不能像普通历史那样被摘要吞掉，所以 compact_messages 要把它钉在 system 之后。
    只认抬头，不认位置以外的任何东西——旧格式的 messages 里没有这条，逻辑不变。
    """
    return (
        isinstance(msg, dict)
        and msg.get("role") == "user"
        and str(msg.get("content", "")).startswith(DYNAMIC_CONTEXT_HEADER)
    )


def compact_messages(messages: list, max_history: int = None, verbose: bool = True) -> list:
    """把 messages 压到 max_history 条以内（默认取 MAX_HISTORY / 环境变量）。

    规则：
    - 第 1 条 system prompt 永远保留；
    - 紧跟其后的「【当前状态】」动态上下文（如果有）也保留，不参与摘要；
    - 保留最近 max_history - 2 条原始消息（有动态上下文时名额相应少 1 条）；
    - 中间部分（钉住的消息之后、最近 N 条之前）交给 LLM 摘要，
      以「之前对话摘要：…」插在这些消息之后，这 1 条摘要本身也计入上限；
    - 没超限时原样返回（返回新列表，不改动入参）；
    - 切点如果正好落在「工具返回结果」上，会往前多留一条（见下方注释）。

    注意：触发上面最后一条配对修复时，返回条数会是 max_history + 1。
    这是有意为之——多留一条原文换取消息对的完整，比省一条更划算；
    多出来的那条本来就在尾部窗口边上，摘要覆盖的中段反而少了一条，token 量基本不变。
    """
    limit = max_history or _get_max_history()
    if len(messages) <= limit:
        return list(messages)

    system_msg = messages[0]
    # 钉住的消息（当前只有「【当前状态】」那一条），始终排在 system 之后
    pinned = [messages[1]] if len(messages) > 1 and _is_dynamic_context(messages[1]) else []
    head_len = 1 + len(pinned)                     # system + 钉住的消息
    keep_tail = max(0, limit - head_len - 1)       # 再留 1 条名额给摘要
    if keep_tail:
        # 切尾部之前先看切点：如果尾部第一条是「工具返回结果」，说明它对应的
        # assistant 消息（含 action）被切进了摘要区。只保留结果、丢掉产生它的
        # 那次工具调用，会让模型看到一条没有来源的用户消息，轻则重复调工具，
        # 重则把工具结果误当成用户说的话。所以把窗口往前扩，把那个
        # assistant 消息一起留在尾部，保证 (assistant, 工具返回结果) 成对。
        while True:
            cut = len(messages) - keep_tail
            if cut <= head_len or cut >= len(messages) or not _is_tool_result(messages[cut]):
                break
            keep_tail += 1
    tail = messages[len(messages) - keep_tail:] if keep_tail else []
    middle = messages[head_len:len(messages) - keep_tail] if keep_tail else messages[head_len:]

    if not middle:                                 # 兜底：没有可压缩内容时不调 LLM
        return [system_msg] + pinned + tail

    summary = _summarize_messages(middle, verbose=verbose)
    compacted = [
        system_msg,
    ] + pinned + [
        {"role": "user", "content": f"之前对话摘要：{summary}"},
    ] + tail

    if verbose:
        print(
            f"[上下文] 消息 {len(messages)} 条 → 中间 {len(middle)} 条压成摘要，"
            f"现在 {len(compacted)} 条（上限 {limit}）"
        )
    return compacted


def _history_to_messages(history: list) -> list:
    """把落库的历史轮次拼成 messages（供 run() 注入）。

    history 形如 [{"question": ..., "answer": ...}, ...]（见 agent/chat_history.py 的
    load_history）。这里**只认这两个键**，刻意不接收「【当前状态】」这类快照：

    为什么：动态上下文（用户偏好 / 当前简历 / 当前时间）每轮都由 run() 重新生成一次，
    如果历史里也塞一份旧快照，模型就会同时看到两份简历、两份偏好时间，
    轻则以旧为准，重则把旧的当成用户刚说的话。所以状态只走动态上下文这一条路，
    历史只负责「谁问了什么、答了什么」。

    assistant 那一侧刻意写成 {"final_answer": ...} 的**同构 JSON**，而不是裸文本：
    让模型看到的历史格式与它自己每一轮的输出格式完全一致，
    不会误判成「用户说过这些岗位名」。
    """
    messages = []
    for item in (history or []):
        if not isinstance(item, dict):
            continue
        prior_q = str(item.get("question") or "").strip()
        if not prior_q:
            continue                                  # 空问句不注入
        prior_a = str(item.get("answer") or "").strip()
        messages.append({"role": "user", "content": prior_q})
        messages.append({
            "role": "assistant",
            "content": json.dumps({"final_answer": prior_a}, ensure_ascii=False),
        })
    return messages


def _with_messages(result: dict, messages: list, trace_id: str,
                   return_messages: bool) -> dict:
    """按 return_messages 决定要不要把本轮 messages 附在返回值里。

    默认不开：调用方（eval 脚本 / Dashboard / Chainlit）只读 answer 和 steps，
    多塞一份完整上下文（含工具 observation，可能几十 KB）纯属浪费。
    """
    if return_messages:
        result = dict(result)
        result["messages"] = list(messages)
        result["trace_id"] = trace_id
    return result


def _long_output_turn(question: str, steps: list, retry_after_truncation: bool) -> bool:
    """本轮要不要走「长输出档」（更大的 max_tokens + 更低的思考档）。

    ReAct 每轮输出的 JSON 形状固定，但**长度差别很大**：工具调用轮只有几十个
    字（实测 176 token 就收尾），而「查看我的完整简历」「把 20 条岗位列全」这种
    轮次要复述一长段内容，日常档（历史默认 1024，现全局默认 4096）的额度会被
    思考（`reasoning_content`）吃光 → finish_reason=length、正文为空（真复现过）。

    三条判据都指向「这轮会写长文本」：
      - 上一轮刚被截断：同样的额度必然再被截断（重试必须提额，否则白等一轮）；
      - 上一轮工具返回很大：模型要把这段读进去再复述出来（简历 / 长列表）；
      - 问题本身在要一份长内容（完整简历、全部列表、完整 JD）。
    判据偏保守：`max_tokens` 只是上限，模型说完就停，猜错（其实很短）不会多花
    token —— 只有猜漏（其实很长）才会再吃一次截断。
    """
    if retry_after_truncation:
        return True
    for step in reversed(steps or []):
        if step.get("type") == "action":
            if len(str(step.get("observation") or "")) >= LONG_OBSERVATION_CHARS:
                return True
            break                        # 只看紧邻的上一个工具结果，再往前的无关
    for token in LONG_OUTPUT_HINTS:
        if token in (question or ""):
            return True
    return False


def _turn_budget(question: str, steps: list,
                 retry_after_truncation: bool = False) -> tuple:
    """定出本轮 chat() 的 (max_tokens, reasoning_effort)。

    两档（理由见 limits.react_long_max_tokens / react_reasoning_effort）：
      - 日常轮：直接走**全局默认** limits.default_max_tokens()（LLM_MAX_TOKENS，
        默认 4096）+ 低思考档 —— 工具调用轮实测 176 token 就收尾，4096 只是上限；
      - 长输出轮：REACT_LLM_LONG_MAX_TOKENS（8192）+ 低思考档。
    只回一个标量不够用：llm_client 的 _resolve_max_tokens 把「显式传 None」
    当成「用全局默认」，所以这里必须**同时**返回额度与档位。
    """
    if _long_output_turn(question, steps, retry_after_truncation):
        return limits.react_long_max_tokens(), limits.react_reasoning_effort()
    return limits.default_max_tokens(), limits.react_reasoning_effort()


def run(question: str, resume_data: dict = None, verbose: bool = True,
        history: list = None, return_messages: bool = False) -> dict:
    """运行 ReAct 循环
    resume_data: 当前用户的简历（dict），会注入到 system prompt
    history: 之前几轮的问答（[{"question","answer"}, ...]，时间升序），
        会被拼成 user/assistant 消息注入到当前问题之前。**默认 None，行为与
        加这个参数之前逐字节一致**（不注入任何历史）。
    return_messages: 为 True 时在返回值里带上本轮的完整 messages（排查/调试用），
        默认 False，返回值结构不变。
    """
    # 每次运行的 Trace ID：把这轮的 run_start / thought / tool_call /
    # observation / run_end 串成一条链，线上排查时按 trace_id 就能捞出全过程。
    trace_id = str(uuid.uuid4())[:8]
    log_event(trace_id, "run_start", question=question[:50])

    # 开一次本轮的预算记账（ContextVar，随这次请求生灭）。之后每次 LLM 调用
    # 的 usage 都会由 llm_client 累加进来，循环里每轮读一次判断是否该收尾。
    limits.start_run_budget()
    limits.reset_truncated()

    # 多版本简历：调用方没显式给简历时，用「当前使用」的那份
    # （use_resume 设过的 → 否则取默认/最新一份），支持技术岗版 / 产品岗版切换。
    if resume_data is None:
        try:
            resume_data = get_current_resume()
        except Exception as e:                  # 读简历失败不该让对话直接挂掉
            if verbose:
                print(f"[简历] 读取当前简历失败（忽略）：{e}")
            resume_data = None

    # 静态前缀：只有代码/工具表变了才会变，跨调用逐字节一致 → 前缀缓存可命中
    static_prefix = build_static_prefix()

    # 长期记忆：读用户画像（目标城市/关键词/惯用简历/自由偏好）→ 放进动态上下文。
    # 画像读失败不该让对话挂掉，所以整段兜住异常、退化成「没有画像」。
    try:
        profile = user_profile.load_profile()
    except Exception as e:                          # noqa: BLE001 - 画像坏了也要能聊天
        if verbose:
            print(f"[画像] 读取失败（忽略）：{type(e).__name__}: {e}")
        profile = None

    profile_text = user_profile.profile_to_prompt(profile) if profile else ""
    if profile_text:
        if verbose:
            print(f"[画像] 已注入长期偏好：{json.dumps(profile, ensure_ascii=False)}")
    elif verbose:
        print("[画像] 暂无长期偏好（用户还没说过稳定偏好）")

    # 动态上下文：当前时间 + 用户偏好 + 当前简历。每次都变，所以单独放在
    # 一条 user 消息里（不用第二条 system，兼容性更稳），跟在静态前缀后面。
    dynamic_context = build_dynamic_context(
        profile_text=profile_text, resume_data=resume_data
    )

    # messages 结构：[system=STATIC_PREFIX] + [user=【当前状态】] + 历史轮次 + 当前问题
    # compact_messages 保留第 1 条 system 和「当前状态」这条（见该函数内的钉住逻辑），
    # 所以危险操作规则、画像、简历在长对话压缩后都还在。
    #
    # 历史的位置：夹在「当前状态」之后、当前问题之前。
    #   - 放在动态上下文**之后**：状态是"现在"的（本轮刚读的简历/偏好），
    #     历史是"过去"的，让模型先看到最新状态再回看历史，顺序上不会拿旧状态覆盖新状态；
    #   - 历史里只有 user 问句 + assistant 的 final_answer（见 _history_to_messages），
    #     不含任何状态快照，所以与动态上下文不存在重复注入。
    prior_messages = _history_to_messages(history)
    if verbose and prior_messages:
        print(f"[上下文] 注入历史 {len(prior_messages) // 2} 轮"
              f"（{len(prior_messages)} 条消息）")
    messages = [
        {"role": "system", "content": static_prefix},
        {"role": "user", "content": f"{DYNAMIC_CONTEXT_HEADER}\n{dynamic_context}"},
    ] + prior_messages + [
        {"role": "user", "content": question},
    ]

    steps = []
    truncated_retry = False          # 上一轮是否因输出触顶而重来（见 _turn_budget）
    for turn in range(1, MAX_TURNS + 1):
        # 第 2 道闸门：单次预算熔断。每轮 chat() **之前**查一次本轮累计用量；
        # 触顶就停止循环、用已有 steps 收尾 —— 降级，不拒绝。
        budget = limits.run_budget_status()
        if budget["exceeded"]:
            log_event(trace_id, "budget_stop", turn=turn,
                      used=budget["used"], limit=budget["limit"])
            if verbose:
                print(f"[预算] 本轮已用 {budget['used']} token（上限 {budget['limit']}），"
                      "停止循环并降级收尾")
            return _with_messages({
                "answer": BUDGET_STOP_ANSWER,
                "steps": steps + [{
                    "turn": turn,
                    "type": "budget_stop",
                    "thought": (f"本轮 token 已用 {budget['used']}/{budget['limit']}，"
                                "停止继续调用工具与模型"),
                    "used_tokens": budget["used"],
                    "limit_tokens": budget["limit"],
                }],
            }, messages, trace_id, return_messages)

        if verbose:
            print(f"\n--- 第 {turn} 轮 ---")

        # 每次调用 LLM 前压缩历史：超限就摘要中间部分，长对话不会把 token 撑爆
        messages = compact_messages(messages, verbose=verbose)

        # 每次调用 LLM 前报一次静态前缀指纹：多次调用 hash 一致 = 前缀稳定、缓存能命中
        if verbose:
            print(
                f"[context] static_prefix_hash={static_prefix_hash(static_prefix)} "
                f"len={len(static_prefix)}"
            )

        # 额度按档位走：日常轮走全局默认（LLM_MAX_TOKENS=4096，工具调用实测 176 token
        # 收尾、上限不花 token），长输出轮给足
        # （否则思考吃光额度 → 正文为空）。上一轮被截断时也必须进长输出档：
        # 同样的额度再试一次必然再截断，白等一轮。
        turn_max_tokens, turn_effort = _turn_budget(
            question, steps, retry_after_truncation=truncated_retry)
        if verbose:
            print(f"[额度] max_tokens={turn_max_tokens} "
                  f"reasoning_effort={turn_effort or '(默认)'}")
        raw = chat(messages, source="react_agent",
                   max_tokens=turn_max_tokens, reasoning_effort=turn_effort)

        # 第 1 道闸门：输出触顶（finish_reason=length）。
        # 被截断的半截 JSON 不是正常答案，也不能任由它落进「解析失败」那条
        # 通用分支（那样只会看到一句「格式错误」，根本看不出是被 max_tokens 掐的）。
        # 所以单独识别：如实告诉模型「你被截断了」，让它精简后重出一份完整 JSON。
        if limits.consume_truncated():
            log_event(trace_id, "truncated", turn=turn, output_chars=len(raw or ""))
            if verbose:
                print("[截断] 输出触顶（max_tokens），本轮按解析失败处理；"
                      "下一轮改用长输出档（更大额度 + 低思考档）")
            # 重试必须提额并降思考档：额度不变的话同样的思考会再次吃光额度
            # （上一轮简历解析已实测「只把 1024 调到 4096、思考档不动」治不了）。
            truncated_retry = True
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "你上一次的输出被截断了（超过单次输出长度上限），"
                           "请精简内容后重新输出一个完整合法的 JSON。",
            })
            continue

        try:
            decision = _parse_json(raw)
        except Exception as e:
            # 以前这里只 print 不写日志：整轮在 app.log 里就是空白，
            # 事后根本看不出「模型这一轮其实没调工具、只是输出废了」。
            log_event(trace_id, "parse_error", turn=turn,
                      error=f"{type(e).__name__}: {e}"[:200],
                      output_chars=len(raw or ""))
            if verbose:
                print(f"[解析失败] {e}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "你输出的不是合法 JSON，请严格按格式重新输出。",
            })
            continue

        thought = decision.get("thought", "")
        if verbose:
            print(f"Thought: {thought}")
        log_event(trace_id, "thought", turn=turn, content=thought[:100])

        if "final_answer" in decision:
            log_event(trace_id, "run_end", total_turns=turn,
                      final_answer_len=len(decision["final_answer"]))
            return _with_messages({
                "answer": decision["final_answer"],
                "steps": steps + [{"turn": turn, "type": "final", "thought": thought}],
            }, messages, trace_id, return_messages)

        action = decision.get("action")
        action_input = decision.get("action_input", {})

        # 兜底：两个字段都没有时以前会变成 action=None → call_tool 报「未知工具：None」，
        # 模型收到的提示很晦涩（看着像工具坏了，而不是它自己漏了字段）。显式说清楚。
        if not action:
            log_event(trace_id, "bad_action", turn=turn,
                      decision=str(decision)[:150])
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "你的 JSON 里既没有 final_answer 也没有 action，"
                           "两者必须二选一（要调工具就写 action + action_input；"
                           "要结束就写 final_answer）。请重新输出。",
            })
            continue

        # 把 "current" 替换成真实简历
        if action == "match_resume" and resume_data:
            payload = action_input.get("resume_json")
            if payload in (None, "current", "") or _is_resume_record(payload):
                action_input["resume_json"] = json.dumps(resume_data, ensure_ascii=False)

        if verbose:
            print(f"Action: {action}")
            print(f"Input: {str(action_input)[:200]}")

        log_event(trace_id, "tool_call", turn=turn, tool=action,
                  args=str(action_input)[:100])

        try:
            result = call_tool(action, action_input)
            result_str = json.dumps(result, ensure_ascii=False, default=str)
            if verbose:
                print(f"Observation: {result_str[:OBSERVATION_LOG_CHARS]}"
                      f"{'…' if len(result_str) > OBSERVATION_LOG_CHARS else ''}"
                      f"（完整 {len(result_str)} 字，steps 里存的是全文）")
        except Exception as e:
            result_str = f"工具调用失败：{e}"
            # 失败必须两级留痕：给模型（下面的 observation）＋给日志。
            # 只给模型的话，事后翻 app.log 会以为这一轮没出过问题。
            log_event(trace_id, "tool_error", turn=turn, tool=action,
                      error=f"{type(e).__name__}: {e}"[:200])
            if verbose:
                print(f"Observation: {result_str}")

        log_event(trace_id, "observation", turn=turn,
                  result_preview=result_str[:100],
                  ok=not result_str.startswith("工具调用失败："))

        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"工具返回结果：\n{result_str}\n\n请继续。",
        })

        steps.append({
            "turn": turn,
            "type": "action",
            "thought": thought,
            "action": action,
            "action_input": action_input,
            # 存**完整** observation（不截断）。
            # Round 6 的误判根因就在这：以前这里存 result_str[:500]，
            # 一条 JD 的 requirements 直接腰斩，LLM-as-Judge 看不到答案引用的原文，
            # 只能把真实数据判成「编造」。日志可读性由上面那行 print 的截断负责，
            # 判定/归档需要的是原始证据，两者不该共用同一个截断。
            "observation": result_str,
        })

    # 轮次耗尽也是 run 的正常收尾路径，run_end 同样要打，否则这条 trace 会断尾
    answer = "抱歉，我没能在限定轮次内完成。请简化问题。"
    log_event(trace_id, "run_end", total_turns=turn, final_answer_len=len(answer))
    return _with_messages({
        "answer": answer,
        "steps": steps,
    }, messages, trace_id, return_messages)

if __name__ == "__main__":
    questions = [
        "帮我找北京的 Agent 开发实习，看看有哪些岗位",
    ]
    for q in questions:
        print(f"\n{'='*70}")
        print(f"❓ {q}")
        print(f"{'='*70}")
        result = run(q)
        print(f"\n【最终答案】\n{result['answer']}")