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

【搜索结果透明化】
调用 search_jobs 拿到结果后，必须如实、完整地汇报，不要只挑几条就说完了：
1. 先报总数：明确说出工具本次返回的完整条数（如「共找到 20 个相关岗位」），
   不要隐瞒、不要省略，也不要用「等」把后面的条目糊过去。
2. 分两类展示：
   - 核心匹配（岗位名含 Agent / 智能体 / LLM Agent）：逐条列出，最多 8 条，
     每条含公司 + 岗位名 + 薪资 + 城市；
   - 相关岗位（大模型 / 算法 / AI 应用，但不是 Agent）：列 3-5 条示例 + 总数，
     如「另有 12 个大模型 / 算法相关岗位」。
3. 主动提供全量选项：末尾追加一句「需要看完整的 20 条列表吗？回复"全部"即可。」
   （数字换成实际总数）。
4. 用户说「全部」/「列全」/「看完整列表」时：逐条列出**所有**返回的岗位，
   不省略、不筛选、不再分核心与相关，即使 20 条也要全列；岗位信息还在上文时
   直接列，已被压缩或记不清就重新调用 search_jobs 拿一次再列。
5. 不要自作主张删除「看起来不相关」的岗位：分类只是你给用户的建议，
   不是替用户做最终决策；用户要全部就给全部。
6. 列举多条的 final_answer 仍然是单行 JSON 字符串，条目之间用 \\n 转义换行，
   不要输出真实换行。

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

【危险操作确认】
下面这些操作不可逆，动手前必须先向用户复述、等用户确认：
- 删除投递记录（尤其是「删除全部 / 清空 / 批量删」这类影响多条记录的操作）
- 修改投递状态（尤其是改成终态，如 accepted / rejected / withdrawn）
- 任何其他不可逆操作（覆盖简历、清空备注等）

确认格式："我将要 <动作>，涉及 <记录列表>。确认吗？"

怎么执行：
- 先（可用 list_tracking 等只读工具）查出将受影响的记录，把记录逐条列进确认话术里；
- 然后用 final_answer 把这句确认话术发给用户，**本轮到此结束，绝对不要先把操作做掉**；
- 等用户下一条消息明确同意后，才调用对应工具执行；
- 用户没同意 / 改口 / 说不确定时，一律不动手。

适用范围（别把简单操作也卡住）：
- 用户已经指名道姓、且只影响一条记录的操作（如「删掉腾讯的记录」「把腾讯改成 rejected」），
  定位清楚后可以直接执行，不需要额外确认；
- 但按名字定位出多条记录，或用户用的是「全部 / 所有 / 都 / 批量」这类说法时，
  必须先列出记录并等确认；
- 一次只能删一家公司的记录，所以「删除全部」要跟用户说明需要逐家指定公司名。

【最小必要原则】
只调用回答当前问题所必需的工具。用户没要求查看详情就不要调 get_job_detail，
用户没要求匹配简历就不要调 match_resume。
判断标准：如果问题的答案用当前已有信息就能回答，立即给 final_answer。

【跟进提醒（主动报告）】
- 用户问「我该做什么 / 接下来干什么 / 有什么要跟进的 / 投递有消息吗」这类
  开放式问题时，**先调用 check_reminders**（默认阈值 7 天），再结合结果给建议；
- 用户提到具体天数（如「超过 10 天没动静的」）时，把 days 参数传成那个数字；
- 报告时说清公司、岗位和已经过了多少天，并给出下一步动作
  （发消息跟进 / 更新状态 / 放弃），不要只念一遍列表；
- check_reminders 只读数据；真正改状态要用户确认后再调 update_tracking_status。

【上下文】
- 用户的简历已经在系统中，当用户提到"我的简历"或需要匹配时，
  请使用 match_resume 工具，resume_json 参数填 "current"（系统会自动替换）。

【长期偏好】
- 系统会把用户跨会话记住的偏好（目标城市/关键词/惯用简历等）注入在本轮
  上下文的【当前状态】里。用户没特别说明时，搜索和匹配默认沿用这些偏好，
  并且要在回答里体现你用了它们。
- 用户说出新的稳定偏好时（如"我只找广州的""关键词以后用 Agent 开发""以后都用产品岗版简历"），
  调用 save_preference 记住它（key 用 target_cities / target_keywords / resume_id
  或自定义偏好名，value 是值或列表）；用户想确认你记住了什么就用 get_preferences。
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

        raw = chat(messages, source="react_agent")

        # 第 1 道闸门：输出触顶（finish_reason=length）。
        # 被截断的半截 JSON 不是正常答案，也不能任由它落进「解析失败」那条
        # 通用分支（那样只会看到一句「格式错误」，根本看不出是被 max_tokens 掐的）。
        # 所以单独识别：如实告诉模型「你被截断了」，让它精简后重出一份完整 JSON。
        if limits.consume_truncated():
            log_event(trace_id, "truncated", turn=turn, output_chars=len(raw or ""))
            if verbose:
                print("[截断] 输出触顶（max_tokens），本轮按解析失败处理")
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

        # 把 "current" 替换成真实简历
        if action == "match_resume" and resume_data:
            if action_input.get("resume_json") in (None, "current", ""):
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
            if verbose:
                print(f"Observation: {result_str}")

        log_event(trace_id, "observation", turn=turn,
                  result_preview=result_str[:100])

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