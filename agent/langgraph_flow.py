"""
LangGraph 工作流定义（阶段 1：固定基础流程 + 反思节点）。

为什么单独一个文件：`agent/react_agent.py` **保持原样做兜底**（万一 LangGraph 版
有问题，把环境变量 `AGENT_ENGINE=react` 一设就切回去）。这里只放「图」本身 ——
State / 节点函数 / 边 / 编译好的图；对外入口在 `agent/react_agent_lg.py`。

两张图
------
1. **搜岗位（把「模型自决」换成「流程固定」）**::

       [接收] → [提取参数] → [搜岗位] → [筛选] → [返回列表]

   旧版这条链路是「模型自己决定调哪个工具、调几次、怎么排版」：同一句话可能
   搜两遍、可能把列表重排后重新编号（真实故障：用户看到的第 1 行 ≠ index=1，
   于是投错岗）。现在每一步都是确定的函数，输出与旧版逐字同格式。

2. **匹配 + 反思（本项目核心亮点）**::

       [接收] → [定位岗位] → [匹配打分] → [反思] ─(合理)────────→ [输出]
                                            └─(不合理, ≤2 次)─→ [修正] → 重新打分

   反思节点输入「简历 + 岗位 + 打分结果」，评估「这个分数合理吗」，
   输出 `{合理, 理由, 建议修正}`。典型场景：简历里没有任何 RAG 经验、
   岗位 JD 却把 RAG 列为硬性要求，打分却给了 90 分 → 反思节点标记
   「分数可能虚高」并给出下调建议，图走「重新分析」边回去重打（最多 2 次）。

反思节点的判据是**两层**的：一层是确定性的「岗位硬性要求 vs 简历证据」覆盖检查
（事实层，不依赖模型），一层是 LLM 的语义判断（润色理由、判断分寸）。
两层都为「虚高」时才下调，避免模型一句话就把合理分数改掉。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent import tools_registry as reg
from shared import limits
from shared.job_type import (detect_query_type, normalize_job_type,
                             strip_query_type_words)
from shared.llm_client import chat
from shared.logger import log_event


# --------------------------------------------------------------------------
# 常量 / 小工具
# --------------------------------------------------------------------------

#: 搜索默认条数（与旧版 search_jobs 工具默认值一致，保证新旧版返回同样的列表长度）
DEFAULT_LIMIT = 20
#: 列表默认展示条数：与 DEFAULT_LIMIT 对齐（先展示前 20 条，其余回「全部」）。
#: 注意：这只控制**展示多少条**，不控制"搜到多少条"——真实总数由
#: `true_total` / `true_by_city` 单独统计并如实报告（见 respond_search）。
DISPLAY_LIMIT = 20
#: 多城市展示名额下限：每个城市至少先给这么多条，单城不足时余量转给别的城市。
MIN_PER_CITY_DISPLAY = 10
#: 反思最多触发几次「重新分析 → 重新打分」
MAX_REFLECTION_RETRIES = 2
#: 打高分却存在「岗位硬性要求完全没在简历里出现」时的虚高警戒线
INFLATION_SCORE = int(os.getenv("LG_REFLECT_INFLATION_SCORE", "85") or 85)
#: 关键项大面积缺失（覆盖 < 50%）时降到的低警戒线（甲：判据按证据强度分档）
INFLATION_LINE_LOW = int(os.getenv("LG_REFLECT_INFLATION_SCORE_LOW", "70") or 70)
#: 收敛条件（乙）：建议修正的绝对值小于它就认为分数已经稳定，不再反思重打
CONVERGENCE_DELTA = int(os.getenv("LG_REFLECT_CONVERGENCE_DELTA", "5") or 5)
#: 关键项覆盖率高于它 → 用高警戒线；否则用低警戒线
HARD_COVERAGE_OK = 0.5

#: 城市池（与 config/scraping.yaml 的 cities 保持同口径，另留常用城市做识别）
CITY_POOL = (
    "广州", "深圳", "北京", "上海", "杭州", "成都",
    "武汉", "南京", "西安", "长沙", "重庆", "天津",
    "苏州", "厦门", "珠海", "东莞", "佛山", "合肥", "郑州", "青岛",
)

#: 从用户问题里剥掉的口语填充词（确定性兜底路径用）
_FILLERS = (
    "帮我找一下", "帮我搜一下", "帮我看看", "帮我找", "帮我搜", "帮我查", "帮忙找",
    "帮我投简历", "帮我投递", "帮我投", "我想投递", "我想投", "我要投递", "我要投",
    "打算投", "想投",
    "帮我", "帮忙", "给我", "我想找", "我想", "我想要", "我要", "找一下", "搜一下",
    "查一下", "找找", "搜索", "查找", "看看", "有没有", "哪里有", "推荐", "相关",
    # 并列城市之间的连词：「北京和广州的 Agent 岗位」摘掉城市后还剩「和」，
    # 不摘就会变成关键词 "和 Agent"（当作两个词 AND），一条都搜不到。
    "以及", "和", "或", "与",
    "一些", "几个", "份", "个", "的", "岗位", "职位", "实习", "工作", "机会", "呗", "吧", "呀",
)

#: 英文常见虚词（从 JD 里抽「硬性要求」时排除，避免把 and / with 当技术要求）
_EN_STOPWORDS = frozenset("""
a an the and or but with for you your our we they it its this that these those as at by from
in on of to is are be been being will would shall should can could may might must have has had
do does did not no nor if then than so such very more most other others any all each both
job jobs role position candidate candidates intern internship work working experience year years
team teams good strong familiar knowledge understand understanding ability skills
skill plus preferred requirement requirements responsibility responsibilities
""".split())

#: 中文技术名词（JD 里出现、但简历里可能完全没写的「硬性要求」候选）
_CN_TECH_TERMS = (
    "大模型", "向量检索", "向量数据库", "知识库", "微调", "提示词", "智能体",
    "多模态", "机器学习", "深度学习", "推荐系统", "搜索引擎", "分布式", "高并发",
    "数据挖掘", "数据分析", "强化学习", "模型部署", "推理优化", "检索增强",
)

#: 「硬性要求」的措辞标记：出现这些词的句子里的技术项才算**关键项**（甲的依据）
_HARD_MARKERS = (
    "必须", "精通", "熟练掌握", "熟练使用", "熟练", "掌握", "熟悉", "具备",
    "要求", "硬性", "至少", "会用", "有…经验", "有经验", "相关经验",
)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
# 至少 3 个字符：2 个字符的 "ai" / "os" 这类短词几乎出现在每份 JD 里，
# 却几乎不可能出现在简历里，会把「虚高」判成常态（误报）。
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#._\-]{2,}")
_ORDINAL_HINT_RE = re.compile(r"\[系统提示 · 岗位序号\][^\n]*?index=(\d+)")
_JOB_ID_HINT_RE = re.compile(r"job_id=([A-Za-z0-9_\-]+)")
#: 系统提示块（`[系统提示 · xxx]` 到下一个空行/行尾）：只给「意图识别 / 定位岗位」用，
#: **不能进检索与生成** —— 它让问句从 11 字变 59 字，越过查询理解的长度门槛后被重写成
#: 子查询、走 `_retrieve_merged` 的**岗位级去重**，同一条 JD 只留一个 chunk（实测只剩
#: 「岗位职责」那段），答案里引用的「任职要求」原文没有证据可核 → 被判「无依据」。见
#: `answer_followup`。
_HINT_BLOCK_RE = re.compile(r"\[系统提示 · [^\]]*\][^\n]*(\n\s*)*")


def _clamp(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, value))


def _parse_json_loose(raw: str) -> dict:
    """容错解析 LLM 输出：剥 ```json 围栏，再退一步取第一个平衡的 {...}。"""
    text = (raw or "").strip()
    text = _JSON_FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:                                   # noqa: BLE001 - 走下面的兜底
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError(f"没有 JSON 对象：{text[:120]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(text[start:i + 1])
                if isinstance(data, dict):
                    return data
                break
    raise ValueError(f"JSON 解析失败：{text[:120]}")


def _llm_json(messages: list, source: str, verbose: bool = False) -> dict:
    """一次结构化 LLM 调用。

    **必须带额度与思考档**：`glm-5.3-flash` 是思考模型，`reasoning_content`
    与正文共用 `max_tokens`，当年不传就是全局默认 1024（现全局兜底已抬到 4096
    + low）—— 真实简历下思考照样能把额度吃掉大半，所以这里不依赖兜底，
    直接复用 ReAct 的「长输出档」getter：`RATE_LIMIT_ENABLED=false` 时它
    返回 0/""，payload 形状与改造前一致。
    """
    raw = chat(
        messages,
        source=source,
        max_tokens=limits.react_long_max_tokens(),
        reasoning_effort=limits.react_reasoning_effort(),
    )
    if verbose:
        print(f"[{source}] {(raw or '')[:200]}")
    return _parse_json_loose(raw)


def _step(state: dict, node: str, thought: str,
          payload: dict = None, observation: str = "") -> list:
    """追加一条执行步骤。

    `type` 故意用 `"action"`：agent/app.py 的 `_format_step_log` 只渲染
    `type == "action"` 的步骤，复用它能**不改 app.py 一行**就把 LangGraph
    的执行过程显示进 CoT 面板；`node` 字段标明这是图节点而不是 ReAct 轮次。
    """
    steps = list(state.get("steps") or [])
    steps.append({
        "turn": len(steps) + 1,
        "type": "action",
        "engine": "langgraph",
        "node": node,
        "thought": thought,
        "action": node,
        "action_input": payload or {},
        "observation": observation,
    })
    return steps


def _progress(state: dict, message: str) -> None:
    """给「这一轮」推一条进度（节点开始时的「正在 XXX…」）。

    只对**慢操作**调（匹配打分 / 模糊搜索）；精确搜索、追问不调。
    `progress_cb` 由 `react_agent_lg.run` 从 app 透传进 state —— 离线单测 / 评测
    脚本里没有它，是**静默 no-op**，行为与改造前逐字一致。
    """
    cb = state.get("progress_cb") if isinstance(state, dict) else None
    if cb is None:
        return
    try:
        cb(str(message))
    except Exception as exc:                            # noqa: BLE001 - 进度是旁路
        print(f"[progress] 推送失败（忽略）：{type(exc).__name__}: {exc}")


def _resume_blob(resume_data: Any) -> str:
    """把简历压成一段可做「有没有提到某技术」判断的小写文本。"""
    if not isinstance(resume_data, dict):
        return ""
    parts = [str(resume_data.get("name") or "")]
    parts += [str(s) for s in (resume_data.get("skills") or [])]
    parts.append(str(resume_data.get("education") or ""))
    parts.append(str(resume_data.get("city") or ""))
    for group in ("educations", "experience", "projects"):
        for item in (resume_data.get(group) or []):
            if isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
    return " ".join(parts).lower()


def _resume_blob_zh(resume_data: dict) -> str:
    """给反思 prompt 用的可读版简历（保留中文，不转义）。"""
    if not isinstance(resume_data, dict):
        return "（没有简历）"
    lines = [
        f"姓名：{resume_data.get('name') or '匿名'}",
        f"技能：{'、'.join(str(s) for s in (resume_data.get('skills') or [])) or '（空）'}",
        f"学历：{resume_data.get('education') or '（空）'}",
        f"城市：{resume_data.get('city') or '（空）'}",
        f"教育经历：{json.dumps(resume_data.get('educations') or [], ensure_ascii=False)}",
        f"实习经历：{json.dumps(resume_data.get('experience') or [], ensure_ascii=False)}",
        f"项目经历：{json.dumps(resume_data.get('projects') or [], ensure_ascii=False)}",
    ]
    return "\n".join(lines)


class _SkipLLMReflect(Exception):
    """事实层已判虚高 → 反思节点跳过这一轮 LLM 复核（省一次调用，结论不变）。"""


def _job_blob(detail) -> str:
    if detail is None:
        return ""
    return " ".join([
        str(getattr(detail, "title", "") or ""),
        str(getattr(detail, "requirements", "") or ""),
        str(getattr(detail, "description", "") or ""),
    ])


def _collect_jd_terms(text: str) -> list:
    """从一段 JD 文本里抽技术项（ASCII 技术词 + 中文技术名词），保序去重。"""
    terms: list = []
    for token in _ASCII_TOKEN_RE.findall(text or ""):
        low = token.lower()
        if low in _EN_STOPWORDS or low in terms:
            continue
        terms.append(low)
    for term in _CN_TECH_TERMS:
        if term in (text or "") and term not in terms:
            terms.append(term)
    return terms


def _hard_text(detail) -> str:
    """只保留「硬性要求」那部分 JD 文本（甲：关键项从这儿抽）。

    来源两处：① `requirements` 字段（JD 的「任职要求」段，整段都算硬性）；
    ② description 里带「熟悉 / 必须 / 精通 / 要求」等措辞的句子。
    两处都抽不到关键项时由调用方退回「全部技术项都算关键项」。
    """
    if detail is None:
        return ""
    parts = [str(getattr(detail, "requirements", "") or "")]
    desc = str(getattr(detail, "description", "") or "")
    for seg in re.split(r"[\n。；;!！?？]+", desc):
        if any(marker in seg for marker in _HARD_MARKERS):
            parts.append(seg)
    return " ".join(p for p in parts if p)


# ==========================================================================
# 图 1：搜岗位  [接收] → [提取参数] → [搜岗位] → [筛选] → [返回列表]
# ==========================================================================

#: 会话态里存「上一轮搜索条件」的键（修 Bug 1：筛选句丢关键词）
LAST_SEARCH_KEY = "last_search_params"


def last_search_params() -> dict:
    """读上一轮搜索条件（keyword / cities / job_type / semantic / limit）。

    为什么要有：「筛选出广州的」这种句子**只说了新条件、没说岗位方向**，
    参数提取节点（LLM）抽出来 keyword 是空串，SQL 就退化成 `city='广州'`，
    返回广州全部 400 条（里面大量「算法工程师」，与 Agent 无关）。
    上轮条件存在会话态，筛选句继承 keyword，城市 / 类型按本轮新说的覆盖。
    """
    try:
        data = reg._session_state().get(LAST_SEARCH_KEY) or {}
    except Exception:                                     # noqa: BLE001 - 会话态不可用不当错误
        return {}
    return dict(data) if isinstance(data, dict) else {}


def remember_search_params(params: dict) -> None:
    """把本轮**实际生效**的搜索条件落进会话态，供下一句筛选句继承。"""
    try:
        reg._session_state()[LAST_SEARCH_KEY] = dict(params or {})
    except Exception as exc:                              # noqa: BLE001 - 写失败不影响本轮
        print(f"[lg_search] 会话态写入失败（忽略）：{type(exc).__name__}: {exc}")


class SearchState(TypedDict, total=False):
    question: str
    history: list
    keyword: str
    #: 筛选句（「筛选出广州的」）：关键词从会话态继承，不被本轮的空抽取覆盖
    inherit_search: bool
    city: str
    cities: list
    limit: int
    semantic: bool
    job_type: str
    extract_source: str
    rows: list
    filtered: list
    dropped: int
    total: int
    #: 真实命中总数（不按 limit 截断）与分城市真实命中数：展示层如实报告用。
    #: 旧字段 `total` 是"截断后留下的条数"，混用正是「搜什么都是 20 条」的根因。
    true_total: int
    true_by_city: dict
    answer: str
    steps: list
    trace_id: str
    verbose: bool
    progress_cb: Any          # 长任务进度回调（见 `_progress`）；没有就是 None


def receive_search(state: SearchState) -> dict:
    """[接收] 归一化输入：记 trace、洗净问题文本、兜住默认值。"""
    question = (state.get("question") or "").strip()
    trace_id = state.get("trace_id") or str(uuid.uuid4())[:8]
    log_event(trace_id, "lg_node", node="receive", engine="langgraph",
              graph="search", question=question[:50])
    return {
        "question": question,
        "trace_id": trace_id,
        "limit": int(state.get("limit") or DEFAULT_LIMIT),
        "rows": [],
        "filtered": [],
        "dropped": 0,
        "steps": _step(state, "接收", f"收到问题：{question[:40]}"),
    }


#: 并列结构里的多城市切分：逗号 / 顿号 / 空格 / 「和」「或」「与」都算分隔符
_MULTI_CITY_SPLIT_RE = re.compile(r"[，,、;；/\\|~～\s]+|(?:和|或|与|以及)")
#: 城市名匹配（长名优先；池内城市无包含关系，取非重叠即可）
_CITY_RE = re.compile("|".join(CITY_POOL))


def _cities_in(text: str) -> list[str]:
    """扫出文本里出现的**全部**已知城市，按首次出现位置排序；没有就返回空列表。

    为什么按出现位置而不是按 CITY_POOL 顺序：池子是有序的，`for name in CITY_POOL`
    等于「北京优先于广州」，用户问题里先写了广州也会被北京抢走（长句里更明显），
    顺序还决定多城市合并时的展示顺序（用户写「北京，广州」就先列北京）。

    多城市（本轮 Bug）：「北京，广州」以前只被 `_first_city_in` 取走第一个城市，
    第二个整段丢掉，于是检索只按北京过滤 —— 这里是修法的第一层。
    切分只用来**判定并列**（保证与「按分隔符切段」的直觉一致），命中的城市名
    本身仍按全串扫，所以「上海的 Agent 岗位，广州的也行」这类写法也能全部认出来。
    """
    text = text or ""
    hits = [(m.start(), m.group(0)) for m in _CITY_RE.finditer(text)]
    cities: list[str] = []
    for _, name in sorted(hits):
        if name not in cities:
            cities.append(name)
    return cities


def _first_city_in(text: str) -> str:
    """扫出文本里**最早出现**的单个已知城市；没有就返回空串。

    为什么要有这个函数：模型会从长句里挑出**库外**地名（如实测的「石景山区」，
    北京的一个区）当城市，而真正的「广州」被丢掉 —— 见 extract_params 的兜底。
    """
    cities = _cities_in(text)
    return cities[0] if cities else ""


def _plural_cities(city) -> list[str]:
    """把参数里的城市统一成列表：None/"" → []；"广州" → ["广州"]；列表去重保序。

    为什么要有：`city` 是历史参数名（单值），`cities` 是本轮新增的多城市参数。
    两者在链路上并存（旧调用方只传 city、测试替身只认 city），所以入口处
    统一归一，之后一律按列表处理，不会再出现「第二个城市被无声丢掉」。
    """
    if isinstance(city, (list, tuple, set)):
        raw = [str(c or "").strip() for c in city]
    else:
        text = str(city or "").strip()
        if not text:
            return []
        # 单值里也可能塞着并列串（旧调用方直接传用户原话）：按分隔符再切一次
        raw = _MULTI_CITY_SPLIT_RE.split(text)
    out: list[str] = []
    for name in raw:
        name = str(name or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _extract_params_rules(question: str) -> dict:
    """确定性参数提取（LLM 不可用时的兜底，也是 LLM 结果的校验基准）。

    岗位类型也在这里抽：用户说「实习」就必须只给实习岗（本轮修的 bug 就是
    类型被忽略、正式岗混进结果）。类型词同时要**从关键词里摘掉** ——
    否则「广州 agent 实习岗位」会退化成关键词 LIKE '%实习%'，把牛客那批
    标题不带「实习」的实习岗（实测占 49%）全部漏掉。
    """
    text = question or ""
    cities = _cities_in(text)
    city = cities[0] if cities else ""
    limit = 50 if re.search(r"(全部|所有|列全|完整列表|都列出来)", text) else DEFAULT_LIMIT
    job_type = detect_query_type(text)

    stripped = strip_query_type_words(text)
    for name in cities:                       # 每个城市都要从关键词里摘掉，不只第一个
        stripped = stripped.replace(name, " ")
    for filler in sorted(_FILLERS, key=len, reverse=True):
        stripped = stripped.replace(filler, " ")
    stripped = re.sub(r"[\s,，。.、；;：:!！?？~～\-—]+", " ", stripped).strip()

    if stripped and len(stripped) <= 10:
        return {"keyword": stripped, "city": city, "cities": cities, "limit": limit,
                "semantic": False, "job_type": job_type}
    # 描述型需求（「想找偏大模型落地、能写工程代码的实习」）没有可直接检索的词 →
    # 交给语义路，整句当 keyword（与旧版 prompt 的语义分流规则同口径）。
    # ⚠️ 用**剥过填充词**的 stripped，不是原始 text：原句里的「帮我 / 我要投 / 广州的」
    # 这些口语成分对语义检索只有噪声（改前「我要投墨泊可士的 AI Agent 岗位」整句进
    # 向量检索，公司名被动词和「岗位」稀释掉，重排后只剩「AI」在起作用）。
    return {"keyword": stripped or text.strip(), "city": city, "cities": cities,
            "limit": limit, "semantic": True, "job_type": job_type}


_EXTRACT_PROMPT = """你是「岗位搜索」工作流的**参数提取节点**。从用户问题里抽出检索参数。

只输出一个 JSON，不要任何其他内容：
{{"keyword": "检索关键词", "cities": ["城市1", "城市2"], "limit": 20, "semantic": false, "job_type": "实习"}}

规则：
1. keyword 必须是**可直接检索**的技术词 / 岗位词（如 Agent、大模型、Java、算法工程师、产品经理），
   **只能填一个词**。把「帮我 / 找 / 广州 / 的 / 岗位」这些口语成分全部去掉。
   **不要**把「实习 / 正式 / 兼职」这类类型词放进 keyword（那是 job_type 的事）。
   **绝对不要**把用户提到的多个方向拼成一个关键词（如 "Agent RAG Milvus"）：那串字符在岗位库里
   不存在，精确检索会是 0 条。用户描述了多个方向、或整句话是「什么样的岗位」而不是一个词时，
   keyword 填**用户原句**并把 semantic 设为 true（见规则 2）。
2. 用户描述的是「什么样的岗位」而**没有给出具体关键词**时（如「想找偏大模型落地、
   能写工程代码的实习」），keyword 填**用户的整句描述**，并把 semantic 设为 true。
3. cities 是**城市数组**，只填用户明确说出的城市（如广州 / 北京）：
   - 用户说了多个城市（「北京，广州」「北京、广州、深圳」「北京和广州」用逗号 /
     顿号 / 空格 / 和 / 或 并列）时，**每个城市都要放进数组**，不能只填第一个；
   - 只说一个城市就填只有一个元素的数组；**一个都没说填空数组 []**，
     不要用"全国"，也不要把省 / 区 / 商圈当城市（如"石景山区"不是城市）。
4. limit 默认 20；用户说「全部 / 所有 / 列全」时填 50。
5. 用户只说了城市没说岗位方向时，keyword 填空串。
6. job_type 只填 **实习 / 正式 / 兼职** 三者之一：用户说要实习岗就填「实习」，
   说正式岗 / 社招就填「正式」，说兼职填「兼职」；**没提岗位类型时填空串**
   （空串 = 不限类型）。不要自己推断。

用户问题：{question}"""


def _param_cache_enabled() -> bool:
    """参数缓存开关（`LG_PARAM_CACHE=0` 关掉；离线单测可据此绕开缓存）。"""
    return str(os.getenv("LG_PARAM_CACHE", "1")).strip().lower() not in (
        "0", "false", "no", "off", "disable", "disabled")


def _param_cache_key(question: str) -> str:
    """按**用户原句 + 提取 prompt 版本**做键。

    为什么键是原句而不是关键词：Bug 2 的另一半根因就在这一步 —— 同一个问题喂给
    `glm` 思考模型，抽出来的 keyword 会漂（实测同一个问题三次里有一次把
    「偏大模型落地、能写工程代码的实习」整句留下、另两次只留「偏大模型落地…」），
    关键词一变，SQL 候选池就变，后面排序再稳也没用。把结论按原句钉住，
    同一句话在**任何对话、任何进程**里都得到同一组检索参数。
    prompt 版本与**模型名**都进哈希：改 prompt / 换模型都不会读到旧结论。
    """
    try:
        from shared.llm_client import ZHIPU_CHAT_MODEL as _model
    except Exception:                                    # noqa: BLE001
        _model = ""
    raw = hashlib.md5((_EXTRACT_PROMPT + str(_model)).encode("utf-8")).hexdigest()[:8]
    return "extract:" + raw + ":" + hashlib.md5(
        (question or "").strip().encode("utf-8")).hexdigest()


def _param_cache_get(question: str) -> dict:
    if not _param_cache_enabled() or not (question or "").strip():
        return {}
    try:
        return reg.kv_cache_get(_param_cache_key(question)) or {}
    except Exception:                                    # noqa: BLE001 - 缓存不可用就现算
        return {}


def _param_cache_set(question: str, params: dict) -> None:
    if not _param_cache_enabled() or not (question or "").strip():
        return
    try:
        reg.kv_cache_set(_param_cache_key(question), dict(params))
    except Exception as exc:                             # noqa: BLE001 - 缓存写失败不是错误
        print(f"[lg_extract_params] 参数缓存写入失败（忽略）：{type(exc).__name__}: {exc}")


def _llm_cities(data: dict, fallback: dict) -> list[str]:
    """解析模型给的 `cities`（兼容旧的单值 `city`），只保留池内城市、去重保序。

    为什么要在这里挑：模型可能给字符串（旧格式 / 不听话）、给库外地名（实测
    「石景山区」）、或干脆漏给。三种情况都不能让城市整个丢空 —— 漏给 / 全不合法
    时退回规则层扫出来的城市，库外的丢掉、真实的留下（模型给「石景山区」而用户
    问题里写着「广州」时，只有规则层认得广州）。
    """
    raw = data.get("cities")
    if isinstance(raw, str):
        raw = _MULTI_CITY_SPLIT_RE.split(raw)
    if not isinstance(raw, (list, tuple, set)):
        raw = [data.get("city")] if data.get("city") else []
    out: list[str] = []
    for item in raw:
        name = str(item or "").strip()
        # 「北京市」这类带上行政区后缀的写法归一一下，仍按池内城市校验
        if name not in CITY_POOL and name.rstrip("市") in CITY_POOL:
            name = name.rstrip("市")
        if name in CITY_POOL and name not in out:
            out.append(name)
    return out or list(fallback.get("cities") or [])


def _inherit_params(state: SearchState, fallback: dict) -> dict:
    """筛选句的参数：**关键词从会话态继承**，只把本轮新说的城市 / 类型叠上去。

    这是 Bug 1 的修法本体。「筛选出广州的」里没有岗位方向，LLM 抽出来的 keyword
    是空串，旧行为直接拿空关键词去搜 → 广州全部 400 条（含大量「算法工程师」）。
    现在：keyword 取上一轮搜索的（如 Agent），城市取本轮说的（广州），
    类型 / 条数 / 语义开关能继承的也继承，再交回同一条检索链路。
    """
    prev = last_search_params()
    keyword = _primary_keyword(str(prev.get("keyword") or state.get("keyword") or "").strip())
    cities = (_plural_cities(state.get("cities")) or _plural_cities(state.get("city"))
              or _plural_cities(fallback.get("cities")) or _plural_cities(prev.get("cities")))
    city = cities[0] if cities else ""
    job_type = (normalize_job_type(state.get("job_type")) or fallback.get("job_type")
                or normalize_job_type(prev.get("job_type")))
    semantic = bool(prev.get("semantic"))
    try:
        limit = int(state.get("limit") or prev.get("limit") or fallback["limit"])
    except (TypeError, ValueError):
        limit = fallback["limit"]
    city_text = "、".join(cities) if cities else "不限"
    return {
        "keyword": keyword,
        "city": city,
        "cities": cities,
        "limit": limit,
        "semantic": semantic,
        "job_type": job_type,
        "extract_source": "inherit",
        "steps": _step(
            state, "提取参数",
            f"筛选句 → 关键词从上一轮搜索继承「{keyword}」，城市「{city_text}」，"
            f"类型「{job_type or '不限'}」，语义重排 {'开' if semantic else '关'}"
            f"（来源：inherit）",
            payload={"keyword": keyword, "city": city, "cities": cities,
                     "limit": limit, "semantic": semantic, "job_type": job_type},
        ),
    }


_FILTER_WORDS = ("筛选", "筛出", "筛一下", "筛一遍", "筛掉", "过滤", "只要", "只看",
                 "限定", "挑出", "挑一下", "剩下")


def _is_filter_sentence(question: str) -> bool:
    """是不是「在上一轮结果上再加条件」的筛选句（与 `react_agent_lg._REFINE_WORDS` 同源）。

    只看词，不看会话态 —— 调用方自己确认「上一轮有关键词可继承」。
    """
    text = question or ""
    return any(word in text for word in _FILTER_WORDS)


_KEYWORD_SPLIT_RE = re.compile(r"[\s,，、/\\|+;；]+")


def _primary_keyword(keyword: str) -> str:
    """把继承来的关键词收窄成**可精确检索的主词**。

    为什么需要：上一轮的关键词可能是多智能体 Planner 拼出来的口语串
    （实测：「那帮我找这个类型的岗位，Agent / 智能体开发」里 Planner 给的
    keyword 是 `Agent / 智能体开发`）。精确路是 `LIKE '%整串%'`，
    `Agent / 智能体开发` 在广州只命中 **1** 条；按主词 `Agent` 才是用户要的
    「广州的 Agent 岗位」（82 条）。与 `extract_params` 对多词关键词的处理同口径：
    取首个实词，其余概念不靠 LIKE 兜（筛选句本来就是在收窄条件）。
    括号里的补充说明（「Agent（智能体）」）也一并剥掉。
    """
    text = re.sub(r"[（(][^）)]*[）)]", " ", str(keyword or ""))
    for token in _KEYWORD_SPLIT_RE.split(text):
        token = token.strip(" 　·-—~～*\"'“”‘’[]【】《》<>")
        if token:
            return token
    return str(keyword or "").strip()


def extract_params(state: SearchState) -> dict:
    """[提取参数] 按问题查缓存 → LLM 抽取 + 确定性规则兜底 + 合法性校验。"""
    question = state.get("question") or ""
    fallback = _extract_params_rules(question)
    if state.get("inherit_search"):
        # 筛选句（「筛选出广州的」）：不走 LLM 抽取，直接继承上一轮关键词
        return _inherit_params(state, fallback)
    source = "rules"
    params = dict(fallback)
    cached_params = _param_cache_get(question)
    if cached_params.get("keyword") is not None:
        params = dict(cached_params)
        source = "cache"
    else:
        try:
            # 只有走到 LLM 抽取（没有缓存）才是慢操作，才推这一条
            _progress(state, "正在理解查询…")
            data = _llm_json(
                [{"role": "user", "content": _EXTRACT_PROMPT.format(question=question)}],
                source="lg_extract_params",
                verbose=bool(state.get("verbose")),
            )
            keyword = str(data.get("keyword") or "").strip()
            try:
                limit = int(data.get("limit") or DEFAULT_LIMIT)
            except (TypeError, ValueError):
                limit = DEFAULT_LIMIT
            semantic = bool(data.get("semantic"))
            # 多城市（本轮 Bug 的修法第一层）：模型必须给出 `cities` 数组；
            # 给旧格式的单值 city / 给库外地名 / 漏给，都由 `_llm_cities` 兜到规则层。
            cities = _llm_cities(data, fallback)
            city = cities[0] if cities else ""
            # 岗位类型：模型给的先按同一套词表归一，认不出来就用规则层的结果
            # （规则层直接从原句里认「实习 / 正式 / 兼职」，比模型更不容易漏）
            job_type = normalize_job_type(data.get("job_type")) or fallback["job_type"]
            if not keyword and not city:
                keyword = fallback["keyword"]               # 两者都空 → 退回规则结果
                semantic = fallback["semantic"]
            if not semantic and len(str(keyword).split()) > 1:
                # 模型把多个方向拼成了一个 keyword（实测长句下的 "Agent RAG Milvus"）：
                # 精确路是 LIKE '%整串%'，岗位库里没有这串字 → 0 命中。改成「首词 + 语义路」：
                # 既保住一个可精确检索的主词，又让整句进查询理解（重写 + 多子查询），
                # 其余概念不会被丢掉。单关键词的题完全不受影响（split 后长度为 1）。
                keyword = str(keyword).split()[0]
                semantic = True
            # 关键词里若还残留类型词（模型没听话），摘干净，避免 LIKE 把牛客那批
            # 标题不带「实习」的实习岗滤掉。
            # ⚠️ 只在**精确路**摘：语义路的 keyword 是用户整句描述
            # （「想找偏大模型落地、能写工程代码的实习」），摘词会破坏语义。
            if not semantic:
                keyword = strip_query_type_words(keyword).strip()
            limit = max(1, min(50, limit))
            params = {"keyword": keyword, "city": city, "cities": cities, "limit": limit,
                      "semantic": semantic, "job_type": job_type}
            source = "llm"
        except Exception as e:                              # noqa: BLE001 - 抽取失败不该让对话挂掉
            if state.get("verbose"):
                print(f"[lg_extract_params] 退回规则提取：{type(e).__name__}: {e}")
        # 只在**真的调过 LLM 并成功**时落缓存（退回规则 / 临时代理的兜底结果不落，
        # 否则一次网络抖动会把「规则兜底」钉死 24 小时）
        if source == "llm":
            _param_cache_set(question, params)

    # 筛选句的**第二道防线**（Bug 1 复发）：万一这轮没走到 refine 路由（上一轮是
    # 多智能体 / ReAct 搜的、或句子被别的入口塞进来），关键词会被抽成空串 ——
    # 那样 SQL 就退化成 `city='广州'`，返回广州**全库**。这里再认一次「筛选类」句子，
    # 把上一轮关键词补回来；本轮自己抽出了关键词（如「筛选出广州的 Java 岗位」）
    # 就一字不动，避免把明确换了方向的句子按上一轮关键词搜。
    if not state.get("inherit_search") and _is_filter_sentence(question) \
            and not str(params.get("keyword") or "").strip():
        prev = last_search_params()
        if str(prev.get("keyword") or "").strip():
            params = dict(params)
            params["keyword"] = _primary_keyword(prev["keyword"])
            params["semantic"] = bool(prev.get("semantic"))
            if not _plural_cities(params.get("cities")):
                params["cities"] = list(prev.get("cities") or [])
            source = f"{source}+筛选兜底"

    # `cities` 是本轮新增的多城市参数；旧缓存 / 旧调用方只有 `city`，这里补一次归一，
    # 保证「单城市」与「多城市」在下游是同一套逻辑（city 恒等 cities[0]，兼容旧字段）。
    cities = _plural_cities(params.get("cities")) or _plural_cities(params.get("city"))
    city = cities[0] if cities else ""
    city_text = "、".join(cities) if cities else "不限"

    return {
        "keyword": params["keyword"],
        "city": city,
        "cities": cities,
        "limit": params["limit"],
        "semantic": params["semantic"],
        "job_type": params.get("job_type", ""),
        "extract_source": source,
        "steps": _step(
            state, "提取参数",
            f"关键词「{params['keyword']}」，城市「{city_text}」，"
            f"类型「{params.get('job_type') or '不限'}」，"
            f"条数 {params['limit']}，语义重排 {'开' if params['semantic'] else '关'}"
            f"（来源：{source}）",
            payload={"keyword": params["keyword"], "city": city, "cities": cities,
                     "limit": params["limit"], "semantic": params["semantic"],
                     "job_type": params.get("job_type", "")},
        ),
    }


def _true_counts(keyword: str, cities: list, job_type, semantic: bool) -> tuple[dict, int]:
    """统计**真实命中数**：`({城市: 命中数}, 去重后总数)`，与 limit / 截断无关。

    为什么必须单独统计（本任务根因 3）：`limit` 之前同时充当"搜多少"和"展示多少"，
    于是命中的 116 条被截成 20 条，展示层只能看到 20，就报"共 20 条" ——
    用户以为库里只有 20 个岗位。这里用 `limit=0`（口径见
    `job_search._apply_limit`：<=0 表示不限）取**全部**命中再数，只数不展示。

    直接复用工具层的 `reg._search_summary`：图与工具层（planner 路径）必须给出
    **同一个数字**，各写一份迟早漂移。summary 内部逐城市取全量、按 job_id 去重，
    所以同一岗位被两个城市同时命中不会被算两次。
    """
    summary = reg._search_summary(keyword or "", cities or [], job_type)
    by_city = {str(k): int(v or 0) for k, v in (summary.get("by_city") or {}).items()}
    return by_city, int(summary.get("total") or 0)


def _allocate_slots(by_city: dict, display: int) -> dict:
    """把展示名额分给各城市：**以真实命中数比例为主，每城保底 `MIN_PER_CITY_DISPLAY`。**

    为什么不继续"平摊"（旧行为 base, rest = divmod(limit, n)）：平摊让
    「北京 116 + 广州 52」各拿 10 条、总数还是 20 —— 用户看到的条数与单城搜
    一模一样，双城合并反而**信息更少**（本任务现象 2）。

    为什么是"比例为主 + 保底为下限"（而不是"先各给 10 再分余量"）：
    先给保底会在名额大时把比例压扁 —— 实测 limit=100、北京 116 / 广州 52 时
    北京只拿到 79、广州 21（公平应为 69/31）。所以先把名额按真实数量比例
    （最大余数法，保证总和恰好等于名额上限）分好，再把低于保底的城市抬到保底，
    抬升的量从高于保底的城市里按比例收回。

    保底仍然保证"每个城市真的露脸"（上一轮修多城市的目的），且名额永不超该城市
    真实命中数：不会出现"要展示 10 条但只有 3 条"的空位，单城不足 10 条时
    余量自然转给另一个城市。
    """
    names = list(by_city.keys())
    total = int(display or 0)
    if not names or total <= 0:
        return {name: 0 for name in names}
    need = {name: max(0, int(by_city.get(name) or 0)) for name in names}
    if len(names) == 1:
        return {names[0]: min(total, need[names[0]])}

    # 上限：名额不可能超过命中数，也不会超过总名额
    cap = {name: min(need[name], total) for name in names}
    real_total = sum(need.values())
    if real_total <= 0:
        return {name: 0 for name in names}
    if total >= real_total:
        return dict(cap)

    floor = max(1, int(MIN_PER_CITY_DISPLAY))
    # 自适应下限：名额比城市数还少时（limit=3 查 5 个城市），保底 10 会超发，
    # 此时退化成"每个城市先到先得地拿 1 条"，总数仍严格不超过名额上限。
    floor = max(1, min(floor, total // len(names)))
    # 第 1 步：按真实比例分（最大余数法：先取整，再把差额给余数最大的城市）
    ideal = {name: total * need[name] / real_total for name in names}
    alloc = {name: min(int(ideal[name]), cap[name]) for name in names}
    left = total - sum(alloc.values())
    order = sorted(names, key=lambda n: (-(ideal[n] - int(ideal[n])), n))
    while left > 0:
        progressed = False
        for name in order:
            if left <= 0:
                break
            if alloc[name] < cap[name]:
                alloc[name] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    # 第 2 步：保底是**下限**：命中数够却分得低于保底的城市抬到保底
    for name in names:
        want = min(floor, cap[name])
        if alloc[name] < want:
            alloc[name] = want
    # 第 3 步：抬升超出的量，从不低于保底、且仍有余量的城市按比例收回
    over = sum(alloc.values()) - total
    if over > 0:
        donors = sorted((n for n in names if alloc[n] > min(floor, cap[n])),
                        key=lambda n: (-alloc[n], n))
        i = 0
        while over > 0 and donors:
            name = donors[i % len(donors)]
            lowest = min(floor, cap[name])
            if alloc[name] > lowest:
                alloc[name] -= 1
                over -= 1
            else:
                donors = [n for n in donors if alloc[n] > min(floor, cap[n])]
                if not donors:
                    break
                i = 0
                continue
            i += 1
    return alloc


def _merge_city_rows(per_city: list, limit: int) -> list:
    """把「每个城市各一份结果」轮转交错合并、按 job_id 去重、再截断到 limit。

    为什么不是简单拼起来（北京 20 条 + 广州 20 条 = 40 条）：展示层只列前
    `DISPLAY_LIMIT` 条、取前 `limit` 条，若先到先得地把北京排前面，广州那批
    即使在结果里也永远看不到 —— 用户看到的仍然是「只搜了北京」。
    所以按城市**轮转取**（北京1 广州1 北京2 广州2 …），先保证每个城市都有露脸，
    再按 publish_date 降序排在同一次轮转内做稳定排序（job_id 二级键，跨对话一致）。
    去重按 job_id：同一岗位可能被两个城市的匹配规则同时命中（「全国」/ 城市串包含）。
    """
    seen: set = set()
    merged: list = []
    for position in range(max((len(rows) for rows in per_city), default=0)):
        for rows in per_city:
            if position >= len(rows):
                continue
            row = rows[position]
            job_id = str(row.get("job_id") or "")
            key = job_id or f"{row.get('title')}|{row.get('company')}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    merged.sort(key=lambda r: (bool(r.get("publish_date")), str(r.get("publish_date") or "")),
                reverse=True)
    return merged[:limit] if limit and limit > 0 else merged


def search_jobs_node(state: SearchState) -> dict:
    """[搜岗位] 调既有的检索链路（`_search` 与旧版工具是同一份实现）。

    语义路候选为空时用关键词路重试一次：搜索接口本身没有「搜不到」的不可恢复
    错误，重试是为了不把「候选池空了」当成「真的没有岗位」。

    多城市（本轮 Bug）：`_search` 的 `city` 是**单值**、SQL 里是 `city = ?` 的等值
    过滤，所以「北京，广州」按北京查出来的 20 条全是北京的 —— 广州一条都进不来。
    这里改成**每个城市分别查、结果按城市轮转合并**（见 `_merge_city_rows`）：
      * 单城市（cities 长度 <= 1，含旧调用方只传 city）= 只调一次，行为逐字不变；
      * 多城市 = 逐城市调 `_search`（keyword / limit / semantic / job_type 一字不改
        地复用同一条链路，所以类型过滤、语义重排、结果缓存都仍然生效），
        再把 `limit` 平摊到各城市（北京 10 + 广州 10），保证每个城市都真的出现。
    """
    keyword = state.get("keyword") or ""
    # 旧入口可能只带 `city`（字符串，甚至是一整串「北京，广州」）：统一归一成列表
    cities = _plural_cities(state.get("cities")) or _plural_cities(state.get("city"))
    limit = int(state.get("limit") or DEFAULT_LIMIT)
    semantic = bool(state.get("semantic"))
    job_type = normalize_job_type(state.get("job_type"))
    rows: list = []
    error = ""
    # 【如实报告】先统计真实命中数（不截断）。limit 从此只控制"展示多少"，
    # 不再控制"搜到多少"——这是「搜什么都是 20 条」的修法本体。
    true_by_city, true_total = _true_counts(keyword, cities, job_type, semantic)
    if not cities:
        # 不限城市：真实总数就是全库命中数，不再有分城明细
        true_by_city = {}
    # 多城市按「每城保底 10 + 余量按真实数量比例」切展示名额（见 _allocate_slots），
    # 不再是旧行为的平摊（北京 10 + 广州 10 = 总 20，与单城一样多）。
    if len(cities) > 1:
        alloc = _allocate_slots(true_by_city, limit)
        per_city_limits = [(name, max(1, int(alloc.get(name) or 0))) for name in cities]
    elif cities:
        per_city_limits = [(cities[0], limit)]
    else:
        per_city_limits = [(None, limit)]
    try:
        # 模糊搜索（semantic）在这一步最慢：候选召回 + 查询理解 + 融合，
        # 中间没有数据往来就会超过 60s 前端超时；先推一条把窗口打开。
        if semantic:
            _progress(state, "正在检索岗位库（语义召回）…")
        per_city: list = []
        for name, per_limit in per_city_limits:
            got = reg._search(keyword, name, per_limit, semantic, job_type) or []
            # 语义路候选为空时用关键词路重试一次（与单城市行为一致，逐城市各试一次）
            if not got and semantic:
                got = reg._search(keyword, name, per_limit, False, job_type) or []
            per_city.append(got)
        if not any(per_city) and semantic:
            semantic = False
        rows = per_city[0] if len(per_city) == 1 else _merge_city_rows(
            per_city, max(limit, sum(n for _, n in per_city_limits)))
    except Exception as e:                              # noqa: BLE001 - 检索失败如实告知
        error = f"{type(e).__name__}: {e}"
        if state.get("verbose"):
            print(f"[lg_search] 检索失败：{error}")

    city = cities[0] if cities else None
    city_text = "、".join(cities) if cities else ""
    # 记住本轮**实际生效**的搜索条件：下一句「筛选出广州的」从这里继承关键词
    # （修 Bug 1 的另一半，见 last_search_params / _inherit_params）。
    remember_search_params({"keyword": keyword, "cities": cities, "limit": limit,
                            "semantic": semantic, "job_type": job_type})
    log_event(state.get("trace_id") or "-", "lg_node", node="search", engine="langgraph",
              graph="search", keyword=keyword, city=city_text, cities=cities,
              semantic=semantic, job_type=job_type or "", hits=len(rows),
              true_total=true_total, true_by_city=true_by_city, error=error)
    return {
        "rows": rows,
        "cities": cities,
        "semantic": semantic,
        "true_total": true_total,
        "true_by_city": true_by_city,
        "steps": _step(
            state, "搜岗位",
            f"调用 search_jobs(keyword={keyword!r}, cities={cities or '不限'!r}, "
            f"limit={limit}, semantic={semantic}, job_type={job_type or '不限'!r})，"
            f"真实命中 {true_total} 条，本轮取回 {len(rows)} 条",
            payload={"keyword": keyword, "city": city_text, "cities": cities,
                     "limit": limit, "semantic": semantic, "job_type": job_type or ""},
            observation=error or f"真实命中 {true_total} 条（取回 {len(rows)} 条）",
        ),
    }


def filter_jobs(state: SearchState) -> dict:
    """[筛选] 去重 / 丢残缺 / 城市一致性 / **重新编号**。

    为什么要重新编号：`index` 是「用户说第 N 个」的唯一坐标系（app.py 的
    `_ordinal_job_hint` 按 index 查表）。筛选丢了几条之后如果还留着旧 index，
    展示顺序与 index 就会错位 —— 那正是历史上「投错岗」的根因。
    筛完统一调 `reg._number_jobs()` 重排，会话态的 `last_job_list` 一并刷新。
    """
    rows = state.get("rows") or []
    # 多城市：只要岗位落在**任一**被请求的城市里就保留（旧行为是单城市等值比较）。
    wanted_cities = _plural_cities(state.get("cities")) or _plural_cities(state.get("city"))
    wanted_city = wanted_cities[0] if wanted_cities else ""
    seen: set = set()
    kept: list = []
    dropped = 0
    for row in rows:
        job_id = str(row.get("job_id") or "")
        title = str(row.get("title") or "").strip()
        company = str(row.get("company") or "").strip()
        city = str(row.get("city") or "").strip()
        if not job_id or job_id in seen or not title or not company:
            dropped += 1
            continue
        if wanted_cities:
            if city and city != "全国" and city not in wanted_cities:
                dropped += 1
                continue
        seen.add(job_id)
        kept.append(row)

    kept = reg._number_jobs(kept)                       # 重新编号 + 刷新会话态
    tokens = [t for t in re.split(r"[\s,，、/]+", str(state.get("keyword") or "")) if t]
    for row in kept:
        title = str(row.get("title") or "").lower()
        row["core"] = bool(tokens) and all(t.lower() in title for t in tokens)

    log_event(state.get("trace_id") or "-", "lg_node", node="filter", engine="langgraph",
              graph="search", kept=len(kept), dropped=dropped)
    return {
        "filtered": kept,
        "dropped": dropped,
        "total": len(kept),
        "steps": _step(
            state, "筛选",
            f"去重 / 去残缺 / 城市一致性校验后保留 {len(kept)} 条，丢弃 {dropped} 条",
            observation=f"保留 {len(kept)} 条",
        ),
    }


def _job_line(row: dict) -> str:
    """单条岗位的展示行，格式 = 旧版 prompt 规定 + **岗位类型**：
    `3. [岗位名](url) — 公司 · 薪资 · 城市 · 类型 （核心匹配）`

    为什么必须显式带类型：牛客实习频道里**近一半**岗位标题不带「实习」二字
    （「算法工程师」「大模型算法」「AI算法工程师」），只读标题会把它们当成正式岗。
    用户本轮报的「搜实习仍返回正式岗」，举的例子正是牛客实习频道的
    「算法工程师 — 上海信投智联科技」（该条 live 复核在 recruitType=2 实习频道，
    薪资 300-500/天）—— 链路过滤没错，错在列表没把类型显示出来。
    """
    title = str(row.get("title") or "").strip()
    url = str(row.get("url") or "").strip()
    company = str(row.get("company") or "").strip()
    salary = str(row.get("salary") or "").strip()
    city = str(row.get("city") or "").strip()
    job_type = normalize_job_type(row.get("job_type"))
    name = f"[{title}]({url})" if url else title
    tail = " · ".join(p for p in (company, salary, city, job_type) if p)
    line = f"{row.get('index')}. {name}"
    if tail:
        line += f" — {tail}"
    line += "（核心匹配）" if row.get("core") else "（相关）"
    return line


def respond_search(state: SearchState) -> dict:
    """[返回列表] 渲染最终回答。格式 = 旧版 prompt 的【搜索结果透明化】。"""
    rows = state.get("filtered") or []
    keyword = state.get("keyword") or ""
    # 多城市（本轮 Bug）：头行/空结果提示必须把**每个**城市都写出来，
    # 否则用户看到「城市「北京」」会以为系统又把广州丢了。
    cities = _plural_cities(state.get("cities")) or _plural_cities(state.get("city"))
    city = "、".join(cities)
    dropped = int(state.get("dropped") or 0)

    if not rows:
        scope = f"关键词「{keyword}」" if keyword else "当前条件"
        where = f"，城市「{city}」" if city else ""
        answer = (
            f"没有找到符合条件的岗位（{scope}{where}）。\n\n"
            "可以试试：\n"
            "- 换一个更常见的关键词（如 Agent / 大模型 / Java / 算法）\n"
            "- 去掉城市限制再搜一次\n\n"
            "（搜不到就是搜不到，我不会凭空编造岗位。）"
        )
        log_event(state.get("trace_id") or "-", "lg_node", node="respond",
                  engine="langgraph", graph="search", total=0)
        return {
            "answer": answer,
            "steps": _step(state, "返回列表", "没有命中岗位，如实告知用户",
                           observation="0 条"),
        }

    # 【如实报告】`total` 是"本轮取回/截断后还有多少条"，`true_total` 才是
    # 库里真实命中多少条。以前只报前者，于是 116 条被截成 20 条后显示
    # 「共找到 20 个相关岗位」，用户以为库里就 20 个岗位。
    total = len(rows)
    true_total = int(state.get("true_total") or 0) or total
    true_by_city = {str(k): int(v or 0) for k, v in (state.get("true_by_city") or {}).items()}
    shown = rows[:DISPLAY_LIMIT]
    detail_scope = ""
    if keyword:
        detail_scope += f"关键词「{keyword}」"
    if city:
        detail_scope += ("，" if detail_scope else "") + f"城市「{city}」"
    # 把类型过滤**写进标题**：用户说「找实习」时若结果里出现标题不带「实习」的
    # 岗位（牛客实习频道近一半如此），只有把「类型「实习」」写在头行，
    # 用户才知道这是过滤后的结果、而不是过滤没生效。
    job_type = normalize_job_type(state.get("job_type"))
    if job_type:
        detail_scope += ("，" if detail_scope else "") + f"类型「{job_type}」"
    scope = f"（{detail_scope}）" if detail_scope else ""
    if len(true_by_city) > 1:
        # 多城市：逐城市报数字 + 报总数，用户一眼看出"是不是某个城市被吞了"
        parts = "，".join(f"{name}搜到 {n} 条" for name, n in true_by_city.items())
        head = f"{parts}，共 {true_total} 条相关岗位{scope}"
    else:
        head = f"共找到 {true_total} 个相关岗位{scope}"
    if showed_only := len(shown) < true_total:
        head += f"，先展示前 {len(shown)} 条："
    else:
        head += "，已全部列出："
    if dropped:
        head += f"\n（已过滤 {dropped} 条重复 / 信息不全的岗位）"

    lines = [head, ""]
    if len(true_by_city) > 1:
        # 分城市分组展示：让"双城合并"看得出两边各有几条、分别是什么。
        # 展示名额已由 search 节点按 `_allocate_slots` 分好，这里只按城市归组。
        known = set(true_by_city)
        for name in true_by_city:
            group = [r for r in shown if str(r.get("city") or "") == name]
            if not group:
                continue
            lines.append(f"【{name} {len(group)} 条】")
            lines += [_job_line(row) for row in group]
            lines.append("")
        others = [r for r in shown if str(r.get("city") or "") not in known]
        if others:
            # 「全国」这类不属于任何被请求城市的岗位：单列，别混进城市分组里
            lines.append(f"【其他城市 {len(others)} 条】")
            lines += [_job_line(row) for row in others]
            lines.append("")
        if not any(str(r.get("city") or "") in known for r in shown) and not others:
            lines += [_job_line(row) for row in shown]
            lines.append("")
    else:
        lines += [_job_line(row) for row in shown]
        lines.append("")
    rest = true_total - len(shown)
    if rest > 0:
        lines.append(f"共 {true_total} 条，这里先列前 {len(shown)} 条，"
                     f"还有 {rest} 条，要全部回复\"全部\"即可。")
    else:
        lines.append(f"共 {true_total} 条，已全部列出。")
    lines.append("（列表序号与系统内部 index 一一对应，说「我想投第 N 个」即可直接定位。）")

    log_event(state.get("trace_id") or "-", "lg_node", node="respond",
              engine="langgraph", graph="search", total=true_total,
              shown=len(shown), true_by_city=true_by_city)
    return {
        "answer": "\n".join(lines),
        "steps": _step(state, "返回列表",
                       f"按 index 升序输出 {len(shown)}/{true_total} 条（真实命中 {true_total} 条）",
                       observation=f"真实命中 {true_total} 条，展示 {len(shown)} 条"),
    }


def build_search_graph():
    """编译搜岗位图：5 个节点 / 5 条边（含入口与 END），全程无分支。"""
    graph = StateGraph(SearchState)
    graph.add_node("receive", receive_search)
    graph.add_node("extract", extract_params)
    graph.add_node("search", search_jobs_node)
    graph.add_node("filter", filter_jobs)
    graph.add_node("respond", respond_search)
    graph.set_entry_point("receive")
    graph.add_edge("receive", "extract")
    graph.add_edge("extract", "search")
    graph.add_edge("search", "filter")
    graph.add_edge("filter", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


SEARCH_GRAPH = build_search_graph()


# ==========================================================================
# 图 2：匹配 + 反思
# [接收] → [定位岗位] → [匹配打分] → [反思] →(合理)→ [输出]
#                                   └(不合理, ≤2)→ [修正] → 重新打分
# ==========================================================================

class MatchState(TypedDict, total=False):
    question: str
    history: list
    job_id: str
    detail: Any
    resume_data: dict
    score: int
    base_score: int
    bias: int
    dimensions: dict
    gaps: list
    highlights: list
    attempts: int
    reflections: list
    reflection: dict
    location_note: str
    #: 用户**点名**的岗位在库里不存在时的如实回执（有值 → 不打分、不 fallback）。
    locate_error: str
    error: str
    answer: str
    steps: list
    trace_id: str
    verbose: bool
    progress_cb: Any          # 长任务进度回调（见 `_progress`）；没有就是 None


_REFLECT_PROMPT = """你是「匹配打分」工作流的**反思节点**。你的任务不是重新打分，
而是**审阅**下面这份打分结果是否合理，尤其是**有没有虚高**。

【简历】
{resume}

【岗位】
公司：{company}
岗位：{title}
城市：{city}
学历要求：{education}
任职要求：
{requirements}

【打分结果】
总分：{score}/100
各维度：{dimensions}
亮点：{highlights}
差距：{gaps}

【系统给的确定性证据（事实层，不要推翻）】
岗位要求中**完全没有在简历里出现**的技术项：{missing}
结论：{det_reason}

【审阅要求】
1. 逐条核对「岗位任职要求」里的硬性技术项，简历里有没有**真实证据**
   （技能列表 / 项目 / 实习里出现过才算，不能靠猜、不能靠"应该会"）。
2. 只有「岗位明确要求 → 简历完全没有 → 分数却很高」才算**虚高**，给出下调建议。
   反之，若简历证据充分而分数偏低，也可以给出**上调**建议。
3. 不要因为措辞、格式这类表面问题就判不合理；分数在合理区间内就判合理。
4. 建议修正是**调整量**（负数=应下调，正数=应上调，0=维持），绝对值不超过 40。

只输出一个 JSON，不要任何其他内容：
{{"合理": true 或 false, "理由": "一句话说明判定依据", "建议修正": 0}}"""


def receive_match(state: MatchState) -> dict:
    """[接收] 归一化输入 + 读当前简历。"""
    question = (state.get("question") or "").strip()
    trace_id = state.get("trace_id") or str(uuid.uuid4())[:8]
    log_event(trace_id, "lg_node", node="receive", engine="langgraph",
              graph="match", question=question[:50])
    # 匹配打分是长任务（实测 80s+）：第一条进度在这里推，用户立刻看到
    # 「在读简历 / 分析 JD」，之后打分、复核各一条（见 score_node / reflect_node）。
    _progress(state, "正在读简历 / 分析岗位 JD…")

    resume_data = state.get("resume_data")
    if not isinstance(resume_data, dict) or not resume_data:
        try:
            resume_data = reg.get_current_resume()
        except Exception as e:                          # noqa: BLE001 - 读不到就如实说
            if state.get("verbose"):
                print(f"[lg_match] 读取当前简历失败：{type(e).__name__}: {e}")
            resume_data = None
    has_resume = isinstance(resume_data, dict) and bool(resume_data)
    return {
        "question": question,
        "trace_id": trace_id,
        "resume_data": resume_data if has_resume else {},
        "bias": int(state.get("bias") or 0),
        "attempts": int(state.get("attempts") or 0),
        "reflections": list(state.get("reflections") or []),
        "steps": _step(
            state, "接收",
            "收到匹配请求" + ("，已载入当前简历" if has_resume
                          else "，但当前会话没有可用简历"),
        ),
    }


def _extract_job_id(question: str, job_id: str = "") -> str:
    """从显式入参 / 问题里的系统提示中拿 job_id。"""
    if job_id:
        return str(job_id).strip()
    match = _JOB_ID_HINT_RE.search(question or "")
    return match.group(1) if match else ""


# ==========================================================================
# 「用户点名了岗位」的识别（修：指定不存在的岗位时 fallback 到别的岗位）
#
# 故障：用户说「帮我匹配火星科技的量子芯片岗位」，库里没有这条，`locate_job` 却
# 一路 fallback 到「最近一次搜索的第 1 条」（墨泊可士 · AI Agent 开发），打出 83 分，
# 用户以为在评估他点名的那条 —— 这是最严重的一类幻觉（**认错了评估对象**）。
#
# 判据：只在用户**确实点了名**（说了公司名 / 岗位名）时返回非空。
# 没点名的请求（「给简历匹配打分」「第 1 个」）返回空，定位仍走会话态 / 最近一次搜索，
# 老行为一个字节都不变。
# ==========================================================================

#: 「<公司>的<岗位名>岗位」——最常见的点名句式（公司名允许空格：Mars Tech）。
_DESIGNATION_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9（）()·&\-\. ]{2,24}?)的"
    r"([\u4e00-\u9fffA-Za-z0-9+#（）()·&\-\. ]{2,30}?)(?:岗位|职位|工作|岗|JD|jd)")
#: 「<公司>的岗位」——点了公司没点岗位名（库里有多条 → 让用户挑，不替他挑）。
_COMPANY_ONLY_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9（）()·&\-\. ]{2,24}?)的(?:岗位|职位|工作|岗)")
#: 「<公司>的<岗位名>」句末写法（岗位名自带后缀，如「墨泊可士的AI大模型应用开发」）。
_TAIL_TITLE_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9（）()·&\-\. ]{2,24}?)的"
    r"([\u4e00-\u9fffA-Za-z0-9+#（）()·&\-\. ]{2,30})$")
#: 只点了岗位名、没点公司：「帮我匹配量子芯片岗位」。最少 3 字 ——
#: 「AI」这类 2 字碎片在整个库里命中几百条，拿它当"点名"只会把没点名的请求也拦下来。
_BARE_TITLE_RE = re.compile(
    r"(?:匹配|评估|对比|打分|分析)[^，。、？!！\s]{0,8}?"
    r"([\u4e00-\u9fffA-Za-z0-9+#（）()·&\-\. ]{3,24}?)(?:岗位|职位|工作|岗|JD|jd)")
#: 句首的动作 / 客套词：从公司名里剥掉（「帮我匹配火星科技」→「火星科技」）。
#: ⚠️「投」必须在这里，且**不能**吞掉后面的「递 / 了 / 个 / 份」：投递意向句
#: 「我要投墨泊可士的 AI Agent 岗位」里，公司名是「墨泊可士」，不是「投墨泊可士」；
#: 而「投递记录 / 投了 3 家」里的「投递 / 投了」是完整动词，剥成「递记录」会毁掉语义。
_LEAD_IN_RE = re.compile(
    r"^(?:帮我|帮忙|麻烦|请|给我|我要|我想|想要|我需要|然后|再|顺便|能不能|可不可以|"
    r"可以|你|评估|打分|匹配度|匹配|配对|对比|分析|看看|看下|看一下|查一下|找一下|"
    r"针对|关于|一下|的|找|搜|查|看|投(?!递|了|个|份|过|这|那|简|简历)|，|,|、|\s)+")
#: 指代词 / 对象词：这些不是公司名。
_DESIGNATION_STOPWORDS = (
    "这个", "那个", "哪个", "这些", "那些", "该", "此", "上述", "上面", "下面", "之前",
    "刚才", "我们", "我", "你", "他", "它", "本人", "自己", "全部", "所有", "任何",
    "什么", "哪家", "岗位", "这个岗位", "那个岗位", "该岗位", "此岗位", "上述岗位",
)
#: 城市只是限定词（「广州的 Agent 岗位」里的「广州」不是公司名）。
_CITY_KEYS = {
    "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "西安", "苏州",
    "长沙", "重庆", "天津", "合肥", "厦门", "郑州", "青岛", "大连", "福州", "济南",
    "昆明", "宁波", "无锡", "东莞", "佛山", "珠海", "南昌", "贵阳", "太原", "石家庄",
}
#: 出现这些字 → 这是"用户在描述需求"，不是公司名 / 岗位名（「适合我的岗位」「这个岗位」）。
_NAME_JUNK_CHARS = ("我", "你", "他", "它", "这", "那", "适合", "自己", "本人",
                    "所有", "全部", "任何", "什么", "哪些", "怎么", "如何", "简历",
                    "岗位", "职位")
#: 这些词常出现在「<公司>的X」里，但 X 不是岗位名。
_NON_TITLE_WORDS = (
    "要求", "职责", "待遇", "薪资", "信息", "内容", "详情", "情况", "介绍", "城市",
    "学历", "分数", "评价", "建议", "条件", "标准", "面试", "门槛",
)
_JUNK_TITLE_RE = re.compile(
    r"^(?:第\s*[0-9一二三四五六七八九十两]+\s*[个条份家]?|这\s*个|那\s*个|哪\s*个|"
    r"一\s*个|几\s*个|一\s*些)")


def _strip_lead_in(text: str) -> str:
    """剥掉句首的动作 / 客套词（"帮我匹配墨泊可士" → "墨泊可士"）。"""
    out = str(text or "").strip()
    while True:
        stripped = _LEAD_IN_RE.sub("", out, count=1).strip()
        if stripped == out:
            return out.strip("的 ").strip()
        out = stripped


def _norm_key(text) -> str:
    """与 `tools_registry._norm_text` 同口径（去空白 + 小写），用于文本比对。"""
    return re.sub(r"\s+", "", str(text or "")).lower()


def _is_company_like(name: str) -> bool:
    """这个字符串像不像公司名（不是指代词 / 城市 / "我的简历" / "一个后端"）。"""
    raw = _strip_lead_in(str(name or ""))
    key = _norm_key(raw)
    if len(key) < 2 or key in _CITY_KEYS:
        return False
    if key in {_norm_key(word) for word in _DESIGNATION_STOPWORDS}:
        return False
    if _JUNK_TITLE_RE.match(raw):                       # 「一个 / 第 2 条」是量词，不是公司
        return False
    return not any(word in raw for word in _NAME_JUNK_CHARS)


def _is_title_like(name: str, min_len: int = 2) -> bool:
    """这个字符串像不像岗位名（不是「这个 / 第 1 个 / 适合我的 / 要求」）。"""
    raw = _strip_lead_in(str(name or "").strip())
    key = _norm_key(raw)
    if len(key) < min_len or raw in _NON_TITLE_WORDS:
        return False
    if _JUNK_TITLE_RE.match(raw):
        return False
    if key in {_norm_key(word) for word in _DESIGNATION_STOPWORDS}:
        return False
    return not any(word in raw for word in _NAME_JUNK_CHARS)


def _designation_from_text(text: str) -> tuple:
    """抽用户点名的 (公司, 岗位名)；没点名返回 ("", "")。

    四条句式按「信息量从多到少」试：公司+岗位名+「岗位」后缀 → 只有公司 → 公司+句末
    岗位名 → 只有岗位名。公司名被判定为「不是公司」（城市 / 指代词）时**不再退回只按
    岗位名找** —— 「广州的 Agent 岗位」是一句带限定词的泛需求，不该被当成点名后报"没找到"。
    """
    raw = _HINT_BLOCK_RE.sub(" ", text or "").strip().rstrip("。！？!?.,，、 ")

    match = _DESIGNATION_RE.search(raw)
    if match and _is_company_like(match.group(1)):
        company = _strip_lead_in(match.group(1))
        title = _strip_lead_in(match.group(2))
        return company, title if _is_title_like(title) else ""
    if match:
        return "", ""                       # 公司部分不是公司名 → 不算点名

    match = _COMPANY_ONLY_RE.search(raw)
    if match and _is_company_like(match.group(1)):
        return _strip_lead_in(match.group(1)), ""

    match = _TAIL_TITLE_RE.search(raw)
    if match and _is_company_like(match.group(1)):
        company = _strip_lead_in(match.group(1))
        title = _strip_lead_in(match.group(2))
        if _is_title_like(title):
            return company, title

    bare = _BARE_TITLE_RE.search(raw)
    if bare:
        title = _strip_lead_in(bare.group(1))
        if _is_title_like(title, min_len=3):
            return "", title
    return "", ""


def _named_job_not_found_reply(state: MatchState, label: str, reason: str,
                               for_package: bool = False) -> str:
    """点名了却找不到 → 如实说"没找到"，并给出可行路径（**绝不给分数**）。

    `for_package=True` 时是**投递意向**（「我要投 XX」）的回执：收尾要说明
    「没有改去搜岗位列表」而不是「没有换别的岗位来打分」。
    """
    lines = [f"我在岗位库里没找到「{label}」这个岗位。"]
    if reason:
        lines.append(f"（{reason}）")
    lines += [
        "",
        "你可以：",
        "- 确认名称是否正确（公司名 / 岗位名任一有出入都可能对不上）",
        "- 粘贴岗位链接或 JD 文本给我（抓不到的平台也能匹配/出包）",
    ]
    try:
        has_list = bool(reg.lookup_job_ordinal(1))
    except Exception:                                   # noqa: BLE001 - 没有列表就不提
        has_list = False
    if has_list:
        lines.append("- 或从刚才搜索的列表里选一个（如「第 1 个」）")
    if for_package:
        lines += ["", "**我没有改去搜别的岗位** —— 你说的是要投这一条，"
                      "搜出一堆别的公司的岗位对你没有用。"]
    else:
        lines += ["", "**我没有替你换成别的岗位来打分** —— 那样你会以为评的是你点名的这条。"]
    return "\n".join(lines)


def locate_job(state: MatchState) -> dict:
    """[定位岗位] job_id → 序号 → **点名岗位** → 会话里正在看的岗位 → 最近一次搜索第 1 条。

    这一步是确定性的：旧版由模型自己决定匹配哪条岗位，跨轮会让模型「自己数行」，
    历史上既投错过岗也匹配错过岗。这里把优先级写死，拿不到就**如实说拿不到**。

    ⚠️ 用户点了名（说了公司 / 岗位名）时**只有精确命中才继续**：命中不了就报错返回，
    不再往下 fallback。fallback 只适用于「用户没点名」的请求（「给简历匹配打分」
    「第 1 个」）——那是"接着说当前这条"的正常上下文，不是替换用户点名的岗位。
    """
    question = state.get("question") or ""
    job_id = _extract_job_id(question, state.get("job_id") or "")
    note = ""

    ordinal = _ORDINAL_HINT_RE.search(question)
    if not job_id and ordinal:
        entry = reg.lookup_job_ordinal(int(ordinal.group(1)))
        if entry and entry.get("job_id"):
            job_id = str(entry["job_id"])
            note = f"按用户说的「第 {ordinal.group(1)} 个」定位"

    detail = None
    if job_id:
        try:
            detail = reg._resolve_match_detail(job_id)
        except Exception as e:                          # noqa: BLE001 - 落到会话态
            if state.get("verbose"):
                print(f"[lg_match] job_id={job_id} 定位失败：{e}")
            detail = None

    if detail is None:
        named_company, named_title = _designation_from_text(question)
        if named_company or named_title:
            located = {}
            try:
                located = reg.match_named_job(named_company, named_title)
            except Exception as e:                      # noqa: BLE001 - 查库失败=没找到
                located = {"found": False,
                           "reason": f"岗位库查询失败（{type(e).__name__}）"}
            if located.get("found"):
                detail = located.get("detail")
                note = located.get("note") or "按用户点名的岗位定位"
            else:
                label = f"{named_company} · {named_title}" \
                    if named_company and named_title else (named_company or named_title)
                ask = located.get("ask") or ""
                # 问题 1：同名岗位多城市都有 → 直接把问题抛给用户（不替他挑城市）
                reply = ask or _named_job_not_found_reply(
                    state, label, located.get("reason") or "")
                log_event(state.get("trace_id") or "-", "lg_node", node="locate",
                          engine="langgraph", graph="match", found=False, named=True)
                return {
                    "detail": None,
                    "job_id": "",
                    "locate_error": reply,
                    "steps": _step(state, "定位岗位",
                                   (f"用户点名的「{label}」在多个城市都有 → 先问用户投哪个"
                                    if ask else
                                    f"用户点名的「{label}」在库里没有 → 如实报错，不换别的岗位"),
                                   observation=ask or located.get("reason")
                                   or f"未找到 {label}"),
                }

    if detail is None:
        # 「用户当前在看的岗位」优先于「最近一次搜索的第 1 条」：
        # 投递包生成时会把那条岗位写进会话态 current_job（tools_registry._set_current_job），
        # 粘贴的 JD 走另一个槽位，两者取更近的那个（get_focus_job）。
        detail, source = reg.get_focus_job()
        if detail is not None:
            note = source or "会话里正在看的岗位"
    if detail is None:
        last = reg.lookup_job_ordinal(1)
        if last and last.get("job_id"):
            try:
                detail = reg._resolve_match_detail(last["job_id"])
                note = "系统按最近一次搜索的第 1 条岗位"
            except Exception:                           # noqa: BLE001
                detail = None

    if detail is None:
        log_event(state.get("trace_id") or "-", "lg_node", node="locate",
                  engine="langgraph", graph="match", found=False)
        return {
            "detail": None,
            "job_id": "",
            "steps": _step(state, "定位岗位", "没有拿到要匹配的岗位，如实告知用户",
                           observation="未定位到岗位"),
        }

    log_event(state.get("trace_id") or "-", "lg_node", node="locate",
              engine="langgraph", graph="match", found=True,
              job_id=str(getattr(detail, "job_id", "")))
    label = f"{getattr(detail, 'company', '')} · {getattr(detail, 'title', '')}"
    return {
        "detail": detail,
        "job_id": str(getattr(detail, "job_id", "") or job_id),
        "location_note": note,
        "steps": _step(state, "定位岗位", f"锁定岗位：{label}" + (f"（{note}）" if note else ""),
                       observation=label),
    }


def _default_scorer(job_id: str, resume_json) -> dict:
    """默认打分器：**直接复用工具层那条链路**（`tools_registry._match`）。

    复用而不是另写一套，是为了让 LangGraph 版和 ReAct 版对同一个「简历 + 岗位」
    给出可比的分数 —— 差异只应来自流程，不应来自两套打分实现。
    """
    return reg._match(job_id, resume_json)


def score_node_factory(scorer):
    def score_node(state: MatchState) -> dict:
        # 没定位到岗位（用户点名的岗位库里没有）→ **一个分都不打**。
        # 否则 scorer 会拿着空 job_id 走会话态兜底，把用户点名的岗位换成另一条打分。
        if state.get("detail") is None or state.get("locate_error"):
            return {
                "steps": _step(state, "匹配打分", "没有定位到岗位，不进行打分",
                               observation="跳过打分（无岗位）"),
            }
        resume_data = state.get("resume_data") or {}
        if not resume_data:
            return {
                "score": 0, "dimensions": {}, "gaps": [], "highlights": [],
                "steps": _step(state, "匹配打分", "没有简历，无法打分",
                               observation="缺少简历"),
            }
        job_id = state.get("job_id") or ""
        try:
            # 打分是最慢的一跳（读 JD + LLM 打分，实测单次 40s+）：
            # 这一条在 LLM 调用**之前**推，把最长的静默窗口盖住。
            _progress(state, "正在给简历与岗位打分…")
            result = scorer(job_id, json.dumps(resume_data, ensure_ascii=False))
        except Exception as e:                          # noqa: BLE001 - 打分失败如实告知
            log_event(state.get("trace_id") or "-", "lg_node", node="score",
                      engine="langgraph", graph="match",
                      error=f"{type(e).__name__}: {e}"[:200])
            return {
                "score": 0, "dimensions": {}, "gaps": [], "highlights": [],
                "error": f"{type(e).__name__}: {e}",
                "steps": _step(state, "匹配打分", f"打分失败：{e}",
                               observation=f"失败：{e}"),
            }

        base = int(result.get("score") or 0)
        bias = int(state.get("bias") or 0)
        score = _clamp(base + bias)
        note = f"（含反思修正 {bias:+d}）" if bias else ""
        attempt = int(state.get("attempts") or 0)
        log_event(state.get("trace_id") or "-", "lg_node", node="score",
                  engine="langgraph", graph="match", base_score=base, bias=bias,
                  score=score, attempt=attempt)
        return {
            "base_score": base,
            "score": score,
            "dimensions": result.get("dimensions") or {},
            "gaps": list(result.get("gaps") or []),
            "highlights": list(result.get("highlights") or []),
            "steps": _step(
                state, "匹配打分",
                (f"初次打分 {score}/100{note}" if not attempt
                 else f"第 {attempt} 次重新打分 {score}/100{note}"),
                payload={"job_id": job_id, "bias": bias},
                observation=json.dumps(result, ensure_ascii=False)[:300],
            ),
        }
    return score_node


def deterministic_check(resume_data: dict, detail, score: int) -> dict:
    """确定性覆盖检查：岗位**关键项**里，哪些完全没有在简历中出现。

    这是反思节点的**事实层**，不依赖模型。本轮修的是「判据太严 → 不收敛」：

    - **甲（放宽判据）**：只有 **关键项**（`requirements` 段 / 带「熟悉、必须、精通、
      要求」等硬性措辞的句子里的技术项）缺失才算数；JD 里顺带提到的非关键项缺失
      不再单独作为「虚高」理由。而且虚高要求分数**严格高于**警戒线：
      关键项覆盖 >= 50% 用 `INFLATION_SCORE`(85)，覆盖 < 50% 用
      `INFLATION_LINE_LOW`(70)；回到线下就不再判虚高（修正目标 = 回到线）。
    - 这样「判虚高 → 下调到线 → 再判」天然收敛，不需要靠重试上限兜。
    """
    jd_text = _job_blob(detail)
    hard_text = _hard_text(detail)
    resume_text = _resume_blob(resume_data)
    resume_squeezed = re.sub(r"[\s\-_.]+", "", resume_text)
    terms = _collect_jd_terms(jd_text)[:20]
    hard_all = set(_collect_jd_terms(hard_text))

    def _covered(term: str) -> bool:
        if term in resume_text:
            return True
        squeezed = re.sub(r"[\s\-_.]+", "", term)
        return bool(squeezed) and squeezed in resume_squeezed

    missing = [t for t in terms if not _covered(t)]
    # 抽不到硬性段（有些 JD 没有 requirements，描述里也没有硬性措辞）时退回全部技术项，
    # 保证事实层不会因为「认不出关键项」而整体失效。
    hard_terms = [t for t in terms if t in hard_all] or list(terms)
    hard_missing = [t for t in hard_terms if not _covered(t)]
    covered = len(hard_terms) - len(hard_missing)
    coverage = (covered / len(hard_terms)) if hard_terms else 1.0
    line = INFLATION_SCORE if coverage >= HARD_COVERAGE_OK else INFLATION_LINE_LOW
    inflation = bool(hard_missing) and score > line

    if inflation:
        reason = (f"岗位关键项（{'、'.join(hard_missing[:5])}）在简历里找不到证据，"
                  f"关键项只覆盖 {covered}/{len(hard_terms)}，"
                  f"分数 {score} 却高于这条证据支持的警戒线 {line}，分数可能虚高")
    elif hard_missing:
        reason = (f"关键项有 {len(hard_missing)} 项未在简历中体现"
                  f"（覆盖 {covered}/{len(hard_terms)}），但 {score} 分未超过警戒线 "
                  f"{line}，不再下调")
    else:
        reason = "岗位关键项在简历里都能找到对应证据"

    return {
        "terms": terms,
        "missing": missing,
        "hard_terms": hard_terms,
        "hard_missing": hard_missing,
        "covered": covered,
        "coverage": round(coverage, 3),
        "line": line,
        "inflation": inflation,
        "reason": reason,
    }


def reflect_node(state: MatchState) -> dict:
    """[反思] 评估「这个分数合理吗？」，输出 {合理, 理由, 建议修正}。

    两层判据 + 一条收敛规则（本轮修复的核心）：
      1. **事实层**（`deterministic_check`）：只有「**关键项**缺失 + 分数**高于**
         证据支持的警戒线」才判虚高（甲）；命中时无论模型怎么说都判**不合理**，
         且修正量至少把分数拉回警戒线 —— 拉回线下之后下一轮不再判虚高，
         所以**天然收敛**，不是靠重试上限硬停；
      2. **语义层**（LLM）：补事实层看不出的分寸（学历 / 城市 / 年限等），
         但只在**本轮还没改过分**（attempts == 0）时允许它单独提出一次修正（乙）；
         改过一次之后，事实层没判虚高就不再继续下调；
      3. **收敛条件**：建议修正的绝对值 < `CONVERGENCE_DELTA`(5) 时直接判合理，
         分数已经稳定，不再重打（`route_after_reflect` 里还有一道同样的闸门）。
    """
    detail = state.get("detail")
    if detail is None or state.get("locate_error"):
        # 没岗位就没什么可反思的 —— 这一步以前也会白调一次 LLM（占输出耗时的大头）。
        return {"steps": _step(state, "反思", "没有岗位，跳过反思（不烧 LLM 轮次）",
                               observation="跳过反思（无岗位）")}
    resume_data = state.get("resume_data") or {}
    score = int(state.get("score") or 0)
    attempts = int(state.get("attempts") or 0)
    det = deterministic_check(resume_data, detail, score)

    reason = det["reason"]
    delta = 0
    reasonable = not det["inflation"]
    llm_used = False
    converged_by = ""
    # 第 2 道闸门在这里也生效：单次预算用尽时不再多打一次 LLM 反思，
    # 直接用确定性证据下结论（与 react_agent 的「降级不拒绝」同口径）。
    #
    # 第 3 道（省一次 LLM）：事实层已判虚高时，下面那个分支会**直接覆盖**模型的
    # 「合理 / 建议修正」——模型这一轮改不了任何结论，只影响理由措辞。这种情况
    # 白等一次 LLM（输出慢的主因），直接跳过。
    _progress(state, "正在复核打分是否合理…")
    budget = limits.run_budget_status()
    try:
        if det["inflation"]:
            raise _SkipLLMReflect("事实层已判定虚高，LLM 复核改变不了结论")
        if budget.get("exceeded"):
            raise RuntimeError(
                f"本轮 token 预算已用尽（{budget.get('used')}/{budget.get('limit')}）")
        data = _llm_json(
            [{"role": "user", "content": _REFLECT_PROMPT.format(
                resume=_resume_blob_zh(resume_data),
                company=getattr(detail, "company", ""),
                title=getattr(detail, "title", ""),
                city=getattr(detail, "city", ""),
                education=getattr(detail, "education", ""),
                requirements=(getattr(detail, "requirements", "") or "")[:2000],
                score=score,
                dimensions=json.dumps(state.get("dimensions") or {}, ensure_ascii=False),
                highlights=json.dumps(state.get("highlights") or [], ensure_ascii=False),
                gaps=json.dumps(state.get("gaps") or [], ensure_ascii=False),
                missing="、".join(det["hard_missing"][:10]) or "（无）",
                det_reason=det["reason"],
            )}],
            source="lg_reflect",
            verbose=bool(state.get("verbose")),
        )
        llm_used = True
        llm_reasonable = bool(data.get("合理", True))
        llm_reason = str(data.get("理由") or "").strip()
        try:
            delta = int(data.get("建议修正") or 0)
        except (TypeError, ValueError):
            delta = 0
        delta = max(-40, min(40, delta))
        if det["inflation"]:
            # 事实层已判定虚高 → 不允许被"合理"的措辞翻盘；修正量至少要把分数
            # 拉回警戒线（回到线下就不再判虚高 ⇒ 收敛），但也不会比警戒线更低。
            reasonable = False
            required = max(CONVERGENCE_DELTA, score - int(det["line"]))
            step = max(required, abs(delta) if delta < 0 else 0)
            delta = -min(40, step)
            reason = f"{det['reason']}；{llm_reason}" if llm_reason else det["reason"]
        elif llm_reasonable:
            reasonable = True
            delta = 0
            reason = llm_reason or det["reason"]
        elif abs(delta) < CONVERGENCE_DELTA:
            # 乙：模型想改，但幅度已经小于收敛阈值 → 接受当前分
            converged_by = f"调整量 {abs(delta)} < {CONVERGENCE_DELTA}"
            reasonable = True
            delta = 0
            reason = (f"{llm_reason}（但建议修正小于收敛阈值 "
                      f"{CONVERGENCE_DELTA}，按收敛条件接受当前分数）"
                      if llm_reason else det["reason"])
        elif attempts == 0:
            # 首次反思：允许模型基于事实层看不到的维度（学历 / 城市 / 年限）提出一次修正
            reasonable = False
            reason = llm_reason or det["reason"]
        else:
            # 甲 + 乙：已经改过一次，事实层又没判虚高 → 不再无限 -10
            reasonable = True
            delta = 0
            converged_by = "已修正过一次且事实层未判虚高"
            reason = (f"{llm_reason}（但关键项覆盖与分数已匹配、本轮已修正过一次，"
                      f"按收敛条件接受当前分数）" if llm_reason else det["reason"])
    except _SkipLLMReflect as e:                        # 事实层已定论：省掉这一轮 LLM
        if state.get("verbose"):
            print(f"[lg_reflect] 跳过 LLM 复核（省一轮）：{e}")
        required = max(CONVERGENCE_DELTA, score - int(det["line"]))
        delta = -min(40, required)
        reasonable = False
        reason += "（事实层已判定虚高、直接给出下调量，LLM 复核改变不了结论 → 本轮跳过 LLM）"
    except Exception as e:                              # noqa: BLE001 - 模型不可用走事实层
        if state.get("verbose"):
            print(f"[lg_reflect] LLM 反思不可用，只用确定性证据：{type(e).__name__}: {e}")
        if det["inflation"]:
            required = max(CONVERGENCE_DELTA, score - int(det["line"]))
            delta = -min(40, required)
        else:
            delta = 0
        reasonable = not det["inflation"]
        reason += f"（LLM 反思不可用，仅确定性证据：{type(e).__name__}）"

    reflection = {
        "合理": bool(reasonable),
        "理由": reason,
        "建议修正": 0 if reasonable else int(delta),
        "证据": det,
        "llm_used": llm_used,
        "复核分数": score,
        "警戒线": int(det["line"]),
        "收敛依据": converged_by,
    }
    log_event(state.get("trace_id") or "-", "lg_node", node="reflect",
              engine="langgraph", graph="match", score=score,
              line=int(det["line"]), reasonable=reflection["合理"],
              delta=reflection["建议修正"], converged_by=converged_by,
              hard_missing="、".join(det["hard_missing"][:5]), llm_used=llm_used)
    return {
        "reflection": reflection,
        "reflections": list(state.get("reflections") or []) + [reflection],
        "steps": _step(
            state, "反思",
            f"分数 {score} 复核（警戒线 {det['line']}，关键项覆盖 "
            f"{det['covered']}/{len(det['hard_terms'])}）："
            f"{'合理' if reflection['合理'] else '不合理'} —— {reason}"
            + (f"［收敛判据：{converged_by}］" if converged_by else ""),
            payload={"score": score, "missing": det["hard_missing"][:5],
                     "line": det["line"], "llm_used": llm_used},
            observation=reason,
        ),
    }


def revise_node(state: MatchState) -> dict:
    """[修正] 走「重新分析」边：把反思给的下调量累进 bias，然后回到 [匹配打分]。

    `attempts` 每进这里一次 +1，由 `route_after_reflect` 卡在
    `MAX_REFLECTION_RETRIES` 次以内 —— 防的是「反思说不行 → 改 → 还说不行」的
    死循环，也防 token 被无限烧掉。
    """
    reflection = state.get("reflection") or {}
    delta = int(reflection.get("建议修正") or 0)
    bias = int(state.get("bias") or 0) + delta
    attempts = int(state.get("attempts") or 0) + 1
    log_event(state.get("trace_id") or "-", "lg_node", node="revise",
              engine="langgraph", graph="match", delta=delta, bias=bias,
              attempts=attempts)
    return {
        "bias": bias,
        "attempts": attempts,
        "steps": _step(
            state, "修正",
            f"按反思建议调整 {delta:+d} 分（累计 {bias:+d}），回到「匹配打分」重新分析"
            f"（第 {attempts}/{MAX_REFLECTION_RETRIES} 次）",
            payload={"delta": delta, "bias": bias},
        ),
    }


def reflection_converged(state: MatchState) -> str:
    """乙（收敛条件）：分数已经稳定就不该再走「重新分析」边。

    两个判据，任一成立即认为收敛：
      1. 本轮反思给的调整量存在且绝对值 < `CONVERGENCE_DELTA`；
      2. 已经复核过两次，且最近两次复核的分数变化 < `CONVERGENCE_DELTA`
         （"连续两次修正后分数变化 <5 就接受当前分"）。
    返回空串表示未收敛。
    """
    reflections = list(state.get("reflections") or [])
    latest = (reflections[-1] if reflections else state.get("reflection")) or {}
    raw_delta = latest.get("建议修正")
    if raw_delta is not None:
        try:
            if abs(int(raw_delta)) < CONVERGENCE_DELTA:
                return f"调整量 {abs(int(raw_delta))} < {CONVERGENCE_DELTA}"
        except (TypeError, ValueError):
            pass
    if len(reflections) >= 2:
        try:
            now = int(reflections[-1].get("复核分数") or 0)
            prev = int(reflections[-2].get("复核分数") or 0)
        except (TypeError, ValueError):
            return ""
        if abs(now - prev) < CONVERGENCE_DELTA:
            return f"连续两次复核分数变化 {abs(now - prev)} < {CONVERGENCE_DELTA}"
    return ""


def route_after_reflect(state: MatchState) -> str:
    """反思之后走哪条边：合理 / 已收敛 → 输出；否则还有重试额度 → 重新分析。"""
    reflection = state.get("reflection") or {}
    if reflection.get("合理", True):
        return "respond"
    if reflection_converged(state):
        return "respond"
    if int(state.get("attempts") or 0) >= MAX_REFLECTION_RETRIES:
        return "respond"
    return "revise"


def respond_match(state: MatchState) -> dict:
    """[输出] 最终回答：分数 + 维度 + 亮点/差距 + **反思自评日志**。"""
    detail = state.get("detail")
    if state.get("locate_error"):
        # 用户点名的岗位不存在 → 如实告知，**不给任何分数**（这是本轮的修复点）。
        return {"answer": state["locate_error"],
                "steps": _step(state, "输出", "点名的岗位没找到，如实告知（不打分）",
                               observation="未找到用户点名的岗位")}
    if detail is None:
        answer = (
            "我还不知道要匹配哪个岗位。请先让我搜一个岗位"
            "（例如「帮我找广州的 Agent 岗位」），或者把岗位链接 / JD 文本粘贴过来，"
            "然后说「帮我匹配简历」。"
        )
        return {"answer": answer,
                "steps": _step(state, "输出", "没有岗位，给出获取岗位的指引",
                               observation="缺少岗位")}

    if not state.get("resume_data"):
        answer = (
            "我这边还没有你的简历，没法打分。\n\n"
            "可以用 `/resume 你的简历文本`，或者直接上传 PDF / Word 简历附件，"
            "再回来说「帮我匹配简历」。"
        )
        return {"answer": answer,
                "steps": _step(state, "输出", "没有简历，给出设置简历的指引",
                               observation="缺少简历")}

    if state.get("error"):
        answer = (f"匹配打分没能完成：{state['error']}\n\n"
                  "可以稍后再试一次；如果一直失败，请把简历文本重新发我一份。")
        return {"answer": answer,
                "steps": _step(state, "输出", "打分失败，如实告知",
                               observation=state["error"])}

    score = int(state.get("score") or 0)
    base = int(state.get("base_score") or score)
    bias = int(state.get("bias") or 0)
    dims = state.get("dimensions") or {}
    reflections = list(state.get("reflections") or [])
    note = state.get("location_note") or ""

    lines = [
        f"## 匹配打分：{score}/100",
        "",
        f"**岗位**：{getattr(detail, 'company', '')} · {getattr(detail, 'title', '')}"
        f"（{getattr(detail, 'city', '') or '城市未标注'}）"
        + (f"　—　{note}" if note else ""),
        "",
    ]
    if dims:
        lines.append("**各维度**：" + " · ".join(f"{k} {v}" for k, v in dims.items()))
        lines.append("")
    if state.get("highlights"):
        lines.append("**匹配亮点**：")
        lines += [f"- {h}" for h in state["highlights"]]
        lines.append("")
    if state.get("gaps"):
        lines.append("**差距**：")
        lines += [f"- {g}" for g in state["gaps"]]
        lines.append("")

    if reflections:
        lines.append("### 🔍 反思节点自评")
        for i, item in enumerate(reflections, start=1):
            verdict = "✅ 合理" if item.get("合理") else "⚠️ 不合理（分数可能虚高）"
            lines.append(f"{i}. 复核分数 {item.get('复核分数')} → {verdict}")
            lines.append(f"   - 理由：{item.get('理由')}")
            if not item.get("合理"):
                lines.append(f"   - 建议修正：{int(item.get('建议修正') or 0):+d} 分")
        lines.append("")
        if bias:
            lines.append(f"**最终分数已按反思建议修正**：初次 {base} 分 → {bias:+d} → "
                         f"**{score} 分**（共重新分析 {state.get('attempts', 0)} 次）。")
        else:
            lines.append(f"**最终分数**：{score} 分（反思节点认可，未做修正）。")
        lines.append("")

    answer = "\n".join(lines).rstrip()
    log_event(state.get("trace_id") or "-", "lg_node", node="respond",
              engine="langgraph", graph="match", score=score,
              attempts=state.get("attempts", 0))
    return {
        "answer": answer,
        "steps": _step(state, "输出",
                       f"输出打分 {score}/100，反思 {len(reflections)} 次",
                       observation=f"score={score}"),
    }


def build_match_graph(scorer=None):
    """编译「匹配 + 反思」图。

    scorer 可注入（默认走 `tools_registry._match`，与旧版工具同一条链路）——
    单测里用假打分器把分数钉在 90 分，才能稳定复现「简历没 RAG、JD 要 RAG、
    却给了 90 分」的虚高场景，验证反思节点真的能识别。
    """
    scorer = scorer or _default_scorer
    graph = StateGraph(MatchState)
    graph.add_node("receive", receive_match)
    graph.add_node("locate", locate_job)
    graph.add_node("score", score_node_factory(scorer))
    graph.add_node("reflect", reflect_node)
    graph.add_node("revise", revise_node)
    graph.add_node("respond", respond_match)
    graph.set_entry_point("receive")
    graph.add_edge("receive", "locate")
    graph.add_edge("locate", "score")
    graph.add_edge("score", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect,
                                {"revise": "revise", "respond": "respond"})
    graph.add_edge("revise", "score")
    graph.add_edge("respond", END)
    return graph.compile()


MATCH_GRAPH = build_match_graph()


# ==========================================================================
# 图 3：追问岗位（Bug 1 修复）
# [接收] → [定位岗位] → [读 JD + 生成回答（带引用）]
#
# 故障：用户看完列表后问「第 1 个岗位要求什么技术？」——
# `is_search_intent` 只看「岗位」这类名词 + 长度兜底就判成「要列表」，于是**又搜一遍**，
# 用户的问题（第 1 个要什么技术）一个字都没回答。
#
# 这条图与搜索图的根本区别：**不重新检索**。岗位从会话态 `last_job_list` 里按序号取
# （`reg.lookup_job_ordinal`，与「我想投第 3 个」同一条确定性链路），
# 正文从库里按 job_id 读，回答交给 LLM 生成并挂 [n] 引用（复用 RAG 层引用溯源）。
# ==========================================================================

class FollowupState(TypedDict, total=False):
    question: str
    history: list
    job_id: str
    detail: Any
    location_note: str
    hits: list
    answer: str
    steps: list
    trace_id: str
    verbose: bool
    #: 追问是快操作（实测 4s），**故意不推**进度 —— 字段只为与另外两张图同形
    progress_cb: Any


def receive_followup(state: FollowupState) -> dict:
    """[接收] 归一化输入 + 记 trace（与搜索图的 receive 同形，只换 graph 名）。"""
    question = (state.get("question") or "").strip()
    trace_id = state.get("trace_id") or str(uuid.uuid4())[:8]
    log_event(trace_id, "lg_node", node="receive", engine="langgraph",
              graph="followup", question=question[:50])
    return {
        "question": question,
        "trace_id": trace_id,
        "steps": _step(state, "接收", f"收到追问：{question[:40]}（不重新搜索）"),
    }


#: 命中这些词说明用户在问「这条岗位要什么 / 干什么」——追问的分析对象
_ANALYSIS_WORDS = (
    "要求", "技术要求", "技能", "技术栈", "需要", "职责", "做什么", "干什么",
    "负责", "会什么", "内容", "详情", "介绍", "怎么样", "是什么", "怎么投",
    "多少", "薪资", "待遇", "学历", "门槛", "加分", "出勤", "实习时长",
)


def _strip_hints(text: str) -> str:
    """去掉注入的 ``[系统提示 · ...]`` 块，只留用户真正说的那句话。

    追问链路里系统提示是**给定位岗位用的**（`locate_job` 已经消费掉），但它会把
    问句从 11 字撑到 59 字：越过查询理解的门槛 → 触发重写 + 子查询 →
    `_retrieve_merged` 按 job_id 去重，同一条 JD 只留一个 chunk。于是「任职要求」
    那段永远进不了证据集，答案里逐字引用的句子没有可比对的原文，被判「无依据」。
    """
    return _HINT_BLOCK_RE.sub("", str(text or "")).strip()


def _followup_target_from_prompt(text: str, job_id: str = "") -> str:
    """把「第 N 个」翻成 job_id：显式 job_id > 系统提示里的 index=N > 序号提示。"""
    if job_id:
        return str(job_id).strip()
    match = _JOB_ID_HINT_RE.search(text or "")
    if match:
        return match.group(1)
    ordinal = _ORDINAL_HINT_RE.search(text or "")
    if ordinal:
        entry = reg.lookup_job_ordinal(int(ordinal.group(1)))
        if entry and entry.get("job_id"):
            return str(entry["job_id"])
    return ""


def _jd_blob_zh(detail) -> str:
    """JD 全文（给 LLM 当唯一依据）：公司 / 岗位 / 城市 / 薪资 / 学历 / 职责 / 要求。"""
    if detail is None:
        return ""
    parts = [
        f"公司：{getattr(detail, 'company', '') or '（未知）'}",
        f"岗位：{getattr(detail, 'title', '') or '（未知）'}",
        f"城市：{getattr(detail, 'city', '') or '（未知）'}",
        f"薪资：{getattr(detail, 'salary', '') or '（未写）'}",
        f"学历：{getattr(detail, 'education', '') or '（未写）'}",
        f"链接：{getattr(detail, 'url', '') or '（无）'}",
    ]
    desc = str(getattr(detail, "description", "") or "").strip()
    req = str(getattr(detail, "requirements", "") or "").strip()
    bonus = str(getattr(detail, "bonus", "") or "").strip()
    if desc:
        parts.append(f"岗位职责：\n{desc}")
    if req:
        parts.append(f"任职要求：\n{req}")
    if bonus:
        parts.append(f"加分项：\n{bonus}")
    return "\n".join(parts)


#: 结构性句子（小标题 / 客套收尾）——它们**不是事实断言**，不该进「⚠️无依据」名单。
#: `split_sentences` 按换行切句，所以这些句子能逐字对上；归一化只去空白与标点，
#: 长度 <= 14 的门槛保证「技术要求」这类小标题命中、而正文长句不会被误排。
_STRUCTURAL_LINES = (
    "技术要求", "技术要求如下", "岗位职责", "任职要求", "加分项", "岗位要求",
    "职责要求", "技能要求", "学历要求", "薪资待遇", "依据来自这条jd",
    "依据来自这条岗位jd", "依据来自该岗位jd", "以上依据来自这条岗位的jd",
    "以上依据来自岗位jd", "以上依据均来自该岗位的jd", "以下依据来自该岗位的jd",
    "以上信息来自该岗位的jd", "依据这条岗位的jd", "依据来自上面这条岗位的jd",
)
_JUNK_RE = re.compile(r"[\s，。、：:；;！!？?（）()\[\]【】]+")


def _followup_is_junk(text: str) -> bool:
    """小标题 / 客套收尾句（`⚠️无依据` 名单只该放真正的事实断言）。"""
    squeezed = _JUNK_RE.sub("", str(text or "")).replace("JD", "jd").lower()
    return len(squeezed) <= 14 and squeezed in _STRUCTURAL_LINES


_ANSWERING_PROMPT = """你是「岗位解读助手」。用户刚看完一份岗位列表，现在针对**其中一条**追问。

只依据下面这份 JD 原文回答，**不要引入 JD 里没有的信息**（不要编造技术栈、薪资、学历）。
JD 里没写到的，直接说「JD 里没有写」。

写法：
1. 第一行先点明你回答的是哪个岗位（公司 · 岗位名），**只写这一行、不加句号**；
2. 再分点列出用户问的内容（技术要求 / 职责 / 薪资 / 学历等），每点都是 JD 里的原话或紧贴原话的概括；
3. **不要**写「依据来自这条 JD」这类收尾说明（系统会自动附来源表）。
4. 直接输出回答正文，不要输出 JSON、不要写「根据你提供的资料」这类客套话。

用户的问题：{question}

JD 原文：
{jd}"""


def answer_followup(state: FollowupState) -> dict:
    """[读 JD + 生成回答] 按 job_id 读正文 → LLM 生成答案 → 挂 [n] 引用。

    引用走 RAG 层现成的链路（`rag.citation`）：
      `retrieve(追问原句, allowed_job_ids=[job_id])` 只在这条岗位的 chunk 里召回，
      再用 `build_citation_pack` 做句级引用标注 + faithfulness 核验。
    检索/引用任何一步失败都只降级成「无引用的回答」，不能把追问带崩。
    """
    detail = state.get("detail")
    job_id = str(state.get("job_id") or getattr(detail, "job_id", "") or "")
    # 用户的原始问句（去系统提示）：检索 / 生成 / 引用标注**都用它** —— 见 `_strip_hints`。
    question = _strip_hints(state.get("question") or "")
    label = f"{getattr(detail, 'company', '')} · {getattr(detail, 'title', '')}"

    jd = _jd_blob_zh(detail)
    if not jd:
        if state.get("locate_error"):
            return {"answer": state["locate_error"],
                    "steps": _step(state, "生成回答", "点名的岗位没找到，如实告知",
                                   observation="未找到用户点名的岗位")}
        return {
            "answer": "我没能读到这条岗位的 JD 正文，换个说法再试一次（或把 JD 文本粘贴给我）。",
            "steps": _step(state, "生成回答", "JD 正文为空，如实告知用户",
                           observation="无 JD 正文"),
        }

    hits = []
    try:
        # 候选池给大一点（8）：追问已经用 job_id 限定了岗位，多取几条只是让
        # 「任职要求 / 岗位职责」两段都能进证据集，避免答案引用了却没有可比对的原文。
        hits = reg._retrieve(question, top_k=8, allowed_job_ids=[job_id]) or []
    except Exception as e:                              # noqa: BLE001 - 引用是增强
        if state.get("verbose"):
            print(f"[lg_followup] 取引用片段失败（不影响回答）：{type(e).__name__}: {e}")
        hits = []

    try:
        answer = str(chat(
            [{"role": "user", "content": _ANSWERING_PROMPT.format(question=question, jd=jd)}],
            source="lg_followup_answer",
            max_tokens=limits.react_long_max_tokens(),
            reasoning_effort=limits.react_reasoning_effort(),
        ) or "").strip()
    except Exception as e:                              # noqa: BLE001 - 生成失败如实说
        log_event(state.get("trace_id") or "-", "lg_node", node="followup_answer",
                  engine="langgraph", graph="followup", error=f"{type(e).__name__}: {e}"[:200])
        return {
            "answer": f"读到了岗位「{label}」，但生成回答失败（{type(e).__name__}）。请再试一次。",
            "steps": _step(state, "生成回答", f"LLM 生成失败：{type(e).__name__}",
                           observation=f"失败：{e}"),
        }
    if not answer:
        answer = "\n".join(l for l in (jd.splitlines()) if l.strip())

    cited = False
    if hits:
        try:
            from rag.citation import (build_citation_pack, coverage_ok, enabled,
                                      render_citation_block)

            if enabled() and coverage_ok():
                pack = build_citation_pack(answer, hits)
                answer = str(pack.get("answer_with_citations") or answer).strip()
                # faithfulness 的「无依据句子」提示与搜索路同一口径（纯文本通道用 ⚠️ 代替标红）,
                # 但**小标题 / 客套收尾**不是事实断言（如第一行「公司 · 岗位名」、
                # 「技术要求：」），把它们列进「无依据」是误报 —— 这里先滤掉。
                report = dict(pack.get("faithfulness") or {})
                report["unsupported"] = [
                    r for r in (report.get("unsupported") or [])
                    if not _followup_is_junk(r.get("text"))
                ]
                tail = render_citation_block({"answer_with_citations": "",
                                              "faithfulness": report,
                                              "sources": pack.get("sources") or {},
                                              "sentences": pack.get("sentences") or []})
                if tail.strip():
                    answer = f"{answer}\n\n{tail.strip()}"
                cited = True
        except Exception as e:                          # noqa: BLE001 - 引用坏了不阻断
            if state.get("verbose"):
                print(f"[lg_followup] 引用溯源不可用（不影响回答）：{type(e).__name__}: {e}")

    note = state.get("location_note") or "会话里最近一次搜索的岗位"
    log_event(state.get("trace_id") or "-", "lg_node", node="followup_answer",
              engine="langgraph", graph="followup", job_id=job_id,
              hits=len(hits), cited=cited)
    return {
        "hits": hits,
        "answer": answer,
        "steps": _step(
            state, "生成回答",
            f"按 {job_id} 读 JD 原文（{len(jd)} 字）→ LLM 生成回答"
            f"（{'带引用' if cited else '无引用'}，检索片段 {len(hits)} 条）",
            payload={"job_id": job_id, "citations": cited},
            observation=f"{label}",
        ),
    }


def build_followup_graph():
    """编译追问图：3 个节点，全程无色 —— 追问只有一条正确路径。

    节点顺序要紧：`locate_job` 里那句「job_id 优先、其次序号、其次会话态岗位」
    与「我想投第 N 个」用的是同一份实现，两条路径不可能给出不同的岗位。
    """
    graph = StateGraph(FollowupState)
    graph.add_node("receive", receive_followup)
    graph.add_node("locate", locate_job)
    graph.add_node("answer", answer_followup)
    graph.set_entry_point("receive")
    graph.add_edge("receive", "locate")
    graph.add_edge("locate", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


FOLLOWUP_GRAPH = build_followup_graph()
