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

import json
import os
import re
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent import tools_registry as reg
from shared import limits
from shared.llm_client import chat
from shared.logger import log_event


# --------------------------------------------------------------------------
# 常量 / 小工具
# --------------------------------------------------------------------------

#: 搜索默认条数（与旧版 search_jobs 工具默认值一致，保证新旧版返回同样的列表长度）
DEFAULT_LIMIT = 20
#: 列表默认只展示前 N 条（旧版 prompt 的展示规则：默认 10 条 + 一句总数邀请）
DISPLAY_LIMIT = 10
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
    "帮我", "帮忙", "给我", "我想找", "我想", "我想要", "我要", "找一下", "搜一下",
    "查一下", "找找", "搜索", "查找", "看看", "有没有", "哪里有", "推荐", "相关",
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
    与正文共用 `max_tokens`，不传就是全局默认 1024 —— 真实简历下必然被思考
    吃光、正文为空（交接单 5.6 / 5.7 / 5.11 反复踩过）。这里复用 ReAct 的
    「长输出档」getter：`RATE_LIMIT_ENABLED=false` 时它返回 0/""，
    payload 形状与改造前一致。
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

class SearchState(TypedDict, total=False):
    question: str
    history: list
    keyword: str
    city: str
    limit: int
    semantic: bool
    extract_source: str
    rows: list
    filtered: list
    dropped: int
    total: int
    answer: str
    steps: list
    trace_id: str
    verbose: bool


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


def _extract_params_rules(question: str) -> dict:
    """确定性参数提取（LLM 不可用时的兜底，也是 LLM 结果的校验基准）。"""
    text = question or ""
    city = ""
    for name in CITY_POOL:
        if name in text:
            city = name
            break
    limit = 50 if re.search(r"(全部|所有|列全|完整列表|都列出来)", text) else DEFAULT_LIMIT

    stripped = text
    if city:
        stripped = stripped.replace(city, " ")
    for filler in sorted(_FILLERS, key=len, reverse=True):
        stripped = stripped.replace(filler, " ")
    stripped = re.sub(r"[\s,，。.、；;：:!！?？~～\-—]+", " ", stripped).strip()

    if stripped and len(stripped) <= 10:
        return {"keyword": stripped, "city": city, "limit": limit, "semantic": False}
    # 描述型需求（「想找偏大模型落地、能写工程代码的实习」）没有可直接检索的词 →
    # 交给语义路，整句当 keyword（与旧版 prompt 的语义分流规则同口径）
    return {"keyword": text.strip(), "city": city, "limit": limit, "semantic": True}


_EXTRACT_PROMPT = """你是「岗位搜索」工作流的**参数提取节点**。从用户问题里抽出检索参数。

只输出一个 JSON，不要任何其他内容：
{{"keyword": "检索关键词", "city": "城市或空串", "limit": 20, "semantic": false}}

规则：
1. keyword 必须是**可直接检索**的技术词 / 岗位词（如 Agent、大模型、Java、算法工程师、产品经理）。
   把「帮我 / 找 / 广州 / 的 / 岗位」这些口语成分全部去掉。
2. 用户描述的是「什么样的岗位」而**没有给出具体关键词**时（如「想找偏大模型落地、
   能写工程代码的实习」），keyword 填**用户的整句描述**，并把 semantic 设为 true。
3. city 只填用户明确说出的城市（如广州 / 北京）；没说不填，**不要**用"全国"。
4. limit 默认 20；用户说「全部 / 所有 / 列全」时填 50。
5. 用户只说了城市没说岗位方向时，keyword 填空串。

用户问题：{question}"""


def extract_params(state: SearchState) -> dict:
    """[提取参数] LLM 抽取 + 确定性规则兜底 + 合法性校验。"""
    question = state.get("question") or ""
    fallback = _extract_params_rules(question)
    source = "rules"
    params = dict(fallback)
    try:
        data = _llm_json(
            [{"role": "user", "content": _EXTRACT_PROMPT.format(question=question)}],
            source="lg_extract_params",
            verbose=bool(state.get("verbose")),
        )
        keyword = str(data.get("keyword") or "").strip()
        city = str(data.get("city") or "").strip()
        try:
            limit = int(data.get("limit") or DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        semantic = bool(data.get("semantic"))
        if city and city not in CITY_POOL:
            city = ""                                   # 只认已知城市，避免模型瞎编地名
        if not keyword and not city:
            keyword = fallback["keyword"]               # 两者都空 → 退回规则结果
            semantic = fallback["semantic"]
        limit = max(1, min(50, limit))
        params = {"keyword": keyword, "city": city, "limit": limit, "semantic": semantic}
        source = "llm"
    except Exception as e:                              # noqa: BLE001 - 抽取失败不该让对话挂掉
        if state.get("verbose"):
            print(f"[lg_extract_params] 退回规则提取：{type(e).__name__}: {e}")

    return {
        "keyword": params["keyword"],
        "city": params["city"],
        "limit": params["limit"],
        "semantic": params["semantic"],
        "extract_source": source,
        "steps": _step(
            state, "提取参数",
            f"关键词「{params['keyword']}」，城市「{params['city'] or '不限'}」，"
            f"条数 {params['limit']}，语义重排 {'开' if params['semantic'] else '关'}"
            f"（来源：{source}）",
            payload={"keyword": params["keyword"], "city": params["city"],
                     "limit": params["limit"], "semantic": params["semantic"]},
        ),
    }


def search_jobs_node(state: SearchState) -> dict:
    """[搜岗位] 调既有的检索链路（`_search` 与旧版工具是同一份实现）。

    语义路候选为空时用关键词路重试一次：搜索接口本身没有「搜不到」的不可恢复
    错误，重试是为了不把「候选池空了」当成「真的没有岗位」。
    """
    keyword = state.get("keyword") or ""
    city = state.get("city") or None
    limit = int(state.get("limit") or DEFAULT_LIMIT)
    semantic = bool(state.get("semantic"))
    rows: list = []
    error = ""
    try:
        rows = reg._search(keyword, city, limit, semantic) or []
        if not rows and semantic:
            rows = reg._search(keyword, city, limit, False) or []
            semantic = False
    except Exception as e:                              # noqa: BLE001 - 检索失败如实告知
        error = f"{type(e).__name__}: {e}"
        if state.get("verbose"):
            print(f"[lg_search] 检索失败：{error}")

    log_event(state.get("trace_id") or "-", "lg_node", node="search", engine="langgraph",
              graph="search", keyword=keyword, city=city or "", semantic=semantic,
              hits=len(rows), error=error)
    return {
        "rows": rows,
        "semantic": semantic,
        "steps": _step(
            state, "搜岗位",
            f"调用 search_jobs(keyword={keyword!r}, city={city!r}, limit={limit}, "
            f"semantic={semantic})，命中 {len(rows)} 条",
            payload={"keyword": keyword, "city": city or "", "limit": limit,
                     "semantic": semantic},
            observation=error or f"命中 {len(rows)} 条",
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
    wanted_city = state.get("city") or ""
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
        if wanted_city and city and city != wanted_city and city != "全国":
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
    """单条岗位的展示行，格式与旧版 prompt 规定**逐字一致**：
    `3. [岗位名](url) — 公司 · 薪资 · 城市 （核心匹配）`
    """
    title = str(row.get("title") or "").strip()
    url = str(row.get("url") or "").strip()
    company = str(row.get("company") or "").strip()
    salary = str(row.get("salary") or "").strip()
    city = str(row.get("city") or "").strip()
    name = f"[{title}]({url})" if url else title
    tail = " · ".join(p for p in (company, salary, city) if p)
    line = f"{row.get('index')}. {name}"
    if tail:
        line += f" — {tail}"
    line += "（核心匹配）" if row.get("core") else "（相关）"
    return line


def respond_search(state: SearchState) -> dict:
    """[返回列表] 渲染最终回答。格式 = 旧版 prompt 的【搜索结果透明化】。"""
    rows = state.get("filtered") or []
    keyword = state.get("keyword") or ""
    city = state.get("city") or ""
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

    total = len(rows)
    shown = rows[:DISPLAY_LIMIT]
    detail_scope = ""
    if keyword:
        detail_scope += f"关键词「{keyword}」"
    if city:
        detail_scope += ("，" if detail_scope else "") + f"城市「{city}」"
    head = f"共找到 {total} 个相关岗位" + (f"（{detail_scope}）" if detail_scope else "")
    if dropped:
        head += f"\n（已过滤 {dropped} 条重复 / 信息不全的岗位）"

    lines = [head, ""]
    lines += [_job_line(row) for row in shown]
    lines.append("")
    if total > DISPLAY_LIMIT:
        rest = total - DISPLAY_LIMIT
        lines.append(f"共 {total} 条，这里先列前 {DISPLAY_LIMIT} 条，"
                     f"还有 {rest} 条，要全部回复\"全部\"即可。")
    else:
        lines.append(f"共 {total} 条，已全部列出。")
    lines.append("（列表序号与系统内部 index 一一对应，说「我想投第 N 个」即可直接定位。）")

    log_event(state.get("trace_id") or "-", "lg_node", node="respond",
              engine="langgraph", graph="search", total=total)
    return {
        "answer": "\n".join(lines),
        "steps": _step(state, "返回列表", f"按 index 升序输出 {len(shown)}/{total} 条",
                       observation=f"共 {total} 条"),
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
    error: str
    answer: str
    steps: list
    trace_id: str
    verbose: bool


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


def locate_job(state: MatchState) -> dict:
    """[定位岗位] 显式 job_id → 序号提示 → 会话里正在看的岗位 → 上一次搜索的第 1 条。

    这一步是确定性的：旧版由模型自己决定匹配哪条岗位，跨轮会让模型「自己数行」，
    历史上既投错过岗也匹配错过岗。这里把优先级写死，拿不到就**如实说拿不到**。
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
        resume_data = state.get("resume_data") or {}
        if not resume_data:
            return {
                "score": 0, "dimensions": {}, "gaps": [], "highlights": [],
                "steps": _step(state, "匹配打分", "没有简历，无法打分",
                               observation="缺少简历"),
            }
        job_id = state.get("job_id") or ""
        try:
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
    budget = limits.run_budget_status()
    try:
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
