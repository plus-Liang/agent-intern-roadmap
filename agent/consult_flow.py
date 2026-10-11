"""
咨询分析流程 + 开放式回答分支（修「只会走预设意图，开放问题答非所问」）。

为什么要这两张图
----------------
入口路由以前只有「搜岗位 / 匹配打分 / 追问 / 投递包 / 看日志 / 复杂任务」六类，
剩下的一律丢给 ReAct（**带工具**的循环）。于是：

* 「我的简历比较匹配哪些岗位？」里有「匹配」二字 → 被 `is_match_intent` 抢走 →
  落到 `MATCH_GRAPH`；用户没点名岗位，`locate_job` 就按「最近一次搜索的第 1 条」
  兜底 → 对一条**不相干**的岗位打了个分。用户要的是「岗位类型清单 + 匹配理由」，
  拿到的是「某条岗位的分数」—— 这就是「像客服机器人，只会固定问题」的根源。
* 「今天天气」这类跟求职无关的话没有出口，只能进 ReAct 空转一轮工具循环。

修法（两条互补的路）：

1. `CONSULT_GRAPH`（咨询分析）::

       [接收] → [汇总岗位类别] → [LLM 分析画像/匹配] → [输出清单]

   读当前简历 → 从岗位库按**岗位名关键词**归类统计（Agent / 大模型算法 / 后端 …）
   → LLM 拿「简历画像 + 类别统计」产出**岗位类型清单 + 每类匹配度 + 理由 + 建议**。
   统计口径是**全库**、不是「第 1 条」，所以不会再答非所问。

2. `OTHER_GRAPH`（开放式兜底）::

       [接收] → [通用回答（不调任何工具）]

   命中「other」时**一个工具都不调**：把简历技能、最近一次搜索结果摘要当上下文，
   让 LLM 直接说话。prompt 里写死了三条约束：不编造岗位 / 需要数据先说
   「我需要先搜岗位」/ 与求职无关的问题友好回应。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent import tools_registry as reg
from agent.langgraph_flow import _llm_json, _progress, _resume_blob_zh, _step
from shared.llm_client import chat
from shared.logger import log_event


# ==========================================================================
# 岗位类别词表：**按优先级**先到先得，一条岗位只归一类（计数之和 <= 全库总数）。
# 词表是从岗位库的真实岗位名里归纳的（2842 条实测：大模型 421 / agent 176 /
# 后端 130 / 前端 85 / 测试 127 / 产品 184 …），不是凭空想的类目。
# ==========================================================================

CATEGORIES: tuple = (
    ("Agent / 智能体开发", ("agent", "智能体", "ai agent", "agent平台", "多智能体")),
    ("大模型算法 / LLM", ("大模型", "llm", "nlp", "语言模型", "多模态", "vla",
                        "预训练", "微调", "强化学习")),
    ("AI 应用 / 算法工程", ("ai应用", "ai开发", "ai 应用", "算法", "机器学习", "深度学习",
                        "推荐", "计算机视觉", "cv", "数据挖掘")),
    ("数据开发 / 数据分析", ("数据", "sql", "数仓", "etl", "bi", "爬虫")),
    ("后端 / 服务端开发", ("后端", "服务端", "java", "golang", "go开发", "c++", ".net",
                      "微服务", "架构")),
    ("前端 / 全栈 / 客户端", ("前端", "全栈", "web", "vue", "react", "android", "安卓",
                        "ios", "客户端", "小程序")),
    ("测试 / 运维 / 硬件", ("测试", "qa", "运维", "devops", "实施", "技术支持", "硬件",
                       "嵌入式", "电气", "机械")),
    ("产品 / 运营 / 非技术", ("产品", "运营", "市场", "设计", "人力", "hr", "财务", "法务",
                        "管培", "战略", "商务", "销售", "采购", "行政")),
)
#: 一类都没命中的归到这里（保证各类计数之和 == 全库总数，如实可核）
OTHER_CATEGORY = "其他"


def _load_job_rows() -> list:
    """只读岗位库的「岗位名 + 城市」两列来归类统计。

    为什么不走 `search_jobs` / `get_all_jobs`：那两条路会把 JD 正文一起拉进内存
    （3000 条 × 正文几 KB），只为数一数「多少条是 Agent 岗」不值当。
    库不可用 / 表不存在时返回空列表，调用方据此如实说「我还没有岗位数据」。
    """
    try:
        from rag.data import db as _db
    except Exception as exc:                            # noqa: BLE001 - 归类是增强
        print(f"[consult] 岗位库不可用：{type(exc).__name__}: {exc}")
        return []
    path = getattr(_db, "DB_PATH", None)
    table = getattr(_db, "TABLE", "jobs")
    if path is None:
        return []
    try:
        if not path.exists():
            return []
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(f"SELECT title, city FROM {table}").fetchall()
        finally:
            conn.close()
    except Exception as exc:                            # noqa: BLE001 - 数不出来就不编
        print(f"[consult] 岗位归类统计失败：{type(exc).__name__}: {exc}")
        return []
    return [{"title": (r["title"] or ""), "city": (r["city"] or "")} for r in rows]


def categorize_titles(rows: list) -> dict:
    """把岗位按岗位名归类：返回 `{total, categories:[{name,count,cities,samples}]}`。

    一条岗位只进**第一个**命中的类别（先到先得），所以各类 count 之和 <= total，
    差额是「其他」；LLM 拿到的数字与这里逐字一致，不许自己编数量。
    """
    buckets: dict = {name: [] for name, _ in CATEGORIES}
    buckets[OTHER_CATEGORY] = []
    for row in rows or []:
        title = str(row.get("title") or "")
        low = title.lower()
        for name, keywords in CATEGORIES:
            if any(kw in low for kw in keywords):
                buckets[name].append(row)
                break
        else:
            buckets[OTHER_CATEGORY].append(row)

    ordered: list = []
    for name, _ in CATEGORIES:
        items = buckets.get(name) or []
        if not items:
            continue
        cities: dict = {}
        for item in items:
            city = str(item.get("city") or "").strip()
            if city:
                cities[city] = cities.get(city, 0) + 1
        ordered.append({
            "name": name,
            "count": len(items),
            "cities": [c for c, _ in sorted(cities.items(), key=lambda kv: -kv[1])][:3],
            "samples": [str(i.get("title") or "") for i in items[:3]],
        })
    ordered.sort(key=lambda c: -c["count"])
    return {"total": len(rows or []), "categories": ordered,
            "other": len(buckets.get(OTHER_CATEGORY) or [])}


def _resume_text(resume_data: dict) -> str:
    """简历文本 = 结构化字段 + 纯文本原文（两种形态都要能用）。

    会话里的简历有两种：app 侧解析过的结构化 dict，以及只带 `_plain` 的原文记录
    （`/resume` 直接贴文本时就是这种）。以前只看结构化字段，`_plain` 形态会被
    当成「没有简历」，把用户已经给的简历白白丢掉。
    """
    blob = _resume_blob_zh(resume_data)
    plain = str((resume_data or {}).get("_plain") or "").strip()
    return f"{blob}\n{plain}".strip() if plain else blob


def _resume_has_content(resume_data) -> bool:
    """有没有**可用内容**（只有姓名 / 空壳记录不算简历）。"""
    if not isinstance(resume_data, dict) or not resume_data:
        return False
    for key in ("skills", "experience", "projects", "educations", "education", "_plain"):
        value = resume_data.get(key)
        if isinstance(value, (list, tuple, dict)) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


# ==========================================================================
# 图 A：咨询分析 —— [接收] → [汇总岗位类别] → [LLM 分析] → [输出清单]
# ==========================================================================

class ConsultState(TypedDict, total=False):
    question: str
    history: list
    resume_data: dict
    stats: dict
    analysis: dict
    answer: str
    steps: list
    trace_id: str
    verbose: bool
    progress_cb: Any


_ANALYZE_PROMPT = """你是「求职定位分析师」。用户想知道**自己的简历匹配哪些岗位类型**，
不是给某一条岗位打分。请结合下面的「简历」与「岗位库类别统计」做定位分析。

只输出 JSON（不要解释、不要 markdown 围栏）：
{{
  "画像": {{"强项": ["..."], "待补": ["..."]}},
  "岗位类型": [
    {{"类型": "岗位库类别统计里的类别名（照抄）", "匹配度": 90,
     "理由": "为什么匹配（必须引用简历里的具体技能 / 项目名）",
     "建议": "投什么方向、优先补什么"}}
  ],
  "总结": "一句话定位"
}}

规则（违反即错）：
1. 「类型」只能从下面的**岗位库类别**里挑，名字照抄，不要发明新类别；
2. 按匹配度从高到低排，最多 5 类；简历里完全没有依据的类别**不要**写进来；
3. 匹配度是 0-100 的整数，理由必须落到简历里的**具体**技能 / 项目，禁止空话
   （「有较强的学习能力」这类一律不要）；
4. **不许编造**岗位、公司、薪资、学历、数量：要写数量只能用「岗位库类别」里的数字；
5. 岗位数为 0 或统计为空时，「岗位类型」给空列表，并在总结里说明需要先搜岗位。

简历：
{resume}

岗位库类别（按岗位名关键词归类，全库共 {total} 条）：
{categories}"""


def receive_consult(state: ConsultState) -> dict:
    """[接收] 归一化问题 + 读当前简历（没简历也要继续：至少能给岗位类型分布）。"""
    question = (state.get("question") or "").strip()
    trace_id = state.get("trace_id") or str(uuid.uuid4())[:8]
    log_event(trace_id, "lg_node", node="receive", engine="langgraph",
              graph="consult", question=question[:50])
    _progress(state, "正在读简历 / 汇总岗位库类别…")

    resume_data = state.get("resume_data")
    if not isinstance(resume_data, dict) or not resume_data:
        try:
            resume_data = reg.get_current_resume()
        except Exception as exc:                        # noqa: BLE001 - 读不到就如实说
            if state.get("verbose"):
                print(f"[consult] 读取当前简历失败：{type(exc).__name__}: {exc}")
            resume_data = None
    has_resume = _resume_has_content(resume_data)
    return {
        "question": question,
        "trace_id": trace_id,
        "resume_data": resume_data if has_resume else {},
        "steps": _step(state, "接收",
                       "收到定位咨询" + ("，已载入当前简历" if has_resume
                                    else "，但当前会话没有可用简历")),
    }


def categorize_node(state: ConsultState) -> dict:
    """[汇总岗位类别] 数全库（不是「最近一次搜索第 1 条」），这是答对题的关键。"""
    stats = categorize_titles(_load_job_rows())
    names = "、".join(f"{c['name']}{c['count']}" for c in stats["categories"][:4])
    return {
        "stats": stats,
        "steps": _step(state, "汇总岗位类别",
                       f"全库 {stats['total']} 条岗位，按岗位名归成 "
                       f"{len(stats['categories'])} 类（{names}…）",
                       observation=f"total={stats['total']}"),
    }


def _fallback_analysis(resume_data: dict, stats: dict) -> dict:
    """LLM 不可用时的确定性兜底：用简历技能词与类别关键词的重叠度排序。

    宁可给一个粗糙但有依据的清单，也不给一句「分析失败」—— 用户问的就是清单。
    """
    blob = _resume_text(resume_data).lower()
    items: list = []
    for cat in stats.get("categories") or []:
        keywords = dict(CATEGORIES).get(cat["name"], ())
        hits = [kw for kw in keywords if kw in blob]
        if not hits:
            continue
        items.append({
            "类型": cat["name"],
            "匹配度": min(95, 55 + 10 * len(hits)),
            "理由": "简历里出现了该方向的技能词：" + "、".join(hits[:4]),
            "建议": "优先投这一类岗位；拿具体 JD 做一次匹配打分再定。",
        })
    items.sort(key=lambda i: -int(i["匹配度"]))
    skills = [str(s) for s in (resume_data.get("skills") or [])][:8]
    return {
        "画像": {"强项": skills or ["（简历里没解析出技能字段）"], "待补": []},
        "岗位类型": items[:5],
        "总结": "（模型分析这次没跑成，上面是按简历技能与岗位类别关键词的重叠度给的粗排）",
    }


def analyze_consult(state: ConsultState) -> dict:
    """[LLM 分析] 简历画像 + 类别统计 → 岗位类型清单（JSON）。"""
    resume_data = state.get("resume_data") or {}
    stats = state.get("stats") or {"total": 0, "categories": []}

    if not resume_data:
        return {"analysis": {}, "steps": _step(
            state, "分析定位", "没有简历 → 不做画像分析，如实引导用户先上传简历")}
    if not stats.get("total") or not stats.get("categories"):
        # 库里没数据：**不许编岗位**。按约束先说「我需要先搜岗位」。
        return {"analysis": {"画像": {"强项": [str(s) for s in
                                              (resume_data.get("skills") or [])][:8]},
                            "岗位类型": [],
                            "总结": "我这边还没有岗位数据 —— 我需要先搜岗位，"
                                    "才知道库里现在有哪些类型在招。"},
                "steps": _step(state, "分析定位",
                               "岗位库为空 → 不编岗位类型，只给简历画像 + 提示先搜岗位")}

    brief = [{"类别": c["name"], "岗位数": c["count"],
              "城市": c["cities"], "示例岗位": c["samples"]}
             for c in stats["categories"]]
    prompt = _ANALYZE_PROMPT.format(
        resume=_resume_text(resume_data),
        total=stats["total"],
        categories=json.dumps(brief, ensure_ascii=False, indent=1),
    )
    try:
        data = _llm_json([{"role": "user", "content": prompt}],
                         source="lg_consult", verbose=bool(state.get("verbose")))
    except Exception as exc:                            # noqa: BLE001 - 降级到确定性兜底
        log_event(state.get("trace_id") or "-", "lg_node", node="consult_analyze",
                  engine="langgraph", graph="consult", error=f"{type(exc).__name__}: {exc}"[:200])
        data = _fallback_analysis(resume_data, stats)
    if not isinstance(data, dict) or not data.get("岗位类型"):
        data = _fallback_analysis(resume_data, stats)
    return {"analysis": data,
            "steps": _step(state, "分析定位",
                           f"画像+匹配分析完成，给出 {len(data.get('岗位类型') or [])} 类岗位",
                           observation=(data.get("总结") or "")[:100])}


def _number(value) -> int:
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return 0


def respond_consult(state: ConsultState) -> dict:
    """[输出] 岗位类型清单 + 每类匹配度 + 理由 + 建议（**不给任何单条岗位分数**）。"""
    resume_data = state.get("resume_data") or {}
    stats = state.get("stats") or {}
    analysis = state.get("analysis") or {}
    total = int(stats.get("total") or 0)
    by_name = {c["name"]: c for c in (stats.get("categories") or [])}

    if not resume_data:
        lines = ["## 你的简历匹配哪些岗位？",
                 "",
                 "我这边**还没有你的简历**，做不了画像分析 —— 先把简历给我："
                 "用 `/resume 你的简历文本`，或直接上传 PDF / Word 简历附件，"
                 "然后再说一句「我的简历匹配哪些岗位」。", ""]
        if by_name:
            lines += ["先把库里**现在的岗位类型分布**给你参考（按岗位名归类）：", ""]
            lines += [f"- {c['name']}：{c['count']} 条"
                      + (f"（{'、'.join(c['cities'])}）" if c["cities"] else "")
                      for c in stats["categories"][:8]]
            lines += ["", "说「帮我找广州的 Agent 岗位」我就去搜具体岗位。"]
        answer = "\n".join(lines)
        return {"answer": answer,
                "steps": _step(state, "输出", "没有简历 → 如实说明并给出岗位类型分布")}

    lines = ["## 你的简历匹配哪些岗位", "", "**简历画像**"]
    profile = analysis.get("画像") or {}
    strengths = [str(s) for s in (profile.get("强项") or []) if str(s).strip()]
    gaps = [str(s) for s in (profile.get("待补") or []) if str(s).strip()]
    lines.append("- 强项：" + ("、".join(strengths) if strengths else "（简历里没解析出技能字段）"))
    if gaps:
        lines.append("- 待补：" + "、".join(gaps))
    lines.append("")

    items = analysis.get("岗位类型") or []
    if items:
        lines += ["**匹配的岗位类型**（按匹配度排序，依据全库岗位类别统计）", ""]
        for i, item in enumerate(items, start=1):
            name = str(item.get("类型") or "").strip()
            cat = by_name.get(name) or {}
            head = f"{i}. **{name}** — 匹配度 {_number(item.get('匹配度'))}/100"
            if cat.get("count"):
                head += f"（库里同类 {cat['count']} 条）"
            lines.append(head)
            reason = str(item.get("理由") or "").strip()
            advice = str(item.get("建议") or "").strip()
            if reason:
                lines.append(f"   - 为什么匹配：{reason}")
            if advice:
                lines.append(f"   - 建议：{advice}")
            if cat.get("samples"):
                lines.append("   - 库里示例：" + "、".join(cat["samples"]))
        lines.append("")
    elif total:
        lines.append("从简历技能与现有岗位类别的重叠看，还没有足够证据指向某一类岗位。\n")

    if total:
        dist = " · ".join(f"{c['name']} {c['count']}" for c in stats["categories"][:8])
        lines += [f"**库内分布**（共 {total} 条，按岗位名归类）：{dist}", ""]
    else:
        lines += ["⚠️ 我这边还没有岗位数据 —— 我需要先搜岗位，"
                  "才能按真实在招的岗位给你定位。", ""]

    summary = str(analysis.get("总结") or "").strip()
    if summary:
        lines += [f"**一句话定位**：{summary}", ""]
    lines += ["**下一步**：说「帮我找广州的 Agent 岗位」我去搜具体岗位；"
              "或者指定一条岗位说「帮我匹配第 1 个」，我再做**单条打分**"
              "（上面这份是类型清单，不是某一条的分数）。"]
    answer = "\n".join(lines).rstrip()
    log_event(state.get("trace_id") or "-", "lg_node", node="respond",
              engine="langgraph", graph="consult", categories=len(items))
    return {"answer": answer,
            "steps": _step(state, "输出", f"输出 {len(items)} 类岗位清单",
                           observation=f"types={len(items)}")}


def build_consult_graph() -> Any:
    graph = StateGraph(ConsultState)
    graph.add_node("receive", receive_consult)
    graph.add_node("categorize", categorize_node)
    graph.add_node("analyze", analyze_consult)
    graph.add_node("respond", respond_consult)
    graph.set_entry_point("receive")
    graph.add_edge("receive", "categorize")
    graph.add_edge("categorize", "analyze")
    graph.add_edge("analyze", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


CONSULT_GRAPH = build_consult_graph()


# ==========================================================================
# 图 B：开放式兜底 —— [接收] → [通用回答（不调任何工具）]
#
# 这一条是「意图识别失败时的出口」：**不猜用户要什么流程、不调工具**，
# 直接把「已有上下文 + 用户原话」交给 LLM。prompt 里写死不许编造、需要岗位数据
# 时先说「我需要先搜岗位」。
# ==========================================================================

class OtherState(TypedDict, total=False):
    question: str
    history: list
    resume_data: dict
    context: str
    answer: str
    steps: list
    trace_id: str
    verbose: bool
    progress_cb: Any


_OTHER_PROMPT = """你是「求职助手」的**通用对话**分支。这一轮**不使用任何工具**，
直接用你自己的话回答用户。

硬约束（违反即错）：
1. 只能基于下面「已有上下文」里写的信息回答；上下文里没有的事实**不要猜**；
2. **不许编造**岗位、公司、薪资、学历、数量、链接 —— 一条都不许编；
3. 需要岗位数据才能回答时，**先说一句「我需要先搜岗位」**，再告诉用户怎么说
   （例：「帮我找广州的 Agent 岗位」），不要拿「最近一次搜索的第 1 条」硬套；
4. 与求职无关的问题（闲聊 / 常识 / 问你是谁 / 系统怎么用）就友好正常地回应，
   一两句说清就行，**不要**硬扯岗位、也不要拒绝回答；
5. 不要输出 JSON，不要提「工具 / 函数 / 节点」这类内部词，直接说人话，简短。

已有上下文：
{context}

用户的问题：{question}"""


def _last_search_context() -> str:
    """最近一次搜索的**摘要**（给通用回答当上下文，避免它凭空编岗位）。"""
    try:
        rows = reg._session_state().get("last_job_list") or []
    except Exception:                                   # noqa: BLE001 - 没有会话态就当没有
        return ""
    if not rows:
        return ""
    head = "；".join(f"{r.get('company')} · {r.get('title')}" for r in rows[:3])
    return f"最近一次搜索共 {len(rows)} 条，前几条是：{head}"


def _context_blob(state: OtherState) -> str:
    resume_data = state.get("resume_data") or {}
    parts = []
    if _resume_has_content(resume_data):
        skills = "、".join(str(s) for s in (resume_data.get("skills") or [])[:12])
        parts.append(f"用户当前简历：{resume_data.get('name') or '匿名'}；"
                     f"技能：{skills or '（未解析出技能字段，见下面原文）'}；"
                     f"城市：{resume_data.get('city') or '（未写）'}")
        plain = str(resume_data.get("_plain") or "").strip()
        if plain:
            parts.append("简历原文（节选）：" + plain[:300])
    else:
        parts.append("用户当前**没有**上传简历（不要假装看过简历）")
    last = _last_search_context()
    parts.append(last if last else "本次会话还没有搜过岗位")
    return "\n".join(parts)


def receive_other(state: OtherState) -> dict:
    """[接收] 归一化 + 读简历（只当上下文用，不做任何工具调用）。"""
    question = (state.get("question") or "").strip()
    trace_id = state.get("trace_id") or str(uuid.uuid4())[:8]
    log_event(trace_id, "lg_node", node="receive", engine="langgraph",
              graph="other", question=question[:50])
    resume_data = state.get("resume_data")
    if not isinstance(resume_data, dict) or not resume_data:
        try:
            resume_data = reg.get_current_resume()
        except Exception:                               # noqa: BLE001 - 上下文而已
            resume_data = None
    return {
        "question": question,
        "trace_id": trace_id,
        "resume_data": resume_data if isinstance(resume_data, dict) else {},
        "steps": _step(state, "接收", f"收到开放问题：{question[:40]}（不调任何工具）"),
    }


def answer_other(state: OtherState) -> dict:
    """[通用回答] 一次 LLM，零工具调用；失败时给一句诚实的话，不编内容。"""
    question = state.get("question") or ""
    context = _context_blob(state)
    prompt = _OTHER_PROMPT.format(context=context, question=question)
    try:
        answer = str(chat([{"role": "user", "content": prompt}],
                          source="lg_other_answer") or "").strip()
    except Exception as exc:                            # noqa: BLE001 - 如实告知
        log_event(state.get("trace_id") or "-", "lg_node", node="other_answer",
                  engine="langgraph", graph="other", error=f"{type(exc).__name__}: {exc}"[:200])
        answer = ""
    if not answer:
        answer = ("这一句我没太理解成某个具体任务。你可以让我：\n"
                  "- 找岗位（说清城市 / 方向，如「帮我找广州的 Agent 岗位」）\n"
                  "- 用简历匹配打分（如「帮我匹配第 1 个」）\n"
                  "- 分析你的简历适合哪类岗位（如「我的简历匹配哪些岗位」）")
    return {"answer": answer,
            "steps": _step(state, "通用回答", "不调工具，直接基于上下文回答",
                           observation=answer[:80])}


def build_other_graph() -> Any:
    graph = StateGraph(OtherState)
    graph.add_node("receive", receive_other)
    graph.add_node("answer", answer_other)
    graph.set_entry_point("receive")
    graph.add_edge("receive", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


OTHER_GRAPH = build_other_graph()
