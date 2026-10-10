"""RAG 查询理解（第 2 周）：查询重写 + 子查询分解 + 触发规则 + 进程内缓存。

为什么要有这一层
----------------
BM25 + 向量双路召回吃的是「字面 / 语义相似度」，对**模糊需求**很吃亏。用户说
「找偏大模型落地能写工程代码的实习」时，整句里没有任何一条 JD 会同时出现这些词：
向量路能捞到一点，但召回被**单一表述**锁死 ——「大模型落地」和「LLM 工程」其实
是一回事，检索器并不知道。kyopark2014/rag-with-reflection 的做法是在 LangGraph
里加一个 `rewrite` 节点，把口语化问题改写成更适合检索的问法；本模块把同一思路
落在 RAG 层：**重写**（更具体、更贴 JD 用词）+ **分解**（拆成 1~3 个可各自独立
检索的子查询），再把多条子查询的召回一起交给 RRF 融合（见 `rag/retriever.py`）。

参考设计：kyopark2014/rag-with-reflection（LangGraph 编排 rewrite / retrieve /
reflect）。它的 rewrite 节点用的是 LangGraph 自反思 RAG 的标准问法 ——「你是问题
改写器，把输入问题改写成更适合向量检索的版本：先推理原问题背后的语义意图，再给出
改进后的问题」。本模块 prompt 沿用同一设计（**只输出改写后的问题，不回答**），
并按中文 JD 检索补上「补全省略、贴近岗位用语、不许新增限定」三条要求。

三条硬约束
----------
1. **只加在 RAG 层**：Agent 层（`agent/`）一行不改。入口是 `rag.retriever.retrieve`，
   Agent 的语义重排路径本来就走这里，天然受益。
2. **纯 LLM 调用 + 进程内缓存**：不引入新的存储 / 依赖；同一个原始问题 1 小时内
   只算一次（`_CACHE` + TTL），连「触发判定结果」一起缓存。
3. **不打扰精确查询**：只有「问题长度 > 15 字」或「命中 >= 2 个概念词」才触发
   （`should_understand`）。「广州 Python」这类精确查询直接走原路径，**一次 LLM
   都不调**。

失败降级
--------
任何 LLM 异常 / JSON 不合法，都退化成「原问题 + 空子查询」，调用方按原路径检索。
查询理解是**增强**，不是必需环节 —— 它坏了不能把检索带崩。

跑法（自检）::

    python -m rag.query_understanding "找偏大模型落地能写工程代码的实习"
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

from shared.llm_client import chat

# --------------------------------------------------------------------------
# 常量与开关
# --------------------------------------------------------------------------

#: 子查询个数上限（需求：上限 3 个；多了会放大召回噪声、也放大 RRF 的票数）
MAX_SUB_QUERIES = 3

#: 触发重写的长度阈值：问题长度 **> 15 字** 才算模糊 / 复杂
TRIGGER_MIN_CHARS = 15

#: 触发重写需要的概念词个数（>= 2 视为「含多概念」）
TRIGGER_MIN_CONCEPTS = 2

#: 进程内缓存 TTL：1 小时
CACHE_TTL_SECONDS = 3600.0

#: 单次 LLM 输出上限。重写 / 分解都是**短输出**，但这里是**上限不是消耗**：
#: glm 系思考模型的 max_tokens 同时卡住思考（reasoning_content）与正文，实测
#: 1024 档下分解那一步会 finish_reason=length、正文为空（思考把额度吃光），
#: 4096 + reasoning_effort=low 才能稳定拿到 JSON 数组。
#: 这与 shared/limits.py 里 match / cover_letter 的实测结论是同一回事。
DEFAULT_MAX_TOKENS = 4096

#: 调用来源标记（token 用量按来源聚合用）
LLM_SOURCE = "rag_query_understanding"

#: 概念词表：命中 >= 2 个不同概念 → 判为「含多概念」。
#: 刻意**不含城市名、也不含「实习 / 岗位 / 方向 / 偏 / 帮我」这类通用词**：
#: 「广州 Python」「北京 大模型实习」都是精确查询（城市 + 一个技术词），
#: 把它们算成两个概念会让精确查询无谓地触发重写。
CONCEPT_TERMS = (
    # — 方向 / 技术栈 —
    "大模型", "大语言模型", "语言模型", "LLM", "GPT", "Agent", "智能体", "RAG",
    "检索增强", "LangChain", "LangGraph", "Milvus", "向量", "多模态", "AIGC",
    "Prompt", "提示词", "Python", "Java", "Go", "C++", "前端", "后端", "全栈",
    "微服务", "架构", "算法", "推荐", "搜索", "测试", "产品", "运营", "运维",
    "爬虫", "数据分析", "数据开发", "深度学习", "机器学习", "NLP", "CV",
    "训练", "微调", "推理", "部署", "工程", "开发", "落地", "代码", "编码",
    "接口", "服务端", "移动端", "嵌入式", "量化", "风控", "游戏",
    # — 诉求 / 属性 —
    "远程", "转正", "高薪", "大厂", "可转正",
)

#: 关掉查询理解的开关（排查 / 回归对比用）：
#: `RAG_QUERY_UNDERSTANDING=off` → `retrieve()` 完全走改造前的原路径。
ENV_ENABLED = "RAG_QUERY_UNDERSTANDING"

_OFF_VALUES = ("off", "0", "false", "no", "disable", "disabled")


def enabled() -> bool:
    """查询理解总开关：默认开。

    每次调用现读环境变量（不固化在模块常量里）：验证脚本要在同一个进程里
    「关掉 → 跑一遍 → 打开 → 再跑一遍」，固化在 import 时就切不动了。
    """
    raw = os.getenv(ENV_ENABLED, "on")
    return str(raw).strip().lower() not in _OFF_VALUES


def _max_tokens() -> int:
    try:
        return max(0, int(os.getenv("RAG_QU_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOKENS


def _reasoning_effort() -> str:
    """思考档位：默认 low（同 match / cover_letter 的取值口径）。

    重写与分解都是**固定形状的短输出**，长思考的边际收益远小于它挤掉正文的代价
    （思考模型下正文被吃光就只剩空串）。空串 = 不注入该参数。
    """
    return str(os.getenv("RAG_QU_REASONING_EFFORT", "low")).strip().lower()


# --------------------------------------------------------------------------
# 触发规则
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[\s,，、。；;：:！!？?（）()【】\[\]\"'“”‘’/\\|~`@#$%^&*+=<>_-]+")


def normalize(question: str) -> str:
    """判定用的归一化文本：去掉空白与标点（只留实义字符）。

    标点不计入「长度」：「找偏大模型落地、能写工程代码的实习」里的顿号、
    空格不该贡献字数，否则一条精确查询会因为多打几个空格就触发重写。
    """
    return _PUNCT_RE.sub("", str(question or ""))


def concepts(question: str) -> list[str]:
    """命中的概念词（去重、按词表顺序稳定）。"""
    text = str(question or "").upper()
    hit = []
    for term in CONCEPT_TERMS:
        if term.upper() in text and term not in hit:
            hit.append(term)
    return hit


def trigger_reasons(question: str) -> list[str]:
    """返回触发理由列表；空列表 = 不触发（精确查询）。"""
    text = normalize(question)
    if not text:
        return []
    reasons = []
    if len(text) > TRIGGER_MIN_CHARS:
        reasons.append(f"长度 {len(text)} > {TRIGGER_MIN_CHARS}")
    hit = concepts(question)
    if len(hit) >= TRIGGER_MIN_CONCEPTS:
        reasons.append(f"多概念 {len(hit)} 个 {hit}")
    return reasons


def should_understand(question: str) -> bool:
    """是否需要查询理解：问题长度 > 15 字 **或** 含多概念（>= 2）。"""
    return bool(trigger_reasons(question))


# --------------------------------------------------------------------------
# LLM 节点：查询重写
# --------------------------------------------------------------------------

#: 重写 prompt（参考 kyopark2014/rag-with-reflection 的 rewrite 节点设计：
#: 让模型先想清「用户到底想找什么」，再输出一个更适合检索的问法；
#: 这里按中文 JD 检索补了「补全省略 / 贴近 JD 用语 / 不许新增限定」三条）。
REWRITE_PROMPT = """你是一个查询改写器：把用户的原始问题改写成**更适合岗位 JD 检索**的版本。

规则：
1. 先在心里推理用户真正想找什么（语义意图），但**不要输出推理过程**；
2. 补全省略的信息（「偏 X 落地」= 看重 X 的实际项目 / 工程经验）；
3. 贴近岗位 JD 里的用语（「写工程代码」→「工程开发 / 编码能力」）；
4. **不要改变用户意图**，不要添加原问题里没有的限定（城市、学历、薪资、年限等）；
5. 只输出改写后的问题本身：一行，不带引号、不带解释、不带编号、不带前缀。

原始问题：{question}

改写后的问题："""


def _clean_rewrite(text: str, fallback: str) -> str:
    """清洗重写结果：去代码块 / 引号 / 前缀，取第一行。

    模型偶尔会回「改写后的问题：xxx」或带引号、带换行解释 —— 这些都是噪声，
    直接喂给检索会把前缀也当成关键词。
    """
    out = str(text or "").strip()
    if out.startswith("```"):
        out = out.strip("`")
        if out.lower().startswith("json"):
            out = out[4:]
    out = out.strip().strip('"').strip("'").strip("“”").strip()
    for prefix in ("改写后的问题：", "改写后的问题:", "改写后的查询：", "改写：", "问题："):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
    out = out.splitlines()[0].strip() if out else ""
    return out[:200] or fallback


def rewrite_question(question: str) -> str:
    """查询重写节点：输入用户原问题 → 输出更具体、更适合检索的问题。

    LLM 失败 / 输出为空时**返回原问题**（不抛异常）：重写是增强，不是必需环节。
    """
    text = str(question or "").strip()
    if not text:
        return text
    try:
        content = chat(
            [{"role": "user", "content": REWRITE_PROMPT.format(question=text)}],
            source=LLM_SOURCE,
            max_tokens=_max_tokens(),
            reasoning_effort=_reasoning_effort(),
        )
    except Exception as e:                       # noqa: BLE001 - 重写失败就退回原问题
        print(f"[query_understanding] 重写失败，退回原问题：{type(e).__name__}: {e}")
        return text
    return _clean_rewrite(content, text)


# --------------------------------------------------------------------------
# LLM 节点：子查询分解
# --------------------------------------------------------------------------

#: 分解 prompt：输出 JSON 数组，1~3 个可独立检索的名词短语
DECOMPOSE_PROMPT = """你是子查询分解器：把一个问题拆成 1~{max_items} 个**可以各自独立检索**的短查询。

规则：
1. 每个子查询是一个检索用的**名词短语**（方向 / 技能 / 岗位类型），不要整句、不要疑问句；
2. 覆盖问题里的不同侧面，彼此不要重复；
3. 问题里有多个侧面时（例如「哪个方向 + 什么落地/工程要求 + 什么岗位」）**给满 {max_items} 个**；
   只有问题确实只描述了一个侧面时，才给 1 个；
4. 只输出 JSON 数组，不要任何其他内容。示例：
   ["大模型应用开发", "LLM 工程实习", "Agent 开发实习"]

问题：{question}

JSON 数组："""


def _extract_json_array(text: str) -> list:
    """从模型输出里抠出第一个 JSON 数组（容忍代码块 / 前后废话）。"""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return []
    return data if isinstance(data, list) else []


def _clean_sub_queries(items, fallback: str) -> list[str]:
    """清洗 + 截断到 `MAX_SUB_QUERIES`：去空、去重、去与原问题完全相同的项。"""
    out: list[str] = []
    for item in items or []:
        if not isinstance(item, str):
            continue
        text = item.strip().strip('"').strip("'").strip()
        text = " ".join(text.split())[:60]
        if not text or text == fallback or text in out:
            continue
        out.append(text)
    return out[:MAX_SUB_QUERIES]


def decompose_question(question: str, max_items: int = MAX_SUB_QUERIES) -> list[str]:
    """子查询分解节点：输入（重写后的）问题 → 输出 1~3 个子查询。

    LLM 失败 / JSON 不合法时返回 `[]`（= 不分解），由调用方只拿重写后的问题检索。
    注意 `[]` 与 `["xxx"]` 的区别：前者是「分解没成功」，后者是「模型认为一个问题就够」。
    """
    text = str(question or "").strip()
    if not text:
        return []
    limit = max(1, min(int(max_items or MAX_SUB_QUERIES), MAX_SUB_QUERIES))
    try:
        content = chat(
            [{"role": "user",
              "content": DECOMPOSE_PROMPT.format(question=text, max_items=limit)}],
            source=LLM_SOURCE,
            max_tokens=_max_tokens(),
            reasoning_effort=_reasoning_effort(),
        )
    except Exception as e:                       # noqa: BLE001 - 分解失败就只检索重写后的问题
        print(f"[query_understanding] 分解失败，只检索重写后的问题：{type(e).__name__}: {e}")
        return []
    return _clean_sub_queries(_extract_json_array(content), fallback=text)[:limit]


# --------------------------------------------------------------------------
# 进程内缓存（TTL 1 小时）
# --------------------------------------------------------------------------

#: {原始问题: {"at": 写入时间戳, "plan": plan dict}}
_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()


def cache_get(question: str):
    """取缓存；过期或不存在返回 None（过期条目顺手清掉）。"""
    key = str(question or "").strip()
    if not key:
        return None
    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item is None:
            return None
        if now - float(item.get("at") or 0.0) > CACHE_TTL_SECONDS:
            _CACHE.pop(key, None)
            return None
        return dict(item.get("plan") or {})


def cache_put(question: str, plan: dict) -> None:
    key = str(question or "").strip()
    if not key:
        return
    with _CACHE_LOCK:
        _CACHE[key] = {"at": time.time(), "plan": dict(plan)}


def clear_cache() -> None:
    """清空缓存（单测 / 验证脚本用）。"""
    with _CACHE_LOCK:
        _CACHE.clear()


def cache_info() -> dict:
    with _CACHE_LOCK:
        return {"size": len(_CACHE), "ttl_seconds": CACHE_TTL_SECONDS}


# --------------------------------------------------------------------------
# 总入口
# --------------------------------------------------------------------------

def _log(plan: dict) -> None:
    """打一行日志：精确查询也要留痕（验证「精确查询不触发」靠的就是它）。"""
    where = "缓存命中" if plan.get("cached") else "新算"
    if plan.get("triggered"):
        print(f"[query_understanding] {where} 触发重写（{plan.get('reason')}）："
              f"「{plan.get('question')}」→「{plan.get('rewritten')}」；"
              f"子查询 {len(plan.get('sub_queries') or [])} 个：{plan.get('sub_queries')}")
    else:
        print(f"[query_understanding] {where} 精确查询，跳过重写"
              f"（{plan.get('reason')}）：「{plan.get('question')}」")


def plan_queries(question: str, force: bool = False) -> dict:
    """查询理解总入口（带 1 小时进程内缓存）。

    返回 plan（**永不为 None**，也永不抛异常）：
        question      原始问题
        triggered     是否触发重写 / 分解
        reason        触发（或未触发）的理由，写日志用
        rewritten     重写后的问题；未触发 / 失败时 = 原问题
        sub_queries   子查询列表（0 ~ 3 个）；未触发 / 失败时 = []
        cached        本次结果是否来自缓存

    force=True 时跳过长度 / 概念判定强制走一次 LLM（验证脚本做前后对比用）。
    """
    text = str(question or "").strip()
    plan = {
        "question": text,
        "triggered": False,
        "reason": "",
        "rewritten": text,
        "sub_queries": [],
        "cached": False,
    }
    if not text:
        plan["reason"] = "空问题"
        return plan

    reasons = trigger_reasons(text)
    # force=True 且规则说不该触发：这是一次「对照实验」，既不读也不写缓存 ——
    # 否则会把一个精确查询的重写结论写进缓存，之后自动路径也跟着触发。
    forced = bool(force and not reasons)
    if forced:
        reasons = ["强制触发（force=True）"]

    cached = None if forced else cache_get(text)
    if cached is not None and force and not cached.get("triggered"):
        # 缓存里是「不触发」的旧结论，而这次明确要求强制 → 忽略它，重算
        cached = None
    if cached is not None:
        cached["cached"] = True
        _log(cached)
        return cached

    if not enabled():
        plan["reason"] = f"查询理解已关闭（{ENV_ENABLED}=off）"
        _log(plan)
        return plan

    if not reasons:
        plan["reason"] = (f"长度 {len(normalize(text))} <= {TRIGGER_MIN_CHARS} "
                          f"且概念 {len(concepts(text))} < {TRIGGER_MIN_CONCEPTS}")
        cache_put(text, plan)
        _log(plan)
        return plan

    plan["triggered"] = True
    plan["reason"] = " / ".join(reasons)
    plan["rewritten"] = rewrite_question(text) or text
    plan["sub_queries"] = decompose_question(plan["rewritten"])
    if not forced:
        cache_put(text, plan)
    _log(plan)
    return plan


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    questions = args or ["广州 Python", "找偏大模型落地能写工程代码的实习"]
    for q in questions:
        print(f"\n{'=' * 70}\n❓ {q}")
        p = plan_queries(q)
        print(f"   触发：{p['triggered']}（{p['reason']}）")
        print(f"   重写：{p['rewritten']}")
        print(f"   子查询：{p['sub_queries']}")
    print(f"\n缓存：{cache_info()}")
