"""RAG 引用溯源（第 3 周）：句子级引用标注 + faithfulness 验证。

参考项目 auditrag 的三件事，本模块把它们落在 **RAG 层**（`rag/`），Agent 层一行不改：

1. **句子级引用**（auditrag 的 sentence-level citation）
   答案生成完之后再过一层后处理：把答案切成句子，逐句和**检索到的 chunk** 对齐，
   在句尾标 ``[1][2]``；编号就是 chunk 在本次检索结果里的位次，
   通过 ``{platform}:{job_id}:{chunk_index}`` 这个确定性 id 一路指回具体岗位。
   为什么要后处理而不是让 LLM 自己写 ``[n]``：模型写编号时**没有稳定锚点**
   （它看到的片段顺序、它的复述顺序、和最终展示的顺序不一定一致），实测经常标错位；
   后处理用的是同一份 hits 列表，编号天然一致，而且**不额外花 token**。

2. **faithfulness 验证**（auditrag 的 faithfulness check）
   两道闸门，先便宜后贵：
   * **确定性证据闸**（零成本）：句子的字符 n-gram 覆盖率、实义词命中率 ——
     纯粹靠字面证据，能抓住「这句话的内容在给定 chunk 里根本没出现过」；
   * **LLM 闸**（第二次调用，按需）：只在确定性闸**拿不准**时才调，
     让它逐句回答「这一句是不是被引用的 chunk 支撑」，并给出理由。
   输出统一成 ``{"supported": [...], "unsupported": [...]}``。

3. **确定性 chunk id**（auditrag 的 deterministic chunk id）
   见 ``rag/vector_store.make_chunk_id``：``{platform}:{job_id}:{chunk_index}``。
   本模块的 ``sources`` 直接带出这个 id，引用编号 → 岗位身份的链路是**可机读**的，
   不是靠人眼看公司名。

设计取舍（写清楚免得被误用）
----------------------------
* **只在「搜岗位」场景触发**：其他场景（面试、简历、闲聊）没有可溯源的岗位片段，
  标出来的 ``[n]`` 没有意义。触发点在 ``agent.tools_registry._search_rows`` 的
  semantic 分支（**Agent 层的调用逻辑不改**，只是 RAG 多返回一份引用包）。
* **不支持的句子不改写、不删除**：faithfulness 只做**标注**（``unsupported`` 列表 +
  渲染时的 ``⚠️无依据`` 标记），要不要拿掉由上游决定 —— 后处理层擅自改答案，
  会把「模型确实编了」这件事藏起来，评测就测不出来了。
* **任何 LLM 异常都降级**：LLM 闸挂了就退回「确定性闸的结论」，
  引用标注本身仍然可用（它不依赖 LLM）。

跑法（自检，不联网也能看确定性那部分）::

    python -m rag.citation
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

from shared.llm_client import chat

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

#: 句尾最多标几个引用（需求例子是两个；再多会淹没句子本身）
MAX_CITATIONS_PER_SENTENCE = 3

#: 单句引用标注的相似度下限。低于它宁可不标（标错比不标更伤人）。
CITE_MIN_SCORE = 0.10

#: 第 2、3 个引用的相对门槛：得分必须 >= ``CITE_MIN_SCORE * CITE_RELATIVE_FACTOR``。
#: 为什么需要相对门槛：中文短句和**任意**中文 JD 片段之间天然有 0.05~0.10 的
#: 字符 n-gram 重合（"要求/经验/岗位"这类通用搭配），只卡绝对阈值时几乎每句都会
#: 凑出两三条引用，把「哪段才是依据」淹掉。实测自检里第二句就被多标了一个 [2]。
CITE_RELATIVE_FACTOR = 1.5

#: 引用溯源整体开关（``RAG_CITATION=0`` 关掉后完全不跑后处理）
ENV_ENABLE = "RAG_CITATION"

#: faithfulness 的 LLM 闸开关（``RAG_CITATION_LLM=0`` 只留确定性闸，不花 token）
ENV_LLM = "RAG_CITATION_LLM"

#: 确定性闸的判定门槛。**实测标定**（scripts/verify_citation.py + 一次真实答案 20 句）：
#:   真实句子：coverage 中位数 0.96、最小 0.32；term_hit 中位数 0.96、最小 0.21
#:   编造句子（火星基地 / 8 年 K8s / 99999 元）：coverage 最大 0.17、term_hit 最大 0.15
#: 即真实句与幻觉句之间有 0.17~0.32 的空隙，门槛取空隙中间，两边都不贴边：
#: 阈值定高一点（0.45）会把「结论 + 依据」里的过渡句也判成无依据（误报），
#: 定低一点（0.10）会让编造内容混过去（漏报）。两个条件都要满足才算「有依据」。
GROUNDED_MIN_COVERAGE = 0.25   #: 句子 n-gram 被引用 chunk 覆盖的比例
GROUNDED_MIN_TERMS = 0.35      #: 实义词命中率

#: 长度短于这个字数的句子不做引用标注（"结论："、"另外，" 这类碎片）
MIN_SENTENCE_CHARS = 6

#: n-gram 长度（字符级）。中文没有空格，字符 n-gram 比词 n-gram 稳。
NGRAM_N = 2

#: LLM 闸单次输出上限。沿用 query_understanding 的实测结论：
#: glm 系思考模型 max_tokens 要留够，否则思考把额度吃光、正文为空。
DEFAULT_MAX_TOKENS = 4096


def _max_tokens() -> int:
    try:
        return max(0, int(os.getenv("RAG_CITATION_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOKENS


def _reasoning_effort() -> str:
    """思考档位：默认 low（与 match / cover_letter / query_understanding 同口径）。

    判定「这句话有没有依据」是**短输出**任务，长思考的边际收益远小于它挤掉正文的
    代价 —— 思考模型下正文被吃光就只剩空串，JSON 解析直接失败、闸门形同虚设
    （实测就是这么挂的：``finish_reason=length`` + 正文为空）。空串 = 不注入。
    """
    return str(os.getenv("RAG_CITATION_REASONING_EFFORT", "low")).strip().lower()

#: 调用来源标记（token 用量按来源聚合）
LLM_SOURCE = "rag_citation_faithfulness"

#: 单句支撑判定的挂起上限：LLM 闸最多检查多少句（成本闸门）
MAX_LLM_CHECK_SENTENCES = 12

#: 进程内缓存：同一句 × 同一组证据的判定结果 1 小时内不重复问 LLM
_CACHE: dict = {}
_CACHE_TTL = 3600.0
_CACHE_LOCK = threading.Lock()

#: 只保留有信息量的字符做证据比对（去掉标点/空白/装饰符号）
_KEEP = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")

#: 引用标记 ``[1]`` / ``【2】``（模型自己先标过一次时也要能识别）
_MARKER = re.compile(r"[\[【]\s*(\d+)\s*[\]】]")

#: 实义词停用词（这些词命中不能证明有依据）
_STOPWORDS = frozenset("""
的 了 和 与 或 及 是 在 有 为 对 也 都 就 很 更 最 会 能 可 要 需要 要求 具备
这 那 这些 那些 一个 一种 我们 你们 他们 它 他 她 我 你 以及 但是 不过 而且
根据 片段 资料 原文 岗位 公司 表示 显示 说明 因此 所以 可以 应该 必须 相关
""".split())


# --------------------------------------------------------------------------
# 文本归一化 / 句子切分
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    """归一化成"只留实义字符"的串：小写、去标点空白。

    Chroma 存的是原文，答案里会有 ``；`` ``、`` 这类差异，不归一化会把
    「字节跳动；要求 RAG」和「字节跳动要求 RAG」判成不同内容。
    """
    return "".join(_KEEP.findall(str(text or "").lower()))


def split_sentences(answer: str) -> list[str]:
    """把答案切成句子（保序、去掉空片段）。

    用「句末标点后切一刀」+「换行也切」的朴素规则，但**方括号内的标点不切**：
    答案里已经带 ``[1]`` / ``[2]`` 时（模型自己标的、或二次处理），
    ``。`` 落在 ``]`` 之前会被误判成句末，切出 ``...。[1]`` 这种碎片。

    引号内的句号被切开也不影响溯源：多标一个引用不会让结论变错，漏标才会。
    """
    text = str(answer or "")
    pieces: list[str] = []
    buf: list[str] = []
    depth = 0
    pending = False          # buf 已含句末标点，等下一个字符来确认边界
    for ch in text:
        # 句末标点只是"待定边界"：紧跟的 ``[1]`` 属于同一句，不能先切。
        if depth == 0 and pending and not ch in "])】）":
            pieces.append("".join(buf))
            buf = []
        pending = False

        if ch in "[【（(":
            depth += 1
        elif ch in "]】）)":
            depth = max(0, depth - 1)

        if ch == "\n" and depth == 0:
            pieces.append("".join(buf))
            buf = []
            continue

        buf.append(ch)
        if depth == 0 and ch in "。！？；!?;":
            pending = True

    pieces.append("".join(buf))

    parts: list[str] = []
    for raw in pieces:
        # 片段内可能还夹着换行（括号跨行的情况），再按换行拆一次
        for line in str(raw).split("\n"):
            line = line.strip()
            if not line:
                continue
            # 纯引用标记的碎片（``[1][2]``）并回上一句：答案里已经带 ``[n]`` 时
            # ``[1]。`` 这种写法会让句号把标记甩在句尾，单独成片就不好看了。
            if parts and not _KEEP.search(line) and _MARKER.search(line):
                parts[-1] = f"{parts[-1]}{line}"
            else:
                parts.append(line)
    return parts


def ngrams(text: str, n: int = NGRAM_N) -> set:
    """字符 n-gram 集合（归一化之后）。比词切分稳，不依赖分词器。"""
    norm = normalize(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i:i + n] for i in range(len(norm) - n + 1)}


def content_terms(text: str) -> set:
    """实义词集合：中文取 2 字滑窗 + 英文/数字整词，去掉停用词。

    只用来做「覆盖率」的分母，不需要精确分词：2 字滑窗对中文够了。
    """
    norm = normalize(text)
    terms: set = set()
    # 英文/数字整词（用原始文本切，归一化会把它们和中文黏在一起）
    for word in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}", str(text or "")):
        word = word.lower()
        if word and word not in _STOPWORDS:
            terms.add(word)
    if len(norm) >= 2:
        for i in range(len(norm) - 1):
            bigram = norm[i:i + 2]
            if bigram not in _STOPWORDS:
                terms.add(bigram)
    return terms


def _similarity(sentence: str, chunk_text: str) -> float:
    """句子与 chunk 的重合度（0~1），只做字面比对、不调模型。

    口径：句子 n-gram 被 chunk 覆盖的比例 —— 这正是 faithfulness 需要的语义
    （「这句话有多少内容能在这段原文里找到」），比 Jaccard 更贴近"有没有依据"：
    JD 片段很长，Jaccard 会被片段本身的长度稀释。
    """
    sent_grams = ngrams(sentence)
    if not sent_grams:
        return 0.0
    chunk_grams = ngrams(chunk_text)
    if not chunk_grams:
        return 0.0
    hit = sent_grams & chunk_grams
    return len(hit) / len(sent_grams)


def _meta_bonus(sentence: str, meta: dict) -> float:
    """句子提到公司名 / 岗位名时给一点加成（对齐「字节的 Agent 岗位」这类句子）。

    加成刻意很小（<= 0.2）：它只用来在几个候选之间**打破平局**，
    不能让「提到公司名但正文没依据」的句子凭空变成有依据。
    """
    bonus = 0.0
    norm_sent = normalize(sentence)
    for key, weight in (("company", 0.12), ("title", 0.06)):
        value = normalize((meta or {}).get(key) or "")
        if value and len(value) >= 2 and value in norm_sent:
            bonus += weight
    return min(bonus, 0.20)


# --------------------------------------------------------------------------
# 句子级引用标注
# --------------------------------------------------------------------------
def annotate_answer(answer: str, hits: list[dict],
                    max_citations: int = MAX_CITATIONS_PER_SENTENCE) -> dict:
    """给答案逐句标 ``[n]``，返回带引用的答案 + 逐句明细。

    参数：
        answer: LLM 生成的答案（原始文本）
        hits:   **本次检索**的 chunk 列表（``retrieve()`` 的返回，顺序即编号顺序）
        max_citations: 每句最多标几个

    返回：
        {
          "answer": 原答案,
          "answer_with_citations": 逐句带 [n] 的答案,
          "sentences": [{index, text, citations, rendered, top_score}],
          "sources": {编号(str): {chunk_id, job_id, platform, company, title, city,
                                 url, score, text}},
        }

    编号口径：``hits`` 里的第 i 条（0-based）就是 ``[i+1]``，与 ``format_context``
    里 ``[片段i+1]`` 完全一致 —— 生成时模型看到的锚点和最终标注的锚点是同一个。
    """
    hits = list(hits or [])
    answer = str(answer or "")

    sources: dict = {}
    for i, hit in enumerate(hits, start=1):
        meta = dict(hit.get("metadata") or {})
        sources[str(i)] = {
            "chunk_id": str(hit.get("id") or ""),
            "job_id": str(meta.get("job_id") or ""),
            "platform": str(meta.get("platform") or ""),
            "company": str(meta.get("company") or ""),
            "title": str(meta.get("title") or ""),
            "city": str(meta.get("city") or ""),
            "url": str(meta.get("url") or ""),
            "score": hit.get("score"),
            "text": str(hit.get("text") or ""),
        }

    sentences = []
    rendered_lines = []
    for index, sentence in enumerate(split_sentences(answer), start=1):
        # 模型先标过的 [n] / 【n】 一律剥掉，编号口径只认「本次 hits 的位次」——
        # 否则会出现 ``[1][2]`` 这种双重标记，而且模型的编号本来就不可信。
        clean_text = _MARKER.sub("", sentence).strip()
        citations: list = []
        top_score = 0.0
        if len(normalize(clean_text)) >= MIN_SENTENCE_CHARS:
            scored = []
            for i, hit in enumerate(hits, start=1):
                meta = hit.get("metadata") or {}
                score = (_similarity(clean_text, hit.get("text") or "")
                         + _meta_bonus(clean_text, meta))
                scored.append((score, i))
                top_score = max(top_score, score)
            scored.sort(key=lambda item: (-item[0], item[1]))
            floor = CITE_MIN_SCORE * CITE_RELATIVE_FACTOR
            for rank, (score, i) in enumerate(scored[:max_citations]):
                if score < CITE_MIN_SCORE:
                    break
                # 第 2 个及以后的引用必须明显强于噪声底（见 CITE_RELATIVE_FACTOR）
                if rank > 0 and score < floor:
                    break
                citations.append(i)

        marker = "".join(f"[{i}]" for i in citations)
        # 统一把标记放在句尾：模型给的位置不可信，而且句内插入会让
        # ``...[1]，薪资 300-500/天 [1]`` 这种多事实句读起来更乱。
        rendered = f"{clean_text}{marker}" if marker else clean_text

        sentences.append({
            "index": index,
            "text": clean_text,
            "citations": citations,
            "rendered": rendered,
            "top_score": round(top_score, 4),
        })
        rendered_lines.append(rendered)

    return {
        "answer": answer,
        "answer_with_citations": "\n".join(rendered_lines),
        "sentences": sentences,
        "sources": sources,
    }


# --------------------------------------------------------------------------
# faithfulness：第一道闸（确定性证据，零成本）
# --------------------------------------------------------------------------
def check_sentence_support(sentence: str, cited_chunks: list[str]) -> dict:
    """判断「这一句有没有被引用到的 chunk 支撑」——只用字面证据，不调模型。

    返回 ``{supported, coverage, term_hit, reason}``：
        coverage  句子字符 n-gram 落在**引用 chunk 合集**里的比例
        term_hit  句子的实义词命中比例（英文术语如 RAG/Python 靠它兜住）
    """
    sent_grams = ngrams(sentence)
    sent_terms = content_terms(sentence)
    if not cited_chunks:
        return {
            "supported": False,
            "coverage": 0.0,
            "term_hit": 0.0,
            "reason": "没有引用任何片段",
        }

    union_grams: set = set()
    union_terms: set = set()
    for chunk in cited_chunks:
        union_grams |= ngrams(chunk)
        union_terms |= content_terms(chunk)

    coverage = (len(sent_grams & union_grams) / len(sent_grams)) if sent_grams else 0.0
    term_hit = (len(sent_terms & union_terms) / len(sent_terms)) if sent_terms else 0.0

    supported = coverage >= GROUNDED_MIN_COVERAGE and term_hit >= GROUNDED_MIN_TERMS
    if supported:
        reason = f"证据充分（字符覆盖 {coverage:.2f}，实义词命中 {term_hit:.2f}）"
    elif coverage < GROUNDED_MIN_COVERAGE:
        reason = f"引用的片段里找不到这句话的内容（字符覆盖 {coverage:.2f}）"
    else:
        reason = f"关键实义词在引用片段中缺失（实义词命中 {term_hit:.2f}）"
    return {
        "supported": supported,
        "coverage": round(coverage, 4),
        "term_hit": round(term_hit, 4),
        "reason": reason,
    }


# --------------------------------------------------------------------------
# faithfulness：第二道闸（LLM 复核，按需）
# --------------------------------------------------------------------------
def _llm_json(prompt: str, retries: int = 2) -> dict:
    """调 LLM 并解析 JSON；失败抛异常（由调用方降级）。

    显式带 ``max_tokens`` + ``reasoning_effort=low``：这里的输出是一个**小 JSON**，
    但思考模型会先花掉一大截额度思考，默认 1024 上正文经常是空串（实测截断）。
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            content = str(chat(
                [{"role": "user", "content": prompt}],
                source=LLM_SOURCE,
                max_tokens=_max_tokens(),
                reasoning_effort=_reasoning_effort(),
            ) or "").strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            return json.loads(content)
        except Exception as exc:                     # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(attempt)
    raise last_err


def _llm_verdict(sentence: str, sources: list[dict]) -> dict:
    """问 LLM：这一句是否被这些片段支撑。返回 ``{supported, reason}``。"""
    evidence = "\n\n".join(
        f"[片段{src['index']}] {src.get('company', '')} | {src.get('title', '')}\n"
        f"{src.get('text', '')}"
        for src in sources
    )
    prompt = f"""你是「事实核验员」。判断下面这一句话是否被给定的 JD 片段支撑。

待核验的句子：
{sentence}

可用的片段（引用关系已由上游确定，只看这些）：
{evidence}

判定规则：
- 句子的**每个具体事实**（薪资、城市、学历、技术栈、出勤、公司名等）都能在片段里找到 → supported=true。
- 只要有一个具体事实在片段里找不到（编造的薪资 / 没提到的技术 / 张冠李戴的公司）→ supported=false。
- 句子的措辞与原文不同不算错，看事实是否一致。
- 只输出 JSON，不要其他内容：
{{"supported": true 或 false, "reason": "一句话理由"}}
"""
    data = _llm_json(prompt)
    return {
        "supported": bool(data.get("supported")),
        "reason": str(data.get("reason") or "").strip(),
    }


# --------------------------------------------------------------------------
# faithfulness 报告（两道闸门合起来）
# --------------------------------------------------------------------------
def verify_faithfulness(cited: dict, use_llm: bool = None) -> dict:
    """逐句核验答案是否被引用支撑，输出 supported / unsupported。

    参数：
        cited:   ``annotate_answer`` 的返回
        use_llm: True 强制走 LLM 闸；False 只用确定性闸；
                 None（默认）= **按需**（确定性闸拿不准的句子才问 LLM）。

    返回：
        {
          "supported":   [{index, text, citations, coverage, term_hit, reason, by}],
          "unsupported": [...同上，reason 说明为什么判无依据],
          "mode": "deterministic" | "llm" | "llm_fallback",
          "checked": 句数,
          "coverage": 全局平均字符覆盖率（0~1，一个总览指标）,
        }

    降级：LLM 抛异常 → mode 记 ``llm_fallback``，结论退回确定性闸，
    **不会**因为模型不可用就把句子判成「有依据」。
    """
    sentences = list(cited.get("sentences") or [])
    sources = cited.get("sources") or {}

    def _cited_texts(nums) -> list[str]:
        return [str((sources.get(str(n)) or {}).get("text") or "") for n in nums]

    supported: list = []
    unsupported: list = []
    unsure: list = []
    coverage_sum = 0.0

    for item in sentences:
        check = check_sentence_support(item["text"], _cited_texts(item["citations"]))
        coverage_sum += float(check["coverage"])
        record = {
            "index": item["index"],
            "text": item["text"],
            "citations": list(item["citations"]),
            "coverage": check["coverage"],
            "term_hit": check["term_hit"],
            "reason": check["reason"],
            "by": "deterministic",
        }
        if check["supported"]:
            supported.append(record)
        else:
            unsupported.append(record)
            # 拿不准的定义：**有引用但证据不足** —— 这才需要 LLM 判一次；
            # 一条引用都没标的句子没有可核验的证据，确定性闸的结论就是终审。
            if item["citations"]:
                unsure.append((record, item))

    mode = "deterministic"
    should_llm = (bool(unsure) if use_llm is None else bool(use_llm))
    if should_llm and unsure:
        mode = "llm"
        for record, item in unsure[:MAX_LLM_CHECK_SENTENCES]:
            evidence = [
                {"index": n, **(sources.get(str(n)) or {})}
                for n in item["citations"]
            ]
            key = (normalize(item["text"]),
                   tuple(str((sources.get(str(n)) or {}).get("chunk_id") or "")
                         for n in item["citations"]))
            try:
                with _CACHE_LOCK:
                    cached = _CACHE.get(key)
                if cached and (time.time() - cached[0]) < _CACHE_TTL:
                    verdict = cached[1]
                else:
                    verdict = _llm_verdict(item["text"], evidence)
                    with _CACHE_LOCK:
                        _CACHE[key] = (time.time(), verdict)
            except Exception as exc:                 # noqa: BLE001 —— LLM 闸不可用只降级
                print(f"[citation] LLM 核验不可用，退回证据闸：{type(exc).__name__}: {exc}")
                mode = "llm_fallback"
                break

            record["by"] = "llm"
            record["reason"] = verdict["reason"] or record["reason"]
            if verdict["supported"]:
                # LLM 认可 → 从 unsupported 挪到 supported（句子级事实核验比字面覆盖更准）
                unsupported = [r for r in unsupported if r is not record]
                supported.append(record)

    total = len(sentences)
    return {
        "supported": sorted(supported, key=lambda r: r["index"]),
        "unsupported": sorted(unsupported, key=lambda r: r["index"]),
        "mode": mode,
        "checked": total,
        "coverage": round(coverage_sum / total, 4) if total else 1.0,
    }


# --------------------------------------------------------------------------
# 对外入口：一次拿到「带引用的答案 + sources + faithfulness 报告」
# --------------------------------------------------------------------------
def format_hits(hits: list[dict]) -> str:
    """把 chunk 拼成生成用的上下文（与 ``rag.retriever.format_context`` 同形）。

    自带实现而不是 import：只依赖 ``{company,title,city,text}`` 这四个字段，
    检索层怎么改都不影响引用层 —— 引用层要的是「模型看到的锚点」和
    「标注用的编号顺序」是同一份数据（都来自同一个 hits 列表）。
    """
    lines = []
    for i, hit in enumerate(hits or [], start=1):
        meta = hit.get("metadata") or {}
        lines.append(
            f"[片段{i}] 来源：{meta.get('company', '')} | {meta.get('title', '')} | "
            f"{meta.get('city', '')}\n{hit.get('text') or ''}"
        )
    return "\n\n---\n\n".join(lines) if lines else "（未检索到相关内容）"


def generate_answer(question: str, hits: list[dict]) -> str:
    """基于检索片段生成答案（引用溯源里「第一次 LLM 调用」的那一步）。

    为什么不直接用 ``rag.generator.generate`` 的默认参数：它走 ``chat()`` 的默认
    ``max_tokens``（1024），而答案常是「多个岗位 + 每个岗位的依据」这种长输出，
    **实测会被截断**（``finish_reason=length``）—— 截断的答案没法做引用溯源：
    最后一句永远是半句，标不准引用、还必然被判无依据。

    这里给足额度并压低思考档位（口径同 ``query_understanding``）；
    generator 不认这些参数时退回原调用，引用溯源本身仍然可用。
    """
    from rag.generator import generate as _generate

    explicit = {"max_tokens": _max_tokens()}
    effort = _reasoning_effort()
    if effort:
        explicit["reasoning_effort"] = effort
    try:
        return _generate(question, format_hits(hits), source=LLM_SOURCE, **explicit)
    except TypeError:
        return _generate(question, format_hits(hits))


def build_citation_pack(answer: str, hits: list[dict],
                        use_llm_faithfulness: bool = None) -> dict:
    """搜岗位场景的引用溯源总入口（只做后处理，不改答案的生成方式）。

    返回：
        {
          "answer":                   原答案,
          "answer_with_citations":    逐句带 [n] 的答案,
          "sentences": [...],
          "sources": {编号: chunk 身份（含确定性 chunk_id / job_id）},
          "faithfulness": {supported, unsupported, mode, checked, coverage},
          "unsupported_indexes": [无依据句子的序号],   # 给展示层标红用
        }
    """
    cited = annotate_answer(answer, hits)
    # 默认口径由环境开关决定：``RAG_CITATION_LLM=0`` 时连"按需"都不问 LLM。
    if use_llm_faithfulness is None:
        use_llm_faithfulness = None if llm_enabled() else False
    report = verify_faithfulness(cited, use_llm=use_llm_faithfulness)
    out = dict(cited)
    out["faithfulness"] = report
    out["unsupported_indexes"] = [r["index"] for r in report["unsupported"]]
    return out


def render_citation_block(pack: dict) -> str:
    """把引用包渲染成一段可直接放进 observation 的文本（带来源表 + 无依据提示）。

    「标红」在纯文本通道里没法真上色，这里用 ``⚠️无依据`` + 来源表代替 ——
    效果等价：用户一眼能看到哪句没有出处。
    """
    lines = [str(pack.get("answer_with_citations") or "").strip()]
    unsupported = (pack.get("faithfulness") or {}).get("unsupported") or []
    if unsupported:
        lines.append("")
        lines.append("⚠️ 以下句子在检索片段中没有找到依据，仅供参考：")
        for item in unsupported:
            lines.append(f"  - 第{item['index']}句：{item['text']}")
    sources = pack.get("sources") or {}
    used = sorted({int(n) for item in (pack.get("sentences") or [])
                   for n in item.get("citations") or []})
    if used:
        lines.append("")
        lines.append("来源：")
        for n in used:
            src = sources.get(str(n)) or {}
            lines.append(
                f"  [{n}] {src.get('company', '')} | {src.get('title', '')} | "
                f"{src.get('city', '')} | chunk_id={src.get('chunk_id', '')}"
            )
    return "\n".join(lines)


def enabled() -> bool:
    """引用溯源开关（``RAG_CITATION=0`` 可关掉）。

    为什么要开关：faithfulness 的 LLM 闸是**第二次调用**，有 token 成本；
    回归 / 批量评测时可能需要关掉它（``RAG_CITATION=0`` 完全不走引用后处理）。
    """
    return os.getenv(ENV_ENABLE, "1").strip().lower() not in ("0", "false", "no", "off")


def llm_enabled() -> bool:
    """faithfulness 的 LLM 闸开关（``RAG_CITATION_LLM=0`` 只用确定性证据闸）。

    默认开：确定性闸能挡住「原文里根本没有的内容」，但挡不住**张冠李戴**
    （把 A 岗位的薪资安到 B 岗位上，字面证据都在、只是配错了引用）。
    那类错误必须让模型读一遍句子和它引用的片段才能发现。
    """
    return os.getenv(ENV_LLM, "1").strip().lower() not in ("0", "false", "no", "off")


#: 向量库 job_id 覆盖率缓存：{checked_at, ok}。覆盖率体检要 collection.get() 全量，
#: 放进热路径每次都读 5000+ 条太亏，缓存 10 分钟足够了。
_COVERAGE_CACHE: dict = {}
_COVERAGE_TTL = 600.0

#: 允许的 job_id 覆盖缺口：低于它视为「引用溯源降级，不挂引用」。
#: 为什么不是 0：少数岗位本身没有 job_id（文本语料回退路径）时，
#: 硬挂引用会把一个查不到岗位的 [n] 展示给用户 —— 那比不展示更糟。
COVERAGE_MIN_RATIO = 0.95


def coverage_ok() -> bool:
    """向量库的 job_id 覆盖率是否达标（决定这条答案挂不挂引用）。

    引用标注本身不依赖 job_id（它靠 chunk 内容对齐），但 ``sources`` 里的
    ``chunk_id`` / ``job_id`` 是「点得回去」的关键。覆盖率不达标时**宁可不挂**，
    避免给出无法溯源的编号。任何异常都返回 True（覆盖率查不到不该阻断回答）。
    """
    now = time.time()
    cached = _COVERAGE_CACHE.get("checked_at")
    if cached is not None and now - cached < _COVERAGE_TTL:
        return bool(_COVERAGE_CACHE.get("ok"))

    ok = True
    try:
        from rag.vector_store import collection_coverage
        stats = collection_coverage()
        ratio = float(stats.get("job_id_coverage") or 0.0)
        ok = ratio >= COVERAGE_MIN_RATIO
        if not ok:
            print(f"[citation] 向量库 job_id 覆盖率 {ratio:.2%} < "
                  f"{COVERAGE_MIN_RATIO:.0%}，本次不挂引用"
                  f"（需重建：python -m rag.vector_store --rebuild）")
    except Exception as exc:                          # noqa: BLE001 —— 查不到就照常挂
        print(f"[citation] 覆盖率体检不可用，按可用处理：{type(exc).__name__}: {exc}")
    _COVERAGE_CACHE["checked_at"] = now
    _COVERAGE_CACHE["ok"] = ok
    return ok


if __name__ == "__main__":
    # Windows 控制台默认 GBK，⚠️ 这类符号会直接把自检打成 UnicodeEncodeError
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                # noqa: BLE001 - 老解释器/重定向失败都无所谓
        pass

    _hits = [
        {
            "id": "shixiseng:inn_001:0",
            "text": "字节跳动 Agent 开发实习生，要求熟悉 RAG 检索增强与 LangChain，"
                    "薪资 300-500/天，Base 北京。",
            "metadata": {"company": "字节跳动", "title": "Agent 开发实习生",
                         "city": "北京", "job_id": "inn_001", "platform": "shixiseng",
                         "chunk_index": "0"},
        },
        {
            "id": "shixiseng:inn_002:0",
            "text": "腾讯大模型算法实习生，负责多模态数据处理，薪资 200-300/天。",
            "metadata": {"company": "腾讯", "title": "大模型算法实习生",
                         "city": "深圳", "job_id": "inn_002", "platform": "shixiseng",
                         "chunk_index": "0"},
        },
    ]
    _answer = ("字节的 Agent 岗位要求 RAG 经验 [1]，薪资 300-500/天 [1]。\n"
               "该岗位还要求 5 年 Kubernetes 集群运维经验，年薪 80 万。")
    _pack = build_citation_pack(_answer, _hits, use_llm_faithfulness=False)
    print("=== 带引用的答案 ===")
    print(_pack["answer_with_citations"])
    print("\n=== faithfulness（确定性闸）===")
    print(json.dumps({
        "supported": [r["index"] for r in _pack["faithfulness"]["supported"]],
        "unsupported": [r["index"] for r in _pack["faithfulness"]["unsupported"]],
        "mode": _pack["faithfulness"]["mode"],
    }, ensure_ascii=False, indent=2))
    print("\n=== 渲染 ===")
    print(render_citation_block(_pack))
