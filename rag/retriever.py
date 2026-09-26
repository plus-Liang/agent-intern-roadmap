"""岗位 JD 混合检索：BM25 + 向量双路召回，RRF 融合。

Round 10 改动（RAG 接入 Agent 层）：
  * `retrieve(..., allowed_job_ids=...)`：只在给定的岗位子集里召回（乙方案
    「SQL 先过滤、子集内语义重排」的落点）；
  * 命中结果补 `score`（RRF 融合分，**不丢**），并按分数降序；
  * `_BM25_CACHE` 加版本校验（`collection.count()` 对比）：夜间增量入库后
    BM25 索引不再陈旧；
  * 重建过程用 `threading.Lock` 保护（Chainlit 多线程会并发调用）。
"""
import threading

import jieba
from rank_bm25 import BM25Okapi

from rag.vector_store import get_collection
from rag.embedder import embed_query

# BM25 索引缓存：{index, ids, docs, count}
# count 是构建时的 collection.count()，即「缓存版本号」——
# 夜间任务增量入库后条数会变，靠它判断缓存是否该重建。
_BM25_CACHE = {}
# 保护缓存重建：Chainlit 是多线程的，两个请求同时进来会各自重建一遍
# （浪费 jieba 分词的全量开销），更要紧的是可能读到写了一半的缓存。
_BM25_LOCK = threading.Lock()


def _collection_count(collection) -> int:
    """取集合当前条数，作为缓存版本号；拿不到就返回 -1（视为「无法判断」）。"""
    try:
        return int(collection.count())
    except Exception:            # noqa: BLE001 —— 版本号拿不到不该让检索挂掉
        return -1


def _get_bm25(collection=None):
    """取 BM25 索引；集合条数与缓存版本号不一致时重建。

    版本号比较是 Round 10 加的：以前缓存一旦建立就永不失效，
    夜间增量入库的新岗位在 BM25 这一路**永远搜不到**。
    """
    if collection is None:
        collection = get_collection()
    current = _collection_count(collection)
    if "index" in _BM25_CACHE and _BM25_CACHE.get("count") == current:
        return _BM25_CACHE

    with _BM25_LOCK:
        # 双检：等锁期间可能已经有人建好了
        if "index" in _BM25_CACHE and _BM25_CACHE.get("count") == current:
            return _BM25_CACHE
        data = collection.get()
        ids = data["ids"]
        docs = data["documents"]
        tokenized = [list(jieba.cut(doc)) for doc in docs]
        _BM25_CACHE["index"] = BM25Okapi(tokenized) if tokenized else None
        _BM25_CACHE["ids"] = ids
        _BM25_CACHE["docs"] = docs
        _BM25_CACHE["count"] = current
        return _BM25_CACHE


# 全量 metadata 缓存：子集过滤要逐条看 job_id，不能一条条查库。
# 与 BM25 缓存同一把锁、同一版本号口径（collection.count()）。
_META_CACHE: dict = {}


def _meta_index(collection=None) -> dict:
    """{chunk_id: metadata} 全量缓存（子集过滤要逐条看 job_id，不能一条条查库）。

    与 BM25 缓存同版本号口径：条数变了就重建。
    """
    if collection is None:
        collection = get_collection()
    current = _collection_count(collection)
    cached = _META_CACHE.get("map")
    if cached is not None and _META_CACHE.get("count") == current:
        return cached
    with _BM25_LOCK:
        cached = _META_CACHE.get("map")
        if cached is not None and _META_CACHE.get("count") == current:
            return cached
        data = collection.get()
        mapping = {
            cid: (meta or {})
            for cid, meta in zip(data["ids"], data.get("metadatas") or [])
        }
        _META_CACHE["map"] = mapping
        _META_CACHE["count"] = current
        return mapping


def _job_id_of(chunk_id: str) -> str:
    return str((_meta_index().get(chunk_id) or {}).get("job_id") or "")


def _bm25_search(query: str, top_k: int = 20, allowed: set = None) -> list[str]:
    """BM25 检索；allowed 非 None 时**只在子集内打分**。

    子集打分（而不是「全量打分再过滤」）是有意的：全量打分时，若前 top_k 名
    恰好都不在子集里，过滤后就会得到空结果 —— 明明子集里有匹配项却搜不到。
    """
    cache = _get_bm25()
    if cache.get("index") is None:
        return []
    tokenized_query = list(jieba.cut(query))
    scores = cache["index"].get_scores(tokenized_query)

    candidates = []
    for i, score in enumerate(scores):
        cid = cache["ids"][i]
        if allowed is not None and _job_id_of(cid) not in allowed:
            continue
        candidates.append((score, cid))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [cid for _, cid in candidates[:top_k]]


def _vector_search(query: str, top_k: int = 20) -> list[str]:
    collection = get_collection()
    query_vec = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )
    return results["ids"][0]


def _rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict:
    """RRF 融合，返回 {id: 融合分}（不再只给排名列表，分要留给调用方）。"""
    scores = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            scores[id_] = scores.get(id_, 0) + 1 / (k + rank + 1)
    return scores


def retrieve(query: str, top_k: int = 12, allowed_job_ids=None) -> list[dict]:
    """检索 JD 片段。

    参数：
        query:           自然语言查询
        top_k:           返回条数上限
        allowed_job_ids: 可选的岗位 id 集合；给定时**只在这些岗位的 chunk 里召回**
                         （乙方案的「SQL 先过滤、子集内语义重排」）。None = 不限制。

    返回：[{id, text, metadata, score}]，**按 score 降序**。
    """
    allowed = None
    if allowed_job_ids is not None:
        allowed = {str(x) for x in allowed_job_ids if str(x).strip()}
        if not allowed:
            # 空子集：明确返回空，不要退化成"全库检索"（那会把不在子集里的岗位
            # 也召回，反而违背调用方的过滤意图）。
            return []

    bm25_ids = _bm25_search(query, top_k=20, allowed=allowed)
    vector_ids = _vector_search(query, top_k=20)
    if allowed is not None:
        vector_ids = [cid for cid in vector_ids if _job_id_of(cid) in allowed]

    fused = _rrf_fuse([bm25_ids, vector_ids])
    ranked = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)
    unique_ids = ranked[:top_k]
    if not unique_ids:
        # 子集过滤后可能一条都不剩（allowed 里的岗位还没有 chunk）。
        # 必须在这里返回：Chroma 的 collection.get(ids=[]) 会抛
        # "Expected IDs to be a non-empty list"，空结果不该是异常。
        return []

    collection = get_collection()
    data = collection.get(ids=unique_ids)
    id_to_pos = {id_: i for i, id_ in enumerate(data["ids"])}

    hits = []
    for id_ in unique_ids:
        if id_ not in id_to_pos:
            continue
        i = id_to_pos[id_]
        hits.append({
            "id": data["ids"][i],
            "text": data["documents"][i],
            "metadata": data["metadatas"][i],
            "score": fused[id_],
        })
    return hits


def format_context(hits: list[dict]) -> str:
    if not hits:
        return "（未检索到相关内容）"
    lines = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        lines.append(
            f"[片段{i}] 来源：{m['company']} | {m['title']} | {m['city']}\n"
            f"{h['text']}"
        )
    return "\n\n---\n\n".join(lines)


if __name__ == "__main__":
    for q in ["哪些岗位要求 Python？", "腾讯的薪资多少？", "会踢足球"]:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        hits = retrieve(q, top_k=12)
        for h in hits:
            m = h["metadata"]
            print(f"  {h['score']:.4f}  {m['company']} | {m['title']} "
                  f"| job_id={m.get('job_id', '')}")
