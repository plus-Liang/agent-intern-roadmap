"""岗位 JD 混合检索：BM25 + 向量双路召回，RRF 融合。

Round 10 改动（RAG 接入 Agent 层）：
  * `retrieve(..., allowed_job_ids=...)`：只在给定的岗位子集里召回（乙方案
    「SQL 先过滤、子集内语义重排」的落点）；
  * 命中结果补 `score`（RRF 融合分，**不丢**），并按分数降序；
  * `_BM25_CACHE` 加版本校验（`collection.count()` 对比）：夜间增量入库后
    BM25 索引不再陈旧；
  * 重建过程用 `threading.Lock` 保护（Chainlit 多线程会并发调用）。

Round 13 改动（工具并行调用 + 48 秒性能修复）：
  * **双路召回改并行**：BM25 与向量互相独立，用 `asyncio.gather` 同时跑，
    RRF 融合等两边到齐（原来串行，向量路白等 BM25）；
  * **修掉 48 秒真凶**：`_job_id_of()` 以前每个 chunk 都重新 `get_collection()`
    （Chroma 客户端构造 ~11ms × 4392 chunk ≈ 48s）。现改为缓存集合句柄 +
    预计算扁平 `{chunk_id: job_id}` 映射，热循环退化成纯 dict 查表。
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

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

# 双路召回的并行执行池。为什么固定 2 个常驻 worker 而不是每次 new 一个池：
# `retrieve()` 是热路径（每次语义搜岗位都会走），per-call 建池/销毁线程的开销
# 会把并行收益吃干净。2 个 worker 正好对应 BM25 / 向量两条腿。
_RECALL_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="recall")

# 集合句柄缓存（见 _get_collection 的长注释：这是 48 秒的真凶）。
_COLLECTION = None
_COLLECTION_LOCK = threading.Lock()


def _get_collection():
    """取（并缓存）Chroma 集合句柄。

    ⚠️ 这不是「顺手优化」，是性能关键路径：
    `vector_store.get_collection()` 每次都 `chromadb.PersistentClient(...)` +
    `get_or_create_collection(...)`，实测单次 **~11ms**；而 BM25 的子集过滤要对
    **每个 chunk** 调一次 `_job_id_of()`，4392 个 chunk × 11ms ≈ **48 秒** ——
    这就是「语义搜岗位要等 48 秒」的真凶（向量路本身只要 23ms）。

    集合句柄本身无状态，缓存是安全的；数据更新仍由 `count()` 版本号感知
    （与 `_BM25_CACHE` 同一套口径，不是"缓存了就不再更新"）。
    """
    global _COLLECTION
    if _COLLECTION is None:
        with _COLLECTION_LOCK:
            if _COLLECTION is None:
                _COLLECTION = get_collection()
    return _COLLECTION


def reset_collection_cache() -> None:
    """丢掉缓存的集合句柄。

    缓存句柄的唯一代价：`vector_store.reset_collection()` 会删掉集合再重建，
    此时旧句柄指向一个已删除的集合。全量重建本来跑在独立进程（CLI `--rebuild`），
    但同进程重建（脚本 / 单测 / 注入临时库）需要这个入口手动清一次。
    """
    global _COLLECTION
    with _COLLECTION_LOCK:
        _COLLECTION = None


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
        collection = _get_collection()
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
    同时预计算一份**扁平的** `{chunk_id: job_id}`（存进 `_META_CACHE["job_ids"]`）：
    子集过滤是每个 chunk 都要问一次「它在不在 allowed 里」，走 `_job_id_of()`
    再套一层 metadata dict 查表没有必要；扁平映射让热循环变成纯 dict 查表。
    """
    if collection is None:
        collection = _get_collection()
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
        _META_CACHE["job_ids"] = {
            cid: str((meta or {}).get("job_id") or "")
            for cid, meta in zip(data["ids"], data.get("metadatas") or [])
        }
        _META_CACHE["count"] = current
        return mapping


def _job_ids() -> dict:
    """{chunk_id: job_id} 扁平映射（热循环用；确保缓存已建）。"""
    _meta_index()
    return _META_CACHE.get("job_ids") or {}


def _job_id_of(chunk_id: str) -> str:
    return str(_job_ids().get(chunk_id) or "")


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

    # 子集映射**在进入循环前取一次**：以前是循环体里逐 chunk 调 _job_id_of()，
    # 而它每次都重新拿集合句柄（~11ms），4392 个 chunk 就是 48 秒。
    job_ids = _job_ids() if allowed is not None else {}
    candidates = []
    for i, score in enumerate(scores):
        cid = cache["ids"][i]
        if allowed is not None and job_ids.get(cid, "") not in allowed:
            continue
        candidates.append((score, cid))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [cid for _, cid in candidates[:top_k]]


def _vector_search(query: str, top_k: int = 20) -> list[str]:
    collection = _get_collection()
    query_vec = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )
    return results["ids"][0]


async def _recall_legs_async(query: str, top_k: int, allowed):
    """双路召回**并行**：BM25 与向量同时跑，`asyncio.gather` 等两边都到齐。

    为什么可以并行：两条腿只共享只读的集合句柄与 BM25 缓存，谁先跑完都不影响
    另一条；RRF 融合本来就是「必须两边都到齐才能算」——原来串行执行等于让
    向量路白等 BM25 的 20-40ms（缓存未建时是 4.6s）。
    """
    loop = asyncio.get_running_loop()
    bm25_future = loop.run_in_executor(_RECALL_POOL, _bm25_search, query, top_k, allowed)
    vector_future = loop.run_in_executor(_RECALL_POOL, _vector_search, query, top_k)
    return await asyncio.gather(bm25_future, vector_future)


def _recall_legs(query: str, top_k: int, allowed):
    """`_recall_legs_async` 的同步入口，返回 `(bm25_ids, vector_ids)`。

    `retrieve()` 是同步函数（被 tools_registry / dashboard / 脚本直接调用），
    所以分两种情形：
      * **没有正在运行的事件循环**（工具子线程 / Streamlit / 单测 / CLI）→
        用 `asyncio.run` 真正走 `asyncio.gather`；
      * **已经在事件循环里**（有人直接在 async 回调中调 retrieve）→
        `asyncio.run` 会抛 RuntimeError（不能嵌套），改在同一个常驻线程池里
        直接并行提交。两条路径都是并行，行为一致。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_recall_legs_async(query, top_k, allowed))

    bm25_future = _RECALL_POOL.submit(_bm25_search, query, top_k, allowed)
    vector_future = _RECALL_POOL.submit(_vector_search, query, top_k)
    return bm25_future.result(), vector_future.result()


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

    # 两条腿并行（asyncio.gather），融合在两边都到齐之后做
    bm25_ids, vector_ids = _recall_legs(query, 20, allowed)
    if allowed is not None:
        job_ids = _job_ids()
        vector_ids = [cid for cid in vector_ids if job_ids.get(cid, "") in allowed]

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
