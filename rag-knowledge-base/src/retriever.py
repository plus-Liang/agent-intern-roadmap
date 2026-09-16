import jieba
from rank_bm25 import BM25Okapi

from vector_store import get_collection
from embedder import embed_query

# BM25 索引缓存（懒加载）
_BM25_CACHE = {}


def _get_bm25():
    """从 Chroma 读出所有 chunk，构建 BM25 索引"""
    if "index" in _BM25_CACHE:
        return _BM25_CACHE

    collection = get_collection()
    data = collection.get()

    ids = data["ids"]
    docs = data["documents"]

    # jieba 分词（中文必须）
    tokenized = [list(jieba.cut(doc)) for doc in docs]
    _BM25_CACHE["index"] = BM25Okapi(tokenized)
    _BM25_CACHE["ids"] = ids
    _BM25_CACHE["docs"] = docs
    return _BM25_CACHE


def _bm25_search(query: str, top_k: int = 20) -> list[str]:
    """BM25 关键词检索，返回 id 列表"""
    cache = _get_bm25()
    tokenized_query = list(jieba.cut(query))
    scores = cache["index"].get_scores(tokenized_query)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [cache["ids"][i] for i in ranked]


def _vector_search(query: str, top_k: int = 20) -> list[str]:
    """向量语义检索，返回 id 列表"""
    collection = get_collection()
    query_vec = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )
    return results["ids"][0]


def _rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """RRF 融合多个排名列表，返回融合后的 id 列表"""
    scores = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            scores[id_] = scores.get(id_, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def retrieve(query: str, top_k: int = 12) -> list[dict]:
    bm25_ids = _bm25_search(query, top_k=20)
    vector_ids = _vector_search(query, top_k=20)

    fused_ids = _rrf_fuse([bm25_ids, vector_ids])[:top_k]

    # 去重（保险）
    seen = set()
    unique_ids = []
    for id_ in fused_ids:
        if id_ not in seen:
            seen.add(id_)
            unique_ids.append(id_)

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
        })
    return hits

def format_context(hits: list[dict]) -> str:
    """把检索结果格式化成可以塞进 Prompt 的文本"""
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
        hits = retrieve(q, top_k=12)   # ← 改成 12
        for h in hits:
            m = h["metadata"]
            print(f"  {m['company']} | {m['title']}")