from sentence_transformers import CrossEncoder

_model = None

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print(f"正在加载 Reranker 模型：{MODEL_NAME}")
        _model = CrossEncoder(MODEL_NAME, max_length=512)
    return _model


def rerank(query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
    """对候选 chunk 精排，返回 Top K"""
    if not hits:
        return []

    model = _get_model()
    pairs = [[query, h["text"]] for h in hits]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(hits, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )
    return [
        {**h, "rerank_score": float(s)}
        for h, s in ranked[:top_k]
    ]


if __name__ == "__main__":
    from retriever import retrieve

    q = "哪个岗位薪资最高？"
    candidates = retrieve(q, top_k=12)
    print(f"召回 {len(candidates)} 条：")
    for c in candidates:
        m = c["metadata"]
        print(f"  {m['company']} | {m['title']}")

    print("\nRerank 后 Top 5：")
    hits = rerank(q, candidates, top_k=5)
    for h in hits:
        m = h["metadata"]
        print(f"  [{h['rerank_score']:+.4f}] {m['company']}")