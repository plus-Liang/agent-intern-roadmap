import requests
from shared.config import ARK_API_KEY, ARK_EMBEDDING_MODEL, ARK_BASE_URL

EMBEDDING_URL = f"{ARK_BASE_URL}/embeddings/multimodal"


def _extract_one_embedding(data: dict) -> list[float]:
    payload = data.get("data", data)
    if isinstance(payload, dict) and "embedding" in payload:
        return payload["embedding"]
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]
    raise ValueError(f"无法识别的返回结构：{data}")


def embed_one(text: str) -> list[float]:
    if not ARK_API_KEY:
        raise ValueError("未找到 ARK_API_KEY")

    payload = {
        "model": ARK_EMBEDDING_MODEL,
        "encoding_format": "float",
        "input": [{"type": "text", "text": text}],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ARK_API_KEY}",
    }
    resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return _extract_one_embedding(resp.json())


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    total = len(texts)
    for i, t in enumerate(texts, 1):
        vec = embed_one(t)
        vectors.append(vec)
        print(f"  [{i}/{total}] 已生成向量（维度 {len(vec)}）")
    return vectors


def embed_query(query: str) -> list[float]:
    return embed_one(query)


if __name__ == "__main__":
    texts = ["熟悉 Python", "会踢足球", "掌握 RAG 技术"]
    vecs = embed_texts(texts)
    print(f"\n共 {len(vecs)} 条，每条向量维度：{len(vecs[0])}")