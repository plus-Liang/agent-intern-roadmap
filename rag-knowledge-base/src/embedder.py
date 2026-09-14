import os
import requests
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_URL = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
EMBEDDING_MODEL = os.getenv("ARK_EMBEDDING_MODEL", "doubao-embedding-vision-251215")


def _extract_one_embedding(data: dict) -> list[float]:
    """从单次返回中提取向量"""
    payload = data.get("data", data)

    # 结构 A：{"data": {"embedding": [...]}}
    if isinstance(payload, dict) and "embedding" in payload:
        return payload["embedding"]

    # 结构 B：{"data": [{"embedding": [...]}]}
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]

    raise ValueError(f"无法识别的返回结构：{data}")


def embed_one(text: str) -> list[float]:
    """单个文本转向量"""
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("未找到 ARK_API_KEY，请检查 .env 文件")

    payload = {
        "model": EMBEDDING_MODEL,
        "encoding_format": "float",
        "input": [{"type": "text", "text": text}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return _extract_one_embedding(resp.json())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量：逐个请求，避免接口不支持批量"""
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