import chromadb
from pathlib import Path

from embedder import embed_texts, embed_query

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "jd_chunks"


def get_client():
    return chromadb.PersistentClient(path=DB_PATH)


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict]):
    """把切块列表存进向量库（使用 API Embedding）"""
    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for i, c in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(c["text"])
        metadatas.append({
            "company": c["company"],
            "title": c["title"],
            "city": c["city"],
            "chunk_index": c["chunk_index"],
        })

    print(f"正在通过 API 计算 {len(documents)} 个 chunk 的向量...")
    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"已写入 {len(ids)} 条到向量库")


def search(query: str, top_k: int = 5) -> list[dict]:
    """检索最相似的 top_k 个 chunk"""
    collection = get_collection()
    query_vec = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    from loader import load_jd_file
    from splitter import split_jds

    jds = load_jd_file(str(BASE_DIR / "data" / "jd_sample.txt"))
    chunks = split_jds(jds)
    add_chunks(chunks)

    print("\n=== 检索测试 ===")
    for q in ["哪些岗位要求 Python？", "有没有测试相关的岗位？", "会踢足球"]:
        print(f"\n问题：{q}")
        results = search(q, top_k=3)
        for r in results:
            m = r["metadata"]
            print(f"  [{r['distance']:.4f}] {m['company']} | {m['title']}")