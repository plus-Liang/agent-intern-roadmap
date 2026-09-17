import chromadb
from shared.config import CHROMA_DIR
from rag.embedder import embed_texts, embed_query

DB_PATH = str(CHROMA_DIR)
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
    collection = get_collection()
    ids, documents, metadatas = [], [], []
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
    from shared.config import DATA_DIR
    from rag.loader import load_jd_file
    from rag.splitter import split_jds

    jds = load_jd_file(str(DATA_DIR / "scraped_jd.txt"))
    chunks = split_jds(jds)
    add_chunks(chunks)

    print("\n=== 检索测试 ===")
    for q in ["哪些岗位要求 Python？", "有没有测试相关的岗位？", "会踢足球"]:
        print(f"\n问题：{q}")
        for r in search(q, top_k=3):
            m = r["metadata"]
            print(f"  [{r['distance']:.4f}] {m['company']} | {m['title']}")