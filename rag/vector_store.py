"""
JD 向量库（ChromaDB）。

两种入库方式：

1. ``add_chunks`` —— 全量重建用的原函数，保持既有调用方式不变。
   默认 replace=False（追加），需要真正重建时传 replace=True（先清空集合再写），
   避免旧 id 残留在库里形成重复。

2. ``add_chunks_incremental`` —— 增量入库（本模块新增）。
   逐条比对 id 与内容：跳过没变的、只重新 embedding 新增/改动的，
   不再每次抓取都把整个库重算一遍，省 API 额度。

ID 方案（两种方式共用，务必保持一致）：
    ``chunk_<sha1(公司|岗位|chunk_index)[:16]>`` —— 只由「这条 chunk 属于哪条 JD 的
    第几段」决定，**不含正文**。
    为什么不含正文：id 必须能回答「还是不是同一段」。若把正文算进 id，
    正文一改 id 就变，增量入库只会看到「旧 id 消失 + 新 id 出现」，
    永远走不到「更新」那条分支——而且旧记录会以孤儿形式留在库里，
    检索时新旧两份同时被召回。
    正文是否变化由入库时逐条比对 document 判断（见 add_chunks_incremental）。

旧版本用的是列表下标 ``chunk_0..chunk_N``：列表增删一条，后面所有 id 全部错位，
增量比对会把没改过的 chunk 也当成新数据重算，增量就白做了。改成按
「公司|岗位|chunk_index」定位后，同一批数据的 id 在多次运行之间是稳定的。

已知边界（诚实说明）：公司名/岗位名本身被修正、或 chunk 切分位置整体变化时，
id 会随之改变，旧 id 会作为孤儿留在库里，需要迁移一次清理：

    python -m rag.vector_store --rebuild            # 清空集合后全量重建
"""

import hashlib

import chromadb
from shared.config import CHROMA_DIR
from rag.embedder import embed_texts, embed_query

DB_PATH = str(CHROMA_DIR)
COLLECTION_NAME = "jd_chunks"

# 参与 id 计算的字段：只放「定位这段 chunk 是谁」的稳定信息，**不放正文**。
# 正文进 id 会导致「正文一改 id 就变」，更新分支永远走不到（见模块 docstring）。
_IDENTITY_FIELDS = ("company", "title", "chunk_index")

# 写进 metadata 的字段（顺序即建 metadatas 的顺序，便于人肉核对）
_META_FIELDS = ("company", "title", "city", "chunk_index")


# ---------------------------------------------------------------------------
# 客户端 / 集合
# ---------------------------------------------------------------------------
def get_client():
    return chromadb.PersistentClient(path=DB_PATH)


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(collection=None):
    """清空向量库（删掉整个集合再按原参数重建），返回新的空集合。

    只在用户明确要求全量重建（CLI ``--rebuild`` / ``add_chunks(replace=True)``）
    时调用；增量入库绝不走这里。
    """
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - 集合本来就不存在（或并发被删）时忽略
        pass
    if collection is not None:
        return collection
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# chunk 归一化：id / document / metadata
# ---------------------------------------------------------------------------
def _chunk_meta(chunk: dict) -> dict:
    """把一条 chunk 的元信息规整成固定字段（缺字段补默认值，值统一成 str/空串）。

    统一成字符串是有意的：比对时必须能和 Chroma 读回来的值逐字段相等，
    否则 3 与 "3" 这种类型差异会把「没变」误判成「变了」，白白重算向量。
    """
    return {
        "company": str(chunk.get("company") or ""),
        "title": str(chunk.get("title") or ""),
        "city": str(chunk.get("city") or ""),
        "chunk_index": str(chunk.get("chunk_index", "") or ""),
    }


def make_chunk_id(chunk: dict, text: str = None, meta: dict = None) -> str:
    """为一条 chunk 生成**稳定 id**：同一条 JD 的同一段，id 永远相同。

    组成：公司 | 岗位 | chunk_index（不含正文，原因见模块 docstring）。
    这三个字段一起回答「这是哪条 JD 的第几段」；正文变没变由调用方比对 document。

    text 参数保留是为了兼容旧签名，当前不参与计算。
    """
    if meta is None:
        meta = _chunk_meta(chunk)

    identity = "|".join(str(meta.get(f, "")) for f in _IDENTITY_FIELDS)
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}"


def _prepare_chunks(chunks: list[dict]) -> list[dict]:
    """把调用方给的 chunk 列表规整成 [{id, text, metadata}]。

    - 空文本直接丢弃（既没有检索价值，也会在 embedding 接口上浪费一次调用）；
    - 同一批里 id 相同**且正文也相同**的 chunk 视为重复，只保留第一条
      （完全相同的段落重复入库没有意义，还会多花一次 embedding）。
      id 相同但正文不同的两条：像「同一条 JD 的两个版本被同时传进来」，
      这种情况两条都保留、第一条挂 id、后面的挂兜底 id，避免互相覆盖。
    """
    prepared = []
    used_ids: dict[str, str] = {}          # id -> 首次占用的正文
    for position, chunk in enumerate(chunks or []):
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue

        meta = _chunk_meta(chunk)
        chunk_id = make_chunk_id(chunk, text, meta)
        previous = used_ids.get(chunk_id)
        if previous is not None:
            if previous == text:
                continue                    # 同一批里的完全重复，丢弃
            chunk_id = f"{chunk_id}_dup{position}"   # 冲突：给个不会撞车的 id
        used_ids[chunk_id] = text

        prepared.append({
            "id": chunk_id,
            "text": text,
            "metadata": meta,
        })
    return prepared


# ---------------------------------------------------------------------------
# 全量写入（原 add_chunks，行为向后兼容）
# ---------------------------------------------------------------------------
def add_chunks(chunks: list[dict], replace: bool = False):
    """全量写入 chunk（原有接口，签名向后兼容）。

    参数：
        chunks:  chunk 列表（company / title / city / chunk_index / text）
        replace: True 时先清空集合再写（真正的「重建」）；
                 默认 False 保持旧行为——直接 add 到现有集合。

    返回：写入条数。
    """
    collection = get_collection()
    if replace:
        collection = reset_collection()
        print("已清空向量库（replace=True），开始全量重建...")

    prepared = _prepare_chunks(chunks)
    if not prepared:
        print("没有可写入的有效 chunk（空列表或正文全为空）")
        return 0

    ids = [item["id"] for item in prepared]
    documents = [item["text"] for item in prepared]
    metadatas = [item["metadata"] for item in prepared]

    print(f"正在通过 API 计算 {len(documents)} 个 chunk 的向量...")
    embeddings = embed_texts(documents)
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"已写入 {len(ids)} 条到向量库")
    return len(ids)


# ---------------------------------------------------------------------------
# 增量写入（新增）
# ---------------------------------------------------------------------------
def _existing_index(collection, ids: list[str]) -> dict:
    """一次性把待查 id 的现状读回来，返回 {id: (document, metadata)}。

    为什么批量查：逐条 collection.get(ids=[x]) 在几千个 chunk 上会慢得离谱，
    这里一次请求拿全部，命中与否在内存里判断。
    """
    index = {}
    if not ids:
        return index
    result = collection.get(ids=ids)
    # Chroma 对不存在的 id 直接不返回（不报错）：用 zip 对齐三个列表即可
    for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        index[cid] = (doc, meta or {})
    return index


def _meta_equal(old: dict, new: dict) -> bool:
    """比对元信息（只比固定字段），字符串化后逐字段相等。"""
    old = old or {}
    return all(str(old.get(f, "")) == str(new.get(f, "")) for f in _META_FIELDS)


def add_chunks_incremental(chunks: list[dict], collection=None) -> dict:
    """增量入库：按 id 判断每条 chunk 是新增 / 更新 / 跳过。

    判定规则（id 由「公司|岗位|chunk_index」决定，见 make_chunk_id）：
        * id 不存在                         -> 新增（add）
        * id 已存在，正文与元信息都相同       -> 跳过（skipped，**不调用 embedding**）
        * id 已存在，但正文或元信息变了       -> 更新（先 delete 再 add，不留旧副本）

    「正文改了」算 updated（同一个位置换了内容），不是新增——
    这正是用户要的语义：改了岗位描述只重新算这一条的向量。

    参数：
        chunks:     chunk 列表（company / title / city / chunk_index / text）
        collection: 可选的 Chroma 集合；不传就取当前集合（测试可注入临时集合）

    返回：
        {"added": N, "updated": N, "skipped": N}
    """
    result = {"added": 0, "updated": 0, "skipped": 0}

    prepared = _prepare_chunks(chunks)
    if not prepared:
        print("增量入库：没有可写入的有效 chunk")
        return result

    if collection is None:
        collection = get_collection()

    existing = _existing_index(collection, [item["id"] for item in prepared])

    to_add, to_update = [], []
    for item in prepared:
        current = existing.get(item["id"])
        if current is None:
            to_add.append(item)
            continue
        old_doc, old_meta = current
        if str(old_doc) == item["text"] and _meta_equal(old_meta, item["metadata"]):
            result["skipped"] += 1
        else:
            to_update.append(item)

    # 更新 = 先删后插（Chroma 的 add 对同 id 是 upsert，但显式删除能保证
    # 文档/元信息/向量三份数据一起被替换，不留半新半旧的记录）
    if to_update:
        collection.delete(ids=[item["id"] for item in to_update])

    writes = to_add + to_update
    if writes:
        documents = [item["text"] for item in writes]
        print(f"增量入库：需要计算向量的 chunk {len(documents)} 个"
              f"（新增 {len(to_add)}，更新 {len(to_update)}）...")
        embeddings = embed_texts(documents)
        collection.add(
            ids=[item["id"] for item in writes],
            documents=documents,
            embeddings=embeddings,
            metadatas=[item["metadata"] for item in writes],
        )

    result["added"] = len(to_add)
    result["updated"] = len(to_update)

    print(
        "增量入库完成：新增 {added} 条，更新 {updated} 条，跳过 {skipped} 条"
        "（跳过的不消耗 embedding 额度）".format(**result)
    )
    return result


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv=None, verbose: bool = True) -> int:
    """命令行入口。

    用法：
        python -m rag.vector_store                  # 全量重建 + 检索自测（原有行为）
        python -m rag.vector_store --incremental    # 增量入库（默认数据文件）
        python -m rag.vector_store --incremental --file rag/data/scraped_jd.txt
        python -m rag.vector_store --rebuild        # 先清空向量库，再全量写入
        python -m rag.vector_store --no-search-test # 只入库，不跑检索自测
    """
    import sys

    from shared.config import DATA_DIR
    from rag.loader import load_jd_file
    from rag.splitter import split_jds

    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(_cli.__doc__)
        return 0

    incremental = "--incremental" in argv
    rebuild = "--rebuild" in argv
    run_search_test = "--no-search-test" not in argv

    data_path = str(DATA_DIR / "scraped_jd.txt")
    if "--file" in argv:
        data_path = argv[argv.index("--file") + 1]

    if verbose:
        print(f"数据文件：{data_path}")

    jds = load_jd_file(data_path)
    chunks = split_jds(jds)
    if verbose:
        print(f"读到 {len(jds)} 条 JD，切成 {len(chunks)} 个 chunk")

    if incremental:
        if rebuild:
            reset_collection()
            print("已清空向量库（--rebuild），按增量逻辑重新写入")
        stats = add_chunks_incremental(chunks, get_collection())
        if verbose:
            print(f"增量结果：{stats}")
    else:
        add_chunks(chunks, replace=rebuild)

    if not run_search_test:
        return 0

    print("\n=== 检索测试 ===")
    for q in ["哪些岗位要求 Python？", "有没有测试相关的岗位？", "会踢足球"]:
        print(f"\n问题：{q}")
        for r in search(q, top_k=3):
            m = r["metadata"]
            print(f"  [{r['distance']:.4f}] {m['company']} | {m['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
