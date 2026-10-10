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
    ``{platform}:{job_id}:{chunk_index}`` —— **确定性**，而且可读：
    看 id 就知道命中哪个平台的哪条岗位的第几段，不用回查 metadata。

    为什么是确定性字符串而不是 hash（第 3 周改造）：
      * **可溯源**：引用标注 [n] 最终要指回「哪条岗位的哪一段」。hash 形式的
        ``chunk_ab12cd34...`` 只能靠查 metadata 反推，日志 / 评测里读不出信息；
      * **永不重生成**：三个字段全部来自数据层身份（platform + job_id）与切分序位，
        同一批数据重复入库 id 完全一致，换进程 / 换机器也一样；
      * **不含正文**：正文一改 id 就变的话，增量入库永远走不到「更新」分支，
        旧记录还会以孤儿形式留在库里被同时召回（老版本踩过的坑）。
        正文是否变化由入库时逐条比对 document 判断（见 add_chunks_incremental）。

旧 id 形态（Round 10）：``chunk_<sha1(platform|job_id|company|title|chunk_index)[:16]>``。
换成新形态后旧 id **全部成为孤儿**，必须全量重建一次：

    python -m rag.vector_store --rebuild            # 清空集合后全量重建

已知边界（诚实说明）：岗位下架、或 chunk 切分位置整体挪位时，旧 id 不再有人引用，
会作为孤儿留在库里，需要清理（``cleanup_orphans``）。
"""

import hashlib

import chromadb
from shared.config import CHROMA_DIR
from rag.embedder import embed_texts, embed_query

DB_PATH = str(CHROMA_DIR)
COLLECTION_NAME = "jd_chunks"

# 历史遗留常量：Round 10 用它算 hash id，第 3 周改成
# 「{platform}:{job_id}:{chunk_index}」直接拼接后，id 计算不再读这个元组。
# 保留是因为 add_chunks_incremental 的注释仍在引用这个口径，删掉会让注释悬空。
#
# 为什么必须带 platform + job_id（Round 10）：
#   只用 (company, title, chunk_index) 时，**两个平台上的同名同司岗位会算出同一个 id**。
#   跨平台 (company,title,city) 现在确实 0 撞车，但那是当前数据的巧合，不是约束：
#   数据层自己的身份口径已经是 `platform|job_id`（见 cleaner._job_identity），
#   向量层的 id 必须同口径，否则两个平台的岗位会互相覆盖成一条。
#   代价：旧库里的 chunk id 是按 3 字段算的，换口径后旧 id 成为孤儿 —— 用 `--rebuild`
#   一次性重建即可（本轮正是这么做的）。
_IDENTITY_FIELDS = ("platform", "job_id", "company", "title", "chunk_index")

# 写进 metadata 的字段（顺序即建 metadatas 的顺序，便于人肉核对）。
# platform / job_id 是 Round 10 新增：有了它，检索命中后能直接反查回 jobs.db 的岗位
# （不再靠 company/title 模糊匹配），也是上面 id 口径的落库形态。
_META_FIELDS = ("company", "title", "city", "chunk_index", "job_id", "platform")


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
    # 通知检索层丢掉缓存过期的集合句柄：retriever 为了性能会缓存
    # get_collection() 的结果（单次构造 ~11ms，热路径上会被调几千次），
    # 而这里刚把集合删掉重建 —— 不清缓存的话旧句柄指向一个已删除的集合。
    # 函数内 import 避免 retriever <-> vector_store 的模块级循环依赖。
    try:
        from rag import retriever as _retriever
        _retriever.reset_collection_cache()
    except Exception:  # noqa: BLE001 - 检索层没被加载 / 老版本没有该函数，都不该影响重建
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
        # ⚠️ 不能用 ``chunk.get("chunk_index") or ""``：0 是合法段号但 falsy，
        # 会被吞成空串，于是第 0 段的 id 变成 ``platform:job_id:``（丢段号）。
        "chunk_index": str(chunk.get("chunk_index", "")
                           if chunk.get("chunk_index") is not None else ""),
        # Round 10：岗位级身份，用于检索命中后反查 jobs.db（必须落进 metadata）。
        "job_id": str(chunk.get("job_id") or ""),
        "platform": str(chunk.get("platform") or ""),
    }


def make_chunk_id(chunk: dict, text: str = None, meta: dict = None) -> str:
    """为一条 chunk 生成**确定性 id**：``{platform}:{job_id}:{chunk_index}``。

    同一条 JD 的同一段，id 永远相同，且**永不重新生成** —— 不掺正文、不掺时间戳、
    不掺列表下标（下标形态的 id 在列表增删一条时会全体错位，增量比对就白做了）。

    兜底：platform 或 job_id 缺失时（回退文本语料的老路径）返回
    ``legacy:<sha1(公司|岗位|段号)[:16]>`` —— 仍然稳定、互不覆盖，但一眼能看出
    它没有岗位身份，不会被误当成可溯源的引用来源。

    text 参数保留是为了兼容旧签名，当前不参与计算。
    """
    if meta is None:
        meta = _chunk_meta(chunk)

    platform = str(meta.get("platform") or "").strip()
    job_id = str(meta.get("job_id") or "").strip()
    # 同样不能用 ``or ""``：0 是合法段号（见 _chunk_meta）
    raw_index = meta.get("chunk_index", "")
    chunk_index = "" if raw_index is None else str(raw_index).strip()

    if platform and job_id:
        # chunk_index 缺失时末尾留一个空段，字段数恒定，便于按 ":" 解析回身份
        return f"{platform}:{job_id}:{chunk_index}"

    identity = "|".join(
        str(meta.get(f, "")) for f in ("company", "title", "chunk_index")
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return f"legacy:{digest}"


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
            # 冲突：挂一个不会撞车的后缀。用 ``#dupN`` 而不是 ``_dupN``，
            # 是为了让原 id（platform:job_id:index）作为前缀完整保留、仍可溯源。
            chunk_id = f"{chunk_id}#dup{position}"
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

    判定规则（id 由「platform:job_id:chunk_index」决定，见 make_chunk_id）：
        * id 不存在                         -> 新增（add）
        * id 已存在，正文与元信息都相同       -> 跳过（skipped，**不调用 embedding**）
        * id 已存在，但正文或元信息变了       -> 更新（先 delete 再 add，不留旧副本）

    「正文改了」算 updated（同一个位置换了内容），不是新增——
    这正是用户要的语义：改了岗位描述只重新算这一条的向量。

    Round 10：元信息比对范围扩到全部 `_META_FIELDS`（含 **job_id / platform**）——
    同一个 (公司, 岗位, 段号) 的 chunk 若换了 job_id（岗位被重新发布 / id 修正），
    必须判成 updated 而不是 skipped，否则 metadata 会停在旧 id 上、反查不到岗位。

    参数：
        chunks:     chunk 列表（company / title / city / chunk_index / job_id / platform / text）
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
# 孤儿清理（保险用：当前不自动调用）
# ---------------------------------------------------------------------------
# 一次 delete 的批量大小：Chroma 对超大 ids 列表会变慢，分批更稳，
# 也避免某些后端对单次请求的 ids 数量有限制。
ORPHAN_DELETE_BATCH = 500


def cleanup_orphans(collection, valid_ids) -> int:
    """删除 collection 里**不在** valid_ids 中的 id（孤儿），返回清理条数。

    孤儿怎么来的：chunk id 由「platform:job_id:chunk_index」决定（见 make_chunk_id），
    岗位下架、或 chunk_index 因重新切分而整体挪位时，旧 id 就没人再引用，
    留在库里会被检索召回，和新数据重复。

    参数：
        collection: Chroma 集合（测试可注入临时集合）
        valid_ids:  当前仍然有效的 id 集合（list / set / tuple 都行）

    返回：实际删除的条数。

    两个刻意的保守取舍（写清楚免得被误用）：
    1) valid_ids 传空（None / [] / set()）时**直接返回 0，不删任何东西**。
       空集合在语义上等于「没有有效 id」，照字面执行就会清空整个库——
       一次失误的调用代价太大，这里选择什么都不做，由调用方自己确认。
    2) valid_ids 会被字符串化后比较：Chroma 读回来的 id 一定是 str，
       调用方若传了 int/其他类型，不做归一化就会把有效 id 全判成孤儿。
    """
    valid = {str(v) for v in (valid_ids or [])}
    if not valid:
        print("cleanup_orphans：valid_ids 为空，按保守策略不删除任何数据")
        return 0

    if collection is None:
        collection = get_collection()

    existing = collection.get()
    all_ids = [str(i) for i in existing.get("ids", [])]
    orphans = [i for i in all_ids if i not in valid]
    if not orphans:
        print(f"cleanup_orphans：无孤儿（库内 {len(all_ids)} 条全部有效）")
        return 0

    deleted = 0
    for start in range(0, len(orphans), ORPHAN_DELETE_BATCH):
        batch = orphans[start:start + ORPHAN_DELETE_BATCH]
        collection.delete(ids=batch)
        deleted += len(batch)

    print(f"cleanup_orphans：库内 {len(all_ids)} 条，有效 {len(valid)} 条，"
          f"删除孤儿 {deleted} 条")
    return deleted


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


def _load_chunks_from_db(chunk_size: int = 600, verbose: bool = True):
    """从 jobs.db 读全部岗位并切成 chunk（带 job_id / platform）；库不可用返回 None。

    Round 10 为什么必须走库：`scraped_jd.txt` 是**纯文本语料**，只有公司/岗位/城市/
    薪资/链接/发布时间，**没有 job_id 字段** —— 从它切的 chunk 拿不到 job_id，
    检索命中后就无法反查回岗位。jobs.db 是岗位身份（platform + job_id）的权威来源，
    所以 `--rebuild` 以库为准；库不可用时才回退文本语料（此时 chunk 无 job_id）。
    """
    try:
        from rag.data import db
        from agent.scrapers.scheduler import build_chunks
    except Exception as exc:                # noqa: BLE001 —— 拿不到就走回退路径
        if verbose:
            print(f"[rebuild] 无法导入库/chunk 构建器（{type(exc).__name__}: {exc}）")
        return None

    try:
        jobs = db.get_all_jobs()
    except Exception as exc:                # noqa: BLE001
        if verbose:
            print(f"[rebuild] 读库失败（{type(exc).__name__}: {exc}）")
        return None

    if not jobs:
        if verbose:
            print("[rebuild] 库为空，回退文本语料")
        return None

    without_id = sum(1 for j in jobs if not str(j.get("job_id") or "").strip())
    if without_id:
        # 有岗位缺 job_id 时不硬编：宁可不带 id，也要让调用方知道覆盖率不完整
        if verbose:
            print(f"[rebuild] 警告：{without_id}/{len(jobs)} 条岗位缺 job_id")

    chunks = build_chunks(jobs, chunk_size=chunk_size)
    if verbose:
        covered = sum(1 for c in chunks if str(c.get("job_id") or "").strip())
        print(f"[rebuild] 数据源=jobs.db：{len(jobs)} 条岗位 → {len(chunks)} 个 chunk，"
              f"job_id 覆盖 {covered}/{len(chunks)}")
    return chunks


def collection_coverage(collection=None) -> dict:
    """体检向量库的 id / job_id 覆盖率；返回可断言的统计 dict。

    第 3 周加这个函数的理由：ID 口径一改，**旧库里的 id 就全成了孤儿**，
    重建之后必须能一眼确认「库里每一条都能溯源到岗位」，否则 citation 给出的
    [n] 会指向一个查不到岗位的 chunk。所以把它做成函数而不是一次性脚本，
    重建 / 夜间增量入库之后都能再跑一遍。

    返回（键名与含义）：
        total           库内 chunk 总数
        with_job_id     metadata 里 job_id 非空的条数
        deterministic   id 形如 ``{platform}:{job_id}:{chunk_index}`` 且与 metadata 一致的条数
        legacy         id 以 ``legacy:`` 开头的条数（无岗位身份，只该在文本语料回退时出现）
        mutated        id 带 ``#dup`` 后缀的条数（同一位置正文冲突的少数派）
        job_id_coverage    with_job_id / total（0.0~1.0；库空时算 1.0）
        id_coverage        deterministic / total
        orphan_ids       id 与 metadata 对不上 / 格式不可解析的样本（最多 20 个）
    """
    if collection is None:
        collection = get_collection()
    data = collection.get()
    ids = [str(x) for x in (data.get("ids") or [])]
    metas = data.get("metadatas") or []

    stats = {
        "total": len(ids),
        "with_job_id": 0,
        "deterministic": 0,
        "legacy": 0,
        "mutated": 0,
        "orphan_ids": [],
    }
    for cid, meta in zip(ids, metas):
        meta = meta or {}
        job_id = str(meta.get("job_id") or "").strip()
        platform = str(meta.get("platform") or "").strip()
        raw_index = meta.get("chunk_index", "")
        index = "" if raw_index is None else str(raw_index).strip()
        if job_id:
            stats["with_job_id"] += 1
        if cid.startswith("legacy:"):
            stats["legacy"] += 1
            continue
        if "#dup" in cid:
            stats["mutated"] += 1
        expected = f"{platform}:{job_id}:{index}"
        if platform and job_id and cid.split("#dup")[0] == expected:
            stats["deterministic"] += 1
        elif len(stats["orphan_ids"]) < 20:
            stats["orphan_ids"].append(cid)

    total = stats["total"]
    stats["job_id_coverage"] = 1.0 if total == 0 else stats["with_job_id"] / total
    stats["id_coverage"] = 1.0 if total == 0 else stats["deterministic"] / total
    return stats


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
        python -m rag.vector_store --verify-coverage # 只体检当前库（不写入，覆盖率必须 100%）

    数据源（Round 10）：默认**优先读 jobs.db**（chunk 才带得上 job_id），
    库不可用时回退 `--file` / `scraped_jd.txt` 文本语料（此时无 job_id）。
    """
    import sys

    from shared.config import DATA_DIR
    from rag.loader import load_jd_file
    from rag.splitter import split_jds

    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(_cli.__doc__)
        return 0

    if "--verify-coverage" in argv:
        stats = collection_coverage()
        print(f"库内 chunk：{stats['total']}")
        print(f"job_id 覆盖：{stats['with_job_id']}/{stats['total']} "
              f"= {stats['job_id_coverage']:.4%}")
        print(f"确定性 id 覆盖：{stats['deterministic']}/{stats['total']} "
              f"= {stats['id_coverage']:.4%}"
              f"（legacy {stats['legacy']}，dup 后缀 {stats['mutated']}）")
        if stats["orphan_ids"]:
            print(f"对不上的 id 样本：{stats['orphan_ids']}")
        ok = stats["job_id_coverage"] >= 1.0 and stats["id_coverage"] >= 1.0
        print(f"[{'PASS' if ok else 'FAIL'}] job_id / id 覆盖率必须都是 100%")
        return 0 if ok else 1

    incremental = "--incremental" in argv
    rebuild = "--rebuild" in argv
    run_search_test = "--no-search-test" not in argv

    data_path = str(DATA_DIR / "scraped_jd.txt")
    if "--file" in argv:
        data_path = argv[argv.index("--file") + 1]

    # 显式给了 --file 就尊重调用方（测试/临时数据源）；否则优先走库
    chunks = None
    if "--file" not in argv:
        chunks = _load_chunks_from_db(verbose=verbose)
    if chunks is None:
        if verbose:
            print(f"数据文件：{data_path}")
        jds = load_jd_file(data_path)
        chunks = split_jds(jds)
        if verbose:
            print(f"读到 {len(jds)} 条 JD，切成 {len(chunks)} 个 chunk（无 job_id）")

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
            print(f"  [{r['distance']:.4f}] {m['company']} | {m['title']} "
                  f"| job_id={m.get('job_id', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
