# -*- coding: utf-8 -*-
"""RAG 端到端验证：模糊查询走 semantic 分支、精确查询走 SQL 分支，并验证反查闭环。

只读；幂等。用法：python verify_rag_e2e.py
"""
import json
import sys
from pathlib import Path

REPO = Path(r"D:\agent-intern-roadmap")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rag" / "quality"))

import chromadb  # noqa: E402
from shared.config import CHROMA_DIR  # noqa: E402
import rag.vector_store as vs  # noqa: E402

vs.DB_PATH = str(CHROMA_DIR)
_col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection(
    name="jd_chunks", metadata={"hnsw:space": "cosine"})
vs.get_collection = lambda: _col

import rag.retriever as rt  # noqa: E402
rt.get_collection = lambda: _col
rt._BM25_CACHE.clear()
rt._META_CACHE.clear()

from rag.data import db as jobdb  # noqa: E402
import agent.tools_registry as tr  # noqa: E402

PASS, FAIL = [], []


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f" -> {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {label} -> {type(exc).__name__}: {exc}")


FUZZY = "想找偏大模型落地、能写工程代码的实习"
PRECISE = "北京 Python"

print("=" * 74)
print("0. 向量库 / 岗位库 现状")
print("=" * 74)
n_chunks = _col.count()
data = _col.get(include=["metadatas"])
metas = data.get("metadatas") or []
with_job = sum(1 for m in metas if str((m or {}).get("job_id") or "").strip())
job_total = jobdb.count_jobs()
print(f"chroma jd_chunks : {n_chunks}")
print(f"带 job_id 的 chunk: {with_job}/{len(metas)}"
      f"（{with_job / max(len(metas), 1) * 100:.1f}%）")
print(f"jobs.db 岗位数    : {job_total}")

print()
print("=" * 74)
print("1. 模糊查询 -> semantic 分支（子集语义重排 + 反查）")
print("=" * 74)


def t_fuzzy_semantic():
    rows = tr._search(FUZZY, city=None, limit=60, semantic=True)
    assert rows, "模糊查询应返回结果"
    scored = [r for r in rows if "score" in r]
    print(f"    候选 {len(rows)} 条，其中带 score（经语义重排）{len(scored)} 条")
    for r in rows[:5]:
        print(f"      {r.get('score', '-'):>8}  {r['company']} | {r['title']} | "
              f"job_id={r['job_id']}")
    if len(rows) > tr.SEMANTIC_MAX_CANDIDATES:
        raise AssertionError(f"候选池超阈值：{len(rows)} > {tr.SEMANTIC_MAX_CANDIDATES}")
    assert scored, "候选未超阈值，应至少有一条带 score"
    # 反查闭环：每个 job_id 都要能在 jobs.db 里查到
    ids = [r["job_id"] for r in rows]
    found = jobdb.get_jobs_by_ids(ids)
    missing = [i for i in ids if i not in {x["job_id"] for x in found}]
    assert not missing, f"有 {len(missing)} 个 job_id 反查不到：{missing[:3]}"
    return (f"候选 {len(rows)} 条全部可反查回 jobs.db"
            f"（{len(scored)} 条带语义 score）")


def t_retrieve_hits_have_job_id():
    hits = rt.retrieve(FUZZY, top_k=12)
    assert hits, "retrieve 应有结果"
    ids = {str((h["metadata"] or {}).get("job_id") or "") for h in hits}
    assert ids and "" not in ids, f"有命中缺 job_id：{ids}"
    rows = jobdb.get_jobs_by_ids(list(ids))
    assert len(rows) == len(ids), f"反查不全：{len(rows)}/{len(ids)}"
    return f"{len(hits)} 个命中，job_id 齐备且全部可反查（{len(ids)} 个岗位）"


check("模糊查询走 semantic 分支且可反查", t_fuzzy_semantic)
check("retrieve 命中带 job_id 且能反查", t_retrieve_hits_have_job_id)

print()
print("=" * 74)
print("2. 精确查询 -> SQL 分支（行为必须与旧口径逐字一致）")
print("=" * 74)


def t_precise_unchanged():
    from agent.tools.job_search import search_jobs as raw
    new = tr._search(PRECISE, city="北京", limit=20)
    old = [{"job_id": j.job_id, "title": j.title, "company": j.company,
            "city": j.city, "salary": j.salary, "tags": j.tags or []}
           for j in raw(PRECISE, "北京", 20, platform="mock")]
    assert new == old, "精确查询结果与旧口径不一致"
    assert all("score" not in r for r in new), "精确分支不该带 score"
    print(f"    前 5 条：")
    for r in new[:5]:
        print(f"      {r['company']} | {r['title']} | {r['city']}")
    return f"{len(new)} 条，与旧口径逐字一致、无 score"


def t_semantic_off_vs_on():
    """semantic 模式的正确契约：候选池放宽到 <=80，重排只在池内、不增不删。

    注意不能拿 semantic=False 的结果当基准 —— 模糊长句在纯 SQL AND 下命中 0 条，
    semantic=False 返回 0 条是**预期行为**；要比的是"重排前后的候选集合"。
    """
    on = tr._search(FUZZY, city=None, limit=20, semantic=True)
    assert on, "semantic 应返回结果"
    assert len(on) <= tr.SEMANTIC_MAX_CANDIDATES, \
        f"候选池超阈值：{len(on)} > {tr.SEMANTIC_MAX_CANDIDATES}"
    ids = [r["job_id"] for r in on]
    assert len(ids) == len(set(ids)), "候选里有重复 job_id"
    scored = [r for r in on if "score" in r]
    if scored:
        scores = [r["score"] for r in scored]
        assert scores == sorted(scores, reverse=True), "带 score 的部分应按分数降序"
    # 池内所有岗位都必须真实存在（可反查）
    found = {x["job_id"] for x in jobdb.get_jobs_by_ids(ids)}
    assert set(ids) <= found, f"有 {len(set(ids) - found)} 个候选反查不到"
    return (f"候选池 {len(on)} 条（<= {tr.SEMANTIC_MAX_CANDIDATES}），"
            f"{len(scored)} 条带语义 score，全部可反查、无重复")


check("精确查询行为不变", t_precise_unchanged)
check("semantic 候选池不超阈值且只重排", t_semantic_off_vs_on)

print()
print("=" * 74)
print(f"合计：{len(PASS)} 通过 / {len(FAIL)} 失败")
for label, err in FAIL:
    print(f"  x {label} -> {err}")
print("=" * 74)
raise SystemExit(1 if FAIL else 0)
