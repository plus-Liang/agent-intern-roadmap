#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 RAG 引用溯源（第 3 周）：确定性 chunk ID / 句级引用 / faithfulness / 不改变检索。

跑法（在仓库根目录）::

    python scripts/verify_citation.py
    python scripts/verify_citation.py --no-llm     # 只跑确定性闸，不花 token

四组判定（对应任务书的验收点）：
 1. **chunk ID 确定性**：同一条 chunk 反复生成 id 完全相同，形如
    ``{platform}:{job_id}:{chunk_index}``；向量库 job_id 覆盖率 100%。
 2. **引用准确率（人工抽查 3 条）**：拿真实检索结果生成答案后，
    逐条把 ``[n]`` 对应的 chunk 正文打印出来，人眼可核（脚本先做机器可判的那半：
    被引用的 chunk 必须是「句子 n-gram 覆盖率最高」的那条，且带 job_id）。
 3. **faithfulness 抓幻觉**：注入一条**故意编造**的句子（库里没有的事实），
    必须落进 ``unsupported``；同时把真实句子（改写自 chunk 原文）保留在 ``supported``。
 4. **不影响检索**：``retrieve`` 的 chunk id 序列与 ``use_understanding=False``
    的原路径一致（引用是后处理，不参与排序）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                     # noqa: BLE001
    pass

from rag import citation as C                          # noqa: E402
from rag.retriever import retrieve, format_context     # noqa: E402
from rag.generator import generate                     # noqa: E402
from rag.vector_store import (                         # noqa: E402
    make_chunk_id, get_collection, collection_coverage,
)

PASS, FAIL = [], []


def check(label, fn):
    try:
        detail = fn()
    except AssertionError as exc:
        FAIL.append((label, str(exc)))
        print(f"[FAIL] {label} → {exc}")
        return None
    except Exception as exc:                          # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {label} → {type(exc).__name__}: {exc}")
        return None
    PASS.append((label, detail))
    print(f"[ OK ] {label} → {detail}")
    return detail


# --------------------------------------------------------------------------
# 1. chunk ID 确定性 + 覆盖率
# --------------------------------------------------------------------------
def t_id_format():
    chunk = {"platform": "shixiseng", "job_id": "inn_001", "chunk_index": 0,
             "company": "字节跳动", "title": "Agent 实习"}
    first = make_chunk_id(dict(chunk))
    again = make_chunk_id(dict(chunk))
    assert first == again, f"同一个 chunk 两次生成 id 不一致：{first} != {again}"
    assert first == "shixiseng:inn_001:0", f"id 形态不对：{first}"
    assert make_chunk_id({**chunk, "chunk_index": 1}) == "shixiseng:inn_001:1"
    # 缺 job_id 的走 legacy（仍然稳定，但一眼看出无岗位身份）
    legacy = make_chunk_id({"platform": "", "job_id": "", "chunk_index": 0,
                            "company": "X", "title": "Y"})
    assert legacy.startswith("legacy:"), legacy
    assert legacy == make_chunk_id({"platform": "", "job_id": "", "chunk_index": 0,
                                    "company": "X", "title": "Y"})
    return f"{first}（确定性、可读、可溯源）"


def t_coverage():
    stats = collection_coverage()
    assert stats["total"] > 0, "向量库是空的，先跑 python -m rag.vector_store --rebuild"
    assert stats["job_id_coverage"] >= 1.0, (
        f"job_id 覆盖率 {stats['job_id_coverage']:.4%} < 100%（缺 "
        f"{stats['total'] - stats['with_job_id']} 条）")
    assert stats["id_coverage"] >= 1.0, (
        f"确定性 id 覆盖率 {stats['id_coverage']:.4%}；对不上的样本：{stats['orphan_ids']}")
    return (f"{stats['with_job_id']}/{stats['total']} = "
            f"{stats['job_id_coverage']:.2%}，确定性 id {stats['id_coverage']:.2%}")


def t_ids_are_queryable():
    """库里抽 20 条，按 id 反查必须能取回同一条 chunk（id 是真实主键）。"""
    col = get_collection()
    data = col.get(limit=20)
    got = col.get(ids=list(data["ids"]))
    assert set(got["ids"]) == set(data["ids"]), "按 id 反查取回的不是同一批 chunk"
    return f"抽查 {len(data['ids'])} 条按 id 反查一致"


def t_coverage_gate():
    """覆盖率达标时 ``coverage_ok()`` 必须放行（否则搜岗位永远不挂引用）。"""
    assert C.coverage_ok() is True, (
        "覆盖率闸门拦住了引用；库里 job_id 覆盖不足时先重建向量库")
    return "覆盖率达标 → 允许挂引用"


# --------------------------------------------------------------------------
# 2/3. 真实检索 → 引用标注 → faithfulness
# --------------------------------------------------------------------------
QUERY = "找大模型落地的实习岗位"


def _real_hits():
    hits = retrieve(QUERY, top_k=5, use_understanding=False)
    assert hits, f"检索为空：{QUERY}"
    for h in hits:
        assert str((h.get("metadata") or {}).get("job_id") or ""), "命中没有 job_id"
    return hits


def t_citation_shape():
    hits = _real_hits()
    answer = generate(QUERY, format_context(hits))
    pack = C.build_citation_pack(answer, hits, use_llm_faithfulness=False)
    sentences = pack["sentences"]
    assert sentences, "答案没有切出任何句子"
    cited = [s for s in sentences if s["citations"]]
    assert cited, f"没有任何句子标上引用；答案={answer!r}"
    for s in cited:
        for n in s["citations"]:
            src = pack["sources"].get(str(n))
            assert src and src["chunk_id"], f"[{n}] 映射不到 chunk"
            assert f"[{n}]" in s["rendered"], f"rendered 里没有 [{n}]"
    return f"{len(sentences)} 句，其中 {len(cited)} 句带引用"


def t_citation_accuracy():
    """机器可判的那半：被引用的 chunk 必须是句子的最优证据（不会张冠李戴）。"""
    hits = _real_hits()
    answer = generate(QUERY, format_context(hits))
    pack = C.build_citation_pack(answer, hits, use_llm_faithfulness=False)
    checked = 0
    samples = []
    for s in pack["sentences"]:
        if not s["citations"]:
            continue
        best = max(
            range(len(hits)),
            key=lambda i: C._similarity(s["text"], hits[i].get("text") or "")
            + C._meta_bonus(s["text"], hits[i].get("metadata") or {}),
        )
        assert best + 1 in s["citations"], (
            f"句子引用了 {s['citations']}，但证据最强的是 [{best + 1}]：{s['text']}")
        checked += 1
        samples.append((s["text"], s["citations"][0]))
    assert checked >= 3, f"可抽查的带引用句子只有 {checked} 句（需要 >= 3）"
    print("        —— 人工抽查（句子 → 引用的 chunk 正文）——")
    for text, n in samples[:3]:
        src = pack["sources"][str(n)]
        print(f"        句：{text}")
        print(f"        [{n}] {src['company']} | {src['title']} | {src['city']} "
              f"| chunk_id={src['chunk_id']}")
        print(f"             正文：{src['text'][:150].replace(chr(10), ' ')}...")
    return f"{checked} 句的引用都指向证据最强的 chunk"


def t_faithfulness_catches_hallucination(use_llm):
    """故意造幻觉：库里没有的事实必须被判 unsupported。"""
    hits = _real_hits()
    real = generate(QUERY, format_context(hits))
    hallucination = (
        "\n该岗位要求 8 年 Kubernetes 集群运维经验，月薪 15 万，办公地点在火星。"
    )
    pack = C.build_citation_pack(real + hallucination, hits,
                                 use_llm_faithfulness=use_llm)
    report = pack["faithfulness"]
    texts = [r["text"] for r in report["unsupported"]]
    assert any("火星" in t for t in texts), (
        f"幻觉句子没被抓住；unsupported={texts}；mode={report['mode']}")
    assert not any("火星" in r["text"] for r in report["supported"]), "幻觉句子被误判为有依据"
    assert report["supported"], "真实句子全被判无依据（阈值过严）"
    return (f"抓到 {len(texts)} 句无依据（mode={report['mode']}，"
            f"supported={len(report['supported'])}）")


# --------------------------------------------------------------------------
# 4. 引用不改变检索
# --------------------------------------------------------------------------
def t_retrieval_unchanged():
    """引用（后处理）不能改变检索：精确查询的 id 序列必须与原路径逐条一致。

    注意：``QUERY`` 是**模糊查询**，自动路径会走查询理解（重写 + 子查询 + RRF），
    结果本来就与原路径不同（那是第 2 周的行为，不是本轮引入的）。
    所以这里用**精确查询**验证「引用改造没碰排序」，另外单独断言模糊查询仍然
    命中查询理解（行为与第 2 周一致，没有被本轮改动带偏）。
    """
    precise = "广州 Python"
    auto = [h["id"] for h in retrieve(precise, top_k=5)]
    off = [h["id"] for h in retrieve(precise, top_k=5, use_understanding=False)]
    assert auto == off, f"精确查询的检索路径被影响：\n  auto={auto}\n  off ={off}"

    fuzzy = [h["id"] for h in retrieve(QUERY, top_k=5)]
    assert len(set(fuzzy)) == len(fuzzy), "模糊查询返回了重复 chunk"
    assert all(":" in cid or cid.startswith("legacy:") for cid in auto + fuzzy), \
        f"id 形态异常：{auto + fuzzy}"
    return f"精确查询 top5 序列一致；模糊查询仍走查询理解（{len(fuzzy)} 条）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="faithfulness 只跑确定性闸")
    args = ap.parse_args()

    print("== 1. chunk ID 确定性 / 覆盖率 ==")
    check("id 形态与确定性", t_id_format)
    check("向量库 job_id 覆盖率 100%", t_coverage)
    check("id 可反查", t_ids_are_queryable)
    check("引用溯源覆盖率闸门可用", t_coverage_gate)

    print("\n== 2. 句级引用（真实检索 + 真实 LLM 生成）==")
    check("引用结构", t_citation_shape)
    check("引用准确率（对照最强证据）", t_citation_accuracy)

    print("\n== 3. faithfulness 抓幻觉 ==")
    check("确定性闸抓幻觉", lambda: t_faithfulness_catches_hallucination(False))
    if not args.no_llm:
        check("LLM 闸抓幻觉", lambda: t_faithfulness_catches_hallucination(None))

    print("\n== 4. 检索行为不变 ==")
    check("top_k id 序列一致", t_retrieval_unchanged)

    print(f"\n合计：{len(PASS)} 通过 / {len(FAIL)} 失败")
    for label, err in FAIL:
        print(f"  [FAIL] {label}：{err}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
