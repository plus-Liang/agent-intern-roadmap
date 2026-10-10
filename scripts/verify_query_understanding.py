#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 RAG 查询理解（第 2 周）：触发规则 / 重写 / 子查询 / 前后检索对比。

跑法::

    python scripts/verify_query_understanding.py

它会真的调 LLM（重写 + 分解各一次，命中缓存后不再调），并真的跑检索，最后打一张
「改造前 vs 改造后」的对比表。判定口径：
  * 触发规则：精确查询必须**不触发**、模糊查询必须**触发**；
  * 精确查询路径：`retrieve(自动)` 与 `retrieve(use_understanding=False)` 返回的
    chunk id 序列必须**逐条相同**（= 没动到原路径）；
  * Agent 语义路径：`_search_rows(semantic=True)` 的**岗位集合**必须相同
    （评测的 search 题判的就是城市 / 类型 / 条数这些集合属性，顺序变化不影响通过率）；
  * 模糊查询：改造后「去重岗位数」与「与问题的平均向量相似度」都不该变差。
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag import query_understanding as QU          # noqa: E402
from rag.retriever import retrieve                 # noqa: E402
from rag.embedder import embed_one, embed_query    # noqa: E402

PRECISE = "广州 Python"
#: 长但只有一个概念的精查询（15 字，Agent 评测里就是这类句子）
PRECISE_LONG = "帮我找广州的 Agent 实习岗位"
FUZZY = "找偏大模型落地能写工程代码的实习"

PASS, FAIL = [], []


def check(label, fn):
    try:
        detail = fn()
    except AssertionError as e:
        FAIL.append((label, str(e)))
        print(f"[FAIL] {label} → {e}")
        return
    except Exception as e:                          # noqa: BLE001
        FAIL.append((label, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {label} → {type(e).__name__}: {e}")
        return
    PASS.append((label, detail))
    print(f"[ OK ] {label} → {detail}")


def ids_of(hits):
    return [h["id"] for h in hits]


def jobs_of(hits):
    return [str((h.get("metadata") or {}).get("job_id") or "") for h in hits]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def similarity(question, hits):
    """每条命中片段与问题的向量相似度（用建库那个本地 embedding 模型算，离线可跑）。"""
    qv = embed_query(question)
    return [cosine(qv, embed_one((h.get("text") or "")[:512])) for h in hits]


def titles(hits, n=5):
    out = []
    for h in hits[:n]:
        m = h.get("metadata") or {}
        out.append(f"{m.get('company')}·{m.get('title')}")
    return out


# ---------------------------------------------------------------------------
print("=" * 78)
print("① 触发规则")
print("=" * 78)

check("精确查询「广州 Python」不触发",
      lambda: "不触发" if not QU.should_understand(PRECISE)
      else (_ for _ in ()).throw(AssertionError("被误判为模糊查询")))

check("精确查询「帮我找广州的 Agent 实习岗位」（15 字 / 1 概念）不触发",
      lambda: "不触发" if not QU.should_understand(PRECISE_LONG)
      else (_ for _ in ()).throw(AssertionError("被误判为模糊查询")))

check("模糊查询「找偏大模型落地能写工程代码的实习」触发",
      lambda: " / ".join(QU.trigger_reasons(FUZZY))
      if QU.should_understand(FUZZY)
      else (_ for _ in ()).throw(AssertionError("没触发")))

check("短句多概念「大模型算法」触发",
      lambda: " / ".join(QU.trigger_reasons("大模型算法"))
      if QU.should_understand("大模型算法")
      else (_ for _ in ()).throw(AssertionError("没触发")))

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("② 查询理解 plan（精确 / 模糊 / 缓存）")
print("=" * 78)

QU.clear_cache()
precise_plan = QU.plan_queries(PRECISE)


def _precise_plan_ok():
    assert not precise_plan["triggered"], precise_plan
    assert precise_plan["rewritten"] == PRECISE, precise_plan
    assert precise_plan["sub_queries"] == [], precise_plan
    return f"triggered=False，rewritten 原样，子查询 0 个：{precise_plan['reason']}"


check("精确查询的 plan：不触发、不改写、不拆子查询", _precise_plan_ok)

fuzzy_plan = QU.plan_queries(FUZZY)


def _fuzzy_plan_ok():
    assert fuzzy_plan["triggered"], fuzzy_plan
    assert fuzzy_plan["rewritten"] and fuzzy_plan["rewritten"] != FUZZY, fuzzy_plan
    subs = fuzzy_plan["sub_queries"]
    assert 1 <= len(subs) <= QU.MAX_SUB_QUERIES, f"子查询 {len(subs)} 个：{subs}"
    return (f"重写「{fuzzy_plan['rewritten']}」；子查询 {len(subs)} 个 {subs}")


check("模糊查询的 plan：重写 + 拆 1~3 个子查询", _fuzzy_plan_ok)


def _cache_hit():
    again = QU.plan_queries(FUZZY)
    assert again["cached"], again
    assert again["rewritten"] == fuzzy_plan["rewritten"], "缓存内容不一致"
    assert again["sub_queries"] == fuzzy_plan["sub_queries"], "缓存内容不一致"
    return f"第二次命中缓存（不重复调 LLM），缓存 {QU.cache_info()}"


check("同样的问题第二次走缓存", _cache_hit)


def _ttl_expires():
    key = FUZZY
    with QU._CACHE_LOCK:
        QU._CACHE[key]["at"] -= QU.CACHE_TTL_SECONDS + 1
    assert QU.cache_get(key) is None, "超过 TTL 还命中缓存"
    QU.clear_cache()
    fresh = QU.plan_queries(FUZZY)                  # 重新算一次，给后面检索用
    assert fresh["triggered"] and not fresh["cached"], fresh
    return f"TTL {QU.CACHE_TTL_SECONDS:.0f}s 后缓存失效并重算；缓存 {QU.cache_info()}"


check("缓存 TTL 1 小时后失效", _ttl_expires)

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("③ 精确查询：原路径未被改动")
print("=" * 78)


def _precise_path_identical():
    auto = retrieve(PRECISE, top_k=10)
    legacy = retrieve(PRECISE, top_k=10, use_understanding=False)
    assert ids_of(auto) == ids_of(legacy), f"自动 {ids_of(auto)} != 原路径 {ids_of(legacy)}"
    forced = retrieve(PRECISE, top_k=10, use_understanding=True)
    return (f"自动/原路径 chunk 序列完全一致（{len(auto)} 条）；"
            f"强制重写后 {len(forced)} 条（仅作对照，不进主流程）")


check("「广州 Python」检索结果与改造前逐条一致", _precise_path_identical)

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("④ 模糊查询：改造前后检索对比")
print("=" * 78)

before = retrieve(FUZZY, top_k=10, use_understanding=False)
# 取「这次真正用于检索」的 plan（上面 TTL 那步重算过，缓存里是最新的一份）
fuzzy_plan = QU.plan_queries(FUZZY)
after = retrieve(FUZZY, top_k=10, use_understanding=True)

sim_before = similarity(FUZZY, before)
sim_after = similarity(FUZZY, after)
avg = lambda xs: sum(xs) / len(xs) if xs else 0.0            # noqa: E731
jobs_before, jobs_after = set(jobs_of(before)), set(jobs_of(after))
jobs_before.discard("")
jobs_after.discard("")

print(f"查询：{FUZZY}")
print(f"重写：{fuzzy_plan['rewritten']}")
print(f"子查询：{fuzzy_plan['sub_queries']}")
print("-" * 78)
print(f"{'':<14}{'片段数':>6}{'去重岗位数':>12}{'top1 相似度':>12}"
      f"{'top3 相似度':>12}{'平均相似度':>12}")
print(f"{'改造前':<14}{len(before):>6}{len(jobs_before):>12}"
      f"{avg(sim_before[:1]):>12.4f}{avg(sim_before[:3]):>12.4f}{avg(sim_before):>12.4f}")
print(f"{'改造后':<14}{len(after):>6}{len(jobs_after):>12}"
      f"{avg(sim_after[:1]):>12.4f}{avg(sim_after[:3]):>12.4f}{avg(sim_after):>12.4f}")
print("-" * 78)
print("（相似度 = 命中片段与原始问题的向量余弦，用建库那个本地 embedding 算，离线可跑）")
print(f"改造前 top5：{titles(before)}")
print(f"改造后 top5：{titles(after)}")
print(f"改造后新增覆盖的岗位：{len(jobs_after - jobs_before)} 个；"
      f"丢失：{len(jobs_before - jobs_after)} 个")


def _merged_dedupes_by_job():
    assert len(jobs_of(after)) == len(jobs_after), "改造后同一岗位出现了两次"
    assert len(jobs_after) == len(after), "改造后没有按 job_id 去重"
    return f"{len(after)} 条片段 = {len(jobs_after)} 个岗位（已按 job_id 去重）"


check("改造后按 job_id 去重", _merged_dedupes_by_job)


def _coverage_not_worse():
    assert len(jobs_after) >= len(jobs_before), \
        f"覆盖岗位数变少：{len(jobs_before)} → {len(jobs_after)}"
    return f"覆盖岗位数 {len(jobs_before)} → {len(jobs_after)}"


check("改造后覆盖岗位数不低于改造前", _coverage_not_worse)


def _relevance_not_worse():
    b, a = avg(sim_before[:3]), avg(sim_after[:3])
    assert a >= b - 0.02, f"top3 平均相似度变差：{b:.4f} → {a:.4f}"
    return f"top3 平均相似度 {b:.4f} → {a:.4f}（整体均值 {avg(sim_before):.4f} → {avg(sim_after):.4f}）"


check("改造后 top3 相关度不明显变差（容差 0.02）", _relevance_not_worse)

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("⑤ 回归：Agent 语义路径（search_jobs semantic=True）的岗位集合不变")
print("=" * 78)


def _agent_semantic_set_unchanged():
    from agent import tools_registry as reg
    os.environ[QU.ENV_ENABLED] = "off"
    try:
        legacy = reg._search_rows(FUZZY, None, 20, True, "实习")
    finally:
        os.environ[QU.ENV_ENABLED] = "on"
    current = reg._search_rows(FUZZY, None, 20, True, "实习")
    legacy_ids = {str(r.get("job_id")) for r in legacy}
    current_ids = {str(r.get("job_id")) for r in current}
    assert legacy_ids == current_ids, \
        f"岗位集合变了：多 {legacy_ids - current_ids}，少 {current_ids - legacy_ids}"
    return (f"候选 {len(legacy_ids)} 个岗位集合一致；"
            f"顺序前 3：{[(r.get('company'), r.get('title')) for r in current[:3]]}")


check("语义搜索的候选集合与改造前一致（只影响排序）", _agent_semantic_set_unchanged)

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
for label, detail in FAIL:
    print(f"  [FAIL] {label} → {detail}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
