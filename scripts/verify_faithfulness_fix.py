#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回归：追问「第 1 个岗位要求什么技术」的 faithfulness 误判（正确引用被判无依据）。

现场（2026-10-10 20:12，会话 39a8f548 / 74dae708）：
    Q: 第 1 个岗位要求什么技术 + [系统提示 · 岗位序号] ... job_id=inn_qpa38aa45nvn
    A: 墨泊可士 · AI Agent开发（可转正）[1]
       - 熟悉 Python/Java/Go/php 中的至少一种...[1]
       - 深入掌握 LangChain、LangGraph、Llamalndex 等...[1]
    → faithfulness 把第 2、3 句标成「没有找到依据」，而这两句逐字来自 chunk 1。

根因：系统提示混进检索问句（11 字 → 59 字），越过查询理解的长度门槛被重写 + 拆子查询，
走 `_retrieve_merged` 的**岗位级去重**（`_dedupe_hits_by_job`），同一条 JD 只剩一个 chunk
（实测只剩「岗位职责」那段，chunk 0）——「任职要求」（chunk 1）从未进入证据集，
逐字引用的句子自然 coverage≈0.13~0.23 < 0.25，被判「无依据」。

跑法： python scripts/verify_faithfulness_fix.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from agent import langgraph_flow as L          # noqa: E402
from agent import tools_registry as reg        # noqa: E402
from rag import citation as C                  # noqa: E402

PASS, FAIL = [], []

JOB_ID = "inn_qpa38aa45nvn"
HINT = ("\n\n[系统提示 · 岗位序号] 用户说的「第 1 个」= 上一次 search_jobs 结果里 "
        f"index=1 的那条岗位：job_id={JOB_ID}，公司=墨泊可士，岗位=AI Agent开发（可转正），"
        "城市=广州，链接=https://www.shixiseng.com/intern/inn_qpa38aa45nvn。"
        "请**直接用它**，不要重新搜索，也不要换成别的岗位。")
QUESTION = "第 1 个岗位要求什么技术"

#: 现场答案的第 2、3 句（逐字来自「任职要求」chunk）
SENT_REQ = "- 熟悉Python/Java/Go/php中的至少一种，具备扎实的代码能力和系统设计能力"
SENT_FRAME = "- 深入掌握LangChain、LangGraph、Llamalndex等主流智能体开发框架，有实际项目落地经验优先"
#: 故意编造的句子（库里没有的技术 / 薪资）
SENT_FAKE = "该岗位还要求 8 年 Kubernetes 集群运维经验，月薪 15 万。"


def check(label, fn):
    try:
        detail = fn()
    except AssertionError as exc:
        FAIL.append((label, str(exc)))
        print(f"[FAIL] {label} → {exc}")
        return None
    except Exception as exc:                        # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {label} → {type(exc).__name__}: {exc}")
        return None
    PASS.append((label, detail))
    print(f"[ OK ] {label} → {detail}")
    return detail


def t_strip_hint():
    """系统提示必须被剥掉：剥完是用户的问句，且短到不会触发查询理解。"""
    stripped = L._strip_hints(QUESTION + HINT)
    assert stripped == QUESTION, f"剥离结果不对：{stripped!r}"
    assert L._strip_hints(QUESTION) == QUESTION, "没有提示时不该改动问句"
    from rag.query_understanding import should_understand
    assert not should_understand(stripped), (
        f"剥离后仍会触发查询理解（长度 {len(stripped)}）")
    assert should_understand(QUESTION + HINT), (
        "带提示的长文本本应触发查询理解（说明这是真实的触发条件）")
    return f"{len(QUESTION)} 字 vs 带提示 {len(QUESTION + HINT)} 字"


def t_retrieval_keeps_all_chunks():
    """修复口径的检索必须把同一条 JD 的两段都召回（去重前的症状是只留 1 条）。"""
    hits = reg._retrieve(L._strip_hints(QUESTION + HINT), top_k=8,
                         allowed_job_ids=[JOB_ID]) or []
    ids = [h["id"] for h in hits]
    assert len(hits) >= 2, f"只召回 {len(hits)} 条：{ids}"
    assert any(i.endswith(":0") for i in ids) and any(i.endswith(":1") for i in ids), (
        f"「岗位职责」/「任职要求」两段没都召回：{ids}")
    return f"召回 {ids}"


def t_sentences_supported():
    """现场那两句（逐字来自 chunk 1）不能再被判「无依据」。"""
    hits = reg._retrieve(L._strip_hints(QUESTION + HINT), top_k=8,
                         allowed_job_ids=[JOB_ID]) or []
    pack = C.build_citation_pack(SENT_REQ + "\n" + SENT_FRAME, hits,
                                 use_llm_faithfulness=False)
    unsupported = [r["text"] for r in pack["faithfulness"]["unsupported"]]
    assert not unsupported, f"仍被误判：{unsupported}"
    assert len(pack["faithfulness"]["supported"]) == 2, (
        f"supported={len(pack['faithfulness']['supported'])}")
    for s in pack["sentences"]:
        assert s["citations"], f"正确引用的句子没有标上 [n]：{s['text']}"
    return "两句都判「有依据」且带引用"


def t_hallucination_still_caught():
    """放宽证据集后，编造的句子必须仍被抓（不能因为修误判而漏抓）。"""
    hits = reg._retrieve(L._strip_hints(QUESTION + HINT), top_k=8,
                         allowed_job_ids=[JOB_ID]) or []
    for use_llm in (False, None):
        pack = C.build_citation_pack(SENT_REQ + "\n" + SENT_FAKE, hits,
                                     use_llm_faithfulness=use_llm)
        bad = [r["text"] for r in pack["faithfulness"]["unsupported"]]
        good = [r["text"] for r in pack["faithfulness"]["supported"]]
        assert any("Kubernetes" in t for t in bad), (
            f"幻觉句没被抓（use_llm={use_llm}）：unsupported={bad} mode={pack['faithfulness']['mode']}")
        assert not any("Kubernetes" in t for t in good), "幻觉句被判成有依据"
        assert any("Python" in t for t in good), (
            f"真实句子被误判（use_llm={use_llm}）：unsupported={bad}")
    return "确定性闸 + LLM 闸都抓到幻觉句、真实句仍在 supported"


def main() -> int:
    print("== 1. 系统提示剥离 ==")
    check("剥离系统提示且不触发查询理解", t_strip_hint)
    print("\n== 2. 检索召回同一条 JD 的全部 chunk ==")
    check("两段都召回", t_retrieval_keeps_all_chunks)
    print("\n== 3. 误判消失 ==")
    check("正确引用的句子不再标「无依据」", t_sentences_supported)
    print("\n== 4. 幻觉仍抓 ==")
    check("编造句子仍在 unsupported", t_hallucination_still_caught)
    print(f"\n合计：{len(PASS)} 通过 / {len(FAIL)} 失败")
    for label, err in FAIL:
        print(f"  [FAIL] {label}：{err}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
