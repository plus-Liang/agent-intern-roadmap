#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测框架自测：不花一分钱、不调 LLM，验证三件交付物是否真的能用。

    python evaluation/test_framework.py

覆盖的验收点
------------
1. test_set.yaml 能加载、题数对得上、每题字段齐全、judge_type 都注册了插件；
2. 三个 ABC 与插件注册表（METRICS / PROVIDERS / REPORTS）齐活；
3. 三种 metric 各自能打分（确定性 / LLM 裁判 / 轨迹）；
4. **轨迹断言能测出「工具调用顺序错」**（正反例都测，含 not_calls_tool 通配、
   call_order 一端缺失不算违规）；
5. TrajectoryRecorder 能记录并从现有工具入口还原（改完不留痕）；
6. 三种 report 能渲染；
7. `--compare` 的 bootstrap 置信区间：真回归能标红、1 题的抖动不标红、可复现。
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent))

# 和 run_eval.py 一样先做隔离：自测即使碰到 agent.*，也不会指向真实简历库 / 投递包
_ART = EVAL_DIR / "_artifacts"
os.environ.setdefault("RESUME_ROOT", str(_ART / "resumes"))
os.environ.setdefault("PACKAGE_DIR", str(_ART / "packages"))
os.environ.setdefault("EXPORT_DIR", str(_ART / "exports"))
os.environ.setdefault("AGENT_ENGINE", "langgraph")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import framework                                                # noqa: E402
import metrics                                                  # noqa: E402
import regression                                               # noqa: E402

CHECKS: list = []


def test(fn):
    CHECKS.append(fn)
    return fn


# ============================== 工具 ==============================

def _report(pass_map: dict, stamp: str = "s", run_at: str = "2026-01-01 00:00:00") -> dict:
    """按 {题号: [每次的 0/1]} 造一份结果 JSON（够 regression 用）。"""
    results = []
    for cid, runs in pass_map.items():
        passed_runs = sum(runs)
        rate = passed_runs / len(runs)
        results.append({"id": cid, "category": "search", "judge_type": "deterministic",
                        "passed": rate >= 0.5, "pass_rate": rate,
                        "passed_runs": passed_runs, "repeat": len(runs),
                        "reason": "", "checks": [], "extra": {},
                        "runs": [{"passed": bool(x)} for x in runs]})
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {"stamp": stamp, "run_at": run_at, "results": results,
            "total": total, "passed": passed,
            "accuracy": passed / total if total else 0.0}


def _case(**kw) -> dict:
    case = {"id": "t-01", "category": "search", "question": "q", "expect": {},
            "judge_type": "deterministic"}
    case.update(kw)
    return case


# ============================== 1. 题库 ==============================

@test
def test_yaml_loads_with_all_cases():
    provider = framework.make_provider("yaml")
    data = provider.load_test_set()
    cases = data["cases"]
    assert len(cases) >= 27, f"题数不对：{len(cases)}"
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "题号有重复"
    for c in cases:
        for key in ("id", "category", "question", "expect", "judge_type"):
            assert key in c and c[key] not in (None, ""), f"{c.get('id')} 缺 {key}"
        assert c["judge_type"] in framework.METRICS, f"{c['id']} 的 judge_type 没插件"
    print(f"      题库 {len(cases)} 题；judge_type 分布 "
          f"{ {t: sum(1 for c in cases if c['judge_type'] == t) for t in sorted(framework.METRICS)} }")
    assert any(c["judge_type"] == "trajectory" for c in cases), "没有任何轨迹题"
    assert any(c.get("trajectory") for c in cases), "没有任何 trajectory 断言"


@test
def test_run_eval_uses_yaml_provider():
    import run_eval                                            # noqa: PLC0415 - 重依赖懒加载
    assert run_eval.TEST_SET.name == "test_set.yaml", run_eval.TEST_SET
    assert run_eval.TEST_SET.is_file()
    assert len(run_eval.make_provider().load_cases()) >= 27


@test
def test_provider_rejects_bad_case():
    provider = framework.make_provider("yaml")
    for bad, why in [
        ([{"id": "x", "category": "search", "expect": {}}], "缺 question/judge_type"),
        ([_case(), _case()], "题号重复"),
        ([_case(judge_type="magic")], "judge_type 没有插件"),
        ([], "空题库"),
    ]:
        try:
            provider.validate(bad)
            raise AssertionError(f"应当报错：{why}")
        except ValueError:
            pass


# ============================== 2. 插件注册表 ==============================

@test
def test_plugins_registered():
    assert set(framework.METRICS) >= {"deterministic", "llm", "trajectory"}
    assert "yaml" in framework.PROVIDERS
    assert set(framework.REPORTS) >= {"json", "markdown", "console"}
    for cls in list(framework.METRICS.values()) + list(framework.PROVIDERS.values()) \
            + list(framework.REPORTS.values()):
        assert issubclass(cls, (framework.BaseMetric, framework.BaseProvider, framework.BaseReport))
    try:
        framework.get_metric("nope")
        raise AssertionError("未知 judge_type 应当报错")
    except KeyError:
        pass


# ============================== 3. metrics ==============================

@test
def test_deterministic_metric():
    metric = framework.DeterministicMetric()
    good = {"checks": [{"name": "城市识别", "ok": True, "detail": ""},
                       {"name": framework.JUDGE_CHECK_NAME, "ok": False, "detail": "裁判挂了"}]}
    assert metric.evaluate(_case(judge_type="llm"), good) == 1.0, "裁判那条不该算进确定性"
    bad = {"checks": [{"name": "城市识别", "ok": False, "detail": "北京 ≠ 广州"}]}
    assert metric.evaluate(_case(), bad) == 0.0 and "城市识别" in metric.detail
    assert metric.evaluate(_case(), {"checks": []}) == 0.0, "没有证据应当不通过"


@test
def test_llm_metric():
    metric = framework.LlmJudgeMetric()
    case = _case(judge_type="llm")
    assert metric.evaluate(case, {"judge": {"pass": True, "reason": "站得住"}}) == 1.0
    assert metric.evaluate(case, {"judge": {"pass": False, "reason": "编造"}}) == 0.0
    assert metric.evaluate(case, {"judge": {}}) == 0.0, "没有裁判结论应当不通过"


@test
def test_trajectory_metric_detects_wrong_order():
    case = _case(judge_type="trajectory",
                 trajectory={"calls_tool": ["search_jobs", "match_resume"],
                             "not_calls_tool": ["generate_application_package"],
                             "call_order": [["search_jobs", "match_resume"]]})
    metric = framework.TrajectoryMetric()
    ok = {"trajectory": {"calls": ["get_job_detail", "search_jobs", "match_resume"]}}
    assert metric.evaluate(case, ok) == 1.0, metric.detail
    wrong = {"trajectory": {"calls": ["match_resume", "search_jobs"]}}
    assert metric.evaluate(case, wrong) == 0.0 and "顺序错" in metric.detail, metric.detail
    leaked = {"trajectory": {"calls": ["search_jobs", "generate_application_package",
                                       "match_resume"]}}
    assert metric.evaluate(case, leaked) == 0.0 and "不应调用" in metric.detail
    missing = {"trajectory": {"calls": ["search_jobs"]}}
    assert metric.evaluate(case, missing) == 0.0 and "缺少调用" in metric.detail
    assert metric.evaluate(case, {}) == 0.0, "拿不到轨迹应当不通过"


@test
def test_trajectory_primitives():
    check = framework.check_trajectory
    assert check({}, ["anything"])[0] is True
    assert check({"not_calls_tool": ["*"]}, [])[0] is True
    ok, detail = check({"not_calls_tool": ["*"]}, ["search_jobs"])
    assert ok is False and "任何工具" in detail
    # call_order 一端缺失 = 不适用（「分数不够就跳过出包」是设计内行为）
    assert check({"call_order": [["match_resume", "generate_application_package"]]},
                 ["match_resume"])[0] is True
    assert check({"call_order": [["a", "b"]]}, ["b", "a", "b"])[0] is False
    assert check({"calls_tool": ["a", "a"]}, ["a"])[0] is True, "同一工具调两次也算调用过"


@test
def test_retrieval_metric_math():
    """检索三指标的口径（Recall@K / MRR / NDCG@K）：纯函数，不花一分钱。"""
    assert metrics.recall_at_k(["a", "b", "c", "d"], ["x", "a", "y", "b", "z", "c"], k=5) == 0.5
    assert metrics.recall_at_k(["a", "b"], ["a", "b"], k=5) == 1.0
    assert metrics.recall_at_k([], ["a"], k=5) == 0.0, "没有相关岗位时不该算「完美召回」"
    assert metrics.recall_at_k(["a"], ["x", "y", "a"], k=2) == 0.0, "K 之外的不算"

    assert abs(metrics.mrr(["c"], ["a", "b", "c"]) - 1 / 3) < 1e-9
    assert metrics.mrr(["a"], ["a", "b"]) == 1.0
    assert metrics.mrr(["z"], ["a", "b"]) == 0.0, "一个都没命中记 0"
    assert abs(metrics.mrr([{"job_id": "a"}], [{"job_id": "b"}, {"job_id": "a"}]) - 0.5) < 1e-9, \
        "检索结果 dict 直接喂也要能算"

    import math
    want = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert abs(metrics.ndcg_at_k(["a", "b"], ["a", "x", "b"], k=10) - want) < 1e-9
    assert metrics.ndcg_at_k(["a"], ["a"], k=10) == 1.0
    assert metrics.ndcg_at_k(["a"], ["x", "y"], k=10) == 0.0
    assert metrics.ndcg_at_k([], ["a"], k=10) == 0.0

    got = metrics.score_retrieval(["a", "b", "c"], ["x", "a", "b", "c"])
    assert got["evaluable"] and got["relevant_count"] == 3 and got["hit_count"] == 3
    assert got["first_rank"] == 2 and abs(got["mrr"] - 0.5) < 1e-9
    assert metrics.score_retrieval([], ["x"])["evaluable"] is False


@test
def test_retrieval_metric_plugins():
    """三个指标插件 + suite 插件都注册了，并且真的按阈值判过/不过。"""
    assert {"recall_at_k", "mrr", "ndcg_at_k", "retrieval"} <= set(framework.METRICS)
    spec = {"recall_k": 5, "ndcg_k": 10, "min_recall_at_k": 0.5, "min_mrr": 0.4,
            "min_ndcg_at_k": 0.6}
    case = _case(judge_type="retrieval", category="retrieval", expect=spec)
    good = {"retrieval": {"recall_at_k": 0.5, "mrr": 0.5, "ndcg_at_k": 0.6,
                          "hit_count": 1, "relevant_count": 2, "first_rank": 1}}
    assert framework.METRICS["recall_at_k"]().evaluate(case, good) == 1.0
    assert framework.METRICS["mrr"]().evaluate(case, good) == 1.0
    assert framework.METRICS["ndcg_at_k"]().evaluate(case, good) == 1.0
    assert framework.METRICS["retrieval"]().evaluate(case, good) == 1.0

    bad = {"retrieval": {"recall_at_k": 0.2, "mrr": 0.5, "ndcg_at_k": 0.6,
                         "hit_count": 1, "relevant_count": 5, "first_rank": 1}}
    metric = framework.METRICS["retrieval"]()
    assert metric.evaluate(case, bad) == 0.0 and "Recall@5" in metric.detail, metric.detail
    assert framework.METRICS["recall_at_k"]().evaluate(case, bad) == 0.0
    # run_eval 的 Case.extra 里也是同一个口径（result.retrieval 缺失时回退 extra）
    assert framework.METRICS["retrieval"]().evaluate(case, {"extra": {"retrieval": bad["retrieval"]}}) == 0.0
    # 不可评（ground truth 里没有相关岗位）必须判不过，不能静默当满分
    assert framework.METRICS["retrieval"]().evaluate(case, {"retrieval": {"skipped": True}}) == 0.0


@test
def test_evaluate_case_pipeline():
    case = _case(judge_type="trajectory", trajectory={"not_calls_tool": ["generate_application_package"]},
                 expect={"check": "no_write_tool"})
    good = {"checks": [{"name": "有回复", "ok": True, "detail": ""}], "judge": {},
            "trajectory": {"calls": ["list_tracking"]}}
    verdict = framework.evaluate_case(case, good)
    assert verdict["passed"] and set(verdict["metrics"]) == {"deterministic", "trajectory"}
    bad = {"checks": [{"name": "有回复", "ok": True, "detail": ""}], "judge": {},
           "trajectory": {"calls": ["generate_application_package"]}}
    verdict = framework.evaluate_case(case, bad)
    assert not verdict["passed"] and "trajectory" in verdict["reason"]
    # 确定性检查挂了，轨迹再干净也整体不过
    rep = {"checks": [{"name": "有回复", "ok": False, "detail": "answer 为空"}], "judge": {},
           "trajectory": {"calls": []}}
    assert not framework.evaluate_case(case, rep)["passed"]


@test
def test_metrics_for_llm_case():
    names = {m.name for m in framework.metrics_for(_case(judge_type="llm"))}
    assert names == {"deterministic", "llm"}, names
    names = {m.name for m in framework.metrics_for(
        _case(judge_type="llm", trajectory={"calls_tool": ["search_jobs"]}))}
    assert names == {"deterministic", "llm", "trajectory"}, "llm 题也能挂轨迹断言"


# ============================== 4. 轨迹记录器 ==============================

@test
def test_recorder_records_all_four_entries_and_restores():
    """四个入口都要记（含 _match/_search 两个别名），退出要还原，嵌套不许重复记。

    只测 `reg.call_tool` 是不够的：把 `reg._search` 改名，那种测试照样绿，
    而它正是 complex 题 `calls_tool` / `call_order` 的唯一数据来源。
    """
    import importlib
    reg = importlib.import_module("agent.tools_registry")
    rea = importlib.import_module("agent.react_agent")
    originals = {"reg.call_tool": reg.call_tool, "reg._search": reg._search,
                 "reg._match": reg._match, "rea.call_tool": rea.call_tool}

    def probe(*args, **kwargs):                                # 不产生副作用、不花钱
        return {"probe": args[0] if args else kwargs.get("name")}

    try:
        reg._search, reg._match, rea.call_tool = probe, probe, probe
        with framework.TrajectoryRecorder() as rec:
            assert reg.call_tool is not originals["reg.call_tool"]
            try:
                reg.call_tool("__probe_unknown_tool__", {})    # 未注册工具 → 立即报错，无副作用
            except Exception:                                  # noqa: BLE001
                pass
            reg._search("kw", city="广州")                      # 别名 → search_jobs
            reg._match("job-1", "{}")                          # 别名 → match_resume
            rea.call_tool("list_tracking", {})                 # ReAct 那条入口
            # 嵌套（run_complex 的 spy 就是这样再包一层，且内层调外层）
            inner = reg.call_tool
            reg.call_tool = (lambda name, args=None, confirmed=False:
                             inner(name, args, confirmed=confirmed))
            try:
                reg.call_tool("search_jobs", {})
            except Exception:                                  # noqa: BLE001
                pass
        assert rec.calls == ["__probe_unknown_tool__", "search_jobs", "match_resume",
                             "list_tracking", "search_jobs"], rec.calls
        assert rec.as_dict()["count"] == 5
        assert reg.call_tool is originals["reg.call_tool"], "退出后必须还原"
        # recorder 还原的是「它挂上去之前的那一份」= 这里的探针；真身由 finally 兜底
        assert rea.call_tool is probe, "ReAct 那条入口也要还原到被包住的那一份"
    finally:
        reg.call_tool, reg._search = originals["reg.call_tool"], originals["reg._search"]
        reg._match, rea.call_tool = originals["reg._match"], originals["rea.call_tool"]


@test
def test_recorder_skips_missing_entry_without_leaking():
    """TARGETS 里某个入口不存在时：跳过它，绝不能把已挂的入口留在模块上。"""
    import importlib
    reg = importlib.import_module("agent.tools_registry")
    original = reg.call_tool
    rec = framework.TrajectoryRecorder()
    rec.TARGETS = [("agent.tools_registry", "call_tool", None),
                   ("agent.tools_registry", "__no_such_entry__", None)]
    with rec:
        assert reg.call_tool is not original
    assert reg.call_tool is original, "缺属性必须只跳过，不许留下假入口"


# ============================== 5. reports ==============================

@test
def test_reports_render():
    results = [{"id": "search-01", "category": "search", "judge_type": "deterministic",
                "question": "q", "passed": True, "pass_rate": 1.0, "passed_runs": 1,
                "repeat": 1, "reason": "全部检查通过", "elapsed": 1.0, "metrics": {},
                "trajectory": {"calls": ["search_jobs"], "ok": True, "detail": "ok"},
                "checks": [], "extra": {}}]
    rep = framework.build_report(results, 1.0, {"version": "2.0"}, repeat=1,
                                test_set_path="evaluation/test_set.yaml")
    md = framework.make_report("markdown").render(rep)
    assert "轨迹断言" in md and "search_jobs" in md and "judge_type" in md
    import json as _json
    assert _json.loads(framework.make_report("json").render(rep))["total"] == 1
    assert "总计 1/1" in framework.make_report("console").render(rep)


@test
def test_provider_save_report(tmp_root=None):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        provider = framework.YamlProvider(results_dir=tmp)
        rep = framework.build_report([], 0.0, {"version": "2.0"})
        paths = provider.save_report(rep)
        names = sorted(p.suffix for p in paths)
        assert names == [".json", ".md"], paths
        for p in paths:
            assert p.is_file() and p.stat().st_size > 0
        loaded = provider.load_report(paths[0])
        assert loaded["total"] == 0 and "plugins" in loaded


# ============================== 6. bootstrap 回归门控 ==============================

@test
def test_bootstrap_detects_regression():
    ids = [f"case-{i:02d}" for i in range(30)]
    prev = _report({i: [1] for i in ids})
    curr = _report({i: ([0] if k < 8 else [1]) for k, i in enumerate(ids)})
    res = regression.compare(prev, curr, iterations=1000, seed=0)
    assert res["overall"]["delta"] < 0
    assert res["overall"]["ci_high"] < 0, res["overall"]
    assert res["overall"]["verdict"] == "regression"
    assert res["regressions"], "回归应当被列出来"
    assert not res["newly_pass"] and len(res["newly_fail"]) == 8


@test
def test_bootstrap_ignores_one_case_noise():
    ids = [f"case-{i:02d}" for i in range(30)]
    prev = _report({i: [1] for i in ids})
    curr = _report({i: ([0] if i == "case-00" else [1]) for i in ids})
    res = regression.compare(prev, curr, iterations=1000, seed=0)
    assert res["overall"]["delta"] < 0, "确实掉了 1 题"
    assert res["overall"]["ci_high"] >= 0, "但区间跨 0，不该判回归"
    assert res["overall"]["verdict"] == "noise"
    assert not res["regressions"], "1 题抖动不该标红"
    # 真在测「区间跨 0」：下界 ≤ 0 ≤ 上界，且区间宽度不塌成一点
    assert res["overall"]["ci_low"] <= 0 <= res["overall"]["ci_high"]
    assert res["overall"]["ci_high"] > res["overall"]["delta"], "区间不该塌成点估计"


@test
def test_bootstrap_handles_empty_intersection():
    res = regression.compare(_report({"a": [1]}), _report({"b": [0]}), iterations=100, seed=0)
    assert res["overall"]["n"] == 0 and res["overall"]["verdict"] == "noise"
    assert res["overall"]["reliable"] is False
    assert res["only_prev"] == ["a"] and res["only_curr"] == ["b"]
    assert "回归门控" in regression.render(res), "没有交集也要能渲染"


@test
def test_case_runs_falls_back_to_passed_runs():
    """没留 runs[] 的老结果：按 passed_runs 还原，不许用 passed 当成全过。"""
    rep = {"results": [
        {"id": "a", "passed": True, "passed_runs": 2, "repeat": 4},
        {"id": "b", "passed": True, "pass_rate": 0.5, "repeat": 4},
        {"id": "c", "passed": False, "passed_runs": 0, "repeat": 1},
        {"id": "d", "passed": True},
        {"category": "search", "passed": True},                # 缺 id → 跳过
    ]}
    got = regression.case_runs(rep)
    assert got == {"a": [1, 1, 0, 0], "b": [1, 1, 0, 0], "c": [0], "d": [1]}, got


@test
def test_bootstrap_detects_improvement_and_repeat_noise():
    ids = [f"case-{i:02d}" for i in range(30)]
    prev = _report({i: ([1, 0, 0] if k < 10 else [1, 1, 1]) for k, i in enumerate(ids)})
    curr = _report({i: [1, 1, 1] for i in ids})
    res = regression.compare(prev, curr, iterations=1000, seed=0)
    assert res["overall"]["verdict"] == "improvement", res["overall"]
    assert res["overall"]["n"] == 30
    # 同一份数据重复跑，结果必须一致（seed 固定）
    again = regression.compare(prev, curr, iterations=1000, seed=0)
    assert again["overall"] == res["overall"]


@test
def test_bootstrap_skips_unshared_cases():
    ids = [f"case-{i:02d}" for i in range(20)]
    prev = _report({i: [1] for i in ids})
    curr = _report({**{i: [0] for i in ids}, "brand-new": [0]})
    res = regression.compare(prev, curr, iterations=500, seed=0)
    assert res["only_curr"] == ["brand-new"], "新增的题要单独列，不参与 delta"
    assert res["overall"]["n"] == 20, res["overall"]


@test
def test_regression_render_text():
    ids = [f"case-{i:02d}" for i in range(12)]
    prev = _report({i: [1] for i in ids})
    curr = _report({i: ([0] if k < 6 else [1]) for k, i in enumerate(ids)})
    text = regression.render(regression.compare(prev, curr, iterations=300, seed=0))
    assert "回归门控" in text and "CI95" in text and "🔴" in text


# ============================== 跑 ==============================

def main() -> int:
    print(f"框架自测：{len(CHECKS)} 项")
    failed = []
    for fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                                 # noqa: BLE001
            failed.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} 项通过"
          + (f"；失败：{', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
