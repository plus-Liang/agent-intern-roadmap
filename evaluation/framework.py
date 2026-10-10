#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测框架：三个 ABC（BaseMetric / BaseProvider / BaseReport）+ 可插拔插件。

为什么要有这一层
----------------
改造前 evaluation/run_eval.py 把「跑题」「判分」「存报告」写死在一个 800 行脚本里，
加一种判分方式（比如轨迹断言）就得改主流程。现在拆成三个可替换的角色：

    Provider  题库与报告的读写（题库 YAML → cases；结果 dict → results/*.json|.md）
    Metric    把一次执行的结果打成 0~1 分（确定性 / LLM 裁判 / 轨迹断言）
    Report    把整轮结果渲染成人看的文本（Markdown / 控制台 / JSON）

三者的 ABC 就是插件的挂载点，注册表见 METRICS / PROVIDERS / REPORTS。
新增一种判分方式 = 写一个 BaseMetric 子类 + `@register_metric`，主流程一行不用改。

一次判分的完整口径（evaluate_case）
----------------------------------
* 所有 metric 都必须得 1.0 分，整题才算 pass；
* **确定性检查永远先跑**（runner 已经产出的 checks），judge_type 只决定「额外」挂哪个
  metric：deterministic → 就它自己；llm → 加 LLM 裁判；trajectory → 加轨迹断言；
* 任何 judge_type 的题都可以再挂一段 `trajectory:` 断言（轨迹是证据，不是另一套题）。

对外只依赖 `result` 这个 dict，字段约定：
    result["checks"]      list[{"name","ok","detail"}]  runner 产出的确定性检查
    result["judge"]       {"pass": bool, "reason": str} | None    runner 调 LLM 裁判的结论
    result["trajectory"]  {"calls": list[str]} | None   工具调用序列（按发生顺序）
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent

#: LLM 裁判在 checks 里的名字：它归 LlmJudgeMetric 管，确定性 metric 不重复计
JUDGE_CHECK_NAME = "裁判复核"

#: by_judge / 报表里的中文标签
JUDGE_LABELS = {"deterministic": "确定性判分", "llm": "LLM 裁判", "trajectory": "轨迹断言",
                "retrieval": "检索指标"}

#: 单题通过率落在 [0.2, 0.8] 视为「不稳定」（绝对值 0% / 100% 反而是稳定结论）
UNSTABLE_LOW = 0.2
UNSTABLE_HIGH = 0.8


# ============================== BaseMetric ==============================

class BaseMetric(ABC):
    """判分插件：`evaluate(case, result) -> score`（0.0~1.0），说明写在 self.detail 里。"""

    #: 注册名（= judge_type）
    name: str = "metric"
    #: 中文标签（报表用）
    label: str = ""

    def __init__(self) -> None:
        self.score: float = 0.0
        self.detail: str = ""

    @abstractmethod
    def evaluate(self, case: dict, result: dict) -> float:
        """给一次执行打分。返回 0.0~1.0，同时把一句话理由写进 self.detail。"""

    def _verdict(self, ok: bool, detail: str) -> float:
        self.score = 1.0 if ok else 0.0
        self.detail = str(detail)[:300]
        return self.score

    def as_dict(self) -> dict:
        return {"metric": self.name, "score": round(self.score, 4), "detail": self.detail}


class DeterministicMetric(BaseMetric):
    """确定性判分：runner 产出的所有 checks 全过才得 1 分（LLM 裁判那条不算）。"""

    name = "deterministic"
    label = "确定性判分"

    def evaluate(self, case: dict, result: dict) -> float:
        checks = [c for c in (result.get("checks") or []) if c.get("name") != JUDGE_CHECK_NAME]
        if not checks:
            return self._verdict(False, "没有任何确定性检查项（runner 没产出证据）")
        bad = [c for c in checks if not c.get("ok")]
        if not bad:
            return self._verdict(True, f"{len(checks)} 项确定性检查全过")
        return self._verdict(False, "；".join(f"{c['name']}：{c.get('detail', '')}" for c in bad))


class LlmJudgeMetric(BaseMetric):
    """LLM 裁判判分：runner 用独立 judge prompt 得到 (是否通过, 理由)，这里只做记账。"""

    name = "llm"
    label = "LLM 裁判"

    def evaluate(self, case: dict, result: dict) -> float:
        judge = result.get("judge") or {}
        if not judge:
            return self._verdict(False, "没有拿到 LLM 裁判结论（judge 为空）")
        return self._verdict(bool(judge.get("pass")), str(judge.get("reason") or ""))


def check_trajectory(spec: dict, calls: list) -> tuple:
    """轨迹断言核心（纯函数，方便单测）。

    支持三条：
      calls_tool      列表里每个工具都至少被调用一次（顺序无关）
      not_calls_tool  列表里每个工具都不能出现；写 "*" = 任何工具都不许调
      call_order      [[A, B], ...]：A 必须早于 B；**任一端缺失时该条不判违规**
                      （例如「分数不够就跳过出包」是设计内行为，不能算顺序错）
    返回 (是否通过, 说明)。
    """
    calls = [str(c) for c in (calls or [])]
    if not spec:
        return True, "本题没有轨迹断言"
    problems = []
    for tool in spec.get("calls_tool") or []:
        if tool not in calls:
            problems.append(f"缺少调用 {tool}")
    for tool in spec.get("not_calls_tool") or []:
        if tool == "*":
            if calls:
                problems.append(f"不应调用任何工具，实际调了 {sorted(set(calls))}")
        elif tool in calls:
            problems.append(f"不应调用 {tool}")
    for pair in spec.get("call_order") or []:
        if len(pair) != 2:
            problems.append(f"call_order 片段不是二元组：{pair}")
            continue
        before, after = str(pair[0]), str(pair[1])
        if before in calls and after in calls and calls.index(before) > calls.index(after):
            problems.append(f"顺序错：{before} 必须早于 {after}")
    if problems:
        return False, "；".join(problems) + f"（实际序列 {calls}）"
    return True, f"轨迹符合断言（实际序列 {calls}）"


class TrajectoryMetric(BaseMetric):
    """轨迹断言：只看工具调用序列，不看最终文本（借 dryfire 的思路）。"""

    name = "trajectory"
    label = "轨迹断言"

    def evaluate(self, case: dict, result: dict) -> float:
        spec = case.get("trajectory") or {}
        if not spec:
            return self._verdict(True, "本题没有轨迹断言")
        trace = result.get("trajectory") or {}
        calls = trace.get("calls")
        if calls is None:
            return self._verdict(False, "没拿到工具调用轨迹（trajectory.calls 为空）")
        ok, detail = check_trajectory(spec, calls)
        return self._verdict(ok, detail)


# ---- 注册表：judge_type → metric 插件 ----

METRICS: dict = {}
PROVIDERS: dict = {}
REPORTS: dict = {}


def register_metric(cls):
    """`@register_metric` 挂载判分插件；类的 name 就是 judge_type。"""
    METRICS[cls.name] = cls
    return cls


def register_provider(cls):
    PROVIDERS[cls.name] = cls
    return cls


def register_report(cls):
    REPORTS[cls.name] = cls
    return cls


register_metric(DeterministicMetric)
register_metric(LlmJudgeMetric)
register_metric(TrajectoryMetric)


def get_metric(judge_type: str) -> BaseMetric:
    cls = METRICS.get(str(judge_type or "deterministic"))
    if cls is None:
        raise KeyError(f"没有注册的判分插件 judge_type={judge_type!r}；"
                       f"可用：{sorted(METRICS)}")
    return cls()


def metrics_for(case: dict) -> list:
    """这一题要挂哪些 metric：确定性检查永远跑，judge_type 决定额外挂谁。"""
    judge_type = str(case.get("judge_type") or "deterministic")
    metrics = [DeterministicMetric()]
    if judge_type != "deterministic":
        metrics.append(get_metric(judge_type))
    # 任何 judge_type 都可以再挂轨迹断言（轨迹是额外证据）
    if judge_type != "trajectory" and case.get("trajectory"):
        metrics.append(TrajectoryMetric())
    return metrics


def evaluate_case(case: dict, result: dict) -> dict:
    """按插件口径判一题：所有 metric 都 1.0 才算 pass。"""
    records = {}
    for metric in metrics_for(case):
        try:
            metric.evaluate(case, result)
        except Exception as e:                                  # noqa: BLE001
            metric.score, metric.detail = 0.0, f"判分异常：{type(e).__name__}: {e}"
        records[metric.name] = metric.as_dict()
    passed = bool(records) and all(r["score"] >= 1.0 for r in records.values())
    score = sum(r["score"] for r in records.values()) / len(records) if records else 0.0
    bad = [f"{r['metric']}：{r['detail']}" for r in records.values() if r["score"] < 1.0]
    return {
        "passed": passed,
        "score": round(score, 4),
        "metrics": records,
        "reason": "；".join(bad)[:400] if bad else "全部检查通过",
    }


# ============================== 调用明细 ==============================

#: 参数值在轨迹里最多留多少字符（轨迹是证据，不是全文备份）
ARG_TEXT_LIMIT = 200

#: 哪些参数值会被记下来（避免把整份简历 / JD 塞进结果 JSON）
DEFAULT_ARG_KEYS = ("keyword", "city", "job_type", "limit", "semantic", "job_id",
                    "company", "title", "name", "resume_id", "ids", "new_status", "url")


def _safe_value(value, limit: int = ARG_TEXT_LIMIT):
    """把参数值转成可 JSON 序列化的短形式（长文本截断，未知类型退化成 type:...）。"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, (list, tuple)):
        return [_safe_value(v, limit) for v in list(value)[:10]]
    if isinstance(value, dict):
        return {str(k): _safe_value(v, limit) for k, v in list(value.items())[:10]}
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


class CallRecord:
    """一次工具调用的明细：调用序号 + 工具名 + 参数。

    第 4 周的轨迹 diff 要回答「**参数**有没有变」（例如 city 从「广州」变
    「火星城市」）与「成本有没有飙升」，光有工具名不够，所以这里在**不改业务代码**
    的前提下把实参记下来：

    * 位置参数按工具名映射（`_match(job_id, resume_json)` → job_id / resume_json）；
    * 只留白名单键（`DEFAULT_ARG_KEYS`）+ 工具名本身，长文本截断 ——
      评测结果 JSON 本来就要进 git，不能把整份简历 / JD 抄进去；
    * 未知参数的键名照记，值用 `_safe_value` 兜底。
    """

    #: 工具名 → 位置参数名（覆盖 TrajectoryRecorder 拦的四个入口 + 常用工具）
    POSITIONAL = {
        "match_resume": ("job_id", "resume_json"),
        "search_jobs": ("keyword", "city", "limit", "semantic", "job_type"),
        "generate_application_package": ("company", "job_id", "title"),
        "add_tracking": ("company", "title", "platform", "url", "status"),
        "update_tracking_status": ("company", "new_status", "note"),
        "save_resume_tool": ("name", "content"),
        "export_resume_pdf_tool": ("resume_id",),
        "job_detail": ("job_id",),
        "resume_match": ("job_id", "resume_json"),
    }

    def __init__(self, index: int, name: str, args: dict, arg_keys: tuple = DEFAULT_ARG_KEYS):
        self.index = int(index)
        self.name = str(name or "?")
        full = {str(k): _safe_value(v) for k, v in (args or {}).items()}
        self.args = {k: v for k, v in full.items() if k in set(arg_keys)}
        self.arg_keys = sorted(full)

    def as_dict(self) -> dict:
        return {"index": self.index, "name": self.name,
                "args": self.args, "arg_keys": self.arg_keys}


# ============================== 轨迹记录器 ==============================

class TrajectoryRecorder:
    """一次 case 执行期间，把工具调用序列记下来（直接读现有 trace，不改业务代码）。

    为什么拦四个入口：多智能体图走 `reg.call_tool`，ReAct 兜底在**导入时**就绑定了
    `react_agent.call_tool`，匹配图与搜岗位图分别直接调 `reg._match` / `reg._search`
    （见 run_eval.run_complex 的同款注释）。漏掉任何一个，轨迹就是缺的。

    用法：`with TrajectoryRecorder() as rec: ... ; rec.calls`
    """

    #: (模块属性所在的模块名, 属性名, 记录的别名)
    TARGETS = [
        ("agent.tools_registry", "call_tool", None),
        ("agent.react_agent", "call_tool", None),
        ("agent.tools_registry", "_match", "match_resume"),
        ("agent.tools_registry", "_search", "search_jobs"),
    ]

    def __init__(self) -> None:
        self.calls: list = []          # 工具名序列（轨迹断言用，保持原口径）
        self.calls_detail: list = []   # CallRecord 明细（轨迹 diff 用）
        self._patched: list = []

    def record(self, name: str, args: dict = None) -> None:
        self.calls.append(str(name or "?"))
        try:
            self.calls_detail.append(CallRecord(len(self.calls), str(name or "?"), args or {}))
        except Exception:                                      # noqa: BLE001
            # 明细是加分项：记不下来也不该把这次评测带走（断言只依赖 calls）
            self.calls_detail.append(CallRecord(len(self.calls), str(name or "?"), {}))

    def _wrap(self, alias, orig):
        def wrapper(*args, **kwargs):
            name = alias
            if name is None:
                if args:
                    name = args[0]
                else:
                    name = kwargs.get("name")
            self.record(name, self._extract_args(name, args, kwargs))
            return orig(*args, **kwargs)
        return wrapper

    @staticmethod
    def _extract_args(name, args, kwargs) -> dict:
        """把一次调用的实参收成 {参数名: 值}。

        三种入口的实参形态不一样，必须分开处理（否则参数记错，轨迹 diff 的参数栏全废）：

        * `reg.call_tool("search_jobs", {"keyword": ...})` —— **第一个**实参是工具名，
          **第二个**是 kwargs 字典，要摊平成参数，不能按位置参数逐个映射
          （否则 keyword 会被写成工具名、city 会被写成整个字典）；
        * `reg._match(job_id, resume_json)` —— 真·位置参数，按 POSITIONAL 映射；
        * `reg._search(keyword, city=...)` —— 位置参数 + 关键字参数混用，两者都要收。
        """
        out = dict(kwargs or {})
        pos = CallRecord.POSITIONAL.get(str(name), ())
        rest = list(args or ())
        # 入口形态一：`call_tool("search_jobs", {"keyword": ...}, confirmed=True)` ——
        # 第一个实参就是工具名，剥掉；紧随其后的字典才是真正的参数。
        if rest and isinstance(rest[0], str) and str(rest[0]) == str(name):
            rest = rest[1:]
        if rest and isinstance(rest[0], dict):
            out.update(rest[0])
            rest = rest[1:]
        # 入口形态二：`_match(job_id, resume_json)` 这类真·位置参数，按工具名映射。
        for i, value in enumerate(rest):
            if i < len(pos):
                out.setdefault(pos[i], value)
        return out

    def __enter__(self):
        import importlib
        # 先把四个入口的原函数全部取到手（缺属性只跳过），再统一挂 —— 否则"挂到一半抛异常"
        # 会把假入口永久留在模块上：之后**每一题**的轨迹都会带上幽灵调用，而报错还被
        # _run_once 吞掉，`not_calls_tool: ["*"]` 这类断言会集体误判。
        targets = []
        for mod_name, attr, alias in self.TARGETS:
            module = importlib.import_module(mod_name)
            if not hasattr(module, attr):
                continue
            targets.append((module, attr, alias, getattr(module, attr)))
        try:
            for module, attr, alias, orig in targets:
                self._patched.append((module, attr, orig))
                setattr(module, attr, self._wrap(alias, orig))
        except Exception:                                      # noqa: BLE001
            self.__exit__()                                    # 回滚已挂上的部分
            raise
        return self

    def __exit__(self, *exc):
        for module, attr, orig in reversed(self._patched):
            setattr(module, attr, orig)
        self._patched.clear()
        return False

    def as_dict(self) -> dict:
        return {"calls": list(self.calls), "count": len(self.calls),
                "calls_detail": [c.as_dict() for c in self.calls_detail]}


# ============================== BaseProvider ==============================

class BaseProvider(ABC):
    """题库 / 结果的读写插件。"""

    name = "provider"

    @abstractmethod
    def load_cases(self) -> list:
        """读题库，返回 case 列表。"""

    @abstractmethod
    def save_report(self, report: dict) -> list:
        """把整轮结果落盘，返回写出的文件路径列表。"""

    def load_report(self, path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))


@register_provider
class YamlProvider(BaseProvider):
    """默认实现：题库读 YAML（test_set.yaml），报告写 results/<时间戳>.json + .md。"""

    name = "yaml"

    REQUIRED = ("id", "category", "question", "expect", "judge_type")

    #: 各分类的 runner 会**直接索引**这些字段（`case["job_id"]`、`expect["score_range"]`…），
    #: 缺了就是运行期 KeyError、还会被 _run_once 吞成「执行异常」。放到题库自检里拦下来。
    REQUIRED_BY_CATEGORY = {
        "search": {"case": (), "expect": ("city",)},
        "match": {"case": ("job_id",), "expect": ("score_range",)},
        "package": {"case": ("company", "job_id"), "expect": ()},
        "interview": {"case": ("company", "title"), "expect": ()},
        "boundary": {"case": (), "expect": ("check",)},
        "complex": {"case": (), "expect": ("search_min",)},
        # 检索类：一次「执行」跑整个 ground truth 查询集，expect 里给指标阈值
        "retrieval": {"case": (), "expect": ("min_recall_at_k",)},
    }
    def __init__(self, test_set_path=None, results_dir=None) -> None:
        self.test_set_path = Path(test_set_path or EVAL_DIR / "test_set.yaml")
        self.results_dir = Path(results_dir or EVAL_DIR / "results")

    # ---- 读 ----

    def load_test_set(self) -> dict:
        if not self.test_set_path.is_file():
            raise FileNotFoundError(f"找不到题库：{self.test_set_path}")
        data = yaml.safe_load(self.test_set_path.read_text(encoding="utf-8")) or {}
        cases = data.get("cases") or []
        self.validate(cases)
        return data

    def load_cases(self) -> list:
        return self.load_test_set()["cases"]

    def validate(self, cases: list) -> None:
        """题库自检：必填字段 / id 唯一 / judge_type 已注册 —— 早失败好过跑一半炸。"""
        if not cases:
            raise ValueError("题库里一题都没有")
        seen = set()
        for case in cases:
            missing = [k for k in self.REQUIRED if k not in case or case[k] in (None, "")]
            if missing:
                raise ValueError(f"题 {case.get('id', '?')} 缺字段：{missing}")
            if case["id"] in seen:
                raise ValueError(f"题号重复：{case['id']}")
            seen.add(case["id"])
            if case["judge_type"] not in METRICS:
                raise ValueError(f"题 {case['id']} 的 judge_type={case['judge_type']!r} 没有对应插件；"
                                 f"可用：{sorted(METRICS)}")
            spec = case.get("trajectory")
            if spec is not None and not isinstance(spec, dict):
                raise ValueError(f"题 {case['id']} 的 trajectory 必须是映射")
            need = self.REQUIRED_BY_CATEGORY.get(case["category"])
            if need is None:
                raise ValueError(f"题 {case['id']} 的 category={case['category']!r} 没有对应 runner")
            expect = case.get("expect") or {}
            miss_case = [k for k in need["case"] if not case.get(k)]
            # 判存在性而不是判真假：expect 里的空串是**合法期望**（新增的
            # search-07「找能远程的 AI 实习」就要求城市为空 = 不限城市），按真值
            # 判会把这类题直接挡在题库外，等于把「故意难倒」的题删掉。
            miss_expect = [k for k in need["expect"] if k not in expect]
            if miss_case or miss_expect:
                raise ValueError(f"题 {case['id']}（{case['category']}）缺字段："
                                 f"case{miss_case} expect{miss_expect}")

    # ---- 写 ----

    def save_report(self, report: dict) -> list:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = report.get("stamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        report["stamp"] = stamp
        out = []
        for name in ("json", "markdown"):
            report_plugin = REPORTS.get(name)
            if report_plugin is None:
                continue
            plugin = report_plugin()
            path = self.results_dir / f"{stamp}{plugin.suffix}"
            path.write_text(plugin.render(report), encoding="utf-8")
            out.append(path)
        return out


# ============================== BaseReport ==============================

class BaseReport(ABC):
    """渲染插件：把整轮结果 dict 变成一段文本。"""

    name = "report"
    suffix = ".txt"

    @abstractmethod
    def render(self, report: dict) -> str:
        """返回要写进文件 / 打到终端的文本。"""


def _pct(x) -> str:
    return f"{float(x or 0) * 100:.1f}%"


@register_report
class JsonReport(BaseReport):
    name = "json"
    suffix = ".json"

    def render(self, report: dict) -> str:
        data = {k: v for k, v in report.items() if k != "stamp"}
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


@register_report
class MarkdownReport(BaseReport):
    name = "markdown"
    suffix = ".md"

    def render(self, report: dict) -> str:
        lines = [f"# 评测报告 {report.get('run_at', '')}", "",
                 f"- 题库：{report.get('test_set_path', '')}（{report.get('test_set_version', '')}）",
                 f"- 引擎：{report.get('engine', '')}    模型：{report.get('model', '')}",
                 f"- 每题重复次数：**{report.get('repeat', 1)}**"
                 "（题级 passed 取多数票；稳定率 = 各题单题通过率的平均）",
                 f"- 总计：**{report.get('passed', 0)}/{report.get('total', 0)} = "
                 f"{_pct(report.get('accuracy'))}**　稳定率 **{_pct(report.get('stability'))}**"
                 f"（耗时 {float(report.get('elapsed_sec') or 0) / 60:.1f} 分钟）", "",
                 "## 分类准确率 / 稳定率", "",
                 "| 类别 | 通过/总数 | 准确率 | 稳定率 |", "|---|---|---|---|"]
        for name, b in (report.get("by_category") or {}).items():
            lines.append(f"| {name} | {b['passed']}/{b['total']} | {b['accuracy'] * 100:.0f}% | "
                         f"{b.get('stability', b['accuracy']) * 100:.0f}% |")
        lines += ["", "## 判分方式（插件）", "",
                  "| judge_type | 通过/总数 | 准确率 | 稳定率 |", "|---|---|---|---|"]
        for name, b in (report.get("by_judge") or {}).items():
            lines.append(f"| {JUDGE_LABELS.get(name, name)}（{name}） | {b['passed']}/{b['total']} | "
                         f"{b['accuracy'] * 100:.0f}% | {b.get('stability', b['accuracy']) * 100:.0f}% |")

        unstable = report.get("unstable_cases") or []
        lo, hi = [int(x * 100) for x in (report.get("unstable_range") or [UNSTABLE_LOW, UNSTABLE_HIGH])]
        lines += ["", f"## 不稳定题（单题通过率 {lo}%-{hi}%）", ""]
        if unstable:
            lines += ["| 题号 | 类别 | 通过/次数 | 通过率 |", "|---|---|---|---|"]
            for u in unstable:
                lines.append(f"| {u['id']} | {u['category']} | {u['passed_runs']}/{u['total_runs']} | "
                             f"{u['pass_rate'] * 100:.0f}% |")
        else:
            lines.append("无（所有题要么全过、要么全不过）")

        traj = report.get("trajectory_cases") or []
        lines += ["", "## 轨迹断言（工具调用序列）", ""]
        if traj:
            lines += ["| 题号 | 结果 | 实际调用序列 | 断言 |", "|---|---|---|---|"]
            for t in traj:
                calls = " → ".join(t.get("calls") or []) or "（无工具调用）"
                lines.append(f"| {t['id']} | {'✅' if t.get('ok') else '❌'} | {calls} | "
                             f"{str(t.get('detail') or '')[:120]} |")
        else:
            lines.append("无")

        rep_label = "" if report.get("repeat", 1) == 1 else "（N 次里过几次）"
        lines += ["", "## 逐题", "",
                  "| 题号 | 类别 | judge_type | 结果 | 通过率 | 原因 | 耗时 |",
                  "|---|---|---|---|---|---|---|"]
        for r in report.get("results") or []:
            rate = (f"{r.get('passed_runs', 0)}/{r.get('repeat', 1)} = "
                    f"{r.get('pass_rate', 0) * 100:.0f}%{rep_label}")
            lines.append(f"| {r['id']} | {r['category']} | {r.get('judge_type', '')} | "
                         f"{'✅' if r['passed'] else '❌'} | {rate} | {r['reason'][:120]} | "
                         f"{r.get('elapsed', 0)}s |")
        lines += ["", "> 每一次跑的完整详情（判词 / 检查项 / extra / 耗时）见同名 JSON 的 "
                      "`results[].runs[]`；插件口径见 `results[].runs[].metrics`。"
                      "`--repeat > 1` 时，逐题表里的 `checks` / `metrics` / `trajectory` 取"
                      "**第一次失败的那次**（没有失败就取第一次），所以表里的原因和失败现场"
                      "一致，不代表 N 次都是这样。"]
        return "\n".join(lines) + "\n"


@register_report
class ConsoleReport(BaseReport):
    name = "console"
    suffix = ".txt"

    def render(self, report: dict) -> str:
        lines = ["", "=" * 62, "评测结果", "=" * 62]
        if report.get("repeat", 1) > 1:
            lines.append(f"每题重复次数：{report['repeat']}（题级 passed = 多数票，通过率 ≥ 50%）")
        lines.append(f"{'类别':<12}{'通过/总数':<12}{'准确率':<10}{'稳定率':<10}")
        for name, b in (report.get("by_category") or {}).items():
            lines.append(f"{name:<12}{b['passed']}/{b['total']:<10}{b['accuracy'] * 100:.0f}%"
                         f"{'':<4}{b.get('stability', b['accuracy']) * 100:.0f}%")
        lines.append("-" * 62)
        for name, b in (report.get("by_judge") or {}).items():
            lines.append(f"{JUDGE_LABELS.get(name, name):<12}{b['passed']}/{b['total']:<10}"
                         f"{b['accuracy'] * 100:.0f}%"
                         f"{'':<4}{b.get('stability', b['accuracy']) * 100:.0f}%")
        lines.append("-" * 62)
        lines.append(f"总计 {report.get('passed', 0)}/{report.get('total', 0)} = "
                     f"{_pct(report.get('accuracy'))}"
                     f"    稳定率 {_pct(report.get('stability'))}"
                     f"    耗时 {float(report.get('elapsed_sec') or 0) / 60:.1f} 分钟")

        if report.get("repeat", 1) > 1:
            lines.append("\n单题通过率：")
            for r in report.get("results") or []:
                rate = r.get("pass_rate", 1.0 if r["passed"] else 0.0) * 100
                flag = "  ⚠️ 不稳定" if UNSTABLE_LOW <= r.get("pass_rate", 0.0) <= UNSTABLE_HIGH else ""
                lines.append(f"  {r['id']:<14}{r.get('passed_runs', 0)}/{r.get('repeat', 1)}"
                             f" = {rate:.0f}%{flag}")
            unstable = report.get("unstable_cases") or []
            lo, hi = [int(x * 100) for x in (report.get("unstable_range") or [])]
            lines.append(f"\n不稳定题（通过率 {lo}%-{hi}%）："
                         + (", ".join(f"{u['id']}（{u['passed_runs']}/{u['total_runs']}）"
                                      for u in unstable) if unstable else "无"))

        traj = report.get("trajectory_cases") or []
        if traj:
            lines.append("\n轨迹断言：")
            for t in traj:
                calls = " → ".join(t.get("calls") or []) or "（无工具调用）"
                lines.append(f"  {'OK  ' if t.get('ok') else 'FAIL'} {t['id']:<14}{calls}")

        failed = [r for r in (report.get("results") or []) if not r["passed"]]
        lines.append("\n失败题：")
        if failed:
            for r in failed:
                lines.append(f"  - {r['id']} [{r['category']}/{r.get('judge_type', '')}]："
                             f"{r['reason'][:150]}")
        else:
            lines.append("  无")
        return "\n".join(lines) + "\n"


def build_report(results: list, elapsed: float, test_set: dict, repeat: int = 1,
                 test_set_path: str = "") -> dict:
    """把逐题结果汇总成报告 dict（Provider / Report 之间流转的统一结构）。"""
    def bucket(key):
        out: dict = {}
        for r in results:
            b = out.setdefault(r[key], {"total": 0, "passed": 0, "_rate": 0.0})
            b["total"] += 1
            b["passed"] += 1 if r["passed"] else 0
            b["_rate"] += float(r.get("pass_rate", 1.0 if r["passed"] else 0.0))
        for b in out.values():
            b["accuracy"] = round(b["passed"] / b["total"], 4)
            b["stability"] = round(b["_rate"] / b["total"], 4)
            del b["_rate"]
        return out

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    rates = [float(r.get("pass_rate", 1.0 if r["passed"] else 0.0)) for r in results]
    unstable = [{"id": r["id"], "category": r["category"],
                 "passed_runs": r.get("passed_runs", 1 if r["passed"] else 0),
                 "total_runs": r.get("repeat", 1), "pass_rate": r.get("pass_rate", 1.0)}
                for r in results if UNSTABLE_LOW <= float(r.get("pass_rate", 0.0)) <= UNSTABLE_HIGH]
    trajectory_cases = [{"id": r["id"], "ok": bool(r.get("trajectory", {}).get("ok")),
                         "calls": (r.get("trajectory") or {}).get("calls") or [],
                         "detail": (r.get("trajectory") or {}).get("detail", "")}
                        for r in results if r.get("trajectory")]
    return {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_set_path": test_set_path,
        "test_set_version": test_set.get("version"),
        "engine": os.getenv("AGENT_ENGINE", "langgraph"),
        "model": os.getenv("ZHIPU_CHAT_MODEL", ""),
        "repeat": repeat,
        "total": total, "passed": passed,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "stability": round(sum(rates) / total, 4) if total else 0.0,
        "unstable_range": [UNSTABLE_LOW, UNSTABLE_HIGH],
        "unstable_cases": unstable,
        "trajectory_cases": trajectory_cases,
        "elapsed_sec": round(elapsed, 1),
        "by_category": bucket("category"),
        "by_judge": bucket("judge_type"),
        "plugins": {"metrics": sorted(METRICS), "providers": sorted(PROVIDERS),
                    "reports": sorted(REPORTS)},
        "results": results,
    }


def make_provider(name: str = "yaml", **kwargs) -> BaseProvider:
    cls = PROVIDERS.get(name)
    if cls is None:
        raise KeyError(f"没有注册的题库插件 provider={name!r}；可用：{sorted(PROVIDERS)}")
    return cls(**kwargs)


def make_report(name: str) -> BaseReport:
    cls = REPORTS.get(name)
    if cls is None:
        raise KeyError(f"没有注册的渲染插件 report={name!r}；可用：{sorted(REPORTS)}")
    return cls()
