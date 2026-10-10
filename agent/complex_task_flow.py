"""
复杂任务编排图（批次 1 · 真多智能体协作）。

与 `agent/langgraph_flow.py` 的关系
----------------------------------
`langgraph_flow.py` 里已经有**两张固定流程的图**（搜岗位 5 步、匹配打分 + 反思），
它们解决的是「单意图任务」：一句话只要求搜岗位，或只要求打分。
本模块**不推翻它们**，而是**新增第三张图**，专门处理「一句话里有多个需求」的复杂任务，
例如「帮我找岗位 + 匹配 + 生成投递包」。入口在 `agent/react_agent_lg.py`：
简单任务仍走原来那两张图，只有判定为复杂任务时才走这里。

三个专职 Agent（不是「三个 prompt 互相聊天」）
---------------------------------------------
| Agent | 职责 | 输入 | 输出 |
|---|---|---|---|
| **Planner**（规划者） | 把用户需求拆成**有序可执行步骤** | 用户原话 | `{"steps": [{action, args, desc, when}], "reason"}` |
| **Executor**（执行者） | 执行**计划中的一步**（真调工具） | 单个步骤 + 前序结果 | 该步的结构化结果 |
| **Critic**（审查者） | 审查执行结果**是否合格** | 步骤 + 结果 | `{"合格": bool, "理由": str, "打回建议": {...}}` |

协作流程::

    [用户需求] → [Planner 出计划] → [Executor 执行第 N 步]
                     ↑                        ↓
                     │                  [Critic 审查]
                     │                        │
                     │     不合格，≤2 次       │ 合格
                     └── [修正参数] ←─────────┘
                                              ↓
                                     [进入下一步 / 输出]

**关键不是"有 3 个 Agent"，而是"审查者能打回重做"**
--------------------------------------------------
如果 Critic 只能「输出一句评审意见」，那它就是一段白烧 token 的旁白 ——
执行结果没有任何改变，这就是"假多智能体"。所以这里做了三件真事：

1. Critic 的判定是**结构化且可执行的**：`打回建议` 里带 `args` 覆盖项
   （例：搜岗位 0 条 → `{"city": ""}` 去掉城市限制；分数 0 分 → 换当前简历重打）。
   `revise_node` 把这些覆盖**真的合并进步骤参数**，然后回到 Executor **重新执行**。
2. Critic 的**事实层是确定性的**（不依赖模型）：0 条结果、分数异常、三件套缺文件
   这些一律判不合格。模型只在"事实层通过、需要判断分寸"时才投一票。
3. 打回**有上限、有降级**：同一步最多重做 `MAX_STEP_RETRIES`(2) 次；
   用完还不合格就**降级**——保留已有结果、在回答里如实说明哪一步没做成，
   绝不无限重试、也绝不把失败包装成成功。

死循环防护（三层，缺一不可）
--------------------------
* 步骤级：`MAX_STEP_RETRIES` 限制单步重做次数；
* 计划级：`MAX_PLAN_STEPS` 限制计划长度（Planner 也是模型，可能胡说）；
* 全局级：`MAX_TOTAL_EXECUTIONS` 兜住"步骤多 × 每步重试多"的组合爆炸。
"""
from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent import tools_registry as reg
from shared import limits
from shared.llm_client import chat
from shared.logger import log_event


# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

#: 同一步骤最多被 Critic 打回重做几次。超过就降级（返回部分结果 + 说明）。
MAX_STEP_RETRIES = int(os.getenv("MULTI_AGENT_MAX_RETRIES", "2") or 2)
#: 计划最多几步（Planner 是模型，必须防它写出 20 步）
MAX_PLAN_STEPS = int(os.getenv("MULTI_AGENT_MAX_PLAN_STEPS", "5") or 5)
#: 全局执行次数上限（"步骤数 × 重试次数"的组合爆炸兜底）
MAX_TOTAL_EXECUTIONS = int(os.getenv("MULTI_AGENT_MAX_EXECUTIONS", "12") or 12)
#: 是否让 Critic 额外投一票（LLM 语义复核）。关掉后 Critic 只走确定性事实层。
CRITIC_LLM_ENABLED = (os.getenv("MULTI_AGENT_CRITIC_LLM", "1") or "1").strip().lower() \
    not in ("0", "false", "no", "off")
#: 总开关：关掉后入口不再分流到本图（等价于改造前行为）
MULTI_AGENT_ENABLED = (os.getenv("MULTI_AGENT_ENABLED", "1") or "1").strip().lower() \
    not in ("0", "false", "no", "off")

#: 本图允许 Executor 调用的工具白名单 —— Planner 只能从中选，写别的直接丢弃。
ALLOWED_ACTIONS = ("search_jobs", "match_resume", "generate_application_package")

#: 匹配分数低于它就**不生成投递包**（计划里 `when` 条件的默认阈值）
DEFAULT_PACKAGE_THRESHOLD = int(os.getenv("MULTI_AGENT_PACKAGE_THRESHOLD", "70") or 70)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
#: `$step2.job_id` / `$prev.score` 这类占位符：让后一步引用前一步的结果
_PLACEHOLDER_RE = re.compile(r"^\$(?P<which>step\d+|prev|best|last)\."
                             r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)$")
#: 条件表达式：只支持 `score > 70` / `score >= 70` 这类单向数值比较（不 eval）
_CONDITION_RE = re.compile(r"^\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*"
                           r"(?P<op>>=|<=|==|>|<)\s*(?P<value>-?\d+)\s*$")


class ComplexState(TypedDict, total=False):
    """复杂任务编排图的状态。"""
    question: str
    history: list
    verbose: bool
    trace_id: str
    # Planner
    plan: list                    # [{id, action, args, desc, when}]
    plan_reason: str
    plan_source: str              # llm / rules
    # Executor
    cursor: int                   # 当前执行到第几步（0 起）
    step_args: dict               # 本步**最终**参数（可能被 revise 改写）
    step_output: Any              # 本步结果（归一化后的 dict）
    step_error: str
    executions: int               # 全局执行次数（防组合爆炸）
    # Critic
    critique: dict                # {合格, 理由, 建议, actionable, attempts}
    critiques: list               # 全部审查记录（含被驳回的）
    # 结果与收尾
    results: list                 # [{id, action, desc, ok, skipped, output, ...}]
    reworks: list                 # 打回重做记录
    degraded: bool
    degrade_reason: str
    answer: str
    steps: list                   # 给 CoT 面板看的节点轨迹


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------

def _step_trace(state: dict, node: str, thought: str,
                payload: dict = None, observation: str = "") -> list:
    """追加一条节点轨迹。

    `type` 用 `"action"`：`agent/app.py` 的 `_format_step_log` 只渲染
    `type == "action"` 的步骤，复用它能**不改 app.py 一行**就把三 Agent 的协作
    过程显示进 CoT 面板（与 `langgraph_flow._step` 同一套约定）。
    """
    steps = list(state.get("steps") or [])
    steps.append({
        "turn": len(steps) + 1,
        "type": "action",
        "engine": "multi_agent",
        "node": node,
        "thought": thought,
        "action": node,
        "action_input": payload or {},
        "observation": observation,
    })
    return steps


def _parse_json_loose(raw: str) -> dict:
    """容错解析模型输出：剥 ```json 围栏，再退一步取第一个平衡的 `{...}`。"""
    text = (raw or "").strip()
    text = _JSON_FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:                                   # noqa: BLE001 - 走下面的兜底
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError(f"没有 JSON 对象：{text[:120]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(text[start:i + 1])
                if isinstance(data, dict):
                    return data
                break
    raise ValueError(f"JSON 解析失败：{text[:120]}")


def _llm_json(messages: list, source: str, verbose: bool = False) -> dict:
    """一次结构化 LLM 调用。

    **必须带额度与思考档**：`glm-5.3-flash` 是思考模型，`reasoning_content`
    与正文共用 `max_tokens`，当年不传就是全局默认 1024（现全局兜底已抬到 4096
    + low）—— 真实简历下思考照样能把额度吃掉大半，所以这里不依赖兜底，
    直接复用 ReAct 的「长输出档」getter：`RATE_LIMIT_ENABLED=false` 时它
    返回 0/""，payload 形状与改造前一致。
    """
    raw = chat(
        messages,
        source=source,
        max_tokens=limits.react_long_max_tokens(),
        reasoning_effort=limits.react_reasoning_effort(),
    )
    if verbose:
        print(f"[{source}] {(raw or '')[:200]}")
    return _parse_json_loose(raw)


def _flatten(results: list) -> dict:
    """把已完成步骤的结果摊平成 `{字段名: 值}`，给条件求值用。

    后写的覆盖先写的（同名键以更晚的步骤为准）—— 「最新一步的分数」
    正是条件判断想要的语义。
    """
    flat: dict = {}
    for item in results or []:
        out = item.get("output")
        if isinstance(out, dict):
            for key, value in out.items():
                flat[key] = value
    return flat


def _lookup(results: list, which: str, field: str):
    """按 `$stepN` / `$prev` / `$best` / `$last` 取字段（取不到返回 None）。

    `$prev` / `$last` 的语义是「**最近一次产出过该字段**的那一步」，
    而不是"字面意义上的上一步"。这一点是真测逼出来的：模型常写
    `$prev.company` / `$prev.job_id` 来引用搜岗位的结果，但紧邻的往往是
    「匹配打分」那一步 —— 它的输出里只有 score/dimensions/gaps，
    没有 company/title/job_id。按"字面上一步"取会拿到 None、占位符原样留着，
    最后 Executor 拿着 `"$prev.company"` 去查岗位，直接报「没找到这个岗位」。
    改成向后查找第一个**非空**的该字段值，既符合模型的本意，也不会取错
    （同一字段后出现的步骤天然覆盖先出现的）。
    """
    if not results:
        return None

    def _scan_backwards():
        for item in reversed(results):
            out = item.get("output")
            if isinstance(out, dict):
                value = out.get(field)
                if value not in (None, "", [], {}):
                    return value
        return None

    if which in ("prev", "last"):
        return _scan_backwards()
    if which == "best":
        best, best_score = None, None
        for item in results:
            out = item.get("output") or {}
            score = out.get("score")
            if isinstance(score, int) and (best_score is None or score > best_score):
                best, best_score = out, score
        return (best or {}).get(field)
    match = re.match(r"^step(\d+)$", which)
    if not match:
        return None
    index = int(match.group(1)) - 1
    if 0 <= index < len(results):
        return (results[index].get("output") or {}).get(field)
    return None


def resolve_args(args: dict, results: list) -> dict:
    """把步骤参数里的 `$stepN.field` / `$prev.field` 占位符换成真实值。

    这是「后一步依赖前一步」的唯一实现方式：Planner 在规划时**不可能知道**
    运行时才会有的 job_id，只能写占位符。解析失败的占位符**保留原样**
    （让 Executor 报明确的参数错误），而不是悄悄丢掉整个参数 ——
    静默丢参数会生成指向错岗位的投递包。
    """
    resolved: dict = {}
    for key, value in (args or {}).items():
        if isinstance(value, str):
            match = _PLACEHOLDER_RE.match(value.strip())
            if match:
                found = _lookup(results, match.group("which"), match.group("field"))
                resolved[key] = found if found is not None else value
                continue
        resolved[key] = value
    return resolved


def check_condition(when: str, results: list) -> bool:
    """求值计划里的 `when` 条件（**只支持单向数值比较**，不做表达式求值）。

    支持 `score > 70` / `>=` / `<` / `<=` / `==`。空条件恒为 True。
    **刻意不 eval**：条件是模型生成的文本，eval 等于把任意代码执行权交出去。
    看不懂的条件**当作不满足**并记进理由 —— 宁可少做一步，也不做错一步。
    """
    text = str(when or "").strip()
    if not text:
        return True
    match = _CONDITION_RE.match(text)
    if not match:
        return False
    field, op = match.group("field"), match.group("op")
    raw_value = int(match.group("value"))
    actual = _flatten(results).get(field)
    if not isinstance(actual, (int, float)):
        return False
    return {
        ">": actual > raw_value,
        ">=": actual >= raw_value,
        "<": actual < raw_value,
        "<=": actual <= raw_value,
        "==": actual == raw_value,
    }[op]


# ==========================================================================
# Agent 1：Planner（规划者）
# ==========================================================================

_PLAN_PROMPT = """你是「复杂任务编排」里的**规划者（Planner）**。把用户需求拆成
**有序、可执行**的步骤，交给执行者一步步做。

可选工具（只能用这三个，别发明别的）：
1. `search_jobs`   参数：keyword(关键词) / city(城市) / limit(条数) / job_type(实习/正式/兼职) / semantic(bool)
2. `match_resume`  参数：job_id(岗位ID) / resume_json(固定填 "current")
3. `generate_application_package` 参数：company(公司) / title(岗位名) / job_id(岗位ID)

只输出一个 JSON，不要任何其他内容：
{{"steps": [{{"id": 1, "action": "search_jobs", "args": {{}}, "desc": "一句话说明这一步干什么", "when": ""}}], "reason": "为什么这么拆"}}

规则：
1. **按依赖顺序排**：必须先搜到岗位，才能匹配；必须先匹配，才能生成投递包。
2. **后一步引用前一步的结果用占位符**（运行时才知道具体值）：
   job_id 写 `"$prev.job_id"` 或 `"$step1.job_id"`，公司写 `"$step1.company"`，
   岗位名写 `"$step1.title"`，分数写 `"$prev.score"`。**不要自己编 job_id**。
3. **`when` 是可选的前置条件**，只支持 `score > 70` / `score >= 70` 这类单向数值比较。
   用户说「分数够了再生成投递包」时，把条件写在生成投递包那一步上；
   用户没提前置条件时 `when` 留空串。
4. 用户**只要一部分**需求（例如只说找岗位）时，**只排那部分步骤**，别擅自多做。
5. 步骤数不超过 {max_steps} 步。用户的话里没有涉及的能力（改简历文件、
   发邮件、投递到平台等）**不要排**，排了也做不成。
6. `resume_json` 一律填 `"current"`（系统会自动替换成当前简历）。

用户需求：{question}"""


def fallback_plan(question: str) -> dict:
    """确定性兜底计划（LLM 不可用 / 输出不合法时用）。

    复用 `langgraph_flow._extract_params_rules` 的规则层抽关键词与城市，
    保证兜底计划与「搜岗位图」对同一句话的理解一致（不会出现兜底版
    搜出另一批岗位的情况）。
    """
    from agent.langgraph_flow import _extract_params_rules

    params = _extract_params_rules(question or "")
    keyword = params.get("keyword") or ""
    city = params.get("city") or ""
    semantic = bool(params.get("semantic"))
    job_type = params.get("job_type") or ""
    threshold = DEFAULT_PACKAGE_THRESHOLD

    search_args: dict = {"keyword": keyword, "city": city, "limit": 20}
    if job_type:
        search_args["job_type"] = job_type
    if semantic:
        search_args["semantic"] = True

    return {
        "steps": [
            {"id": 1, "action": "search_jobs", "args": search_args,
             "desc": "按关键词与城市搜索岗位", "when": ""},
            {"id": 2, "action": "match_resume",
             "args": {"job_id": "$step1.job_id", "resume_json": "current"},
             "desc": "用当前简历给最匹配的那条岗位打分", "when": ""},
            {"id": 3, "action": "generate_application_package",
             "args": {"company": "$step1.company", "title": "$step1.title",
                      "job_id": "$step1.job_id"},
             "desc": f"分数达到 {threshold} 分才生成投递包",
             "when": f"score >= {threshold}"},
        ],
        "reason": "规则兜底：搜岗位 → 匹配打分 → 达标后生成投递包",
    }


def _normalize_plan(raw: dict, question: str) -> tuple:
    """校验并规整 Planner 的计划，返回 `(plan, reason, dropped, dropped_args)`。

    校验是**必须的**，不是走过场：计划内容完全由模型生成，直接把 action
    拿去调工具等于把工具白名单交给模型。逐条：
      * `action` 必须在 `ALLOWED_ACTIONS` 里，否则整步丢弃；
      * `match_resume` 必须带 job_id（占位符也算"有个值"），否则整步丢弃；
      * `generate_application_package` 必须能定位岗位
        （job_id / company / title 至少一个），否则整步丢弃；
      * **参数按工具声明白名单过滤**（`_clean_args`）：模型会把上一步的参数顺手
        带下来（实测 `match_resume` 被塞进 `city` / `keywords`），非法参数会让
        `call_tool` 直接抛 TypeError —— 这种错误重做多少次都一样，只能提前摘掉；
      * 步骤数截到 `MAX_PLAN_STEPS`。
    """
    steps: list = []
    dropped = 0
    dropped_args: list = []
    for item in (raw.get("steps") or []):
        if not isinstance(item, dict):
            dropped += 1
            continue
        action = str(item.get("action") or "").strip()
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        if action not in ALLOWED_ACTIONS:
            dropped += 1
            continue
        if action == "match_resume" and not str(args.get("job_id") or "").strip():
            dropped += 1
            continue
        if action == "generate_application_package" and not any(
                str(args.get(k) or "").strip() for k in ("job_id", "company", "title")):
            dropped += 1
            continue
        args, bad_args = _clean_args(action, args)
        dropped_args.extend(f"{action}.{k}" for k in bad_args)
        steps.append({
            "id": len(steps) + 1,
            "action": action,
            "args": args,
            "desc": str(item.get("desc") or "").strip() or action,
            "when": str(item.get("when") or "").strip(),
        })
        if len(steps) >= MAX_PLAN_STEPS:
            break
    return steps, str(raw.get("reason") or "").strip(), dropped, dropped_args


def plan_node(state: ComplexState) -> dict:
    """**[Planner]** 出计划：LLM 拆解 + 白名单校验 + 规则兜底。"""
    question = (state.get("question") or "").strip()
    plan, reason, source, dropped, dropped_args = [], "", "llm", 0, []
    try:
        data = _llm_json(
            [{"role": "user", "content": _PLAN_PROMPT.format(
                question=question, max_steps=MAX_PLAN_STEPS)}],
            source="ma_planner",
            verbose=bool(state.get("verbose")),
        )
        plan, reason, dropped, dropped_args = _normalize_plan(data, question)
        if not plan:
            source = "rules"
    except Exception as e:                              # noqa: BLE001 - 规划失败不该让整轮挂掉
        if state.get("verbose"):
            print(f"[ma_planner] 退回规则兜底：{type(e).__name__}: {e}")
        plan, source = [], "rules"
        reason = f"LLM 规划不可用（{type(e).__name__}），改用规则兜底"

    if not plan:
        plan = fallback_plan(question)["steps"]
        source = "rules"

    outline = "；".join(f"{s['id']}. {s['desc']}" for s in plan)
    log_event(state.get("trace_id") or "-", "ma_node", node="planner",
              engine="multi_agent", source=source, steps=len(plan), dropped=dropped,
              dropped_args="、".join(dropped_args))
    return {
        "plan": plan,
        "plan_reason": reason,
        "plan_source": source,
        "cursor": 0,
        "results": [],
        "reworks": [],
        "critiques": [],
        "executions": 0,
        "degraded": False,
        "degrade_reason": "",
        "critique": {},
        "step_args": {},
        "step_output": None,
        "step_error": "",
        "steps": _step_trace(
            state, "Planner 规划",
            f"拆出 {len(plan)} 步计划（来源：{source}）：{outline}"
            + (f"｜丢弃非法步骤 {dropped} 条" if dropped else "")
            + (f"｜理由：{reason}" if reason else ""),
            payload={"plan": plan, "source": source},
            observation=f"{len(plan)} 步",
        ),
    }


# ==========================================================================
# Agent 2：Executor（执行者）
# ==========================================================================

def _normalize_output(action: str, raw: Any, args: dict = None) -> dict:
    """把工具返回值归一化成统一形状（占位符解析与 Critic 都按这个形状读）。

    为什么要归一化：`search_jobs` 返回列表、`match_resume` /
    `generate_application_package` 返回 dict，字段名也各不相同。Critic 与
    占位符解析若各读各的，每加一个工具就要改三处。

    `args` 只用来**补出工具自己没返回的身份字段**：`_match` 的返回值里没有
    `job_id`（它只管打分），而后一步的 `$prev.job_id` 又需要它 ——
    从"这一步用的参数"里补回来是最可靠的来源。
    """
    args = args or {}
    if action == "search_jobs":
        rows = raw if isinstance(raw, list) else []
        first = rows[0] if rows and isinstance(rows[0], dict) else {}
        return {
            "rows": rows,
            "count": len(rows),
            "job_id": str(first.get("job_id") or ""),
            "company": str(first.get("company") or ""),
            "title": str(first.get("title") or ""),
            "city": str(first.get("city") or ""),
        }
    if action == "match_resume":
        data = raw if isinstance(raw, dict) else {}
        return {
            "score": data.get("score"),
            "dimensions": data.get("dimensions") or {},
            "gaps": list(data.get("gaps") or []),
            "general_advice": list(data.get("general_advice") or []),
            "highlights": list(data.get("highlights") or []),
            # 打分结果自带岗位身份：后一步（出投递包）可以写 `$prev.job_id`
            "job_id": str(args.get("job_id") or ""),
        }
    if action == "generate_application_package":
        data = raw if isinstance(raw, dict) else {}
        return {
            "package_dir": str(data.get("package_dir") or ""),
            "files": data.get("files") or {},
            "warnings": list(data.get("warnings") or []),
            "job_id": str(data.get("job_id") or ""),
            "company": str(data.get("company") or ""),
            "title": str(data.get("title") or ""),
            "resume_tailored": bool(data.get("resume_tailored")),
        }
    return {"raw": raw}


def _clean_args(action: str, args: dict) -> tuple:
    """按工具**声明的参数**过滤步骤参数，返回 `(clean_args, dropped_keys)`。

    这是白名单校验的最后一环，也是真测逼出来的：模型写计划时会把上一步的参数
    顺手带下来（实测 `match_resume` 那一步被塞进 `city` / `keywords`），
    而 `call_tool` 的 `_validate_args` 会直接抛
    `TypeError: 工具「match_resume」不支持参数：city、keywords`。
    这种错误**重做多少次都一样**（参数是确定性的非法），会被 Critic 当成瞬时故障
    白重试 2 次再降级。所以在进图之前就把非法参数摘掉。
    """
    allowed = set(((reg.TOOLS.get(action) or {}).get("parameters") or {}).keys())
    if not allowed:
        return dict(args or {}), []
    clean = {k: v for k, v in (args or {}).items() if k in allowed}
    dropped = sorted(k for k in (args or {}) if k not in allowed)
    return clean, dropped


def _summarize(action: str, output: Any) -> str:
    """给执行结果一句人类可读的摘要（进 CoT 面板与日志）。"""
    if not isinstance(output, dict):
        return str(output)[:120]
    if output.get("skipped"):
        return str(output.get("reason") or "已跳过")
    if action == "search_jobs":
        return f"命中 {output.get('count', 0)} 条，首条 {output.get('company')} · {output.get('title')}"
    if action == "match_resume":
        return f"得分 {output.get('score')}/100，差距 {len(output.get('gaps') or [])} 项"
    if action == "generate_application_package":
        return f"投递包 {output.get('package_dir')}（{len(output.get('files') or {})} 个文件）"
    return str(output)[:120]


def execute_node(state: ComplexState) -> dict:
    """**[Executor]** 执行计划里的第 `cursor` 步（真调工具）。

    三件事：
      1. 求值 `when` 前置条件 —— 不满足就**跳过**这一步（例如分数没到 70
         就不生成投递包），并在结果里如实写明"条件不满足，已跳过"；
      2. 解析 `$prev.xxx` 占位符，叠加 Critic 打回时写入的参数覆盖；
      3. `call_tool` 执行。工具异常**不往外抛**，而是记成 `step_error`
         交给 Critic 判 —— 「打回重做」本来就是 Critic 的职责，这里抛异常
         等于把失败处理绕过 Critic 直接崩掉整轮。
    """
    plan = state.get("plan") or []
    cursor = int(state.get("cursor") or 0)
    if cursor >= len(plan):
        return {"step_output": None, "step_error": "计划已执行完，没有待执行步骤"}

    step = plan[cursor]
    action = step["action"]
    results = list(state.get("results") or [])
    executions = int(state.get("executions") or 0) + 1
    attempts = int((state.get("critique") or {}).get("attempts") or 0)

    if executions > MAX_TOTAL_EXECUTIONS:
        log_event(state.get("trace_id") or "-", "ma_node", node="executor",
                  engine="multi_agent", action=action, status="abort_budget")
        return {
            "executions": executions,
            "step_output": None,
            "step_error": f"已达全局执行上限 {MAX_TOTAL_EXECUTIONS} 次，停止执行",
            "steps": _step_trace(
                state, "Executor 执行",
                f"已达全局执行上限 {MAX_TOTAL_EXECUTIONS} 次，第 {cursor + 1} 步不再执行",
                observation="超出执行预算"),
        }

    # 1) 条件步骤：`when` 不满足直接跳过（不算失败，也不占重做额度）
    if not check_condition(step.get("when"), results):
        note = f"前置条件「{step['when']}」不满足，已跳过这一步"
        log_event(state.get("trace_id") or "-", "ma_node", node="executor",
                  engine="multi_agent", action=action, status="skipped",
                  when=step.get("when") or "")
        return {
            "executions": executions,
            "step_args": {},
            "step_output": {"skipped": True, "reason": note},
            "step_error": "",
            "steps": _step_trace(
                state, "Executor 执行",
                f"第 {cursor + 1} 步「{step['desc']}」：{note}",
                payload={"action": action, "when": step.get("when") or ""},
                observation=note),
        }

    # 2) 参数：先解析占位符，再叠加 Critic 打回时写入的参数覆盖
    args = resolve_args(step.get("args") or {}, results)
    args.update(state.get("step_args") or {})
    # **最后一道白名单**：`_normalize_plan` 只过滤了 Planner 写的参数，而
    # `revise_node` 会把 Critic（含它的 LLM 语义层）给出的 `建议.args` 合并进来 ——
    # 那条路没走过校验。全量评测实测命中：Critic 建议里带了 `city` / `keywords`，
    # 直接传给 `match_resume` 抛 TypeError。所以调用前必须再过一遍工具声明的参数。
    args, bad_args = _clean_args(action, args)
    if bad_args and state.get("verbose"):
        print(f"[ma_executor] 已摘掉 {action} 不支持的参数：{bad_args}")
    if action == "match_resume":
        # `resume_json` 写 `"current"` 时必须**在这里替换成真实简历**。
        # ReAct 那条链路是在 react_agent 的循环里做的替换（见 reap_agent 的
        # `if payload in (None, "current", "")`）；本图不经过那段代码，
        # 少了这一步 `normalize_resume("current")` 会退化成
        # `{"_plain": "current"}` → 空简历 → **匹配恒 0 分**（真测实测命中，
        # 然后 Critic 打回 2 次、最终降级，用户看到的是一次都没成）。
        raw_resume = str(args.get("resume_json") or "")
        if raw_resume in ("", "current", "None"):
            current = None
            try:
                current = reg.get_current_resume()
            except Exception as e:                      # noqa: BLE001 - 读不到就如实说
                if state.get("verbose"):
                    print(f"[ma_executor] 读取当前简历失败：{type(e).__name__}: {e}")
            if not current:
                # 没有简历就别调工具了：`_match` 拿到空简历只会给 0 分，
                # 那是个误导性结果。直接给明确的失败原因，且**标注不可重试**
                # （重做一百次也还是没有简历），由 Critic 判为降级。
                note = ("缺少简历：当前会话里没有可用简历，无法打分。"
                        "请先上传 PDF / Word 简历，或粘贴简历文本。")
                log_event(state.get("trace_id") or "-", "ma_node", node="executor",
                          engine="multi_agent", action=action, status="no_resume")
                return {
                    "executions": executions,
                    "step_args": args,
                    "step_output": None,
                    "step_error": note,
                    "steps": _step_trace(
                        state, "Executor 执行",
                        f"第 {cursor + 1} 步「{step['desc']}」→ 会话里没有简历，未调用 {action}",
                        payload={"action": action},
                        observation=note),
                }
            args["resume_json"] = json.dumps(current, ensure_ascii=False, default=str)

    # 3) 执行
    output, error = None, ""
    try:
        raw = reg.call_tool(action, args)
        output = _normalize_output(action, raw, args)
    except Exception as e:                              # noqa: BLE001 - 失败交给 Critic 判
        error = f"{type(e).__name__}: {e}"
        if state.get("verbose"):
            print(f"[ma_executor] {action} 失败：{error}")

    log_event(state.get("trace_id") or "-", "ma_node", node="executor",
              engine="multi_agent", action=action,
              status="ok" if not error else "error",
              attempt=attempts, error=error[:200])
    return {
        "executions": executions,
        "step_args": args,
        "step_output": output,
        "step_error": error,
        "steps": _step_trace(
            state, "Executor 执行",
            f"第 {cursor + 1} 步「{step['desc']}」→ 调用 {action}"
            + (f"（第 {attempts + 1} 次尝试）" if attempts else ""),
            payload={"action": action, "args": args},
            observation=error or _summarize(action, output)),
    }


# ==========================================================================
# Agent 3：Critic（审查者）—— 本轮的核心：能打回重做
# ==========================================================================

#: 这些报错来自 `tools_registry._validate_args` / `call_tool` 的**确定性**校验，
#: 重做同样的输入必然得到同样的结果 —— 判为不可重试，直接降级。
_FATAL_ARG_MARKERS = ("不支持参数", "参数必须是 dict", "未知工具")


_CRITIC_PROMPT = """你是「复杂任务编排」里的**审查者（Critic）**。执行者刚做完一步，
你要审查结果是否合格，不合格就打回重做。

【用户的原始需求】
{question}

【这一步要做什么】
{desc}

【执行者用的参数】
{args}

【执行结果】
{output}

【系统给出的确定性检查结论（事实层，不要推翻）】
{deterministic}

【审查要求】
1. 判断这一步的结果**是不是真的完成了用户要的这件事**。
2. 不合格时必须给出**可执行的打回建议**：把参数改成什么才能做对？
   例如「搜岗位 0 条 → 去掉城市限制」「关键词太窄 → 换成更常见的词」。
   如果问题不在参数上（例如工具报错、数据本身没有），把 args 留空对象，
   并在理由里说清"重做也解决不了"，系统会按降级处理。
3. 只有确实不合格才判 false。结果合理、只是没那么完美，就判 true。

只输出一个 JSON，不要任何其他内容：
{{"合格": true 或 false, "理由": "一句话说明判定依据", "打回建议": {{"args": {{}}}}}}"""


def deterministic_critique(action: str, output: Any, error: str,
                           args: dict) -> dict:
    """**事实层**审查（不依赖模型，可复现）。

    这是 Critic 的骨架：绝大多数不合格是确定性可判的（0 条结果、分数异常、
    三件套缺文件、工具报错）。事实层先判的好处有两个：
      * 判据不随模型心情漂移，「能打回」这件事是**可测的**；
      * 事实层已判不合格时**不再调 LLM**，省一次调用（"假多智能体"的成本正是这么来的）。

    返回 `{合格, 理由, 建议, 可重试}`。`建议["args"]` 是**真的会合并进步骤参数**的
    覆盖项，不是一句"建议重试"；`可重试` 表示"参数不用改，但值得原样再来一次"
    （工具超时 / LLM 偶发失败这类瞬时故障）。

    注意 `args` 传的是**本步实际用的参数**（含上一次打回写入的覆盖），不是计划里的
    原始参数 —— 否则「去掉城市后仍然 0 条」会被再一次建议"去掉城市"，
    重做两轮做的是同一件事（纯浪费）。
    """
    def verdict(ok, reason, override=None, retryable=False):
        return {"合格": ok, "理由": reason, "建议": {"args": override or {}},
                "可重试": bool(retryable)}

    if error:
        if "缺少简历" in error:
            # 「没有简历」不是参数问题，重做一百次也还是没简历 → 不可重试，直接降级，
            # 把明确的处置建议（去上传简历）如实交给用户。
            return verdict(False, error)
        # 参数非法是**确定性**错误（同样的参数重做多少次结果都一样），不能当瞬时故障
        # 重试，否则会白烧 2 次重做再降级。正常路径上 `_clean_args` 已经把它挡在进图前，
        # 这里只是兜底（覆盖 `_validate_args` 的三种报错 + 工具名不存在）。
        if any(marker in error for marker in _FATAL_ARG_MARKERS):
            return verdict(False, f"参数不合法（{error}），重做也解决不了")
        # 其余工具异常多为瞬时故障（超时 / 网络 / LLM 截断），原样重试有意义
        return verdict(False, f"工具执行失败（{error}）", retryable=True)

    if not isinstance(output, dict):
        return verdict(False, f"执行结果不是结构化数据：{str(output)[:80]}")

    if output.get("skipped"):
        # 条件不满足是**设计内**的跳过，不是失败
        return verdict(True, str(output.get("reason") or "前置条件不满足，按计划跳过"))

    if action == "search_jobs":
        count = int(output.get("count") or 0)
        if count > 0:
            return verdict(True, f"命中 {count} 条岗位，可以继续")
        # 打回建议：按「限制最强 → 最弱」逐级放宽，每次只放宽一层
        # （收敛快、也不会一步放到底）。读的是**实际生效**的参数，
        # 所以第二轮不会重复建议同一个放宽动作。
        if str(args.get("city") or "").strip():
            return verdict(False,
                           f"关键词「{args.get('keyword')}」在「{args.get('city')}」"
                           "一条都没搜到，建议去掉城市限制重搜",
                           {"city": ""})
        if str(args.get("job_type") or "").strip():
            return verdict(False, "加上岗位类型过滤后一条都没搜到，建议先不限类型重搜",
                           {"job_type": ""})
        if str(args.get("keyword") or "").strip():
            return verdict(False,
                           f"关键词「{args.get('keyword')}」搜不到岗位，"
                           "建议不限关键词（按城市/时间返回近期岗位）",
                           {"keyword": "", "semantic": False})
        return verdict(False, "不带任何条件仍然搜不到岗位，岗位库可能没有数据，重做也解决不了")

    if action == "match_resume":
        score = output.get("score")
        if not isinstance(score, int):
            return verdict(False, f"没有拿到有效分数（score={score!r}）", retryable=True)
        if not 0 <= score <= 100:
            return verdict(False, f"分数 {score} 超出 0-100 的合法区间", retryable=True)
        if score == 0:
            # 有简历却 0 分：这**不是参数问题**（Executor 已经会用当前简历重打，
            # 重做一次拿到的还是同一份简历 + 同一个岗位），所以不给可执行的参数覆盖，
            # 让路由直接降级 —— 否则会白烧 2 次重做去重复同一个动作。
            return verdict(False, "打分为 0 分，而会话里是有简历的 —— 这通常说明简历内容"
                                  "没被解析出来（缺技能/项目/学历），重做同样输入拿不到"
                                  "不同结果，按降级处理更诚实")
        return verdict(True, f"打分 {score}/100，落在合法区间")

    if action == "generate_application_package":
        files = output.get("files") or {}
        missing = [name for name, path in files.items()
                   if not path or not os.path.isfile(str(path))
                   or os.path.getsize(str(path)) == 0]
        if len(files) >= 3 and not missing:
            return verdict(True, f"投递包三件套齐全（{len(files)} 个文件）")
        # 投递包不完整通常是 LLM 偶发失败（交接单 5.16 的"模板兜底"就是这类），
        # 参数不用改，原样重做一次往往就成了。
        return verdict(False, f"投递包不完整（缺失/空文件：{missing or sorted(files)}），"
                              "建议重做一次", retryable=True)

    return verdict(True, "没有针对该动作的确定性判据")


def _critic_llm(question: str, step: dict, args: dict, output: Any,
                det: dict, verbose: bool = False) -> dict:
    """语义层复核（只在事实层通过、且是「需要判断分寸」的步骤时调用）。"""
    data = _llm_json(
        [{"role": "user", "content": _CRITIC_PROMPT.format(
            question=question,
            desc=step.get("desc") or step.get("action"),
            args=json.dumps(args, ensure_ascii=False),
            output=json.dumps(output, ensure_ascii=False)[:1500],
            deterministic=det.get("理由") or "",
        )}],
        source="ma_critic",
        verbose=verbose,
    )
    suggestion = data.get("打回建议") if isinstance(data.get("打回建议"), dict) else {}
    override = suggestion.get("args") if isinstance(suggestion.get("args"), dict) else {}
    return {"合格": bool(data.get("合格", True)),
            "理由": str(data.get("理由") or "").strip(),
            "建议": {"args": override}}


def critic_node(state: ComplexState) -> dict:
    """**[Critic]** 审查执行结果：事实层 → （必要时）语义层 → 打回建议。

    判定顺序刻意「先便宜后昂贵」：
      1. 事实层不合格 → **直接**返回，不再调 LLM（省钱，且判据可复现）；
      2. 事实层通过且是 `match_resume`（唯一需要判断分寸的步骤）→ 调一次 LLM 复核；
      3. 其它通过 → 直接合格。

    `actionable` 是「这条打回能不能真的改变点什么」：有参数覆盖，或值得原样重试。
    为 False 时 `route_after_critic` 会直接降级 —— 不白烧重试次数。
    """
    plan = state.get("plan") or []
    cursor = int(state.get("cursor") or 0)
    step = plan[cursor] if cursor < len(plan) else {}
    action = str(step.get("action") or "")
    output = state.get("step_output")
    error = state.get("step_error") or ""
    attempts = int((state.get("critique") or {}).get("attempts") or 0)

    det = deterministic_critique(action, output, error,
                                 state.get("step_args") or (step.get("args") or {}))
    verdict = {"合格": det["合格"], "理由": det["理由"], "建议": det["建议"],
               "layer": "deterministic"}
    llm_used = False

    need_llm = (det["合格"] and action == "match_resume"
                and CRITIC_LLM_ENABLED and not (output or {}).get("skipped"))
    if need_llm:
        budget = limits.run_budget_status()
        try:
            if budget.get("exceeded"):
                raise RuntimeError(f"本轮 token 预算已用尽"
                                   f"（{budget.get('used')}/{budget.get('limit')}）")
            llm = _critic_llm(state.get("question") or "", step,
                              state.get("step_args") or {}, output, det,
                              verbose=bool(state.get("verbose")))
            llm_used = True
            if not llm["合格"] and (llm["建议"] or {}).get("args"):
                verdict = {"合格": False, "理由": llm["理由"] or det["理由"],
                           "建议": llm["建议"], "layer": "llm"}
            else:
                # 模型说合格，或虽说不合格却给不出**可执行的参数修改** → 采纳事实层结论。
                # 这一步很关键：否则模型一句"分数偏低"就能让流程空转一轮
                # （不合格 → 重做 → 参数没变 → 结果一样 → 再判不合格），
                # 正是"假多智能体"最典型的浪费。
                verdict = {"合格": True,
                           "理由": f"{det['理由']}；语义复核：{llm['理由'] or '无异议'}",
                           "建议": {"args": {}}, "layer": "deterministic+llm"}
        except Exception as e:                          # noqa: BLE001 - 复核不可用走事实层
            if state.get("verbose"):
                print(f"[ma_critic] 语义复核不可用：{type(e).__name__}: {e}")
            verdict["理由"] += f"（语义复核不可用：{type(e).__name__}）"

    verdict["attempts"] = attempts
    verdict["action"] = action
    verdict["llm_used"] = llm_used
    verdict["actionable"] = bool((verdict.get("建议") or {}).get("args")) \
        or bool(det.get("可重试"))

    log_event(state.get("trace_id") or "-", "ma_node", node="critic",
              engine="multi_agent", action=action, ok=verdict["合格"],
              layer=verdict["layer"], llm_used=llm_used, attempts=attempts,
              actionable=verdict["actionable"], reason=verdict["理由"][:200])
    icon = "✅ 合格" if verdict["合格"] else "⚠️ 不合格（打回重做）"
    return {
        "critique": verdict,
        "critiques": list(state.get("critiques") or []) + [dict(verdict)],
        "steps": _step_trace(
            state, "Critic 审查",
            f"第 {cursor + 1} 步「{step.get('desc', '')}」审查结果：{icon} —— {verdict['理由']}"
            + (f"｜打回建议：{json.dumps(verdict['建议'], ensure_ascii=False)}"
               if not verdict["合格"] else ""),
            payload={"action": action, "layer": verdict["layer"],
                     "llm_used": llm_used, "建议": verdict["建议"]},
            observation=verdict["理由"]),
    }


def revise_node(state: ComplexState) -> dict:
    """**[修正 → 打回重做]** 把 Critic 的打回建议合并进步骤参数，回到 Executor。

    这是"真协作"的落点：**Critic 说的话真的改变了下一步执行的输入**。
    如果这里只是把 attempts +1 而参数不动，那就退化成"重试"而不是"打回重做"。
    """
    plan = state.get("plan") or []
    cursor = int(state.get("cursor") or 0)
    step = plan[cursor] if cursor < len(plan) else {}
    critique = state.get("critique") or {}
    override = (critique.get("建议") or {}).get("args") or {}
    attempts = int(critique.get("attempts") or 0) + 1

    rework = {
        "step": cursor + 1,
        "action": step.get("action", ""),
        "attempt": attempts,
        "reason": critique.get("理由", ""),
        "args_override": override,
    }
    change = (f"参数改为 {json.dumps(override, ensure_ascii=False)}"
              if override else "参数不变（上次是瞬时失败，原样重试一次）")
    log_event(state.get("trace_id") or "-", "ma_node", node="revise",
              engine="multi_agent", step=cursor + 1, attempt=attempts,
              override=json.dumps(override, ensure_ascii=False))
    return {
        # 覆盖项**合并**进参数（不整体替换）：只改 Critic 指出的那一项，
        # 其余原参数保留，避免"修一个坏一个"。
        "step_args": {**(state.get("step_args") or {}), **override},
        "reworks": list(state.get("reworks") or []) + [rework],
        # 传入 attempts 让下一轮 Critic 知道"这是第几次重做"，超过上限就降级
        "critique": {"attempts": attempts, "合格": False,
                     "理由": critique.get("理由", ""), "建议": {"args": {}}},
        "steps": _step_trace(
            state, "打回重做",
            f"Critic 判第 {rework['step']} 步不合格，{change}"
            f"（第 {attempts}/{MAX_STEP_RETRIES} 次）—— {critique.get('理由', '')}",
            payload={"args_override": override, "attempt": attempts},
            observation=f"重做第 {attempts} 次"),
    }


def _record_result(state: ComplexState, ok: bool, skipped: bool = False) -> list:
    """把本步结果写进 `results`（`advance` 与 `degrade` 共用）。"""
    plan = state.get("plan") or []
    cursor = int(state.get("cursor") or 0)
    step = plan[cursor] if cursor < len(plan) else {}
    critique = state.get("critique") or {}
    return list(state.get("results") or []) + [{
        "id": cursor + 1,
        "action": step.get("action", ""),
        "desc": step.get("desc", ""),
        "ok": bool(ok),
        "skipped": bool(skipped),
        "attempts": int(critique.get("attempts") or 0),
        "output": state.get("step_output"),
        "error": state.get("step_error") or "",
        "critique": critique.get("理由", ""),
    }]


def advance_node(state: ComplexState) -> dict:
    """**[进入下一步]** 本步合格 → 记结果 + 游标 +1 + 清空打回痕迹。"""
    results = _record_result(state, ok=True,
                             skipped=bool((state.get("step_output") or {}).get("skipped")))
    cursor = int(state.get("cursor") or 0) + 1
    total = len(state.get("plan") or [])
    log_event(state.get("trace_id") or "-", "ma_node", node="advance",
              engine="multi_agent", cursor=cursor, total=total)
    return {
        "results": results,
        "cursor": cursor,
        # 换新步骤：清掉上一步的打回痕迹（attempts 从 0 重新算）
        "critique": {},
        "step_args": {},
        "step_output": None,
        "step_error": "",
        "steps": _step_trace(
            state, "进入下一步",
            f"第 {cursor} 步通过审查" if cursor < total
            else "全部步骤执行完毕，准备汇总输出",
            observation=f"游标 {cursor}"),
    }


def degrade_node(state: ComplexState) -> dict:
    """**[降级]** 同一步重做满 `MAX_STEP_RETRIES` 次仍不合格 → 收下部分结果。

    降级的语义是「**保留已经做成的部分 + 如实说明哪一步没做成**」，
    **不是**把失败包成成功。这是「无死循环」的最后一道闸门。
    """
    results = _record_result(state, ok=False)
    cursor = int(state.get("cursor") or 0)
    plan = state.get("plan") or []
    desc = plan[cursor].get("desc", "") if cursor < len(plan) else ""
    critique = state.get("critique") or {}
    reason = (f"第 {cursor + 1} 步「{desc}」重做 {MAX_STEP_RETRIES} 次后仍不合格："
              f"{critique.get('理由', '')}")
    log_event(state.get("trace_id") or "-", "ma_node", node="degrade",
              engine="multi_agent", reason=reason[:200])
    return {
        "results": results,
        "degraded": True,
        "degrade_reason": reason,
        "steps": _step_trace(
            state, "降级收尾",
            f"{reason}；停止重做，改为返回已完成的部分结果并如实说明",
            observation="已降级"),
    }


# --------------------------------------------------------------------------
# 边：路由函数
# --------------------------------------------------------------------------

def route_after_critic(state: ComplexState) -> str:
    """Critic 之后走哪条边：合格 → 下一步；不合格 → 打回 / 降级。

    三个判据，顺序即优先级：
      1. 合格 → `advance`；
      2. 不合格但**不可执行**（没有参数覆盖、也不值得原样重试）→ 直接 `degrade`，
         不白烧重试次数；
      3. 不合格且可执行 → 还有重做额度就走 `revise`，否则 `degrade`。
    """
    critique = state.get("critique") or {}
    if critique.get("合格", True):
        return "advance"
    if not critique.get("actionable"):
        return "degrade"
    if int(critique.get("attempts") or 0) >= MAX_STEP_RETRIES:
        return "degrade"
    return "revise"


def route_after_advance(state: ComplexState) -> str:
    """下一步：还有步骤就继续执行，否则汇总输出。"""
    cursor = int(state.get("cursor") or 0)
    return "execute" if cursor < len(state.get("plan") or []) else "respond"


# ==========================================================================
# 汇总输出
# ==========================================================================

def _format_search_block(output: dict) -> list:
    """搜岗位结果转成展示行（复用 langgraph_flow 的行格式，口径一致）。"""
    from agent.langgraph_flow import DISPLAY_LIMIT, _job_line

    rows = list(output.get("rows") or [])
    if not rows:
        return ["- 搜岗位：没有命中岗位。"]
    shown = rows[:DISPLAY_LIMIT]
    lines = [f"- 搜岗位：命中 **{len(rows)}** 条，列出前 {len(shown)} 条：", ""]
    for row in shown:
        item = dict(row)
        # `core` 只有搜岗位图的 filter 节点会打；这里补一个等价粗判，
        # 免得 _job_line 的「核心匹配 / 相关」标注永远是「相关」。
        item.setdefault("core", False)
        lines.append("  " + _job_line(item))
    lines.append("")
    return lines


def respond_node(state: ComplexState) -> dict:
    """**[输出]** 汇总各步结果 + 三 Agent 的协作轨迹（含打回记录）。"""
    results = list(state.get("results") or [])
    reworks = list(state.get("reworks") or [])
    degraded = bool(state.get("degraded"))

    lines = ["## 复杂任务执行结果", ""]
    if degraded:
        lines += [f"> ⚠️ **已降级**：{state.get('degrade_reason')}", "",
                  "下面给出**已经完成的部分**；没做成的那一步如实说明，没有编造。", ""]

    for item in results:
        output = item.get("output") or {}
        action = item.get("action")
        mark = "✅" if item.get("ok") else ("⏭️" if item.get("skipped") else "❌")
        lines += [f"### {mark} 第 {item['id']} 步 · {item.get('desc') or action}", ""]
        if item.get("skipped"):
            lines.append(f"- {output.get('reason') or '前置条件不满足，已跳过'}")
        elif not item.get("ok"):
            lines.append(f"- 失败原因：{item.get('error') or item.get('critique') or '未知'}")
        elif action == "search_jobs":
            lines += _format_search_block(output)
        elif action == "match_resume":
            lines.append(f"- **匹配得分：{output.get('score')}/100**")
            dims = output.get("dimensions") or {}
            if dims:
                lines.append("- 各维度：" + " · ".join(f"{k} {v}" for k, v in dims.items()))
            for label, key in (("匹配亮点", "highlights"), ("差距", "gaps"),
                               ("通用建议", "general_advice")):
                items = output.get(key) or []
                if items:
                    lines.append(f"- {label}：")
                    lines += [f"  - {x}" for x in items]
        elif action == "generate_application_package":
            lines.append(f"- 投递包目录：`{output.get('package_dir')}`")
            for name, path in (output.get("files") or {}).items():
                lines.append(f"  - `{name}` → {path}")
            if output.get("warnings"):
                lines.append("- 提醒：" + "；".join(str(w) for w in output["warnings"]))
        lines.append("")

    if reworks:
        lines += ["### 🔁 Critic 打回记录（真协作的证据）", ""]
        for item in reworks:
            lines.append(f"- 第 {item['step']} 步 · 第 {item['attempt']} 次重做：{item['reason']}")
            if item.get("args_override"):
                lines.append(f"  - 参数改为：`{json.dumps(item['args_override'], ensure_ascii=False)}`")
        lines.append("")

    if not results:
        lines += ["没有执行任何步骤（计划为空）。请换种说法再说一次。", ""]

    log_event(state.get("trace_id") or "-", "ma_node", node="respond",
              engine="multi_agent", steps=len(results), reworks=len(reworks),
              degraded=degraded, plan_source=state.get("plan_source") or "")
    return {
        "answer": "\n".join(lines).rstrip(),
        "steps": _step_trace(
            state, "汇总输出",
            f"汇总 {len(results)} 个步骤的结果"
            + (f"，其中 {len(reworks)} 次打回重做" if reworks else "")
            + ("（已降级）" if degraded else ""),
            observation=f"steps={len(results)} reworks={len(reworks)}"),
    }


# ==========================================================================
# 编译图
# ==========================================================================

def build_complex_task_graph(executor=None, critic=None):
    """编译「复杂任务编排图」。

    节点与边::

        plan → execute → critic ─(advance)──→ advance ─(还有步骤)→ execute
                            │                          └(做完了)────→ respond
                            ├(revise)──→ revise → execute（重做同一步）
                            └(degrade)─→ degrade ─────────────────────→ respond

    `executor` / `critic` 可注入（单测用替身，才能稳定复现"执行结果不合格"
    并断言「Critic 真的打回了、参数真的变了、最多只重做 2 次」）。
    """
    graph = StateGraph(ComplexState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", executor or execute_node)
    graph.add_node("critic", critic or critic_node)
    graph.add_node("revise", revise_node)
    graph.add_node("advance", advance_node)
    graph.add_node("degrade", degrade_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "critic")
    graph.add_conditional_edges("critic", route_after_critic,
                                {"advance": "advance", "revise": "revise",
                                 "degrade": "degrade"})
    graph.add_edge("revise", "execute")
    graph.add_conditional_edges("advance", route_after_advance,
                                {"execute": "execute", "respond": "respond"})
    graph.add_edge("degrade", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


COMPLEX_TASK_GRAPH = build_complex_task_graph()


# ==========================================================================
# 入口判断：什么算「复杂任务」
# ==========================================================================

#: 「要投递材料」意图（生成投递包 / 简历定制 / 自荐信）
PACKAGE_WORDS = ("投递包", "投递材料", "一键投递", "简历定制", "定制简历", "自荐信",
                 "求职信", "cover letter", "打包")
#: 明确要求「一条龙 / 整个流程」的说法
FULL_FLOW_WORDS = ("一条龙", "整个流程", "全流程", "完整流程", "从头到尾",
                   "一整套", "全自动帮我", "帮我走完")

#: 「找岗位」意图的词表（**独立于 `react_agent_lg.is_search_intent`**，原因见 detect_intents）
_SEARCH_NOUNS = ("岗位", "职位", "实习", "招聘", "工作机会", "机会")
_SEARCH_VERBS = ("找", "搜", "查", "看看", "有没有", "推荐", "列出", "列一下", "筛选")
#: 「匹配打分」意图的词表
_MATCH_WORDS = ("匹配", "打分", "评分", "match", "匹配度", "对口", "合不合适",
                "符不符合", "能不能过", "适合我吗")


def detect_intents(question: str) -> list:
    """这句话里出现了哪几种**任务意图**（搜索 / 匹配 / 投递材料）。

    ⚠️ 这里**刻意不复用** `react_agent_lg.is_search_intent` / `is_match_intent`：
    那两个函数是**互斥分流**用的 —— 一旦句子里出现「生成 / 投递包 / 匹配」这类
    其它流程的词，它们会**主动返回 False**，好把这句话让给别的流程。
    对单意图分流那是对的；但对「判断这句话里有几种需求」是致命的：
    「找岗位 + 匹配 + 出投递包」会被它们同时否掉，只剩 package 一种意图，
    于是复杂任务永远判不出来。所以这里用**互不排斥**的词表各自识别。
    """
    text = (question or "").strip()
    if not text or text.startswith("/"):
        return []
    low = text.lower()
    intents = []
    if any(noun in text for noun in _SEARCH_NOUNS) \
            and any(verb in text for verb in _SEARCH_VERBS):
        intents.append("search")
    if any(word in low for word in _MATCH_WORDS):
        intents.append("match")
    if any(word in low for word in PACKAGE_WORDS):
        intents.append("package")
    return intents


def is_complex_task(question: str) -> bool:
    """判定「复杂任务」：**一次说了多个需求**，或**明确要求走完整流程**。

    判据（刻意保守，宁可不触发）：
      * 命中 ≥2 种意图（例如"找岗位 + 匹配"、"匹配 + 出投递包"）；
      * 或者出现「一条龙 / 整个流程」这类显式全流程说法，且至少 1 种意图。

    为什么保守：多智能体比单流程贵（Planner + Critic 都是额外 LLM 调用）。
    一句话只要求搜岗位时也走它，就是**为了三个 Agent 而三个 Agent** ——
    加成本、不加质量，正是要避免的"假多智能体"。
    """
    if not MULTI_AGENT_ENABLED:
        return False
    text = (question or "").strip()
    if not text or text.startswith("/"):
        return False
    intents = detect_intents(text)
    if len(intents) >= 2:
        return True
    if intents and any(word in text for word in FULL_FLOW_WORDS):
        return True
    return False
