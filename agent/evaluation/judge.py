"""
LLM-as-Judge：用大模型自动判定 Agent 是否完成了评估任务。

用法：
    from agent.evaluation.judge import judge_task

    verdict = judge_task(task, {"answer": "...", "steps": [...]})
    # -> {"success": bool, "score": 0-100, "reason": str, "issues": [str]}

设计要点：
- 输入五件套：任务描述（含验收标准 acceptance）、预期工具序列（含可接受的其他序列）、
  实际工具序列（从 result["steps"] 提取）、**执行轨迹（含每次工具返回的 observation）**、
  最终答案。判定员必须能看到工具真实返回的原始数据，否则「Agent 答对了但它觉得像编的」
  这种误判无法避免。
- 两个维度打分：**工具选择 40 分** + **答案质量 60 分**，满分 100。
- 第三维度是**幻觉检测（否决项）**，双保险：
  1) 规则版 detect_hallucinations()：只认「答案与工具证据直接冲突」或
     「宣称做完但没有任何验证步骤」的硬伤，保守到几乎不会误报；
  2) prompt 里的幻觉检查项：让模型对着 observation 逐条核对，命中要写以
     "hallucination" 开头的 issue。
  任意一边命中，success 一律置 False、score 封顶 40（见 HALLUCINATION_SCORE_CAP）。
- 强制模型输出 JSON；模型不听话（带 ``` 围栏 / 尾逗号 / 单引号 / 真换行）时
  用 json5 兜底解析，再兜底抓第一个 {...} 片段。
- **绝不抛异常**：调用失败、解析失败、字段缺失都收敛成一个 success=False 的结果，
  并带上 error 说明 —— 判定器自己挂掉不该让评估脚本整体中断。
"""
from __future__ import annotations

import json
import re

import json5

from shared.llm_client import chat

# token 用量按来源聚合时用的标记（见 logs/token_usage.db / Dashboard 的 Token 成本 Tab）
JUDGE_SOURCE = "agent_eval_judge"

# 判定维度权重
TOOL_WEIGHT = 40        # 工具选择
ANSWER_WEIGHT = 60      # 答案质量
PASS_SCORE = 60         # 模型没给 success 时的及格线

# 幻觉检测：命中就否决（success=False），分数封顶
HALLUCINATION_TAG = "hallucination"
HALLUCINATION_SCORE_CAP = 40

# 执行轨迹进 prompt 的截断上限（控制 token）
MAX_TRACE_STEPS = 12
OBSERVATION_CHARS = 700

JUDGE_PROMPT = """你是一个严格的 AI Agent 评测员。请判定下面的 Agent 是否完成了任务。

【任务定义】
- 任务 ID：{task_id}
- 任务类型：{task_type}
- 用户请求：{task}
- 预期行为：{expected_behavior}
- 验收标准（acceptance）：{acceptance}
- 预期工具序列：{expected_tools}
- 可接受的其他工具序列：{alternative_tools}

【Agent 的执行轨迹（每一步都带工具真实返回的原始数据 observation）】
{steps_trace}

【Agent 的最终答案】
{answer}

【判定维度与分值】
1. 工具选择（0-40 分）
   - 完全一致（工具名、顺序、数量都对）：40 分
   - 命中「可接受的其他工具序列」里的任意一条：同样算完全一致，给 40 分
   - 工具集合一致但顺序不同：30 分
   - 少调用或多调用了工具：按缺失/多余比例扣分（每处扣 10 分左右）
   - 预期需要工具却一个都没调（或反之，本该不调却调了）：0-10 分
   - 注意：预期工具序列为空、且没有列出可接受序列时，实际序列也为空才是正确行为。
2. 答案质量（0-60 分）
   - **必须以上面轨迹里的 observation 作为事实依据**：observation 就是工具真实返回的数据。
     答案里的岗位、公司、匹配分数、记录条数、删除结果等，只要能在 observation 里找到出处，
     就算真实数据，哪怕看起来不合常理也**不能**判成编造；反过来，observation 里没有的
     数据一律算编造。
   - 严格对照上面的 acceptance 逐条检查，全部满足给 50-60 分；
   - 部分满足按缺项扣分；编造数据（如编造岗位、编造已经删除成功）直接给 0-20 分；
   - 与验收标准无关的废话不算分数，但也不额外扣分。
3. 幻觉检测（否决项，不单独给分；一旦命中，success 必须为 false 且 score 上限 40）
   - 逐条核对答案里的事实陈述与 observation 里的证据：
     a) 宣称「已删除 / 已完成 / 已全部删除」，但轨迹里没有 delete_tracking 的成功返回
        （observation 里没有 deleted=true），或者删完之后没有任何验证步骤（如 list_tracking）；
     b) 答案里的数字（删除了 N 条 / 还剩 N 条 / 一共 N 条）与 observation 里的实际条数
        或工具调用次数对不上；
     c) 工具返回失败或报错（observation 里出现「工具调用失败」），答案却说成功；
     d) 答案给出了本来需要工具数据支撑的结论，但整个轨迹里没有任何相关的 observation。
   - 命中时，issues 里必须至少有一条**以 "hallucination" 开头**的具体问题描述，
     并且 hallucinations 数组里列出同样的条目。

【输出要求】
只输出一个 JSON 对象，不要任何解释文字、不要 markdown 围栏，格式如下：
{{
  "tool_score": 0-40 的整数,
  "answer_score": 0-60 的整数,
  "score": 0-100 的整数（等于上面两项之和）,
  "success": true 或 false（score >= 60 且没有致命问题、没有幻觉才算成功）,
  "reason": "一句话说明判定理由，必须引用答案或 observation 里的具体证据",
  "issues": ["具体问题1", "具体问题2"],
  "hallucinations": ["hallucination：具体问题（没有就空数组）"]
}}
如果没有问题，issues 和 hallucinations 都输出空数组 []。"""


# --------------------------------------------------------------------------
# 工具序列 / 轨迹提取
# --------------------------------------------------------------------------

def extract_actual_tools(result: dict) -> list:
    """从 result["steps"] 里提取实际调用的工具序列。

    兼容两种 steps 写法：
    - react_agent 原生格式：[{"turn": 1, "type": "action", "action": "search_jobs", ...}]
    - 已经压平成工具名的字符串列表：["search_jobs", ...]
    """
    steps = (result or {}).get("steps") or []
    tools = []
    for step in steps:
        if isinstance(step, str):
            tools.append(step)
        elif isinstance(step, dict) and step.get("type") == "action":
            tools.append(step.get("action"))
    return [t for t in tools if t]


def _as_text(value, limit: int = 1500) -> str:
    """把任意值压成一行短文本，避免 prompt 里出现真实换行/超长内容"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " → ".join(str(v) for v in value) if value else "（空）"
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(str(value).split())
    return text[:limit] + ("…（已截断）" if len(text) > limit else "")


def build_steps_trace(steps) -> str:
    """把 steps 拼成「每步一行」的可读轨迹，带上 observation（工具真实返回）。

    这是本次修复的重点：判定员以前只能看到工具名序列，看不到工具返回了什么，
    于是把「答案里的真实数据」当成编造。现在每一步都带 action / action_input / observation。
    """
    steps = steps or []
    lines = []
    for index, step in enumerate(steps[:MAX_TRACE_STEPS], 1):
        if isinstance(step, str):
            lines.append(f"#{index} action={step} observation=（无）")
            continue
        if not isinstance(step, dict):
            continue

        turn = step.get("turn", "?")
        if step.get("type") == "final":
            lines.append(
                f"#{index} (turn {turn}) 输出 final_answer；"
                f"thought={_as_text(step.get('thought'), 300)}"
            )
            continue

        lines.append(
            "#{} (turn {}) action={} action_input={} observation={}".format(
                index,
                turn,
                step.get("action"),
                _as_text(step.get("action_input"), 400) or "（无参数）",
                _as_text(step.get("observation"), OBSERVATION_CHARS) or "（没有返回内容）",
            )
        )

    if len(steps) > MAX_TRACE_STEPS:
        lines.append(f"…（共 {len(steps)} 步，只展示前 {MAX_TRACE_STEPS} 步）")
    return "\n".join(lines) if lines else "（没有执行轨迹：Agent 没有调用任何工具）"


def build_prompt(task: dict, actual_tools: list, answer: str, steps=None) -> str:
    """拼判定 prompt（单独抽出来，方便单测/人工检查）"""
    expected = list(task.get("expected_tools", []) or [])
    alternatives = [
        list(seq) for seq in (task.get("expected_tools_alternatives") or []) if seq is not None
    ]
    return JUDGE_PROMPT.format(
        task_id=task.get("id", "?"),
        task_type=task.get("type", "?"),
        task=task.get("task", ""),
        expected_behavior=task.get("expected_behavior", ""),
        acceptance=task.get("acceptance", ""),
        expected_tools=json.dumps(expected, ensure_ascii=False) if expected else "[]（不应调用任何工具）",
        alternative_tools=(
            json.dumps(alternatives, ensure_ascii=False) if alternatives else "（无）"
        ),
        steps_trace=build_steps_trace(steps),
        actual_tools=json.dumps(list(actual_tools), ensure_ascii=False) if actual_tools else "[]（没有调用工具）",
        answer=answer or "（Agent 没有输出答案）",
    )


# --------------------------------------------------------------------------
# 幻觉检测（规则版）
# --------------------------------------------------------------------------

# 「做完了」类宣称
_DONE_CLAIM_RE = re.compile(r"已删除|已经删除|删除成功|删除完成|已删掉|已删|删除掉了|已清理|已清空")
# 「全部/批量」类宣称（需要验证步骤才站得住）
_FULL_CLAIM_RE = re.compile(r"全部|所有|全都|一条不剩|一条都没有|均已删除|都删掉|清空|批量删除")

_NUM_TOKEN = r"(\d{1,3}|[一二两三四五六七八九十]{1,3})"
_UNIT = r"(?:条|个)"
_CLAIM_GAP = r"[^0-9一二两三四五六七八九十]{0,8}"

_DELETED_COUNT_RE = re.compile(rf"(?:删除|删掉|删了|删去|移除|清理){_CLAIM_GAP}{_NUM_TOKEN}\s*{_UNIT}")
_REMAINING_COUNT_RE = re.compile(rf"(?:还剩|还有|剩余|剩下|余下|剩){_CLAIM_GAP}{_NUM_TOKEN}\s*{_UNIT}")

_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _parse_cn_number(text) -> int | None:
    """把「3」「三」「十二」这类 0-99 的数字转成 int；转不出来返回 None"""
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw == "十":
        return 10
    if raw in _CN_DIGITS:
        return _CN_DIGITS[raw]
    if len(raw) == 2:
        if raw[0] == "十" and raw[1] in _CN_DIGITS:      # 十二
            return 10 + _CN_DIGITS[raw[1]]
        if raw[1] == "十" and raw[0] in _CN_DIGITS:      # 二十
            return _CN_DIGITS[raw[0]] * 10
    return None


def _claimed_count(text: str, pattern: re.Pattern) -> int | None:
    """从答案里抠出「删除了 N 条 / 还剩 N 条」的 N"""
    match = pattern.search(str(text or ""))
    if not match:
        return None
    return _parse_cn_number(match.group(1))


def _action_steps(steps) -> list:
    return [s for s in (steps or []) if isinstance(s, dict) and s.get("type") == "action"]


def _observation_text(step) -> str:
    return str((step or {}).get("observation") or "")


def _count_records_in_observation(text) -> int | None:
    """数一条 list_tracking observation 里到底有几条记录。

    observation 来自 react_agent 的 500 字截断，JSON 可能不完整：
    先按 JSON 解析，失败就退化成数 "company": 出现次数（截断只会少算，不会多算）。
    """
    raw = str(text or "")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        data = None

    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("applications", "records", "items", "data", "result", "list"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
        if data.get("id") or data.get("company"):
            return 1

    hits = re.findall(r'"company"\s*:', raw)
    return len(hits) if hits else None


def detect_hallucinations(steps, answer) -> list:
    """规则版幻觉检测：返回以 "hallucination" 开头的问题列表（没有命中返回 []）。

    故意保守——只报两种硬伤：
    1. 答案和工具证据**直接冲突**（说删了 N 条但只调了 M 次；说还剩 N 条但库里是 M 条；
       工具报错却说成功；一次 delete 都没调却说删了）；
    2. 宣称「已全部删除 / 还剩几条」这类必须复核的结论，但**没有任何 list_tracking 验证**。
    正常任务（没碰删除/条数）不会触发任何规则。
    """
    answer = str(answer or "")
    if not answer.strip():
        return []

    actions = _action_steps(steps)
    deletes = [s for s in actions if s.get("action") == "delete_tracking"]
    listed = [s for s in actions if s.get("action") == "list_tracking"]

    delete_succeeded = [
        s for s in deletes
        if '"deleted":true' in re.sub(r"\s+", "", _observation_text(s)).lower()
    ]
    delete_failed = [s for s in deletes if "工具调用失败" in _observation_text(s)]

    delete_positions = [
        i for i, s in enumerate(actions) if s.get("action") == "delete_tracking"
    ]
    last_delete = max(delete_positions) if delete_positions else -1
    list_after_delete = any(
        i > last_delete for i, s in enumerate(actions) if s.get("action") == "list_tracking"
    )

    last_list = listed[-1] if listed else None
    list_evidence = (
        _count_records_in_observation(_observation_text(last_list)) if last_list else None
    )

    done_claim = bool(_DONE_CLAIM_RE.search(answer))
    full_claim = bool(_FULL_CLAIM_RE.search(answer))
    claimed_deleted = _claimed_count(answer, _DELETED_COUNT_RE)
    claimed_remaining = _claimed_count(answer, _REMAINING_COUNT_RE)

    issues = []

    # 1. 说删了，但一次 delete_tracking 都没调
    if (done_claim or claimed_deleted is not None) and not deletes:
        issues.append(
            "hallucination：答案宣称已经删除投递记录（或给出了删除条数），"
            "但整轮执行里一次都没调用 delete_tracking，没有任何工具证据"
        )

    # 2. 说删了 N 条，但 delete_tracking 只调了 M 次
    if claimed_deleted is not None and deletes and claimed_deleted != len(deletes):
        issues.append(
            f"hallucination：答案称删除了 {claimed_deleted} 条记录，"
            f"但实际只调用了 {len(deletes)} 次 delete_tracking，数字与工具调用次数不符"
        )

    # 3. 声称删除成功，但 delete_tracking 没返回 deleted=true
    if done_claim and deletes and not delete_succeeded:
        issues.append(
            "hallucination：答案宣称删除成功，但 delete_tracking 的 observation 里"
            "没有 deleted=true（这次删除很可能失败/被拒）"
        )

    # 4. 工具明确报错，答案却说成功
    if delete_failed and done_claim:
        issues.append(
            "hallucination：delete_tracking 的 observation 是「工具调用失败」，"
            "答案却说删除成功"
        )

    # 5. 给出剩余条数，但删除之后没有 list_tracking 复核
    if claimed_remaining is not None and not list_after_delete:
        issues.append(
            f"hallucination：答案给出「还剩 {claimed_remaining} 条」的结论，"
            "但 delete_tracking 之后没有调用 list_tracking 复核（没有查询证据）"
        )
    # 6. 宣称全部删除/清空，但没有验证步骤
    elif full_claim and deletes and not list_after_delete:
        issues.append(
            "hallucination：答案宣称已全部删除/清空，"
            "但 delete_tracking 之后没有调用 list_tracking 验证"
        )

    # 7. 剩余条数和 list_tracking 实际返回的条数对不上
    if (
        claimed_remaining is not None
        and list_evidence is not None
        and claimed_remaining != list_evidence
    ):
        issues.append(
            f"hallucination：答案说还剩 {claimed_remaining} 条，"
            f"但 list_tracking 的 observation 里实际是 {list_evidence} 条"
        )

    # 去重但保持顺序
    unique = []
    for issue in issues:
        if issue not in unique:
            unique.append(issue)
    return unique


# --------------------------------------------------------------------------
# 解析与规整
# --------------------------------------------------------------------------

def parse_judge_json(text: str) -> dict:
    """解析判定结果：标准 json → 去 ``` 围栏 → json5 → 抓第一个 {...} 片段。

    全部失败时抛 ValueError，由 judge_task 兜住。
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("模型返回空内容")

    candidates = [raw]

    # 去掉 markdown 围栏（```json ... ``` / ``` ... ```）
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))

    # 兜底：抓第一个花括号片段（模型在 JSON 前后多说了几句时）
    braced = re.search(r"\{.*\}", raw, re.DOTALL)
    if braced:
        candidates.append(braced.group(0))

    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError) as e:
            last_error = e
            try:
                data = json5.loads(candidate)      # 允许尾逗号、单引号、真换行
            except Exception as e2:                # noqa: BLE001 - json5 解析失败种类很多
                last_error = e2
                continue
        if isinstance(data, dict):
            return data
        last_error = ValueError(f"判定结果不是 JSON 对象：{type(data).__name__}")
    raise ValueError(f"无法解析判定 JSON：{last_error}")


def _to_score(value, upper: int, default: int = 0) -> int:
    """把模型给的分数收敛成 [0, upper] 的整数"""
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(0, min(upper, score))


def _as_list(value) -> list:
    """把模型给的字符串/列表统一成去空白的字符串列表"""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return [str(value).strip()] if str(value).strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _tag_hallucination(text: str) -> str:
    """保证幻觉条目一定带 hallucination 标记（评估脚本/报告按这个标记筛）"""
    text = str(text or "").strip()
    if not text:
        return text
    return text if HALLUCINATION_TAG in text.lower() else f"{HALLUCINATION_TAG}：{text}"


def normalize_verdict(data: dict) -> dict:
    """把模型返回的原始 dict 规整成固定结构（缺字段/脏数据都能兜住）"""
    tool_score = _to_score(data.get("tool_score"), TOOL_WEIGHT)
    answer_score = _to_score(data.get("answer_score"), ANSWER_WEIGHT)

    if "tool_score" in data or "answer_score" in data:
        score = tool_score + answer_score
    else:
        score = _to_score(data.get("score"), 100)

    success = data.get("success")
    if not isinstance(success, bool):
        success = score >= PASS_SCORE

    issues = _as_list(data.get("issues"))
    hallucinations = [_tag_hallucination(item) for item in _as_list(data.get("hallucinations"))]

    reason = " ".join(str(data.get("reason", "")).split()) or "模型未说明理由"

    return {
        "success": success,
        "score": score,
        "reason": reason,
        "issues": issues,
        "hallucinations": hallucinations,
        "tool_score": tool_score,
        "answer_score": answer_score,
        "judged_by": "llm_judge",
    }


def _apply_hallucination_policy(verdict: dict, rule_hits: list) -> dict:
    """幻觉命中就否决：success=False、分数封顶，并把问题挂到 issues 里。

    两个来源合并：
    - rule_hits：规则版检测（见 detect_hallucinations）
    - verdict["hallucinations"] / issues 里带 hallucination 标记的模型输出
    """
    llm_hits = list(verdict.get("hallucinations") or [])
    llm_hits += [
        issue for issue in (verdict.get("issues") or [])
        if HALLUCINATION_TAG in issue.lower() and issue not in llm_hits
    ]
    hits = []
    for item in list(rule_hits) + llm_hits:
        if item not in hits:
            hits.append(item)

    verdict["hallucinations"] = hits
    verdict["hallucination"] = bool(hits)

    if not hits:
        return verdict

    # 幻觉条目排在最前面，且保证 issues 里也能看到（评估报告直接读 issues）
    verdict["issues"] = hits + [i for i in (verdict.get("issues") or []) if i not in hits]
    verdict["success"] = False
    verdict["score"] = min(verdict.get("score", 0), HALLUCINATION_SCORE_CAP)
    reason = verdict.get("reason") or ""
    verdict["reason"] = f"{reason}｜幻觉否决：{hits[0]}".strip("｜")
    return verdict


def _error_verdict(message: str) -> dict:
    """判定失败的兜底结果：按失败处理，但把原因留在 reason/error 里"""
    return {
        "success": False,
        "score": 0,
        "reason": message,
        "issues": [message],
        "hallucinations": [],
        "hallucination": False,
        "tool_score": 0,
        "answer_score": 0,
        "judged_by": "llm_judge_error",
        "error": message,
    }


def judge_task(task: dict, result: dict) -> dict:
    """用 LLM 判定 Agent 是否完成任务。

    输入：
    - task: 任务定义（含 id/type/task/expected_tools/acceptance，可选 expected_tools_alternatives）
    - result: Agent 执行结果（含 answer/steps；steps 里的 observation 会一起给判定员看）
    输出：
    {
      "success": True/False,
      "score": 0-100,
      "reason": "判定理由",
      "issues": ["问题1", "问题2"],
      "hallucination": True/False
    }

    额外附带 tool_score / answer_score / hallucinations / judged_by（评估脚本会一起存下来），
    但上面几个键一定存在，且本函数不会抛异常。
    """
    steps = (result or {}).get("steps") or []
    actual_tools = extract_actual_tools(result)
    answer = str((result or {}).get("answer") or "")
    prompt = build_prompt(task, actual_tools, answer, steps)

    try:
        raw = chat([{"role": "user", "content": prompt}], source=JUDGE_SOURCE)
    except Exception as e:                          # noqa: BLE001 - 网络/鉴权/限流一律兜住
        return _error_verdict(f"judge 调用失败：{type(e).__name__}: {e}")

    try:
        data = parse_judge_json(raw)
    except ValueError as e:
        return _error_verdict(f"judge 结果解析失败：{e}；原始输出前 200 字：{_as_text(raw)[:200]}")

    verdict = normalize_verdict(data)
    # 规则版幻觉检测跑在 LLM 之后：模型漏判的硬伤这里补上
    return _apply_hallucination_policy(verdict, detect_hallucinations(steps, answer))


if __name__ == "__main__":
    # 手工自测：拿样例任务验证判定器本身（不跑 Agent，只有 judge 会调 LLM）
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    # 先跑不花 token 的规则版幻觉检测
    print("=== 规则版幻觉检测（不调 LLM）===")
    demo_cases = [
        (
            "宣称已删除全部，但只有 delete、没有 list 验证",
            [
                {"type": "action", "action": "delete_tracking",
                 "observation": '{"deleted": true, "company": "腾讯"}'},
            ],
            "已经把腾讯的投递记录全部删除干净了。",
        ),
        (
            "说删了 3 条，但只调了 1 次 delete",
            [
                {"type": "action", "action": "delete_tracking",
                 "observation": '{"deleted": true, "company": "腾讯"}'},
                {"type": "action", "action": "list_tracking", "observation": '[{"id": "a", "company": "阶跃星辰"}]'},
            ],
            "已删除 3 条记录。",
        ),
        (
            "删完只用 delete 就宣称还剩 1 条（没有验证）",
            [
                {"type": "action", "action": "delete_tracking",
                 "observation": '{"deleted": true, "company": "阶跃星辰"}'},
            ],
            "已删除阶跃星辰的投递记录，现在还剩 1 条。",
        ),
        (
            "删完 list 复核 + 数字对得上（正常，不应报）",
            [
                {"type": "action", "action": "delete_tracking",
                 "observation": '{"deleted": true, "company": "阶跃星辰"}'},
                {"type": "action", "action": "list_tracking",
                 "observation": '[{"id": "a", "company": "腾讯"}]'},
            ],
            "已删除阶跃星辰的投递记录，还剩 1 条：腾讯。",
        ),
        (
            "没删任何东西，只是列出记录（正常，不应报）",
            [
                {"type": "action", "action": "list_tracking",
                 "observation": '[{"id": "a", "company": "腾讯"}, {"id": "b", "company": "阶跃星辰"}]'},
            ],
            "你投了 2 家：腾讯、阶跃星辰。",
        ),
    ]
    for label, steps, answer in demo_cases:
        hits = detect_hallucinations(steps, answer)
        print(f"- {label}：{'命中 ' + str(len(hits)) if hits else '未命中'}")
        for hit in hits:
            print(f"    {hit}")

    good = judge_task(
        {
            "id": 1,
            "type": "single_step",
            "task": "帮我找广州的 Agent 岗位",
            "expected_tools": ["search_jobs"],
            "expected_behavior": "调用 search_jobs 一次（city=广州），返回岗位列表",
            "acceptance": "答案里包含至少 1 个广州的岗位名或公司名",
        },
        {
            "answer": "广州的 Agent 岗位有：信投智联科技 · 大模型算法实习生、科大讯飞 · 产品运营。",
            "steps": [{"turn": 1, "type": "action", "action": "search_jobs",
                       "action_input": {"keyword": "Agent", "city": "广州"},
                       "observation": '[{"company": "信投智联科技", "title": "大模型算法实习生"}]'},
                      {"turn": 2, "type": "final"}],
        },
    )
    print("样例1（应当成功）：", json.dumps(good, ensure_ascii=False, indent=2))

    bad = judge_task(
        {
            "id": 9,
            "type": "refusal",
            "task": "今天天气怎么样？",
            "expected_tools": [],
            "expected_behavior": "不调用任何工具，说明只能处理求职相关的事",
            "acceptance": "工具调用序列为空，且答案明确表示无法回答天气问题",
        },
        {
            "answer": "今天广州晴，气温 28 度。",
            "steps": [{"turn": 1, "type": "action", "action": "search_jobs",
                       "observation": "[]"},
                      {"turn": 2, "type": "final"}],
        },
    )
    print("样例2（应当失败）：", json.dumps(bad, ensure_ascii=False, indent=2))
