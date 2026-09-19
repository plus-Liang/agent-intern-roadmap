"""
LLM-as-Judge：用大模型自动判定 Agent 是否完成了评估任务。

用法：
    from agent.evaluation.judge import judge_task

    verdict = judge_task(task, {"answer": "...", "steps": [...]})
    # -> {"success": bool, "score": 0-100, "reason": str, "issues": [str]}

设计要点：
- 输入四件套：任务描述（含验收标准 acceptance）、预期工具序列、
  实际工具序列（从 result["steps"] 提取）、最终答案。
- 两个维度打分：**工具选择 40 分** + **答案质量 60 分**，满分 100。
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

JUDGE_PROMPT = """你是一个严格的 AI Agent 评测员。请判定下面的 Agent 是否完成了任务。

【任务定义】
- 任务 ID：{task_id}
- 任务类型：{task_type}
- 用户请求：{task}
- 预期行为：{expected_behavior}
- 验收标准（acceptance）：{acceptance}
- 预期工具序列：{expected_tools}
- 实际工具序列：{actual_tools}

【Agent 的最终答案】
{answer}

【判定维度与分值】
1. 工具选择（0-40 分）
   - 完全一致（工具名、顺序、数量都对）：40 分
   - 工具集合一致但顺序不同：30 分
   - 少调用或多调用了工具：按缺失/多余比例扣分（每处扣 10 分左右）
   - 预期需要工具却一个都没调（或反之，本该不调却调了）：0-10 分
   - 注意：预期工具序列为空时，实际序列也为空才是正确行为。
2. 答案质量（0-60 分）
   - 严格对照上面的 acceptance 逐条检查，全部满足给 50-60 分；
   - 部分满足按缺项扣分；编造数据（如编造岗位、编造已经删除成功）直接给 0-20 分；
   - 与验收标准无关的废话不算分数，但也不额外扣分。

【输出要求】
只输出一个 JSON 对象，不要任何解释文字、不要 markdown 围栏，格式如下：
{{
  "tool_score": 0-40 的整数,
  "answer_score": 0-60 的整数,
  "score": 0-100 的整数（等于上面两项之和）,
  "success": true 或 false（score >= 60 且没有致命问题才算成功）,
  "reason": "一句话说明判定理由，必须引用答案里的具体证据",
  "issues": ["具体问题1", "具体问题2"]
}}
如果没有问题，issues 输出空数组 []。"""


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


def _as_text(value) -> str:
    """把任意值压成一行短文本，避免 prompt 里出现真实换行/超长内容"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " → ".join(str(v) for v in value) if value else "（空）"
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(str(value).split())
    return text[:1500] + ("…（已截断）" if len(text) > 1500 else "")


def build_prompt(task: dict, actual_tools: list, answer: str) -> str:
    """拼判定 prompt（单独抽出来，方便单测/人工检查）"""
    expected = list(task.get("expected_tools", []) or [])
    return JUDGE_PROMPT.format(
        task_id=task.get("id", "?"),
        task_type=task.get("type", "?"),
        task=task.get("task", ""),
        expected_behavior=task.get("expected_behavior", ""),
        acceptance=task.get("acceptance", ""),
        expected_tools=json.dumps(expected, ensure_ascii=False) if expected else "[]（不应调用任何工具）",
        actual_tools=json.dumps(list(actual_tools), ensure_ascii=False) if actual_tools else "[]（没有调用工具）",
        answer=answer or "（Agent 没有输出答案）",
    )


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

    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    issues = [str(i).strip() for i in issues if str(i).strip()]

    reason = " ".join(str(data.get("reason", "")).split()) or "模型未说明理由"

    return {
        "success": success,
        "score": score,
        "reason": reason,
        "issues": issues,
        "tool_score": tool_score,
        "answer_score": answer_score,
        "judged_by": "llm_judge",
    }


def _error_verdict(message: str) -> dict:
    """判定失败的兜底结果：按失败处理，但把原因留在 reason/error 里"""
    return {
        "success": False,
        "score": 0,
        "reason": message,
        "issues": [message],
        "tool_score": 0,
        "answer_score": 0,
        "judged_by": "llm_judge_error",
        "error": message,
    }


def judge_task(task: dict, result: dict) -> dict:
    """用 LLM 判定 Agent 是否完成任务。

    输入：
    - task: 任务定义（含 id/type/task/expected_tools/acceptance）
    - result: Agent 执行结果（含 answer/steps）
    输出：
    {
      "success": True/False,
      "score": 0-100,
      "reason": "判定理由",
      "issues": ["问题1", "问题2"]
    }

    额外附带 tool_score / answer_score / judged_by（评估脚本会一起存下来），
    但上面 4 个键一定存在，且本函数不会抛异常。
    """
    actual_tools = extract_actual_tools(result)
    answer = str((result or {}).get("answer") or "")
    prompt = build_prompt(task, actual_tools, answer)

    try:
        raw = chat([{"role": "user", "content": prompt}], source=JUDGE_SOURCE)
    except Exception as e:                          # noqa: BLE001 - 网络/鉴权/限流一律兜住
        return _error_verdict(f"judge 调用失败：{type(e).__name__}: {e}")

    try:
        data = parse_judge_json(raw)
    except ValueError as e:
        return _error_verdict(f"judge 结果解析失败：{e}；原始输出前 200 字：{_as_text(raw)[:200]}")

    return normalize_verdict(data)


if __name__ == "__main__":
    # 手工自测：拿两个样例任务验证判定器本身（不跑 Agent，只有 judge 会调 LLM）
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
            "steps": [{"turn": 1, "type": "action", "action": "search_jobs"},
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
            "steps": [{"turn": 1, "type": "action", "action": "search_jobs"},
                      {"turn": 2, "type": "final"}],
        },
    )
    print("样例2（应当失败）：", json.dumps(bad, ensure_ascii=False, indent=2))
