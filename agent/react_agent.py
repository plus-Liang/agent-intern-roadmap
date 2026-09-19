import json5

"""
ReAct Agent。
LLM 在循环里自主决策：思考 → 调工具 → 观察 → 再思考。
"""
import json
import os
import re
from shared.llm_client import chat
from agent.tools_registry import list_tools_description, call_tool


MAX_TURNS = 6

# 消息历史软上限：超过就把中间部分压成摘要，防止长对话把 token 撑爆。
# 可用环境变量 MAX_HISTORY 覆盖。
MAX_HISTORY = 10

SUMMARY_PROMPT_TEMPLATE = (
    "以下是之前的对话历史，请用 2-3 句话概括关键信息"
    "（用户问了什么、调用了哪些工具、得到什么结果）：\n\n{history}\n\n摘要："
)


SYSTEM_PROMPT_TEMPLATE = """你是一个求职助手 Agent。你可以调用工具帮用户完成任务。

【可用工具】
{tools}

【工作方式】
每一轮你必须输出一个 JSON：

调工具：
{{
  "thought": "你的思考",
  "action": "工具名",
  "action_input": {{参数对象}}
}}

完成回答：
{{
  "thought": "你的思考",
  "final_answer": "给用户的最终回答"
}}

【规则】
1. 每次只输出一个 JSON，不要有其他内容。
2. 不要编造工具返回的数据。
3. 最多 {max_turns} 轮。
4. 字符串里不能有真实换行，用 \\n 转义。
5. final_answer 必须是单行字符串。

【最小必要原则】
只调用回答当前问题所必需的工具。用户没要求查看详情就不要调 get_job_detail，
用户没要求匹配简历就不要调 match_resume。
判断标准：如果问题的答案用当前已有信息就能回答，立即给 final_answer。

【上下文】
- 用户的简历已经在系统中，当用户提到"我的简历"或需要匹配时，
  请使用 match_resume 工具，resume_json 参数填 "current"（系统会自动替换）。
"""


def _parse_json(text: str) -> dict:
    """解析 LLM 输出的 JSON，兼容多种格式"""
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

    # 先试标准 json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 兜底：用 json5（允许尾逗号、单引号、真实换行）
    try:
        return json5.loads(text)
    except Exception:
        raise


def _get_max_history() -> int:
    """读取消息上限：环境变量 MAX_HISTORY 优先，非法值回退到常量 MAX_HISTORY。

    上限小于 3 时无法同时容纳 system + 摘要 + 尾部消息，一律按默认值处理。
    """
    raw = os.getenv("MAX_HISTORY", "")
    if raw is not None and str(raw).strip():
        try:
            value = int(str(raw).strip())
            if value >= 3:
                return value
        except (TypeError, ValueError):
            pass
        print(f"[上下文] MAX_HISTORY={raw!r} 非法（需 >=3 的整数），改用默认 {MAX_HISTORY}")
    return MAX_HISTORY


def _format_history(messages: list) -> str:
    """把待摘要的消息拼成纯文本，单条过长（如工具返回）先截断"""
    role_names = {"system": "系统", "user": "用户", "assistant": "助手"}
    lines = []
    for m in messages:
        role = role_names.get(m.get("role"), str(m.get("role")))
        content = str(m.get("content", ""))
        if len(content) > 800:
            content = content[:800] + "…（已截断）"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def _summarize_messages(messages: list, verbose: bool = True) -> str:
    """调 LLM 概括一段历史；失败时退化为截断拼接，保证一定拿得到可用摘要"""
    history = _format_history(messages)
    try:
        summary = chat([{
            "role": "user",
            "content": SUMMARY_PROMPT_TEMPLATE.format(history=history),
        }])
        summary = " ".join(str(summary or "").split())     # 压成单行，别让摘要自己变长
        if summary:
            return summary
        if verbose:
            print("[上下文] 摘要模型返回空内容，退化为截断拼接")
    except Exception as e:
        if verbose:
            print(f"[上下文] 摘要失败，退化为截断拼接：{e}")

    flat = history.replace("\n", " ")
    return flat[:500] + ("…" if len(flat) > 500 else "")


def compact_messages(messages: list, max_history: int = None, verbose: bool = True) -> list:
    """把 messages 压到 max_history 条以内（默认取 MAX_HISTORY / 环境变量）。

    规则：
    - 第 1 条 system prompt 永远保留；
    - 保留最近 max_history - 2 条原始消息；
    - 中间部分（system 之后、最近 N 条之前）交给 LLM 摘要，
      以「之前对话摘要：…」插在 system prompt 之后，这 1 条摘要本身也计入上限；
    - 没超限时原样返回（返回新列表，不改动入参）。
    """
    limit = max_history or _get_max_history()
    if len(messages) <= limit:
        return list(messages)

    system_msg = messages[0]
    keep_tail = max(0, limit - 2)                  # 留 1 条名额给摘要
    tail = messages[len(messages) - keep_tail:] if keep_tail else []
    middle = messages[1:len(messages) - keep_tail] if keep_tail else messages[1:]

    if not middle:                                 # 兜底：没有可压缩内容时不调 LLM
        return [system_msg] + tail

    summary = _summarize_messages(middle, verbose=verbose)
    compacted = [
        system_msg,
        {"role": "user", "content": f"之前对话摘要：{summary}"},
    ] + tail

    if verbose:
        print(
            f"[上下文] 消息 {len(messages)} 条 → 中间 {len(middle)} 条压成摘要，"
            f"现在 {len(compacted)} 条（上限 {limit}）"
        )
    return compacted


def run(question: str, resume_data: dict = None, verbose: bool = True) -> dict:
    """运行 ReAct 循环
    resume_data: 当前用户的简历（dict），会注入到 system prompt
    """
    tools_desc = list_tools_description()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        tools=tools_desc, max_turns=MAX_TURNS
    )

    if resume_data:
        system_prompt += f"\n\n【用户当前简历】\n{json.dumps(resume_data, ensure_ascii=False)}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    steps = []
    for turn in range(1, MAX_TURNS + 1):
        if verbose:
            print(f"\n--- 第 {turn} 轮 ---")

        # 每次调用 LLM 前压缩历史：超限就摘要中间部分，长对话不会把 token 撑爆
        messages = compact_messages(messages, verbose=verbose)

        raw = chat(messages)

        try:
            decision = _parse_json(raw)
        except Exception as e:
            if verbose:
                print(f"[解析失败] {e}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "你输出的不是合法 JSON，请严格按格式重新输出。",
            })
            continue

        thought = decision.get("thought", "")
        if verbose:
            print(f"Thought: {thought}")

        if "final_answer" in decision:
            return {
                "answer": decision["final_answer"],
                "steps": steps + [{"turn": turn, "type": "final", "thought": thought}],
            }

        action = decision.get("action")
        action_input = decision.get("action_input", {})

        # 把 "current" 替换成真实简历
        if action == "match_resume" and resume_data:
            if action_input.get("resume_json") in (None, "current", ""):
                action_input["resume_json"] = json.dumps(resume_data, ensure_ascii=False)

        if verbose:
            print(f"Action: {action}")
            print(f"Input: {str(action_input)[:200]}")

        try:
            result = call_tool(action, action_input)
            result_str = json.dumps(result, ensure_ascii=False, default=str)
            if verbose:
                print(f"Observation: {result_str[:200]}...")
        except Exception as e:
            result_str = f"工具调用失败：{e}"
            if verbose:
                print(f"Observation: {result_str}")

        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"工具返回结果：\n{result_str}\n\n请继续。",
        })

        steps.append({
            "turn": turn,
            "type": "action",
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": result_str[:500],
        })

    return {
        "answer": "抱歉，我没能在限定轮次内完成。请简化问题。",
        "steps": steps,
    }

if __name__ == "__main__":
    questions = [
        "帮我找北京的 Agent 开发实习，看看有哪些岗位",
    ]
    for q in questions:
        print(f"\n{'='*70}")
        print(f"❓ {q}")
        print(f"{'='*70}")
        result = run(q)
        print(f"\n【最终答案】\n{result['answer']}")