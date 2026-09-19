import json5

"""
ReAct Agent。
LLM 在循环里自主决策：思考 → 调工具 → 观察 → 再思考。
"""
import json
import re
from shared.llm_client import chat
from agent.tools_registry import list_tools_description, call_tool


MAX_TURNS = 6


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