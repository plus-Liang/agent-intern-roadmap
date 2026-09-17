from shared.llm_client import chat_stream
from planner.prompts import build_system_prompt, build_plan_prompt


def detect_mode(user_input: str, current_mode: str) -> str:
    """根据用户输入判断输出模式：diff 只输出改动，full 输出完整计划"""
    text = user_input.lower()

    if any(k in text for k in [
        "完整", "全部", "都输出", "整个计划", "全文",
        "重新给我一份", "完整版", "从头到尾"
    ]):
        return "full"

    if any(k in text for k in [
        "只输出改动", "只给改动", "不用重复", "只发改动",
        "只列改动", "仅改动"
    ]):
        return "diff"

    return current_mode


def create_initial_messages(goal: str, hours: str, background: str) -> list:
    """创建初始消息列表"""
    return [
        {"role": "system", "content": build_system_prompt("diff")},
        {"role": "user", "content": build_plan_prompt(goal, hours, background)},
    ]


def stream_plan(messages: list):
    """流式生成计划"""
    return chat_stream(messages)