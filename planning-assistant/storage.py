from pathlib import Path
from datetime import datetime

def save_plan(content: str, goal: str, folder: str = "plans") -> str:
    Path(folder).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_goal = goal.replace("/", "-").replace("\\", "-")
    path = Path(folder) / f"plan_{safe_goal}_{timestamp}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)

def save_conversation(messages: list, goal: str, folder: str = "plans") -> str:
    """把完整对话保存为 Markdown 文件"""
    Path(folder).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_goal = goal.replace("/", "-").replace("\\", "-")
    path = Path(folder) / f"conversation_{safe_goal}_{timestamp}.md"

    lines = [f"# 规划对话记录\n", f"目标：{goal}\n"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            continue
        elif role == "user":
            lines.append(f"\n## 用户\n\n{content}\n")
        elif role == "assistant":
            lines.append(f"\n## 助手\n\n{content}\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)