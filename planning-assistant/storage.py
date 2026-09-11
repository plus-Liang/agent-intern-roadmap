from pathlib import Path
from datetime import datetime

def save_plan(goal: str, content: str, folder: str = "plans") -> str:
    """把计划保存为 Markdown 文件，返回文件路径"""
    Path(folder).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(folder) / f"plan_{goal}_{timestamp}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)