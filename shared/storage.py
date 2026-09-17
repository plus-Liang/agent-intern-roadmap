import json
from pathlib import Path
from datetime import datetime


def save_markdown(content: str, folder: Path, prefix: str = "doc") -> str:
    """保存 Markdown 文件，返回路径"""
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{prefix}_{timestamp}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def save_json(data, folder: Path, prefix: str = "data") -> str:
    """保存 JSON 文件，返回路径"""
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{prefix}_{timestamp}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def load_json(path: Path):
    """读取 JSON 文件"""
    return json.loads(path.read_text(encoding="utf-8"))