import re
from pathlib import Path


def load_jd_file(path: str) -> list[dict]:
    """读取 JD 文件，返回结构化列表（每条 JD 包含完整正文）"""
    text = Path(path).read_text(encoding="utf-8")

    # 按「【数字】公司：」为分界切分
    blocks = re.split(r"\n(?=【\d+】公司[:：])", text)

    jds = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # 只保留真正的 JD 块
        if not re.search(r"【\d+】公司[:：]", block):
            continue

        company = _extract(block, r"公司[：:]\s*(.+)")
        title = _extract(block, r"岗位[：:]\s*(.+)")
        city = _extract(block, r"城市[：:]\s*(.+?)\s*[｜|]")

        jds.append({
            "company": company,
            "title": title,
            "city": city,
            "content": block,
        })

    return jds


def _extract(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return "未知"


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    jds = load_jd_file(str(BASE_DIR / "data" / "jd_sample.txt"))
    print(f"共读取 {len(jds)} 条 JD\n")
    for jd in jds:
        print(f"- {jd['company']} | {jd['title']} | {jd['city']} | 正文长度 {len(jd['content'])}")