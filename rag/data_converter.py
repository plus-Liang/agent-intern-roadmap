# -*- coding: utf-8 -*-
"""
清洗结果（JSON）→ RAG 语料（纯文本）

rag/loader.py 读的是 rag/data/jd_sample.txt 这种纯文本，按
    【N】公司：<company>
块切分（正则 r"\\n(?=【\\d+】公司[:：])"），并从每块里抽 company / title / city。
本模块把清洗后的 cleaned_jd.json（Job dict 列表）写成同样的排版，
让 RAG 侧不用改一行 loader.py 就能吃到新抓取的岗位。

单条输出格式（分隔线为 88 个 "-"，与 jd_sample.txt 保持一致）：

    【N】公司：<company>
    岗位：<title>
    城市：<city> ｜ 薪资：<salary>
    来源链接：<url>
    发布时间：<publish_date>
    --------------------------------------------------------------------------------
    <description>
    --------------------------------------------------------------------------------

依赖：仅标准库 json / re / sys / pathlib。
"""

import json
import re
import sys
from pathlib import Path

# 本文件位于 rag/，DATA_DIR = rag/data/
RAG_DIR = Path(__file__).resolve().parent
DATA_DIR = RAG_DIR / "data"

# 默认输入输出路径
DEFAULT_JSON = DATA_DIR / "cleaned_jd.json"
DEFAULT_TXT = DATA_DIR / "scraped_jd.txt"

# 与 jd_sample.txt 对齐的分隔线
SEPARATOR = "-" * 88

# 字段缺失时的占位（与 loader.py 抽不到字段时返回的 "未知" 保持一致）
MISSING = "未知"

# 正文里万一出现 "【12】公司：" 这种行，会骗过 loader 的块切分正则；
# 在这些标记中间插一个空格，保证只有真正的记录头能匹配 ^【\d+】公司[:：]
_FAKE_HEADER = re.compile(r"(?m)^(【\d+】)(公司[:：])")


def _field(job, name) -> str:
    """安全取字段并转成 str；None / 缺失都当 ""。"""
    if not isinstance(job, dict):
        return ""
    value = job.get(name)
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _load_jobs(json_path) -> list[dict]:
    """读 JSON；支持顶层是 list、单个 dict，或 {"jobs": [...]} 包装。"""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("jobs", [data])
    return [item for item in data if isinstance(item, dict)]


def _format_job(index: int, job: dict) -> str:
    """把一条 Job 排成 jd_sample.txt 风格的文本块。"""
    company = _field(job, "company").strip() or MISSING
    title = _field(job, "title").strip() or MISSING
    city = _field(job, "city").strip() or MISSING
    salary = _field(job, "salary").strip()
    url = _field(job, "url").strip()
    publish_date = _field(job, "publish_date").strip()
    description = _FAKE_HEADER.sub(r"\1 \2", _field(job, "description").strip())

    return "\n".join((
        "【%d】公司：%s" % (index, company),
        "岗位：%s" % title,
        "城市：%s ｜ 薪资：%s" % (city, salary),
        "来源链接：%s" % url,
        "发布时间：%s" % publish_date,
        SEPARATOR,
        description,
        SEPARATOR,
    ))


def json_to_rag_format(json_path: str, output_path: str) -> int:
    """
    把 cleaned_jd.json 转成 jd_sample.txt 格式。

    参数：
        json_path:   清洗结果 JSON（Job dict 列表）
        output_path: 输出纯文本路径（如 rag/data/scraped_jd.txt）

    返回：
        转换的条数（写入的记录数）
    """
    jobs = _load_jobs(json_path)
    blocks = [_format_job(index, job) for index, job in enumerate(jobs, start=1)]
    text = "\n\n".join(blocks)
    if text:
        text += "\n"

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n"：避免 Windows 上写出 CRLF，保持和 jd_sample.txt 一致
    target.write_text(text, encoding="utf-8", newline="\n")
    return len(blocks)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    json_path = Path(argv[0]) if argv else DEFAULT_JSON
    output_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_TXT

    count = json_to_rag_format(str(json_path), str(output_path))
    print("=" * 60)
    print("cleaned_jd.json → jd_sample.txt 格式")
    print("=" * 60)
    print("输入：%s" % json_path)
    print("输出：%s" % output_path)
    print("转换条数：%d" % count)
    return count


if __name__ == "__main__":
    main()
