# -*- coding: utf-8 -*-
"""
岗位数据（SQLite / JSON）→ RAG 语料（纯文本）

rag/loader.py 读的是 rag/data/jd_sample.txt 这种纯文本，按
    【N】公司：<company>
块切分（正则 r"\\n(?=【\\d+】公司[:：])"），并从每块里抽 company / title / city。
本模块把岗位数据写成同样的排版，让 RAG 侧不用改一行 loader.py 就能吃到新抓取的岗位。

数据来源：默认路径优先读 SQLite（rag/data/jobs.db），库不可用时回退
rag/data/cleaned_jd.json。为什么先 DB：agent/tools/job_search.py 和 job_detail.py
读的也是 jobs.db，语料和工具必须是同一份数据，否则会出现「工具搜得到、RAG 检索不到」。

单条输出格式（分隔线为 88 个 "-"，与 jd_sample.txt 保持一致）：

    【N】公司：<company>
    岗位：<title>
    城市：<city> ｜ 薪资：<salary>
    来源链接：<url>
    发布时间：<publish_date>
    --------------------------------------------------------------------------------
    <description>
    --------------------------------------------------------------------------------

依赖：仅标准库 json / re / sys / pathlib（SQLite 走标准库 sqlite3，见 rag/data/db.py）。
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

# 本文件位于 <repo_root>/rag/data_converter.py，parents[1] 即仓库根目录
_REPO_ROOT = Path(__file__).resolve().parents[1]

# 本文件位于 rag/，DATA_DIR = rag/data/
RAG_DIR = Path(__file__).resolve().parent
DATA_DIR = RAG_DIR / "data"


def _db_module():
    """懒导入 rag.data.db（顺带保证仓库根在 sys.path 上，便于直接跑脚本）。"""
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from rag.data import db as _db
    return _db

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


def _load_jobs_from_db() -> Optional[list[dict]]:
    """从 SQLite 取全量岗位；None 表示「库不可用 / 库为空」，调用方回退 JSON。"""
    try:
        db = _db_module()
        if not db.ensure_db():
            return None
        rows = db.get_all_jobs()
    except Exception as exc:  # noqa: BLE001 —— 数据源坏了不该让转换整体挂掉
        print("[data_converter] SQLite 不可用，回退 JSON：%s" % exc, file=sys.stderr)
        return None
    return rows or None


def _load_jobs(json_path) -> list[dict]:
    """读岗位数据：默认路径优先读 SQLite，读不到再回退 JSON。

    显式传入非默认路径时（CLI 指定 / 单测临时文件）仍以该 JSON 为准，
    保证旧调用方式语义不变。支持顶层是 list、单个 dict，或 {"jobs": [...]} 包装。
    """
    path = Path(json_path)
    if path.resolve() == DEFAULT_JSON:
        jobs = _load_jobs_from_db()
        if jobs is not None:
            return jobs

    data = json.loads(path.read_text(encoding="utf-8"))
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
    把岗位数据转成 jd_sample.txt 格式（默认读 SQLite，库不可用时读 JSON）。

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
    is_default = Path(json_path).resolve() == DEFAULT_JSON
    print("=" * 60)
    print("岗位数据 → jd_sample.txt 格式")
    print("=" * 60)
    print("输入：%s%s" % (json_path,
                        "（默认路径：优先 SQLite %s，不可用时回退该 JSON）" % _db_module().DB_PATH
                        if is_default else ""))
    print("输出：%s" % output_path)
    print("转换条数：%d" % count)
    return count


if __name__ == "__main__":
    main()
