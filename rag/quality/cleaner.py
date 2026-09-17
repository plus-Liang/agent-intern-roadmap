# -*- coding: utf-8 -*-
"""
岗位数据清洗（只清洗，不改写源数据）

输入：抓取器产出的岗位 dict 列表（如 agent/scrapers/shixiseng_result.json），
字段与 agent/tools/job_search.py 的 Job 对齐：
    platform / job_id / title / company / city / salary / url / publish_date / description

依次应用四道过滤（命中即计数并丢弃，一条只计一次，按下列顺序判定）：
    a) relevance 相关性：标题或正文含 Agent / LLM / 大模型 / RAG / 智能体
       —— 单独出现 "AI" 不作为命中（太宽泛），例外是 "AI Agent" / "AI 大模型"
    b) city      城市：city == 目标城市 或 city == "全国"
    c) age       时间：publish_date 距今天 <= max_age_days（publish_date 为空则不参与时间过滤）
    d) length    正文长度：len(description) >= min_desc_len

保留的岗位按 publish_date 降序排列（最新在前），publish_date 为空的排在最后。

依赖：仅标准库 json / sys / datetime / pathlib。
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path

# 本文件位于 rag/quality/，RAG_DIR = rag/，REPO_DIR = 仓库根目录
QUALITY_DIR = Path(__file__).resolve().parent
RAG_DIR = QUALITY_DIR.parent
REPO_DIR = RAG_DIR.parent

# 默认输入输出路径
DEFAULT_INPUT = REPO_DIR / "agent" / "scrapers" / "shixiseng_result.json"
DEFAULT_OUTPUT = RAG_DIR / "data" / "cleaned_jd.json"

# 相关性关键词。
# - 中文关键词按原文大小写敏感匹配（"AI Agent" / "AI 大模型" 两种写法都列出来）
# - 英文关键词用 lower() 做大小写不敏感匹配（"Agent" / "agent" / "AGENT" 都算）
# - 不收录单独的 "AI"：广州搜索里 "AI原画"、"AI短视频创意"、"AI产品经理" 这类
#   泛 AI 岗位命中率太高，会让 clean_jobs 失去筛选意义
RELEVANCE_KEYWORDS_ZH = ("大模型", "智能体", "AI Agent", "AI 大模型")
RELEVANCE_KEYWORDS_EN = ("agent", "llm", "rag")

# 输出字段顺序 = Job 的字段顺序（不含 tags，抓取结果里也没有 tags）
# 抓取结果里的 description_chars 是抓取器自己的统计字段，不进入清洗结果
JOB_FIELDS = (
    "platform", "job_id", "title", "company", "city",
    "salary", "url", "description", "publish_date",
)

# 城市过滤的"不限城市"占位值
NATIONWIDE = "全国"

# 过滤原因键（removed 字典里始终齐全，未命中为 0）
REMOVAL_REASONS = ("relevance", "city", "age", "length")


def _job_field(job, name):
    """安全取字段并转成 str；None/缺失都当 ""。"""
    if not isinstance(job, dict):
        return ""
    value = job.get(name)
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _is_relevant(text):
    """判断一段文本（标题或正文）是否命中相关性关键词。"""
    if not text:
        return False
    for keyword in RELEVANCE_KEYWORDS_ZH:
        if keyword in text:
            return True
    lowered = text.lower()
    for keyword in RELEVANCE_KEYWORDS_EN:
        if keyword in lowered:
            return True
    return False


def _parse_date(value):
    """把 publish_date 解析成 date；解析不了（含空串、脏数据）返回 None。

    只认 "YYYY-MM-DD"，也容忍 "YYYY/MM/DD" 这类写法。
    """
    if not value:
        return None
    text = str(value).strip().replace("/", "-")
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_fresh(publish_date, today, max_age_days):
    """时间过滤：距今 <= max_age_days 天算新鲜。

    publish_date 为空或解析失败 -> True（不参与时间过滤）。
    未来日期同样按天数差判断（|today - publish_date|），避免把未来的日期
    当成"负年龄"永远放行。
    """
    parsed = _parse_date(publish_date)
    if parsed is None:
        return True
    return abs((today - parsed).days) <= max_age_days


def _normalize_job(job):
    """只保留 Job 定义的字段，保证输出结构稳定。"""
    normalized = {}
    for field in JOB_FIELDS:
        normalized[field] = _job_field(job, field).strip()
    return normalized


def clean_jobs(jobs: list[dict], city: str = "广州",
               max_age_days: int = 30,
               min_desc_len: int = 200,
               today: date | None = None) -> dict:
    """
    清洗岗位列表。

    参数：
        jobs:         原始岗位 dict 列表
        city:         目标城市（city == 该值，或 city == "全国" 时保留）
        max_age_days: publish_date 距今天最大天数
        min_desc_len: 正文最小长度
        today:        基准日期，默认取系统当天；测试时可显式传入

    返回：
        {
          "total_input": 42,
          "total_output": N,
          "removed": {"relevance": X, "city": Y, "age": Z, "length": W},
          "jobs": [清洗后的 Job 列表]   # 按 publish_date 降序，日期为空的在最后
        }
    """
    if today is None:
        today = date.today()

    removed = {reason: 0 for reason in REMOVAL_REASONS}
    kept = []

    for job in jobs or []:
        title = _job_field(job, "title")
        description = _job_field(job, "description")
        job_city = _job_field(job, "city").strip()
        publish_date = _job_field(job, "publish_date").strip()

        # a) 相关性：标题或正文命中关键词
        if not _is_relevant(title) and not _is_relevant(description):
            removed["relevance"] += 1
            continue

        # b) 城市：目标城市或"全国"
        if job_city != city and job_city != NATIONWIDE:
            removed["city"] += 1
            continue

        # c) 时间：空日期不参与过滤
        if not _is_fresh(publish_date, today, max_age_days):
            removed["age"] += 1
            continue

        # d) 正文长度
        if len(description) < min_desc_len:
            removed["length"] += 1
            continue

        kept.append(_normalize_job(job))

    # 按 publish_date 降序；空日期（解析失败）统一排在最后
    kept.sort(
        key=lambda item: (
            _parse_date(item.get("publish_date")) is not None,
            _parse_date(item.get("publish_date")) or date.min,
        ),
        reverse=True,
    )

    return {
        "total_input": len(jobs or []),
        "total_output": len(kept),
        "removed": removed,
        "jobs": kept,
    }


def load_jobs(path) -> list[dict]:
    """读 JSON 岗位文件；支持顶层是 list 或单个 dict。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # 容忍 {"jobs": [...]} 这类包装
        data = data.get("jobs", [data])
    return [item for item in data if isinstance(item, dict)]


def _print_summary(result):
    """打印清洗汇总和前 5 条。"""
    removed = result["removed"]
    print("=" * 60)
    print("岗位数据清洗汇总")
    print("=" * 60)
    print("输入 %d 条 → 输出 %d 条" % (result["total_input"], result["total_output"]))
    print("-" * 60)
    print("按过滤原因移除：")
    labels = {
        "relevance": "相关性（标题/正文无 Agent/LLM/大模型/RAG/智能体）",
        "city": "城市（非目标城市且非全国）",
        "age": "时间（publish_date 超出天数上限）",
        "length": "正文长度（description 太短）",
    }
    for reason in REMOVAL_REASONS:
        print("  %-9s %2d 条  —— %s" % (reason, removed.get(reason, 0), labels[reason]))
    print("  合计移除 %d 条" % sum(removed.values()))
    print("-" * 60)
    print("保留前 5 条（按 publish_date 降序）：")
    for index, job in enumerate(result["jobs"][:5], start=1):
        print("  %d. %s | %s | %s | 正文 %d 字" % (
            index,
            job.get("title") or "(无标题)",
            job.get("company") or "(无公司)",
            job.get("publish_date") or "(无日期)",
            len(job.get("description") or ""),
        ))
    if not result["jobs"]:
        print("  （无）")


def save_cleaned(jobs, path) -> Path:
    """把清洗后的岗位列表写成 JSON（顶层是 list，与输入文件保持一致）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    input_path = Path(argv[0]) if argv else DEFAULT_INPUT
    output_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUTPUT

    jobs = load_jobs(input_path)
    result = clean_jobs(jobs, city="广州", max_age_days=30, min_desc_len=200)
    _print_summary(result)

    saved = save_cleaned(result["jobs"], output_path)
    print("-" * 60)
    print("源文件：%s" % input_path)
    print("已保存：%s" % saved)

    return result


if __name__ == "__main__":
    main()
