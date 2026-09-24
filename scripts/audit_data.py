# -*- coding: utf-8 -*-
"""岗位数据质量审计脚本（只读，零新依赖）。

用途
----
对 rag/data/jobs.db 的 jobs 表做一次全量体检，回答五个问题：
  1. 总量与分布（平台 / 城市 / 月份）；
  2. 重复岗位（job_id 重复、(company, title) 跨平台重复）；
  3. 数据完整性（description / salary / url 等字段缺失）；
  4. 时效性（publish_date 超过 90 天、为空、格式非法）；
  5. 字段异常（company 是占位垃圾值、city 不在标准城市列表）。

用法
----
    python scripts/audit_data.py               # 审计默认库 rag/data/jobs.db
    python scripts/audit_data.py --db PATH     # 指定其它库文件
    python scripts/audit_data.py --json        # 只把 JSON 打到 stdout，不打印报告
    python scripts/audit_data.py --out PATH    # 指定报告输出路径

报告固定落盘到 logs/audit_report.json（--json 时不打印控制台报告，但同样落盘）。

本脚本严格只读：只做 SELECT，不建表、不写入、不删除任何数据。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "rag" / "data" / "jobs.db"
DEFAULT_REPORT = REPO_ROOT / "logs" / "audit_report.json"

TABLE = "jobs"
STALE_DAYS = 90            # publish_date 早于今天 - 90 天算过期
MIN_DESC_CHARS = 200       # description 短于 200 字算残缺
EXAMPLE_LIMIT = 10         # 每个问题最多在报告里留几条样例，避免报告膨胀

# 标准城市列表：一线 + 新一线 + 常见省会/计划单列市。
# 库里出现列表外的城市（或城市为空/带奇怪后缀）视为字段异常。
STANDARD_CITIES = {
    "北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉", "西安", "南京",
    "苏州", "天津", "长沙", "郑州", "东莞", "青岛", "合肥", "佛山", "宁波", "无锡",
    "厦门", "济南", "大连", "福州", "温州", "哈尔滨", "沈阳", "昆明", "长春", "南宁",
    "常州", "泉州", "南昌", "贵阳", "太原", "烟台", "嘉兴", "南通", "金华", "珠海",
    "惠州", "徐州", "海口", "乌鲁木齐", "兰州", "中山", "保定", "临沂", "潍坊", "绍兴",
    "远程", "全国",
}

# company 明显的占位值：短词按整串相等匹配，避免误伤（如「无锡XX科技」不含「无」误判）。
BAD_COMPANY_EXACT = {
    "", "-", "--", "---", "----", ".", "。", "无", "暂无", "未知", "未知公司", "其他",
    "待定", "未填写", "未公开", "保密", "公司", "某公司", "xx", "xxx", "xxxx",
    "none", "null", "nan", "n/a", "na", "test", "unknown", "tbd",
}
BAD_COMPANY_SUBSTR = ("未知", "匿名", "保密", "待定", "暂无", "未公开", "未填写", "n/a", "null")

_PUBLISH_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日", "%Y-%m", "%Y/%m", "%Y年%m月", "%Y%m%d", "%Y",
)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------

def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _clean(value: Any) -> str:
    """去首尾空白 + 把连续空白压成一个空格。"""
    return re.sub(r"\s+", " ", _text(value)).strip()


def parse_publish_date(value: Any) -> Optional[date]:
    """尽量把 publish_date 解析成 date；解析不了返回 None。"""
    raw = _clean(value)
    if not raw:
        return None
    raw = raw.split("T")[0].split(" ")[0]
    for fmt in _PUBLISH_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return parsed.date()
    return None


def normalize_city(value: Any) -> str:
    """城市归一化：去空白、去尾部「市」（与 rag/data/db.py 口径一致）。"""
    city = _clean(value)
    if city.endswith("市"):
        city = city[:-1]
    return city


def company_is_abnormal(value: Any) -> bool:
    company = _clean(value)
    lowered = company.lower()
    if not company or lowered in BAD_COMPANY_EXACT:
        return True
    if len(company) < 2:
        return True
    return any(marker in lowered for marker in BAD_COMPANY_SUBSTR)


def _top(counter: Counter, limit: int = EXAMPLE_LIMIT) -> dict:
    """Counter -> {key: n}（按数量降序，最多 limit 项）。"""
    return {str(key): int(n) for key, n in counter.most_common(limit)}


def _fmt_dist(counter: Counter, limit: int = 8) -> str:
    items = counter.most_common(limit)
    text = ", ".join(f"{key} {n}" for key, n in items)
    if len(counter) > limit:
        text += f", ...(共 {len(counter)} 项)"
    return text or "无"


# --------------------------------------------------------------------------
# 审计主体
# --------------------------------------------------------------------------

def audit(db_path: Path) -> dict:
    """对 db_path 跑一次全量审计，返回报告 dict（只读）。"""
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"数据库不存在：{db_path}")

    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        if TABLE not in tables:
            raise sqlite3.OperationalError(f"库里没有 {TABLE} 表：{db_path}")
        rows = [dict(row) for row in conn.execute(
            f"SELECT job_id, platform, title, company, city, salary, url,"
            f" description, publish_date FROM {TABLE}"
        )]
    finally:
        conn.close()

    today = date.today()
    cutoff = today - timedelta(days=STALE_DAYS)

    by_platform: Counter = Counter()
    by_city: Counter = Counter()
    by_month: Counter = Counter()
    platform_city: Counter = Counter()
    job_id_seen: Counter = Counter()
    company_title: Counter = Counter()
    company_title_platforms: dict[tuple, set] = {}
    company_bad: Counter = Counter()
    city_bad: Counter = Counter()

    desc_empty = desc_short = salary_empty = url_empty = 0
    title_empty = company_empty = city_empty = 0
    date_empty = date_invalid = 0
    stale = 0
    dates: list[date] = []

    for row in rows:
        platform = _clean(row.get("platform")) or "未知"
        city = normalize_city(row.get("city"))
        company = _clean(row.get("company"))
        title = _clean(row.get("title"))
        description = _clean(row.get("description"))

        by_platform[platform] += 1
        by_city[city or "未知"] += 1
        platform_city[f"{platform} × {city or '未知'}"] += 1

        job_id_seen[_clean(row.get("job_id"))] += 1
        if company or title:
            key = (company, title)
            company_title[key] += 1
            company_title_platforms.setdefault(key, set()).add(platform)

        parsed = parse_publish_date(row.get("publish_date"))
        if parsed is None:
            if _clean(row.get("publish_date")):
                date_invalid += 1
            else:
                date_empty += 1
        else:
            dates.append(parsed)
            by_month[parsed.strftime("%Y-%m")] += 1
            if parsed < cutoff:
                stale += 1

        if not description:
            desc_empty += 1
        elif len(description) < MIN_DESC_CHARS:
            desc_short += 1
        if not _clean(row.get("salary")):
            salary_empty += 1
        if not _clean(row.get("url")):
            url_empty += 1
        if not title:
            title_empty += 1
        if not company:
            company_empty += 1
        if not city:
            city_empty += 1

        if company_is_abnormal(company):
            company_bad[company or "(空)"] += 1
        if city not in STANDARD_CITIES:
            city_bad[city or "(空)"] += 1

    dup_job_id = {k: v for k, v in job_id_seen.items() if v > 1}
    dup_pairs = {k: v for k, v in company_title.items() if v > 1}

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": str(Path(db_path).resolve()),
        "params": {"stale_days": STALE_DAYS, "min_desc_chars": MIN_DESC_CHARS},
        "total": len(rows),
        "distribution": {
            "by_platform": dict(by_platform.most_common()),
            "by_city": dict(by_city.most_common()),
            "by_month": dict(sorted(by_month.items())),
            "platform_city": dict(platform_city.most_common()),
        },
        "duplicates": {
            "duplicate_job_id_groups": len(dup_job_id),
            "duplicate_job_id_extra_rows": int(sum(v - 1 for v in dup_job_id.values())),
            "duplicate_job_id_examples": _top(Counter(dup_job_id)),
            "duplicate_company_title_groups": len(dup_pairs),
            "duplicate_company_title_extra_rows": int(sum(v - 1 for v in dup_pairs.values())),
            "duplicate_company_title_examples": [
                {
                    "company": key[0],
                    "title": key[1],
                    "count": int(count),
                    "platforms": sorted(company_title_platforms.get(key, set())),
                }
                for key, count in sorted(dup_pairs.items(), key=lambda kv: -kv[1])[:EXAMPLE_LIMIT]
            ],
        },
        "completeness": {
            "description_empty": desc_empty,
            "description_short_200": desc_short,
            "description_missing_or_short": desc_empty + desc_short,
            "salary_empty": salary_empty,
            "url_empty": url_empty,
            "title_empty": title_empty,
            "company_empty": company_empty,
            "city_empty": city_empty,
            "publish_date_empty": date_empty,
        },
        "timeliness": {
            "stale_over_90_days": stale,
            "publish_date_empty": date_empty,
            "publish_date_invalid": date_invalid,
            "earliest": min(dates).isoformat() if dates else "",
            "latest": max(dates).isoformat() if dates else "",
        },
        "field_anomalies": {
            "company_abnormal": int(sum(company_bad.values())),
            "company_abnormal_values": _top(company_bad),
            "city_non_standard": int(sum(city_bad.values())),
            "city_non_standard_values": _top(city_bad),
            "standard_city_count": len(STANDARD_CITIES),
        },
    }
    return report


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------

def _print_report(report: dict) -> None:
    dist = report["distribution"]
    comp = report["completeness"]
    time_ = report["timeliness"]
    dup = report["duplicates"]
    anom = report["field_anomalies"]

    print("===== 数据审计报告 =====")
    print(f"数据库：{report['db_path']}")
    print(f"总数：{report['total']} 条")
    print(f"按平台：{_fmt_dist(Counter(dist['by_platform']))}")
    print(f"按城市：{_fmt_dist(Counter(dist['by_city']))}")
    print(f"按月份：{_fmt_dist(Counter(dist['by_month']), 12)}")
    print(f"平台×城市：{_fmt_dist(Counter(dist['platform_city']), 10)}")
    print(
        "重复：job_id "
        f"{dup['duplicate_job_id_groups']} 组 / {dup['duplicate_job_id_extra_rows']} 条冗余, "
        f"(company,title) {dup['duplicate_company_title_groups']} 组 / "
        f"{dup['duplicate_company_title_extra_rows']} 条冗余"
    )
    print(
        f"完整性：description 空 {comp['description_empty']}, "
        f"<{report['params']['min_desc_chars']}字 {comp['description_short_200']}, "
        f"salary 空 {comp['salary_empty']}, url 空 {comp['url_empty']}, "
        f"title 空 {comp['title_empty']}"
    )
    print(
        f"时效性：>{report['params']['stale_days']}天 {time_['stale_over_90_days']}, "
        f"日期空 {time_['publish_date_empty']}, 日期非法 {time_['publish_date_invalid']}, "
        f"日期范围 {time_['earliest'] or '-'} ~ {time_['latest'] or '-'}"
    )
    print(
        f"字段异常：company 异常 {anom['company_abnormal']}, "
        f"city 非标准 {anom['city_non_standard']}"
    )
    if anom["company_abnormal_values"]:
        print(f"  company 异常样例：{anom['company_abnormal_values']}")
    if anom["city_non_standard_values"]:
        print(f"  city 非标准样例：{anom['city_non_standard_values']}")
    if dup["duplicate_company_title_examples"]:
        print(f"  跨平台重复样例：{dup['duplicate_company_title_examples']}")
    print("========================")


def _write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="岗位数据质量审计（只读 SQLite，输出控制台报告 + logs/audit_report.json）"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite 库路径（默认 rag/data/jobs.db）")
    parser.add_argument("--json", action="store_true", help="只把 JSON 输出到 stdout，不打印控制台报告")
    parser.add_argument("--out", default=str(DEFAULT_REPORT), help="JSON 报告输出路径（默认 logs/audit_report.json）")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

    db_path = Path(args.db)
    try:
        report = audit(db_path)
    except (FileNotFoundError, sqlite3.Error) as exc:
        print(f"审计失败：{exc}", file=sys.stderr)
        return 2

    _write_report(report, Path(args.out))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
        print(f"JSON 报告已写入：{Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
