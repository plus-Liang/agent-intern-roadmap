# -*- coding: utf-8 -*-
"""岗位数据迁移：cleaned_jd.json → SQLite（rag/data/jobs.db）。

为什么迁移
----------
岗位数据现在存在 rag/data/cleaned_jd.json。每次搜索都要整份读进来 json.loads
再全表过滤，181 条时无感，1w+ 条时会变成百毫秒级 + 几十 MB 内存。换成 SQLite
（Python 标准库自带，不需要装任何依赖）之后，按城市/关键词查询走索引，毫秒级。

怎么用
------
    python rag/data/migrate_json_to_sqlite.py                    # 默认 JSON → 默认 DB
    python rag/data/migrate_json_to_sqlite.py other.json         # 指定输入 JSON
    python rag/data/migrate_json_to_sqlite.py --dry-run          # 只校验+统计，不写库
    python -m rag.data.migrate_json_to_sqlite

脚本幂等：按 job_id upsert，跑第二遍会显示 added=0 / updated=全部，数据不会重复。
迁移不删除 cleaned_jd.json（api/router.py、dashboard 页面还在读它），也不删库里
已有的旧记录——要清过期岗位用 db.delete_older_than(days)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# 允许 `python rag/data/migrate_json_to_sqlite.py` 直接运行：把自己所在仓库根塞进 sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.data import db  # noqa: E402  （必须在 sys.path 处理之后导入）


def load_json_jobs(json_path: Path) -> list[dict]:
    """读 JSON 岗位列表；支持顶层是 list、单个 dict，或 {"jobs": [...]} 包装。"""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("jobs", [data])
    if not isinstance(data, list):
        raise ValueError(f"JSON 结构不是 list：{json_path}")
    return [item for item in data if isinstance(item, dict)]


def migrate(json_path: Path, dry_run: bool = False) -> dict:
    """执行迁移并返回统计 dict（含校验结果）。"""
    jobs = load_json_jobs(json_path)
    json_ids = {str(job.get("job_id")).strip() for job in jobs if job.get("job_id")}

    db.init_db()
    if dry_run:
        stats = {"added": 0, "updated": 0, "skipped": len(jobs) - len(json_ids)}
    else:
        stats = db.upsert_jobs(jobs)

    total = db.count_jobs()
    missing = sorted(json_ids - {str(row["job_id"]) for row in db.get_all_jobs()})
    return {
        "source": str(json_path),
        "db": str(db.DB_PATH),
        "json_count": len(jobs),
        "json_unique_ids": len(json_ids),
        "stats": stats,
        "db_total": total,
        "missing_ids": missing,
        "cities": db.count_by_city(),
        "dry_run": dry_run,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 cleaned_jd.json 迁移进 SQLite（rag/data/jobs.db），幂等可重复执行"
    )
    parser.add_argument("json_path", nargs="?", default=str(db.SOURCE_JSON),
                        help=f"输入 JSON（默认 {db.SOURCE_JSON}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只读 JSON 做校验和统计，不写数据库")
    args = parser.parse_args(argv)

    json_path = Path(args.json_path)
    if not json_path.is_file():
        print(f"[错误] 找不到 JSON 文件：{json_path}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    try:
        result = migrate(json_path, dry_run=args.dry_run)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[错误] 迁移失败：{exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    stats = result["stats"]
    print("=" * 64)
    print("岗位数据迁移：cleaned_jd.json → SQLite" + ("（dry-run，未写库）" if args.dry_run else ""))
    print("=" * 64)
    print(f"输入 JSON ：{result['source']}（{result['json_count']} 条，唯一 job_id {result['json_unique_ids']} 个）")
    print(f"输出 DB   ：{result['db']}")
    print(f"写入统计  ：added={stats['added']}  updated={stats['updated']}  skipped={stats['skipped']}")
    print(f"库内总数  ：{result['db_total']}")
    print(f"城市分布  ：{result['cities']}")
    print(f"耗时      ：{elapsed * 1000:.0f} ms")

    if result["missing_ids"]:
        print(f"[警告] 有 {len(result['missing_ids'])} 个 job_id 没进库：{result['missing_ids'][:5]}", file=sys.stderr)
        return 2
    if result["json_unique_ids"] and result["db_total"] < result["json_unique_ids"]:
        print(f"[警告] 库内条数（{result['db_total']}）少于 JSON 去重后条数"
              f"（{result['json_unique_ids']}）", file=sys.stderr)
        return 2

    print(f"[OK] 全部 {result['json_unique_ids']} 个 job_id 均已在 SQLite 中")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
