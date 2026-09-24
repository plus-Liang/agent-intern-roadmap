# -*- coding: utf-8 -*-
"""清空指定平台的岗位数据（cleaned_jd.json + jobs.db），用于数据重建。

用途
----
某个平台（例如 niuke）的数据抓得不对 / 需要重建时，把该平台的记录一次性
从「JSON 语料」和「SQLite 岗位库」里清掉，其余平台不受影响；清完可以重新
抓取该平台，不会和旧数据混在一起。

用法
----
    python scripts/clear_platform.py --platform niuke             # 真删（默认先备份 .bak）
    python scripts/clear_platform.py --platform niuke --dry-run   # 只统计并打印，不改任何东西
    python scripts/clear_platform.py --platform niuke --resync    # 删完再从 JSON 全量 upsert 回库
    python scripts/clear_platform.py --platform niuke --no-backup # 不写 .bak 备份
    python scripts/clear_platform.py --platform niuke --db PATH   # 指定其它 jobs.db

流程
----
    1) 读 rag/data/cleaned_jd.json（list[dict]，每条带 platform 字段）；
    2) 过滤掉 platform == 指定平台的记录；
    3) 写回 JSON（默认先备份为 cleaned_jd.json.bak）；
    4) SQLite：DELETE FROM jobs WHERE platform = ?；
    5) --resync 时再用 rag.data.db.import_json() 从新 JSON 全量 upsert 回库；
    6) 打印 JSON / SQLite 的剩余条数。

安全
----
    * --platform 必填：不传直接报错退出（绝不"默认清空全部"）；
    * --dry-run 只做 SELECT / 统计，不改文件、不改数据库；
    * 默认写 .bak 备份，误删可回滚。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_JSON = REPO_ROOT / "rag" / "data" / "cleaned_jd.json"
DEFAULT_DB = REPO_ROOT / "rag" / "data" / "jobs.db"
TABLE = "jobs"


# --------------------------------------------------------------------------
# JSON 读写
# --------------------------------------------------------------------------
def _load_json(path: Path) -> tuple[Optional[str], list]:
    """读 JSON，返回 (容器字段名, 记录列表)。

    顶层是 list 时字段名为 None；顶层是 dict（{"jobs": [...]}）时返回那个键，
    写回时保持原结构，不改变文件格式。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return None, data
    if isinstance(data, dict):
        for key in ("jobs", "data", "records"):
            if isinstance(data.get(key), list):
                return key, data[key]
    raise ValueError(f"无法识别的 JSON 结构（顶层既不是 list 也不含 jobs 列表）：{path}")


def _dump_json(path: Path, key: Optional[str], records: list) -> None:
    payload: Any = records if key is None else {key: records}
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def _platform_of(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    return str(record.get("platform") or "").strip()


# --------------------------------------------------------------------------
# SQLite 读写
# --------------------------------------------------------------------------
def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _sqlite_counts(db_path: Path) -> Optional[dict[str, int]]:
    """返回 {platform: 条数}；库不存在或没有 jobs 表时返回 None（视为没有数据）。"""
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(
        f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=10.0
    )
    try:
        if not _table_exists(conn, TABLE):
            return None
        rows = conn.execute(
            f"SELECT platform, COUNT(*) AS n FROM {TABLE} GROUP BY platform"
        ).fetchall()
    finally:
        conn.close()
    return {str(p or "").strip(): int(n) for p, n in rows}


def _sqlite_delete_platform(db_path: Path, platform: str) -> Optional[int]:
    """DELETE FROM jobs WHERE platform = ?；返回删除条数（库/表不存在时返回 None）。"""
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        if not _table_exists(conn, TABLE):
            return None
        cursor = conn.execute(f"DELETE FROM {TABLE} WHERE platform = ?", (platform,))
        removed = int(cursor.rowcount or 0)
        conn.commit()
        return removed
    finally:
        conn.close()


def _total(counts: Optional[dict[str, int]]) -> int:
    return sum(counts.values()) if counts else 0


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="清空指定平台的岗位数据（JSON + SQLite），用于数据重建。",
    )
    parser.add_argument("--platform", default="",
                        help="要清空的平台名，例如 niuke（必填）")
    parser.add_argument("--json", default=str(DEFAULT_JSON),
                        help=f"cleaned_jd.json 路径（默认 {DEFAULT_JSON}）")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help=f"jobs.db 路径（默认 {DEFAULT_DB}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计并打印，不改文件 / 数据库")
    parser.add_argument("--resync", action="store_true",
                        help="删除后从新 JSON 全量 upsert 回 SQLite")
    parser.add_argument("--no-backup", action="store_true",
                        help="不写 .bak 备份（默认会备份 JSON 与 DB）")
    args = parser.parse_args(argv)

    platform = (args.platform or "").strip()
    if not platform:
        print("[错误] 必须显式指定 --platform（例如 --platform niuke）；"
              "本工具不会默认清空任何平台。", file=sys.stderr)
        return 2

    json_path = Path(args.json)
    db_path = Path(args.db)
    if not json_path.is_file():
        print(f"[错误] 找不到 JSON 文件：{json_path}", file=sys.stderr)
        return 1

    try:
        key, records = _load_json(json_path)
    except (OSError, ValueError) as exc:
        print(f"[错误] 读取 JSON 失败：{exc}", file=sys.stderr)
        return 1

    json_before = len(records)
    kept = [r for r in records if _platform_of(r) != platform]
    json_removed = json_before - len(kept)

    db_before = _sqlite_counts(db_path)
    db_removed_expect = (db_before or {}).get(platform, 0)
    db_before_total = _total(db_before)

    print(f"[清空] 平台 = {platform}")
    print(f"JSON   : {json_path}")
    print(f"         {json_before} 条 → 删除 {json_removed} 条 → 剩余 {len(kept)} 条")
    if db_before is None:
        print(f"SQLite : {db_path}")
        print("         （库文件或 jobs 表不存在，跳过数据库操作）")
    else:
        print(f"SQLite : {db_path}")
        print(f"         {db_before_total} 条 → 待删除 {db_removed_expect} 条")

    if args.dry_run:
        print("[dry-run] 未改动任何文件 / 数据库。")
        print(f"JSON 剩余 {len(kept)} 条，SQLite 剩余 {db_before_total} 条（dry-run 预估值）")
        return 0

    # 1) 备份
    if not args.no_backup:
        try:
            shutil.copy2(json_path, json_path.with_suffix(json_path.suffix + ".bak"))
            if db_path.is_file():
                shutil.copy2(db_path, db_path.with_suffix(db_path.suffix + ".bak"))
            print(f"[备份] 已写 {json_path.name}.bak"
                  + (f" 与 {db_path.name}.bak" if db_path.is_file() else ""))
        except OSError as exc:
            print(f"[错误] 备份失败，已中止（避免误删）：{exc}", file=sys.stderr)
            return 1

    # 2) 写回 JSON
    try:
        _dump_json(json_path, key, kept)
    except OSError as exc:
        print(f"[错误] 写回 JSON 失败：{exc}", file=sys.stderr)
        return 1

    # 3) 删除 SQLite 中该平台的记录
    try:
        db_removed = _sqlite_delete_platform(db_path, platform)
    except sqlite3.Error as exc:
        print(f"[错误] 删除 SQLite 记录失败：{exc}", file=sys.stderr)
        return 1

    # 4) 可选：从新 JSON 全量 upsert 回库
    if args.resync:
        try:
            from rag.data import db as job_db  # 延迟导入：db.py 很轻，但保持按需
            if str(db_path) != str(DEFAULT_DB):
                job_db.DB_PATH = db_path
            stats = job_db.import_json(json_path)
            print(f"[resync] 从 JSON 全量同步 SQLite：{stats}")
        except Exception as exc:  # noqa: BLE001 - 同步失败不回滚已完成的清空
            print(f"[警告] resync 失败（JSON 已清空、SQLite 已删除）："
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    db_after = _sqlite_counts(db_path)
    db_after_total = _total(db_after)
    print(f"[完成] JSON 剩余 {len(kept)} 条，SQLite 剩余 {db_after_total} 条"
          f"（JSON 删除 {json_removed} 条，SQLite 删除 "
          f"{db_removed if db_removed is not None else '—'} 条）")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    raise SystemExit(main())
