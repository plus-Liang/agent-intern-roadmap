# -*- coding: utf-8 -*-
"""岗位数据 SQLite 接口（只依赖 sqlite3 标准库）。

为什么要有这个模块
------------------
岗位数据原先存在 rag/data/cleaned_jd.json（181 条 / 365KB）。每次搜索都要把整份
JSON 读进来 json.loads，再在 Python 里全表过滤：数据量到 1w+ 之后，单次查询光解析
就要上百毫秒，还要吃几十 MB 内存。

换成本模块之后：
  * 数据落在 rag/data/jobs.db（SQLite 单文件，标准库直接读写，零新依赖）；
  * city / publish_date / platform 建索引，关键词在 title + description 上 LIKE 匹配；
  * 1w+ 量级下按城市/关键词查询是毫秒级。

数据同步约定
------------
* 迁移入口：rag/data/migrate_json_to_sqlite.py（读 JSON → upsert_jobs）。
* 自动兜底：ensure_db() 在 jobs.db 不存在时自动从 cleaned_jd.json 建库；当
  cleaned_jd.json 比 jobs.db 新（定时抓取刚落盘）时自动增量同步，避免
  「爬虫写 JSON、工具读 DB」两个入口读出两套数据。
* upsert_jobs 只增改不删；清理过期数据请用 delete_older_than()。

字段与 agent/tools/job_search.py 的 Job 对齐：
    job_id / platform / title / company / city / salary / url / description /
    publish_date / tags（库里存 JSON 字符串，读出来还原成 list）。
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

# 数据库文件固定放在 rag/data/ 下（与 cleaned_jd.json 同目录）
DB_PATH = Path(__file__).resolve().parent / "jobs.db"
# 兜底用的 JSON 数据源
SOURCE_JSON = Path(__file__).resolve().parent / "cleaned_jd.json"

TABLE = "jobs"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    platform     TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    company      TEXT NOT NULL DEFAULT '',
    city         TEXT NOT NULL DEFAULT '',
    salary       TEXT NOT NULL DEFAULT '',
    url          TEXT NOT NULL DEFAULT '',
    description  TEXT NOT NULL DEFAULT '',
    publish_date TEXT NOT NULL DEFAULT '',
    tags         TEXT NOT NULL DEFAULT '',
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_jobs_city     ON jobs(city);
CREATE INDEX IF NOT EXISTS idx_jobs_publish  ON jobs(publish_date DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_platform ON jobs(platform);

-- 分批滚动抓取的进度表：每个 (平台, 城市, 关键词) 组合一行，
-- 记录"上次抓取时间"，让调度器每次只抓最久未抓的 N 个组合（见 get_stale_combos）。
CREATE TABLE IF NOT EXISTS scrape_history (
    platform      TEXT NOT NULL,
    city          TEXT NOT NULL,
    keyword       TEXT NOT NULL,
    last_run_at   TEXT,
    run_count     INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_error    TEXT,
    PRIMARY KEY (platform, city, keyword)
);
CREATE INDEX IF NOT EXISTS idx_scrape_history_last_run
    ON scrape_history(last_run_at);
"""

_UPSERT_SQL = """
INSERT INTO jobs (job_id, platform, title, company, city, salary, url,
                  description, publish_date, tags, updated_at)
VALUES (:job_id, :platform, :title, :company, :city, :salary, :url,
        :description, :publish_date, :tags, datetime('now'))
ON CONFLICT(job_id) DO UPDATE SET
    platform     = excluded.platform,
    title        = excluded.title,
    company      = excluded.company,
    city         = excluded.city,
    salary       = excluded.salary,
    url          = excluded.url,
    description  = excluded.description,
    publish_date = excluded.publish_date,
    tags         = excluded.tags,
    updated_at   = datetime('now')
"""

_FIELDS = (
    "job_id", "platform", "title", "company", "city",
    "salary", "url", "description", "publish_date", "tags",
)

# 同一条诊断只打印一次，避免每次查询都往 stderr 刷屏
_warned: set[str] = set()


def _warn(message: str) -> None:
    if message in _warned:
        return
    _warned.add(message)
    print(f"[db] {message}", file=sys.stderr)


# --------------------------------------------------------------------------
# 底层工具
# --------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    """每次调用开一个新连接：Streamlit 会多线程调用，连接不跨线程复用最省心。"""
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _text(value: Any) -> str:
    """把任意字段安全转成 str；None / 缺失都当 ""。"""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _tags_to_text(tags: Any) -> str:
    if isinstance(tags, (list, tuple, set)):
        return json.dumps([_text(t) for t in tags], ensure_ascii=False)
    return _text(tags)


def _tags_from_text(text: Any) -> list[str]:
    raw = _text(text).strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(data, list):
        return [_text(item) for item in data]
    return [raw]


def _normalize_job(job: Any) -> Optional[dict]:
    """把一条 Job dict 规整成可入库的行；没有 job_id 的返回 None（跳过）。"""
    if not isinstance(job, dict):
        return None
    job_id = _text(job.get("job_id")).strip()
    if not job_id:
        return None
    row = {"job_id": job_id}
    for name in _FIELDS:
        if name in ("job_id", "tags"):
            continue
        row[name] = _text(job.get(name))
    row["tags"] = _tags_to_text(job.get("tags"))
    return row


def _row_to_dict(row: sqlite3.Row) -> dict:
    data = {name: _text(row[name]) for name in _FIELDS if name != "tags"}
    data["tags"] = _tags_from_text(row["tags"])
    return data


def _normalize_city(city: Optional[str]) -> str:
    """城市归一化：去空白、去掉结尾的「市」，让「广州」和「广州市」等价。"""
    text = _text(city).strip()
    if text.endswith("市"):
        text = text[:-1]
    return text


def _keyword_terms(keyword: Optional[str]) -> list[str]:
    """关键词拆词：含空格时按多个词处理，全部命中才算命中（与 job_search 一致）。"""
    return [term for term in _text(keyword).strip().split() if term]


def _escape_like(term: str) -> str:
    """转义 LIKE 通配符，避免用户输入 % / _ 时把全表捞出来。"""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _existing_ids(conn: sqlite3.Connection, job_ids: list[str]) -> set[str]:
    """分批查已存在的 job_id，用于区分 added / updated。"""
    found: set[str] = set()
    chunk = 500
    for start in range(0, len(job_ids), chunk):
        batch = job_ids[start:start + chunk]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT job_id FROM {TABLE} WHERE job_id IN ({placeholders})", batch
        ).fetchall()
        found.update(_text(row["job_id"]) for row in rows)
    return found


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------

def init_db() -> None:
    """建表 + 索引，幂等（IF NOT EXISTS）。"""
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def upsert_jobs(jobs: list[dict]) -> dict:
    """批量插入/更新。

    按 job_id 去重：存在则更新（覆盖全部字段），不存在则插入。
    返回 {"added": N, "updated": N, "skipped": N}：
      * added   —— 新插入的条数；
      * updated —— job_id 已存在、被覆盖更新的条数（同一批里重复出现也计入）；
      * skipped —— 无效数据（不是 dict / 没有 job_id）的条数。
    """
    stats = {"added": 0, "updated": 0, "skipped": 0}

    rows: list[dict] = []
    for job in jobs or []:
        row = _normalize_job(job)
        if row is None:
            stats["skipped"] += 1
        else:
            rows.append(row)

    init_db()
    if not rows:
        return stats

    with _connect() as conn:
        existing = _existing_ids(conn, [row["job_id"] for row in rows])
        for row in rows:
            conn.execute(_UPSERT_SQL, row)
            if row["job_id"] in existing:
                stats["updated"] += 1
            else:
                stats["added"] += 1
                existing.add(row["job_id"])
    return stats


def search_jobs(
    keyword: Optional[str] = None,
    city: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """按关键词/城市查询。

    * keyword：在 title 或 description 中做 LIKE 匹配（大小写不敏感）；
      含空格时按多个词处理，要求全部命中。None / "" 表示不限。
    * city：城市匹配，支持「广州」和「广州市」归一化；None / "" 表示不限。
    * limit：>0 截断；<=0 或 None 表示不限。
    * 返回按 publish_date 降序排序（空日期排在最后）的 dict 列表。

    jobs.db 不存在时返回空列表（调用方据此回退 JSON / mock）。
    """
    if not DB_PATH.exists():
        return []

    where: list[str] = []
    params: list[Any] = []

    for term in _keyword_terms(keyword):
        like = f"%{_escape_like(term.lower())}%"
        where.append(
            "(LOWER(IFNULL(title, '')) LIKE ? ESCAPE '\\'"
            " OR LOWER(IFNULL(description, '')) LIKE ? ESCAPE '\\')"
        )
        params.extend([like, like])

    normalized_city = _normalize_city(city)
    if normalized_city:
        where.append(
            "(IFNULL(city, '') <> '' AND ("
            "REPLACE(city, '市', '') = ?"
            " OR city LIKE '%' || ? || '%'"
            " OR ? LIKE '%' || REPLACE(city, '市', '') || '%'))"
        )
        params.extend([normalized_city, normalized_city, normalized_city])

    sql = f"SELECT * FROM {TABLE}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY publish_date DESC, job_id ASC"
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    try:
        with _connect() as conn:
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        _warn(f"查询失败，按无数据处理：{exc}")
        return []
    return [_row_to_dict(row) for row in rows]


def count_by_city() -> dict[str, int]:
    """返回 {"北京": 66, "上海": 58, ...}（按条数降序）。"""
    try:
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT city, COUNT(*) AS n FROM {TABLE}"
                " GROUP BY city ORDER BY n DESC, city ASC"
            ).fetchall()
    except sqlite3.Error as exc:
        _warn(f"统计失败，返回空统计：{exc}")
        return {}
    return {(_text(row["city"]).strip() or "未知"): int(row["n"]) for row in rows}


def get_all_cities() -> list[str]:
    """返回所有出现过的城市（去重、排序，忽略空值）。"""
    try:
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT city FROM {TABLE}"
                " WHERE TRIM(IFNULL(city, '')) <> '' ORDER BY city ASC"
            ).fetchall()
    except sqlite3.Error as exc:
        _warn(f"取城市列表失败，返回空列表：{exc}")
        return []
    return [_text(row["city"]) for row in rows]


def delete_older_than(days: int) -> int:
    """删除 publish_date 早于 N 天的记录，返回删除条数。

    publish_date 为空的记录不动（拿不到日期的岗位不该被过期清理误删）。
    """
    days = int(days)
    if days < 0:
        raise ValueError(f"days 不能为负数：{days}")
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            f"DELETE FROM {TABLE}"
            " WHERE publish_date <> '' AND date(publish_date) < date('now', ?)",
            (f"-{days} days",),
        )
        return cursor.rowcount or 0


# --------------------------------------------------------------------------
# 辅助接口（供 job_search / job_detail / data_converter 复用）
# --------------------------------------------------------------------------

def count_jobs() -> int:
    """当前库里的岗位总数（库/表不存在时返回 0）。"""
    if not DB_PATH.exists():
        return 0
    try:
        with _connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {TABLE}").fetchone()
    except sqlite3.Error:
        return 0
    return int(row["n"]) if row else 0


def get_job(job_id: str) -> Optional[dict]:
    """按 job_id 取单条（主键查询，O(log n)）；查不到返回 None。"""
    if not DB_PATH.exists() or not _text(job_id).strip():
        return None
    try:
        with _connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {TABLE} WHERE job_id = ?", (_text(job_id).strip(),)
            ).fetchone()
    except sqlite3.Error as exc:
        _warn(f"按 id 查询失败：{exc}")
        return None
    return _row_to_dict(row) if row else None


def get_all_jobs() -> list[dict]:
    """取全量岗位（按 publish_date 降序）；库不可用时返回空列表。"""
    return search_jobs(keyword=None, city=None, limit=0)


def import_json(json_path: Optional[Path] = None) -> dict:
    """读 JSON 并 upsert 进库，返回 upsert_jobs 的统计。"""
    path = Path(json_path) if json_path else SOURCE_JSON
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("jobs", [data])
    jobs = [item for item in data if isinstance(item, dict)]
    return upsert_jobs(jobs)


def ensure_db(source_json: Optional[Path] = None) -> bool:
    """确保 jobs.db 可用；必要时自动从 JSON 建库 / 增量同步。返回是否可用。

    触发同步的两种情况：
      1) jobs.db 还不存在（新机器 / Streamlit Cloud 冷启动）；
      2) cleaned_jd.json 的 mtime 比 jobs.db 新（定时抓取刚写完 JSON）。
    任何异常都只记一行 stderr 并返回 False，让调用方回退 JSON / mock。
    """
    source = Path(source_json) if source_json else SOURCE_JSON
    try:
        if not DB_PATH.exists():
            if not source.is_file():
                return False
            stats = import_json(source)
            _warn(f"jobs.db 不存在，已从 {source.name} 建库：{stats}")
        elif source.is_file() and source.stat().st_mtime > DB_PATH.stat().st_mtime:
            stats = import_json(source)
            _warn(f"{source.name} 比 jobs.db 新，已增量同步：{stats}")
        return count_jobs() > 0
    except (sqlite3.Error, OSError, ValueError) as exc:
        _warn(f"初始化/同步数据库失败，回退 JSON：{exc}")
        return False


# --------------------------------------------------------------------------
# 分批滚动抓取：scrape_history 读写
# --------------------------------------------------------------------------

SCRAPE_HISTORY = "scrape_history"


def _combo_key(platform: Any, city: Any, keyword: Any) -> tuple[str, str, str]:
    """统一组合键的字符串口径（None / 空值都记成 ""，避免 None 与 "" 变成两行）。"""
    return (str(platform or ""), str(city or ""), str(keyword or ""))


def get_stale_combos(platforms, cities, keywords, limit: int = 0) -> list[tuple]:
    """按「最久未抓」返回本批要抓的 (platform, city, keyword) 组合。

    排序口径：**没抓过的（last_run_at 为 NULL）最优先**，其次按 last_run_at 升序
    （越久没抓越靠前），同一年龄的按池内原始顺序（platform → city → keyword）
    稳定排序，保证多轮跑下来可复现、不互相插队。

    分批口径：先按 (platform, city) 分组，组内按上面的时序排好，再**组间轮询**
    逐个取（每轮每组取 1 个）直到凑够 limit。这样单个城市即使关键词很多，也不会
    一口气吃光整批，保证每个 (platform, city) 都能稳定分到名额。

    limit <= 0 时返回全部组合（不分批，顺序同为轮询序）；
    全池组合 = platforms × cities × keywords。
    """
    pool = [
        _combo_key(platform, city, keyword)
        for platform in (platforms or [])
        for city in (cities or [])
        for keyword in (keywords or [])
    ]
    if not pool:
        return []

    init_db()
    try:
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT platform, city, keyword, last_run_at FROM {SCRAPE_HISTORY}"
            ).fetchall()
    except sqlite3.Error as exc:
        # 进度表读不到时不分批：宁可多抓一轮，也不要因为历史表异常什么都不抓。
        _warn(f"读取 {SCRAPE_HISTORY} 失败，本次不按历史排序：{exc}")
        rows = []

    history = {
        _combo_key(row["platform"], row["city"], row["keyword"]): row["last_run_at"]
        for row in rows
    }
    ranked = [
        (history.get(combo), index, combo) for index, combo in enumerate(pool)
    ]
    ranked.sort(key=lambda item: (item[0] is not None, item[0] or "", item[1]))

    # 按 (platform, city) 分组：ranked 已按时序排好，dict 保持首次出现顺序，
    # 于是组间顺序 = 各组「最久未抓」的那条谁更久，组内同样保持时序。
    groups: dict[tuple[str, str], list[tuple]] = {}
    for _, _, combo in ranked:
        groups.setdefault((combo[0], combo[1]), []).append(combo)

    # 组间轮询：第 0 轮每组取 1 个（各组最久未抓的），第 1 轮再每组取第 2 个……
    # 只要 (platform, city) 组数 <= limit，每个组合都能稳定分到名额。
    combos: list[tuple] = []
    for depth in range(max(len(bucket) for bucket in groups.values())):
        for bucket in groups.values():
            if depth < len(bucket):
                combos.append(bucket[depth])

    if limit and int(limit) > 0:
        return combos[: int(limit)]
    return combos


def record_scrape(platform, city, keyword, success: bool, error: str = "") -> None:
    """记录一个组合的抓取结果：刷新 last_run_at，累加 run_count / success_count。

    失败时把原因写进 last_error；成功时清空 last_error（表示最近一次是好的）。
    幂等：同一组合反复调用只会累加计数，不会产生重复行（主键 UPSERT）。
    """
    init_db()
    sql = f"""
    INSERT INTO {SCRAPE_HISTORY}
        (platform, city, keyword, last_run_at, run_count, success_count, last_error)
    VALUES (?, ?, ?, datetime('now'), 1, ?, ?)
    ON CONFLICT(platform, city, keyword) DO UPDATE SET
        last_run_at   = datetime('now'),
        run_count     = {SCRAPE_HISTORY}.run_count + 1,
        success_count = {SCRAPE_HISTORY}.success_count + excluded.success_count,
        last_error    = excluded.last_error
    """
    row = _combo_key(platform, city, keyword) + (
        1 if success else 0,
        "" if success else str(error or ""),
    )
    try:
        with _connect() as conn:
            conn.execute(sql, row)
    except sqlite3.Error as exc:
        # 进度记录失败不影响本轮抓取本身（最坏情况是下轮重复抓这几个组合）。
        _warn(f"写入 {SCRAPE_HISTORY} 失败（{platform}/{city}/{keyword}）：{exc}")


def get_history_stats() -> dict:
    """分批进度概览：{total_combos, covered_combos, never_run, last_24h_runs}。

    * total_combos  —— 进度表里出现过的组合数（= 抓过至少一次的组合数）；
    * covered_combos—— last_run_at 非空的组合数（正常等于 total_combos）；
    * never_run     —— last_run_at 为空的组合数（建行但没跑成功过）；
    * last_24h_runs —— 最近 24 小时内抓过的组合数（表里只留 last_run_at，
                       所以这里是"组合数"而不是历史总次数；历史总次数看 run_count）。
    """
    init_db()
    stats = {"total_combos": 0, "covered_combos": 0, "never_run": 0,
             "last_24h_runs": 0}
    try:
        with _connect() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*)                                          AS total_combos,
                    SUM(CASE WHEN last_run_at IS NOT NULL THEN 1 ELSE 0 END)
                                                                      AS covered_combos,
                    SUM(CASE WHEN last_run_at IS NULL THEN 1 ELSE 0 END)
                                                                      AS never_run,
                    SUM(CASE WHEN last_run_at >= datetime('now', '-1 day')
                             THEN 1 ELSE 0 END)                       AS last_24h_runs
                FROM {SCRAPE_HISTORY}
                """
            ).fetchone()
    except sqlite3.Error as exc:
        _warn(f"统计 {SCRAPE_HISTORY} 失败：{exc}")
        return stats
    if row is not None:
        for key in stats:
            stats[key] = int(row[key] or 0)
    return stats


def reset_db() -> None:
    """删库重建（仅测试/本地排查用）。"""
    if DB_PATH.exists():
        DB_PATH.unlink()


if __name__ == "__main__":
    init_db()
    ok = ensure_db()
    print(f"DB_PATH = {DB_PATH}（exists={DB_PATH.exists()}，可用={ok}）")
    print(f"岗位总数：{count_jobs()}")
    print(f"城市分布：{count_by_city()}")
    print(f"所有城市：{get_all_cities()}")
    print("\nsearch_jobs(keyword='Agent', city='广州市', limit=3)：")
    for job in search_jobs("Agent", "广州市", 3):
        print(f"  [{job['company']}] {job['title']} | {job['city']} | {job['publish_date']}")
