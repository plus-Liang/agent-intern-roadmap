"""
Token 用量追踪（SQLite）。

背景：每次对话到底烧了多少 token 之前完全不可见，没法做成本控制。
这个模块只做两件事：
1. record_usage()：把每次 LLM 调用的 usage 落库（由 shared/llm_client.py 自动调用）；
2. query_usage()：按总量 / 模型 / 来源 / 天三个维度聚合，供 Dashboard 展示。

数据库位置：ROOT_DIR / "logs" / "token_usage.db"（logs/ 已在 .gitignore 中）。
可用环境变量 TOKEN_DB_PATH 覆盖——测试请指向临时库，不要污染真实用量数据。

设计原则：追踪是旁路功能，绝不能影响主流程。record_usage / query_usage 内部
对异常做兜底（record_usage 失败只打印一行提示，不抛异常）。
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from shared.config import ROOT_DIR

# 用量库路径：默认 logs/token_usage.db，环境变量可覆盖（测试用临时库）
DB_PATH = Path(os.getenv("TOKEN_DB_PATH", str(ROOT_DIR / "logs" / "token_usage.db")))

# 流式调用拿不到 usage 时，给 source 打上的后缀标记。
# 这样 by_source 聚合里能一眼看出「这次调用没拿到用量，token 数记为 0」，
# 而不是把 0 当成真实的 0 消耗。
STREAM_NO_USAGE_SUFFIX = ":stream_no_usage"

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def now() -> str:
    return datetime.now().strftime(_TS_FORMAT)


def init_token_db():
    """建表（幂等）。每次写入前都会调一次，保证库和表一定存在。"""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            source TEXT,
            request_id TEXT
        )
    """)
    # 查询基本都带时间范围过滤，给 timestamp 建索引
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp ON token_usage(timestamp)"
    )
    conn.commit()
    conn.close()


def _as_int(value) -> int:
    """把 usage 里的值安全转成整数：None / 异常值一律算 0"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def record_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    source: str = "unknown",
    request_id: str = None,
) -> int | None:
    """记录一次 LLM 调用的 token 用量，返回自增 id（失败返回 None）。

    total_tokens = prompt_tokens + completion_tokens。
    source 用来区分调用方（react_agent / react_agent_summary / planner / ...）；
    流式拿不到用量时由 llm_client 拼上 STREAM_NO_USAGE_SUFFIX。

    这个函数不会抛异常——tracking 挂掉不该把用户的对话一起带走。
    """
    try:
        prompt = _as_int(prompt_tokens)
        completion = _as_int(completion_tokens)
        init_token_db()
        conn = _get_conn()
        cursor = conn.execute(
            """INSERT INTO token_usage
            (timestamp, model, prompt_tokens, completion_tokens, total_tokens, source, request_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                now(),
                str(model or "unknown"),
                prompt,
                completion,
                prompt + completion,
                str(source or "unknown"),
                request_id,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id
    except Exception as e:                      # noqa: BLE001 - 旁路功能，不能影响主流程
        print(f"[token] 用量记录失败（忽略）：{type(e).__name__}: {e}")
        return None


def query_usage(days: int = 7) -> dict:
    """聚合最近 days 天的用量（含今天，共 days 天）。

    返回：
    {
        "days": 7,
        "total_tokens": N,
        "total_calls": N,
        "by_model": {model: {"tokens": N, "calls": N}},
        "by_source": {source: {"tokens": N, "calls": N}},
        "daily": [{"date": "YYYY-MM-DD", "tokens": N, "calls": N}, ...],
    }

    daily 会补齐窗口内没有记录的日期（tokens=0, calls=0），
    这样前端画图时 x 轴不会缺天。
    """
    days = max(1, _as_int(days) or 7)
    today = datetime.now().date()
    start_date = today - timedelta(days=days - 1)
    since = start_date.strftime("%Y-%m-%d 00:00:00")

    init_token_db()
    conn = _get_conn()

    total_row = conn.execute(
        """SELECT COALESCE(SUM(total_tokens), 0) AS tokens, COUNT(*) AS calls
        FROM token_usage WHERE timestamp >= ?""",
        (since,),
    ).fetchone()

    def _group(column: str) -> dict:
        # column 只在下面两处传内部常量，不接受外部输入，无注入风险
        rows = conn.execute(
            f"""SELECT COALESCE(NULLIF({column}, ''), 'unknown') AS k,
                       COALESCE(SUM(total_tokens), 0) AS tokens,
                       COUNT(*) AS calls
            FROM token_usage WHERE timestamp >= ?
            GROUP BY k ORDER BY tokens DESC""",
            (since,),
        ).fetchall()
        return {r["k"]: {"tokens": r["tokens"], "calls": r["calls"]} for r in rows}

    by_model = _group("model")
    by_source = _group("source")

    daily_rows = conn.execute(
        """SELECT substr(timestamp, 1, 10) AS d,
                  COALESCE(SUM(total_tokens), 0) AS tokens,
                  COUNT(*) AS calls
        FROM token_usage WHERE timestamp >= ?
        GROUP BY d""",
        (since,),
    ).fetchall()
    conn.close()

    by_day = {r["d"]: {"tokens": r["tokens"], "calls": r["calls"]} for r in daily_rows}
    daily = []
    for offset in range(days):
        day = (start_date + timedelta(days=offset)).strftime("%Y-%m-%d")
        hit = by_day.get(day, {"tokens": 0, "calls": 0})
        daily.append({"date": day, "tokens": hit["tokens"], "calls": hit["calls"]})

    return {
        "days": days,
        "total_tokens": total_row["tokens"],
        "total_calls": total_row["calls"],
        "by_model": by_model,
        "by_source": by_source,
        "daily": daily,
    }


if __name__ == "__main__":
    init_token_db()
    print(f"用量库：{DB_PATH}")
    print(query_usage(7))
