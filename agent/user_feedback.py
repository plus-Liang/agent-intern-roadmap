"""
用户反馈收集（D2 的落地件）。

每条 Agent 回答后面挂 👍 / 👎，用户点了就把「问 + 答 + 当时的工具序列 + 评价」
落进 logs/feedback.db 的 feedback 表。它是评估集的补充：
- 评估集（agent/evaluation/）回答的是「Agent 有没有按预期做」；
- 用户反馈回答的是「用户觉得有没有用」，能捞到评估集没覆盖的真实问题。

用独立的小库（logs/feedback.db），不跟投递追踪库 applications.db 混在一起：
反馈是旁路埋点，写失败也绝不能影响主流程。

API：
    record_feedback(question, answer, tool_sequence, rating, comment="") -> int
    query_feedback(days=7) -> {"total", "up", "down", "down_cases": [...]}
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 允许 `python agent/user_feedback.py` 直接跑自测：把仓库根目录放进 sys.path，
# 否则直接执行脚本时拿不到 shared 包（正常被 import 时这几行也无害）。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import ROOT_DIR  # noqa: E402

# 库路径：<repo_root>/logs/feedback.db（可用环境变量 FEEDBACK_DB_PATH 覆盖，方便测试）
DB_PATH = Path(os.getenv("FEEDBACK_DB_PATH", str(ROOT_DIR / "logs" / "feedback.db")))

# 合法的评价取值
RATING_UP = "up"
RATING_DOWN = "down"
VALID_RATINGS = (RATING_UP, RATING_DOWN)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    question      TEXT    NOT NULL DEFAULT '',
    answer        TEXT    NOT NULL DEFAULT '',
    tool_sequence TEXT    NOT NULL DEFAULT '',
    rating        TEXT    NOT NULL,
    comment       TEXT    NOT NULL DEFAULT ''
)
"""


def _get_conn() -> sqlite3.Connection:
    """打开连接（父目录不存在就建），row_factory 设成 Row 方便按列名取值"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表（幂等）。重复调用不会有副作用。"""
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)


def _norm_rating(rating) -> str:
    """把评价归一成 up/down：认 👍/👎、yes/no、good/bad 这些写法"""
    text = str(rating or "").strip().lower()
    if text in ("up", "👍", "good", "yes", "y", "1", "true", "positive", "赞"):
        return RATING_UP
    if text in ("down", "👎", "bad", "no", "n", "0", "false", "negative", "踩"):
        return RATING_DOWN
    raise ValueError(f"非法的 rating：{rating!r}（只接受 up/down）")


def _norm_tool_sequence(tool_sequence) -> str:
    """工具序列统一存成逗号分隔的字符串（也可以直接传字符串）"""
    if tool_sequence is None:
        return ""
    if isinstance(tool_sequence, (list, tuple)):
        return ", ".join(str(t).strip() for t in tool_sequence if str(t).strip())
    return str(tool_sequence).strip()


def record_feedback(question: str, answer: str, tool_sequence, rating: str,
                    comment: str = "") -> int:
    """记录一条反馈，返回新记录的 id。

    question/answer 超长会截断（各 4000 字），免得一条脏数据把库撑大。
    """
    rating_value = _norm_rating(rating)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)
        cursor = conn.execute(
            "INSERT INTO feedback (timestamp, question, answer, tool_sequence, rating, comment)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                str(question or "")[:4000],
                str(answer or "")[:4000],
                _norm_tool_sequence(tool_sequence),
                rating_value,
                str(comment or "")[:1000],
            ),
        )
        return int(cursor.lastrowid)


def _since(days: int) -> str:
    """days 天前的时间戳字符串（同格式，方便字符串比较）"""
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 7
    return (datetime.now() - timedelta(days=max(0, days))).strftime("%Y-%m-%d %H:%M:%S")


def query_feedback(days: int = 7) -> dict:
    """汇总最近 days 天的反馈。

    返回：
        {
          "total": N,            # 窗口内总条数
          "up": N, "down": N,    # 两种评价各多少条
          "down_cases": [...],   # 差评明细（含 question/answer/tool_sequence/comment）
        }
    没有记录时返回 total=0、down_cases=[]，不抛异常。
    """
    since = _since(days)
    init_db()

    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE timestamp >= ? ORDER BY id DESC",
            (since,),
        ).fetchall()

    up = sum(1 for row in rows if row["rating"] == RATING_UP)
    down = sum(1 for row in rows if row["rating"] == RATING_DOWN)

    down_cases = [
        {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "question": row["question"],
            "answer": row["answer"],
            "tool_sequence": row["tool_sequence"],
            "comment": row["comment"],
        }
        for row in rows
        if row["rating"] == RATING_DOWN
    ]

    return {
        "total": len(rows),
        "up": up,
        "down": down,
        "days": days,
        "since": since,
        "down_cases": down_cases,
    }


if __name__ == "__main__":
    # 自测：写两条（一好一差）再查回来。跑在临时库上，不碰真实 logs/feedback.db。
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="feedback_selftest_"))
    DB_PATH = tmp_dir / "feedback.db"

    good_id = record_feedback("帮我找广州的 Agent 岗位", "找到 3 个岗位：…",
                              ["search_jobs"], "up", "很快")
    bad_id = record_feedback("把投递记录都删掉", "已经把全部记录删除。",
                             ["list_tracking", "delete_tracking"], "👎", "没问我确认")
    print(f"临时库：{DB_PATH}")
    print(f"写入两条：id={good_id}（up）、id={bad_id}（down）")

    stats = query_feedback(days=7)
    print(f"查询结果：total={stats['total']} up={stats['up']} down={stats['down']}")
    assert stats["total"] == 2 and stats["up"] == 1 and stats["down"] == 1, stats
    assert len(stats["down_cases"]) == 1, stats
    print(f"差评案例：{stats['down_cases'][0]['comment']} / {stats['down_cases'][0]['tool_sequence']}")
    print("自测通过")
