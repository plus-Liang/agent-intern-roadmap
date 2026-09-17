"""
投递记录存储（SQLite）。
两张表：
- applications: 投递主表
- events: 状态变更事件
"""
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from shared.config import ROOT_DIR

DB_PATH = ROOT_DIR / "agent" / "data" / "applications.db"


def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id TEXT PRIMARY KEY,
            company TEXT NOT NULL,
            title TEXT NOT NULL,
            platform TEXT,
            url TEXT,
            applied_at TEXT NOT NULL,
            status TEXT NOT NULL,
            next_follow_up TEXT,
            notes TEXT,
            updated_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id TEXT NOT NULL,
            from_status TEXT,
            to_status TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (application_id) REFERENCES applications(id)
        )
    """)

    conn.commit()
    conn.close()
    init_marks_table()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_application(
    company: str,
    title: str,
    platform: str = "",
    url: str = "",
    notes: str = "",
) -> str:
    """新建投递记录，返回 id"""
    app_id = str(uuid.uuid4())[:8]
    ts = now()
    conn = _get_conn()
    conn.execute(
        """INSERT INTO applications
        (id, company, title, platform, url, applied_at, status, next_follow_up, notes, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (app_id, company, title, platform, url, ts, "applied", "", notes, ts),
    )
    conn.execute(
        """INSERT INTO events (application_id, from_status, to_status, note, created_at)
        VALUES (?, ?, ?, ?, ?)""",
        (app_id, None, "applied", "创建记录", ts),
    )
    conn.commit()
    conn.close()
    return app_id


def update_status(app_id: str, to_status: str, note: str = ""):
    """更新状态，同时记录事件"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT status FROM applications WHERE id = ?", (app_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise ValueError(f"未找到记录：{app_id}")

    from_status = row["status"]
    ts = now()
    conn.execute(
        "UPDATE applications SET status = ?, updated_at = ? WHERE id = ?",
        (to_status, ts, app_id),
    )
    conn.execute(
        """INSERT INTO events (application_id, from_status, to_status, note, created_at)
        VALUES (?, ?, ?, ?, ?)""",
        (app_id, from_status, to_status, note, ts),
    )
    conn.commit()
    conn.close()


def update_next_follow_up(app_id: str, date_str: str):
    """更新下次跟进日期"""
    conn = _get_conn()
    conn.execute(
        "UPDATE applications SET next_follow_up = ?, updated_at = ? WHERE id = ?",
        (date_str, now(), app_id),
    )
    conn.commit()
    conn.close()


def update_notes(app_id: str, notes: str):
    """更新备注"""
    conn = _get_conn()
    conn.execute(
        "UPDATE applications SET notes = ?, updated_at = ? WHERE id = ?",
        (notes, now(), app_id),
    )
    conn.commit()
    conn.close()


def get_application(app_id: str) -> dict:
    """获取单条记录"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM applications WHERE id = ?", (app_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def list_applications(status: str = None) -> list[dict]:
    """列出所有记录，可按状态过滤"""
    conn = _get_conn()
    if status:
        rows = conn.execute(
            "SELECT * FROM applications WHERE status = ? ORDER BY applied_at DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY applied_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_events(app_id: str) -> list[dict]:
    """获取某条记录的所有事件"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM events WHERE application_id = ? ORDER BY created_at ASC",
        (app_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_application(app_id: str):
    """删除记录及其事件"""
    conn = _get_conn()
    conn.execute("DELETE FROM events WHERE application_id = ?", (app_id,))
    conn.execute("DELETE FROM applications WHERE id = ?", (app_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"数据库已初始化：{DB_PATH}")


# ========== 岗位标记功能 ==========

def init_marks_table():
    """初始化岗位标记表（与主表分开）"""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_marks (
            job_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            city TEXT,
            salary TEXT,
            url TEXT,
            mark TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def mark_job(job: dict, mark: str):
    """
    标记岗位。
    mark: "want" 想投 / "skip" 不合适 / "untagged" 取消标记
    """
    conn = _get_conn()
    if mark == "untagged":
        conn.execute("DELETE FROM job_marks WHERE job_id = ?", (job["job_id"],))
    else:
        conn.execute("""
            INSERT OR REPLACE INTO job_marks
            (job_id, title, company, city, salary, url, mark, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job["job_id"],
            job.get("title", ""),
            job.get("company", ""),
            job.get("city", ""),
            job.get("salary", ""),
            job.get("url", ""),
            mark,
            now(),
        ))
    conn.commit()
    conn.close()


def get_marked_jobs(mark: str = None) -> list[dict]:
    """获取已标记的岗位"""
    conn = _get_conn()
    if mark:
        rows = conn.execute(
            "SELECT * FROM job_marks WHERE mark = ? ORDER BY created_at DESC",
            (mark,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM job_marks ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_mark(job_id: str) -> str:
    """获取某个岗位的标记"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT mark FROM job_marks WHERE job_id = ?", (job_id,)
    ).fetchone()
    conn.close()
    
    return row["mark"] if row else "untagged"


