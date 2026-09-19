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
import os
from pathlib import Path

DB_PATH = Path(os.getenv(
    "APP_DB_PATH",
    str(ROOT_DIR / "agent" / "data" / "applications.db")
))



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


def find_application(company: str) -> dict | None:
    """按公司名查找投递记录（模糊匹配，不区分大小写）。

    - 子串匹配：传入 "阶跃" 也能命中 "阶跃星辰"
    - 大小写：SQLite 的 LIKE 对 ASCII 不区分大小写，再叠一层 LOWER() 兜底
    - 命中多条时返回最近创建的一条（applied_at 精度到秒，同一秒内按插入顺序取最后一条）
    - 找不到（含 company 为空/空白）返回 None
    """
    if company is None or not str(company).strip():
        return None

    keyword = str(company).strip().lower()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM applications
        WHERE LOWER(company) LIKE ?
        ORDER BY applied_at DESC, rowid DESC""",
        (f"%{keyword}%",),
    ).fetchall()
    conn.close()
    return dict(rows[0]) if rows else None


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


# ========== C1：多版本简历（一份简历一个 JSON 文件） ==========
#
# 为什么不用 SQLite：简历内容是自由文本/JSON，体量小、份数少，
# 一个文件一份最直观，也方便用户直接看/改/备份。
#
# 存放位置：ROOT_DIR/agent/data/resumes/{resume_id}.json
#   {"id": "a1b2c3d4", "name": "技术岗版", "content": {...} 或 "文本", "created_at": "..."}
# 另有一个 _default.json 记录「默认使用哪一份」（只存 id，不存内容）。
# 可用环境变量 RESUME_DIR 覆盖目录——测试请指向临时目录。

RESUME_DIR = Path(os.getenv(
    "RESUME_DIR",
    str(ROOT_DIR / "agent" / "data" / "resumes"),
))

_DEFAULT_RESUME_FILE = "_default.json"


def _ensure_resume_dir():
    RESUME_DIR.mkdir(parents=True, exist_ok=True)


def _resume_path(resume_id: str) -> Path:
    return RESUME_DIR / f"{resume_id}.json"


def _normalize_resume_content(content):
    """内容可以是 dict/list（结构化简历），也可以是字符串。

    字符串先尝试按 JSON 解析（用户在 Dashboard 里粘 JSON 是最常见的用法），
    解析失败就当纯文本原样保存——简历文本后面还要交给 LLM 解析，不该在这里报错。
    """
    if isinstance(content, (dict, list)):
        return content
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return content
        return parsed if isinstance(parsed, (dict, list)) else content
    return "" if content is None else str(content)


def _read_resume_file(path: Path) -> dict | None:
    """读单个简历文件，坏文件返回 None（不让一份脏数据毁掉整个列表）"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("id", path.stem)
    return data


def save_resume(name: str, content) -> str:
    """保存一份简历，返回 resume_id（uuid 前 8 位）。

    content 支持 dict/list（结构化简历）或字符串（JSON 文本 / 纯文本）。
    同名简历不覆盖——每次保存都是新的一份，方便对比不同版本。
    """
    _ensure_resume_dir()
    resume_id = str(uuid.uuid4())[:8]
    record = {
        "id": resume_id,
        "name": str(name or "").strip() or "未命名简历",
        "content": _normalize_resume_content(content),
        "created_at": now(),
    }
    with open(_resume_path(resume_id), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return resume_id


def list_resumes() -> list[dict]:
    """列出所有简历（不含 content），按创建时间倒序（最新的在前）。

    下划线开头的文件是内部文件（_default.json 记录默认简历），不算简历。
    """
    if not RESUME_DIR.is_dir():
        return []

    items = []
    for path in RESUME_DIR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        data = _read_resume_file(path)
        if not data:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        items.append({
            "id": data.get("id", path.stem),
            "name": data.get("name", "未命名简历"),
            "created_at": data.get("created_at", ""),
            # created_at 只精确到秒，同一秒内保存的多份用文件 mtime 兜底排序
            "_mtime": mtime,
        })

    items.sort(key=lambda r: (r["created_at"], r["_mtime"]), reverse=True)
    for item in items:
        item.pop("_mtime", None)
    return items


def get_resume(resume_id: str) -> dict | None:
    """按 id 取一份简历（含 content），不存在返回 None"""
    if not resume_id:
        return None
    path = _resume_path(str(resume_id).strip())
    if not path.is_file():
        return None
    return _read_resume_file(path)


def delete_resume(resume_id: str) -> bool:
    """删除一份简历，成功返回 True，不存在返回 False。

    如果删掉的正好是默认简历，顺手清掉默认指针（避免指向一个不存在的 id）。
    """
    path = _resume_path(str(resume_id or "").strip())
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError:
        return False
    if _read_default_resume_id() == str(resume_id).strip():
        try:
            (RESUME_DIR / _DEFAULT_RESUME_FILE).unlink()
        except OSError:
            pass
    return True


def set_default_resume(resume_id: str) -> bool:
    """把某份简历设为默认使用；id 不存在返回 False（不写坏指针）"""
    if get_resume(resume_id) is None:
        return False
    _ensure_resume_dir()
    with open(RESUME_DIR / _DEFAULT_RESUME_FILE, "w", encoding="utf-8") as f:
        json.dump({"id": str(resume_id).strip()}, f, ensure_ascii=False)
    return True


def _read_default_resume_id() -> str | None:
    path = RESUME_DIR / _DEFAULT_RESUME_FILE
    data = _read_resume_file(path)
    if not data:
        return None
    rid = data.get("id")
    return str(rid) if rid else None


def get_default_resume() -> dict | None:
    """取默认简历：优先 set_default_resume 指定的那份；
    没指定过（或指定的那份已被删）就退回最新保存的一份；一份都没有则 None。
    """
    pointed = _read_default_resume_id()
    if pointed:
        data = get_resume(pointed)
        if data:
            return data
    items = list_resumes()
    if not items:
        return None
    return get_resume(items[0]["id"])


