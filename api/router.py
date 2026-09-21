"""REST 路由骨架（Dashboard → FastAPI 迁移第一步，先占位）。

约定：
- 只读接口直接读文件 / 只读方式打开 sqlite，不改任何现有数据；
- 数据库路径与 agent/data、logs/token_usage.db 保持一致；
- 具体业务逻辑（分页、鉴权、投递包生成等）下一步再补。
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
JOB_DATA = BASE_DIR / "rag" / "data" / "cleaned_jd.json"
APPLICATIONS_DB = BASE_DIR / "agent" / "data" / "applications.db"
TOKEN_USAGE_DB = BASE_DIR / "logs" / "token_usage.db"

router = APIRouter()


def _connect(db_path: Path, readonly: bool = True) -> sqlite3.Connection:
    """打开 sqlite；默认只读（避免误改数据）"""
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"数据库不存在：{db_path}")
    if readonly:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fetch(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list:
    return [dict(r) for r in conn.execute(sql, args)]


# --------------------------------------------------------------------------
# 健康检查
# --------------------------------------------------------------------------

@router.get("/health", summary="健康检查")
def health() -> dict:
    return {"status": "ok"}


# --------------------------------------------------------------------------
# 岗位（rag/data/cleaned_jd.json）
# --------------------------------------------------------------------------

@router.get("/jobs", summary="按城市 / 关键词过滤岗位")
def list_jobs(
    city: Optional[str] = Query(None, description="城市，子串匹配"),
    keyword: Optional[str] = Query(None, description="关键词，匹配岗位名 / 公司 / 描述"),
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    if not JOB_DATA.exists():
        raise HTTPException(status_code=404, detail=f"岗位数据不存在：{JOB_DATA}")

    with JOB_DATA.open(encoding="utf-8") as f:
        jobs: list = json.load(f)

    def hit(job: dict) -> bool:
        if city and city.strip() and city.strip() not in str(job.get("city", "")):
            return False
        if keyword and keyword.strip():
            text = " ".join(str(job.get(k, "")) for k in ("title", "company", "description"))
            if keyword.strip() not in text:
                return False
        return True

    matched = [j for j in jobs if hit(j)]
    return {"total": len(matched), "items": matched[:limit]}


# --------------------------------------------------------------------------
# 投递追踪（agent/data/applications.db）
# --------------------------------------------------------------------------

class StatusUpdate(BaseModel):
    status: str
    note: str = ""


@router.get("/applications", summary="投递列表")
def list_applications(status: Optional[str] = Query(None)) -> dict:
    sql = "select * from applications"
    args: tuple = ()
    if status:
        sql += " where status = ?"
        args = (status,)
    sql += " order by applied_at desc"

    conn = _connect(APPLICATIONS_DB)
    try:
        items = _fetch(conn, sql, args)
    finally:
        conn.close()
    return {"total": len(items), "items": items}


@router.post("/applications/{app_id}/status", summary="更新投递状态")
def update_application_status(app_id: str, payload: StatusUpdate) -> dict:
    conn = _connect(APPLICATIONS_DB, readonly=False)
    try:
        row = conn.execute("select * from applications where id = ?", (app_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"投递记录不存在：{app_id}")

        from_status = row["status"]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with conn:  # 事务：状态 + 事件表一起落库
            conn.execute(
                "update applications set status = ?, updated_at = ? where id = ?",
                (payload.status, now, app_id),
            )
            conn.execute(
                "insert into events (application_id, from_status, to_status, note, created_at)"
                " values (?, ?, ?, ?, ?)",
                (app_id, from_status, payload.status, payload.note, now),
            )
        updated = dict(
            conn.execute("select * from applications where id = ?", (app_id,)).fetchone()
        )
    finally:
        conn.close()
    return {"ok": True, "application": updated}


# --------------------------------------------------------------------------
# Token 用量（logs/token_usage.db）
# --------------------------------------------------------------------------

@router.get("/tokens", summary="最近 N 天 token 用量")
def token_usage(days: int = Query(7, ge=1, le=90)) -> dict:
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")

    conn = _connect(TOKEN_USAGE_DB)
    try:
        by_day = _fetch(
            conn,
            "select substr(timestamp, 1, 10) as day,"
            " count(*) as calls,"
            " sum(prompt_tokens) as prompt_tokens,"
            " sum(completion_tokens) as completion_tokens,"
            " sum(total_tokens) as total_tokens"
            " from token_usage where timestamp >= ? group by day order by day",
            (since,),
        )
        by_source = _fetch(
            conn,
            "select coalesce(source, 'unknown') as source,"
            " count(*) as calls, sum(total_tokens) as total_tokens"
            " from token_usage where timestamp >= ? group by source order by total_tokens desc",
            (since,),
        )
    finally:
        conn.close()

    total = sum(int(r["total_tokens"] or 0) for r in by_day)
    return {"days": days, "since": since, "total_tokens": total,
            "by_day": by_day, "by_source": by_source}
