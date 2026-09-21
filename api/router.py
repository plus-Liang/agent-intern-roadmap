"""REST 路由骨架（Dashboard → FastAPI 迁移第一步，先占位）。

约定：
- 只读接口直接读文件 / 只读方式打开 sqlite，不改任何现有数据；
- 数据库路径与 agent/data、logs/token_usage.db 保持一致；
- 具体业务逻辑（分页、鉴权、投递包生成等）下一步再补。
"""
import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
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


# --------------------------------------------------------------------------
# 定时抓取（APScheduler → agent/scrapers/scheduler.run_daily_job，每天 03:00）
#
# 为什么服务放在 router.py：这个实例要同时被 main.py 的 startup/shutdown 事件
# 和下面两个 /scheduler/* 端点用到。放在 router 里，两边都只是 import 同一个
# 对象，不会让 main 与 router 互相 import 形成循环依赖。
# main.py 只负责在 startup 时 start()、shutdown 时 shutdown()。
# --------------------------------------------------------------------------

logger = logging.getLogger("api.scheduler")

JOB_ID = "daily_scrape"
DAILY_HOUR = int(os.getenv("SCHEDULER_HOUR", "3") or 3)
DAILY_MINUTE = int(os.getenv("SCHEDULER_MINUTE", "0") or 0)
# 启动时立刻跑一次（默认关：真实抓取会拉起 Playwright，不该拖慢每次启动）
RUN_ON_STARTUP = (os.getenv("SCHEDULER_RUN_ON_STARTUP", "0") or "0").strip().lower() in {
    "1", "true", "yes", "on",
}


class SchedulerService:
    """AsyncIOScheduler 的薄封装：懒加载任务函数 + 运行态查询 + 手动触发。

    任务函数与状态文件读取都做懒加载——启动时只建 scheduler，不 import
    playwright / scraper，避免 Web 服务启动被拖住。
    """

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._task: Optional[asyncio.Task] = None

    # -- 运行态 ------------------------------------------------------------
    @property
    def running(self) -> bool:
        """当前是否有一次抓取在跑（含手动触发的那次）"""
        return self._task is not None and not self._task.done()

    async def _run(self) -> dict:
        """执行一次完整任务：同步阻塞的 run_daily_job 丢线程池，别卡事件循环"""
        from agent.scrapers.scheduler import run_daily_job   # 懒加载，避免启动阻塞

        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("[定时抓取] 开始（%s）", started)
        try:
            result = await asyncio.to_thread(run_daily_job)
        except Exception as exc:                       # run_daily_job 自己兜异常，这里保底
            logger.exception("[定时抓取] 失败：%s", exc)
            return {"ok": False, "error": str(exc)}

        ok = bool(result.get("ok"))
        message = "[定时抓取] 结束：%s，抓取 %s 条 / 清洗 %s 条 / 新增 %s 条"
        args = (("成功" if ok else "失败"), result.get("scraped", 0),
                result.get("cleaned", 0), (result.get("incremental") or {}).get("added", 0))
        if ok:
            logger.info(message, *args)
        else:
            logger.error(message + "，错误：%s", *args, result.get("error") or "未知")
        return result

    def _spawn(self) -> bool:
        """把一次执行挂成后台 asyncio 任务，立即返回、不阻塞调用方"""
        if self.running:
            logger.warning("[定时抓取] 上一次还在跑，跳过本次触发")
            return False
        self._task = asyncio.create_task(self._run(), name="daily_scrape")
        return True

    # -- 生命周期 ----------------------------------------------------------
    def start(self) -> AsyncIOScheduler:
        """在 FastAPI 事件循环里启动调度器（重复调用无副作用）"""
        if self._scheduler is not None:
            return self._scheduler

        scheduler = AsyncIOScheduler(event_loop=asyncio.get_running_loop())
        scheduler.add_job(
            self._run,
            trigger=CronTrigger(hour=DAILY_HOUR, minute=DAILY_MINUTE),
            id=JOB_ID,
            name="每日岗位抓取",
            replace_existing=True,
            max_instances=1,          # 抓取很慢，绝不允许叠着跑
            coalesce=True,            # 错过多次只补跑一次
            misfire_grace_time=3600,  # 迟到 1 小时内仍执行
        )
        scheduler.start()
        self._scheduler = scheduler
        logger.info("[定时抓取] 已启动：每天 %02d:%02d 自动执行", DAILY_HOUR, DAILY_MINUTE)

        if RUN_ON_STARTUP:
            logger.info("[定时抓取] SCHEDULER_RUN_ON_STARTUP 已开启，立刻跑一次")
            self._spawn()
        return scheduler

    def shutdown(self) -> None:
        """优雅关闭：停调度器，不再派发新任务"""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
            logger.info("[定时抓取] 调度器已关闭")
        if self.running:
            logger.warning("[定时抓取] 关闭时仍有任务在跑，交由进程退出收尾")

    # -- 对外 ---------------------------------------------------------------
    def trigger(self) -> bool:
        """手动触发一次（异步执行，立即返回是否真的派发成功）"""
        return self._spawn()

    def status(self) -> dict:
        """last_run 取自 scheduler.py 自己的状态文件；next_run 取自 APScheduler"""
        state: dict = {}
        try:
            from agent.scrapers.scheduler import load_state   # 懒加载
            data = load_state()
            if isinstance(data, dict):
                state = data
        except Exception as exc:                              # 状态读不到不算接口失败
            logger.warning("[定时抓取] 读取状态文件失败：%s", exc)

        next_run = None
        if self._scheduler is not None:
            job = self._scheduler.get_job(JOB_ID)
            next_time = getattr(job, "next_run_time", None) if job else None
            if next_time is not None:
                next_run = next_time.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "last_run": state.get("last_run") or None,
            "next_run": next_run,
            "running": self.running,
            "ok": state.get("ok"),
            "error": state.get("error") or "",
            "scraped": state.get("scraped", 0),
            "cleaned": state.get("cleaned", 0),
            "schedule": f"{DAILY_HOUR:02d}:{DAILY_MINUTE:02d}",
        }


scheduler_service = SchedulerService()


@router.get("/scheduler/status", summary="定时抓取状态")
def scheduler_status() -> dict:
    """返回 {last_run, next_run, running, ...}，数据来自 scheduler 状态文件"""
    return scheduler_service.status()


@router.post("/scheduler/trigger", summary="手动触发一次抓取（异步，不阻塞）")
async def scheduler_trigger() -> dict:
    """立即返回 accepted，真实抓取在后台任务里跑"""
    dispatched = scheduler_service.trigger()
    return {"status": "accepted", "dispatched": dispatched,
            "already_running": not dispatched}
