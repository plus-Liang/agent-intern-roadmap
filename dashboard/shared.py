"""求职 Dashboard 的共用层（从原单文件 ``dashboard/app.py`` 抽出来）。

四类东西：

* **路径引导 + 数据库连接** —— 把项目根目录补进 ``sys.path``（``agent`` /
  ``shared`` 都在它下面），并初始化 ``applications.db``。
* **常用查询函数** —— 岗位池 / 投递 / 待跟进 / 累计 token / 最近动态。
* **简历加载** —— 默认简历正文（面试准备用）、简历 PDF 字节（下载用，带缓存）。
* **样式转出** —— 把 ``dashboard.styles`` 里的组件和 :func:`inject_styles`
  统一从这里转出，页面只需要 ``from dashboard.shared import ...`` 一次。

注意（多页面模式）
------------------
Streamlit 的 ``pages/`` 目录下每个文件都是**独立脚本**，各自从零启动。
``from dashboard.shared import ...`` 要能成立，前提是项目根目录已经在
``sys.path`` 里——所以每个页面文件开头都要先跑一遍下面的引导:

    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

本模块自身也做了一遍同样的引导，这样 ``import shared`` / ``import
dashboard.shared`` 两种方式都能用。
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------
# 项目根目录引导（dashboard/ 的上一级）
# ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from agent import reminder, storage
from agent.tools_registry import export_resume_pdf_tool
from dashboard.styles import (  # noqa: F401  —— 这里统一转出，页面不用再单独 import styles
    STATUS_LABELS,
    e,
    icon,
    inject_styles,
    job_row,
    stat_card,
    stat_row,
    status_badge,
    timeline,
)
from shared import token_tracker

__all__ = [
    "ROOT_DIR",
    "MIME_TYPES",
    "TABLE_ROW_HEIGHT",
    "STATUS_LABELS",
    "e",
    "icon",
    "inject_styles",
    "job_row",
    "stat_card",
    "stat_row",
    "status_badge",
    "timeline",
    "render_dataframe",
    "cached_json",
    "read_json_fresh",
    "pipeline_spark",
    "delta_dir",
    "count_marked_jobs",
    "count_applications",
    "recent_applications",
    "follow_up_overdue",
    "total_tokens",
    "resume_for_prep",
    "resume_pdf_bytes",
    "storage",
    "reminder",
    "token_tracker",
]

# 初始化数据库（原 app.py 在 import 期就做了这件事）
storage.init_db()

#: 表格行高：设计稿是 40px（Streamlit 1.5x 才支持 row_height，旧版本忽略）
TABLE_ROW_HEIGHT = 40

#: 投递包里几种文件的 MIME 类型（下载按钮用）
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
}


# ============================================================
# 数据库连接 / 常用查询
# ============================================================


def _connect() -> sqlite3.Connection:
    """applications.db 的连接。

    路径直接复用 ``agent.storage.DB_PATH``，不在这里另写一份，避免两处
    路径漂移。首页的「岗位总数 / 最近动态」需要跨表的小查询，用 SQL 直接
    读比在 storage 里加函数更省事。
    """
    con = sqlite3.connect(storage.DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def count_marked_jobs() -> int:
    """岗位池里的岗位数。

    mock 岗位本身不落库，只有被标记过的才写进 ``job_marks``，所以这就是
    数据库里能拿到的「岗位总数」。
    """
    con = _connect()
    try:
        return int(con.execute("select count(*) from job_marks").fetchone()[0])
    finally:
        con.close()


def count_applications() -> int:
    """投递记录总数。"""
    con = _connect()
    try:
        return int(con.execute("select count(*) from applications").fetchone()[0])
    finally:
        con.close()


def recent_applications(limit: int = 5) -> list:
    """最近投递的 ``limit`` 条（按投递时间倒序），首页「最近动态」用。"""
    con = _connect()
    try:
        rows = con.execute(
            "select id, company, title, status, applied_at from applications "
            "order by applied_at desc, rowid desc limit ?",
            (int(limit),),
        ).fetchall()
    finally:
        con.close()
    return [dict(row) for row in rows]


def follow_up_overdue(days: int | None = None) -> list:
    """待跟进的投递（口径与 ``agent.reminder`` 完全一致）。"""
    return reminder.check_follow_ups(days or reminder.DEFAULT_FOLLOW_UP_DAYS)


def total_tokens(days: int = 3650) -> int:
    """累计 token 用量（默认窗口 10 年 ≈ 全部历史）。"""
    usage = token_tracker.query_usage(days=days) or {}
    return int(usage.get("total_tokens") or 0)


# ============================================================
# 简历加载
# ============================================================


def resume_for_prep() -> dict:
    """取当前默认简历，返回 ``{"name", "text", "note"}``；没有简历时给出明确提示。"""
    resume = storage.get_default_resume()
    if not resume:
        return {
            "name": "",
            "text": "（用户还没有保存任何简历）",
            "note": "当前没有简历：参考答案要点里不会出现任何经历/技能，只给「需要准备的素材」",
        }
    content = resume.get("content")
    if isinstance(content, (dict, list)):
        text = json.dumps(content, ensure_ascii=False, default=str)
    else:
        text = str(content or "")
    return {
        "name": resume.get("name", ""),
        "text": text[:2500] or "（简历内容为空）",
        "note": "",
    }


@st.cache_data(show_spinner=False, ttl=1800, max_entries=32)
def resume_pdf_bytes(resume_id: str) -> bytes:
    """导出某份简历的 PDF 字节（缓存：同一份简历不会每次 rerun 都重新生成）"""
    info = export_resume_pdf_tool(resume_id)
    return Path(info["path"]).read_bytes()


# ============================================================
# 文件读取（带「文件指纹」缓存）
# ============================================================


@st.cache_data(show_spinner=False, max_entries=32)
def cached_json(path: str, mtime: float, size: int) -> Any:
    """读 JSON 并解析；缓存键里带上文件指纹（mtime + size）。

    为什么把指纹塞进参数：``st.cache_data`` 默认只按「函数 + 参数」缓存。只写
    ``@st.cache_data(ttl=...)`` 时，数据文件被重写后页面在 TTL 到期前会一直拿到
    旧内容（旧条数、旧更新时间）。把 mtime/size 当参数传进来，文件一落盘指纹就
    变、缓存立刻失效，下一次 rerun 读到的就是新数据。
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_json_fresh(path: Path) -> Any:
    """读 JSON，保证「文件一变，下次 rerun 就是新数据」。

    文件缺失 / 内容不是合法 JSON 时抛 ``OSError`` / ``ValueError``，由调用方兜底。
    """
    info = path.stat()
    return cached_json(str(path), info.st_mtime, info.st_size)


# ============================================================
# 渲染小工具
# ============================================================


def render_dataframe(df, **kwargs):
    """统一封装的 st.dataframe：设计稿的 40px 行高 + 8px 圆角表格。

    ``row_height`` 是较新版本才有的参数，旧版本直接跳过，不会报错。
    """
    try:
        return st.dataframe(df, width="stretch", hide_index=True,
                            row_height=TABLE_ROW_HEIGHT, **kwargs)
    except TypeError:
        return st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def pipeline_spark(apps: list) -> list:
    """近 7 天每天的投递数（统计卡片的迷你柱状图用）。"""
    today = datetime.now().date()
    buckets = [0] * 7
    for item in apps:
        stamp = str(item.get("applied_at") or "")[:10]
        try:
            day = datetime.strptime(stamp, "%Y-%m-%d").date()
        except ValueError:
            continue
        offset = (today - day).days
        if 0 <= offset < 7:
            buckets[6 - offset] += 1
    return buckets


def delta_dir(count: int) -> str:
    """转化率小标签的方向：>0 绿色上箭头，=0 灰色横杠。"""
    return "up" if count > 0 else "flat"
