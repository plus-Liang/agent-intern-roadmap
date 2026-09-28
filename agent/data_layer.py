# -*- coding: utf-8 -*-
"""Chainlit data layer（需求 3）：把「对话线程」交给 Chainlit 自己持久化。

**为什么要有这个模块**

Chainlit 2.12 的 `thread_id` 是 `auth.threadId or uuid4()`（chainlit/session.py:149）。
没有 data layer 时，前端既不保存 `sessionId` 也不带 `threadId` —— 实测每刷新
一次页面就换一个 thread_id，历史永远读不回来。所以 `agent/app.py` 之前只能
把 thread_id 按 user_id 写死（`user:<user_id>`），代价是**一个人只有一条会话**，
侧边栏里也就一条，没有对话列表、没有搜索、没法开新对话。

接上 data layer 之后：

* 前端会把 `threadId` 带进 websocket 握手，Chainlit 自己往 `threads` / `steps`
  写；侧边栏的「历史对话 / 搜索 / 新建对话」全部由 Chainlit 前端驱动；
* 我们**不动** `chat_history.db`（简历快照 / 面试状态还在那儿），两库并存。

**为什么表要自己建**

SQLAlchemyDataLayer 只会执行 `INSERT/SELECT/UPDATE/DELETE`，包里既没有
`models.py` 也没有 alembic 迁移（grep `create_all|alembic|migration` 零命中），
表不会自动出现。所以这里按它 SQL 语句里用到的列，用 `CREATE TABLE IF NOT EXISTS`
建好（幂等）。

**表结构来源**：`chainlit/data/sql_alchemy.py` 里的 SQL 语句逐条反推，
列名大小写敏感（`"userId"` / `"createdAt"` / `"threadId"` …）。
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent

#: Chainlit 自己的库（线程 / 消息步骤 / 反馈 / 元素）。
#: **不要**和 chat_history.db 合并：那张表里是简历快照与面试状态，是业务数据。
DB_PATH = Path(os.getenv(
    "CHAINLIT_DB",
    str(BASE_DIR / "agent" / "data" / "chainlit.db"),
))


def _database_url() -> str:
    """SQLAlchemy 的 async sqlite 连接串（windows 路径要正斜杠）。"""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url
    return f"sqlite+aiosqlite:///{DB_PATH.as_posix()}"


#: 列名与上面的「表结构来源」一一对应。类型只做参考 —— SQLite 是动态类型。
_SCHEMA = (
    # 用户表：Chainlit 的 PersistedUser 来自这里。
    # `get_user` 会断言 id / identifier / createdAt 都是 str，metadata 是 json 字符串。
    """
    CREATE TABLE IF NOT EXISTS users (
        "id"         TEXT PRIMARY KEY,
        "identifier" TEXT NOT NULL UNIQUE,
        "createdAt"  TEXT NOT NULL,
        "metadata"   TEXT
    )
    """,
    # 线程表：侧边栏列表 = 按 "userId" + "userIdentifier" 过滤这张表
    """
    CREATE TABLE IF NOT EXISTS threads (
        "id"             TEXT PRIMARY KEY,
        "createdAt"      TEXT,
        "name"           TEXT,
        "userId"         TEXT,
        "userIdentifier" TEXT,
        "tags"           TEXT,
        "metadata"       TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_threads_user
        ON threads ("userId", "userIdentifier")
    """,
    # 消息步骤：一问一答 + 工具步骤都落这儿；侧边栏搜索就是搜这张表的 "output"
    """
    CREATE TABLE IF NOT EXISTS steps (
        "id"             TEXT PRIMARY KEY,
        "name"           TEXT,
        "type"           TEXT,
        "threadId"       TEXT,
        "parentId"       TEXT,
        "streaming"      TEXT,
        "waitForAnswer"  TEXT,
        "isError"        TEXT,
        "metadata"       TEXT,
        "tags"           TEXT,
        "input"          TEXT,
        "output"         TEXT,
        "createdAt"      TEXT,
        "start"          TEXT,
        "end"            TEXT,
        "generation"     TEXT,
        "showInput"      TEXT,
        "language"       TEXT,
        "defaultOpen"    TEXT,
        "autoCollapse"   TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_steps_thread
        ON steps ("threadId", "createdAt")
    """,
    """
    CREATE TABLE IF NOT EXISTS feedbacks (
        "id"      TEXT PRIMARY KEY,
        "forId"   TEXT,
        "value"   INTEGER,
        "comment" TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elements (
        "id"          TEXT PRIMARY KEY,
        "threadId"    TEXT,
        "type"        TEXT,
        "chainlitKey" TEXT,
        "url"         TEXT,
        "objectKey"   TEXT,
        "name"        TEXT,
        "display"     TEXT,
        "size"        TEXT,
        "language"    TEXT,
        "page"        TEXT,
        "forId"       TEXT,
        "mime"        TEXT,
        "props"       TEXT,
        "autoPlay"    TEXT,
        "playerConfig" TEXT
    )
    """,
)

#: 老库补列用：`CREATE TABLE IF NOT EXISTS` 对已存在的表不生效，加列只能 ALTER。
#: 少了 `defaultOpen`/`autoCollapse` 时 create_step 的 INSERT 会整条失败
#: （step.to_dict() 一定带这两个键，chainlit/step.py:326-327），
#: 结果就是**消息步骤一条都不落库 → 切回历史会话看不到任何内容**。
_MISSING_COLUMNS = {
    "steps": (("defaultOpen", "TEXT"), ("autoCollapse", "TEXT")),
    "elements": (("autoPlay", "TEXT"), ("playerConfig", "TEXT")),
}


def ensure_schema() -> None:
    """建表 + 补列（都幂等）。同步函数，供 CLI / 测试直接调用。"""
    import sqlite3

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        for statement in _SCHEMA:
            conn.execute(statement)
        for table, columns in _MISSING_COLUMNS.items():
            existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
            for name, kind in columns:
                if name not in existing:
                    conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {kind}')


def _as_bool(value):
    """把库里的布尔列还原成真 bool（None 保持 None）。

    **为什么必须要这一步**：官方 `sql_alchemy.py` 是把 `"isError"` 原样塞进
    `StepDict` 的（:488 / :803 / :930）。Postgres 那边这列是真 boolean，没事；
    我们用的 SQLite 没有布尔类型 —— 写入的是 Python `False`，SQLAlchemy 的
    sqlite 方言把参数转成字符串后**落库成了 '0'**（实测 43 条步骤 `isError`
    全是 text '0'），取回来自然也是字符串。

    前端头像组件只做 `if (isError)` 判断：字符串 `"0"` 是**真值**，于是恢复
    历史时每条消息的助手头像都换成了红色 `circle-alert` 图标 —— 就是用户
    看到的「旧对话前面有红色感叹号」。
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if value is None:
        return None
    return bool(value)


def _build_class():
    """延迟 import：没装 sqlalchemy 时不能让 `import agent.data_layer` 直接炸。"""
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
    from chainlit.types import PaginatedResponse

    class ChatDataLayer(SQLAlchemyDataLayer):
        """只做一件事：在真正读写之前把表建好。

        `execute_sql` 是 SQLAlchemyDataLayer 所有操作的唯一出口（get_user /
        update_thread / create_step … 全走它），所以在这里幂等建表最省事，
        也不用关心事件循环什么时候起来、连的是哪个库。
        """

        def __init__(self) -> None:
            super().__init__(
                conninfo=_database_url(),
                # sqlite 换进程/线程都会碰锁，等 10s 比直接报 "database is locked" 强
                connect_args={"timeout": 10},
                show_logger=os.getenv("CHAINLIT_DATA_LAYER_DEBUG", "").strip()
                in ("1", "true", "yes", "on"),
            )
            self._schema_ready = False
            self._schema_lock: Optional[asyncio.Lock] = None

        async def _ensure_schema_async(self) -> None:
            if self._schema_ready:
                return
            if self._schema_lock is None:
                self._schema_lock = asyncio.Lock()
            async with self._schema_lock:
                if self._schema_ready:
                    return
                await asyncio.to_thread(ensure_schema)
                self._schema_ready = True

        async def execute_sql(self, query, parameters):
            await self._ensure_schema_async()
            return await super().execute_sql(query, parameters)

        async def update_thread(
            self,
            thread_id: str,
            name: Optional[str] = None,
            user_id: Optional[str] = None,
            metadata: Optional[Dict] = None,
            tags=None,
        ):
            """给线程补上归属用户，否则侧边栏一条都列不出来。

            为什么要覆写：Chainlit 只在 `session.user` 是它自己的 `PersistedUser`
            时才把 userId 传进来（emitter.py:237-255）。我们没开 `CHAT_AUTH_ENABLED`
            时 `session.user` 是 None，userId 于是为 NULL；而侧边栏的查询是
            `WHERE t."userId" = :user_id OR t."id" = :thread_id`
            （sql_alchemy.py:680）—— 拿不到当前用户 id 就直接报
            `ValueError: userId is required`（:328）。

            我们的用户身份另有来源（`shared/user_context` 的 ContextVar，由
            `agent/app.py` 每个回调入口 `_bind_user()` 设定），这里照抄一份即可：
            登录开着时用 Chainlit 的 uuid，没开时用业务 user_id，两条路都能列出列表。

            **必须先在 `users` 表里查到人**：`update_thread` 拿到 user_id 会去
            `_get_user_identifer_by_id()` 反查 identifier，查不到就
            `assert result` 直接抛 AssertionError（sql_alchemy.py:146-156），
            而 create_step 第一步就是 `await self.update_thread(...)`
            （sql_alchemy.py:385）—— 身份没落库会让**所有步骤持久化失败**。
            """
            if not user_id:
                identifier = None
                try:
                    from shared.user_context import get_current_user

                    identifier = get_current_user() or None
                except Exception:                   # noqa: BLE001 - 身份拿不到就照旧
                    identifier = None
                if identifier:
                    try:
                        persisted = await self.get_user(identifier=identifier)
                    except Exception:               # noqa: BLE001
                        persisted = None
                    # 只有真实存在的 Chainlit 用户才能当 userId；否则宁可为空，
                    # 让 /project/threads 那条路自己报 "userId is required"，
                    # 也不要因为一个查不到的身份把步骤写入全搞挂。
                    user_id = persisted.id if persisted else None
            return await super().update_thread(
                thread_id=thread_id,
                name=name,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
            )

        async def get_all_user_threads(
            self,
            user_id: Optional[str] = None,
            thread_id: Optional[str] = None,
        ):
            """列表 / 单条 thread 的统一出口，顺手把 `isError` 修成真 bool。

            为什么要覆写：恢复历史（`socket.py:84` resume_thread → `get_thread`
            → 这里）与侧边栏列表都走这个方法，`isError` 的字符串 `'0'` 会让
            前端把每条消息都当错误渲染（见 `_as_bool` 的说明）。在这一层修一
            次，比在 app.py 里逐条重写步骤安全得多 —— 步骤内容一律不动，
            只改这一个标记位。
            """
            threads = await super().get_all_user_threads(
                user_id=user_id,
                thread_id=thread_id,
            )
            if not isinstance(threads, list):
                return threads
            for thread in threads:
                if not isinstance(thread, dict):
                    continue
                for step in thread.get("steps") or []:
                    if isinstance(step, dict) and "isError" in step:
                        step["isError"] = _as_bool(step["isError"])
            return threads

        async def get_step(self, step_id: str):
            """单条步骤同病同治（回放/更新反馈时会单独取步骤）。"""
            step = await super().get_step(step_id)
            if isinstance(step, dict) and "isError" in step:
                step["isError"] = _as_bool(step["isError"])
            return step

        async def list_threads(self, pagination, filters):
            """列表照旧，只是把**会话标题**也纳入搜索。

            官方实现在 Python 侧只对 `step["output"]` 做子串匹配
            （sql_alchemy.py:342-347），所以按标题搜历史会话是搜不到的 ——
            而用户看到的那一行恰恰是标题，搜不到就等于没有搜索。

            这里不猜 SQL 的分页细节：先原样拿一页，再对 `filters.search` 补一次
            标题匹配。命中的标题如果不在这一页里，才单独按 id 把那条捞出来接在
            前面（`get_all_user_threads(thread_id=...)` 就是按 id 取单条）。
            """
            response = await super().list_threads(pagination, filters)
            keyword = ((getattr(filters, "search", None) or "")).strip().lower()
            if not keyword or not isinstance(response, PaginatedResponse):
                return response

            threads = list(response.data or [])
            kept = [t for t in threads if keyword in (t.get("name") or "").lower()]
            if kept:
                response.data = kept
                return response

            # 这一页标题都不匹配 —— 直接在库里按标题找一次，命中就精确取回来
            rows = await self.execute_sql(
                query=(
                    'SELECT "id" FROM threads WHERE lower(COALESCE("name", \'\')) '
                    'LIKE :kw ORDER BY "createdAt" DESC LIMIT :limit'
                ),
                parameters={"kw": f"%{keyword}%", "limit": max(1, int(pagination.first or 20))},
            )
            if not rows:
                response.data = threads
                return response

            found: list = []
            for row in rows:
                thread = await self.get_all_user_threads(thread_id=row["id"])
                if thread:
                    found.append(thread[0])
            response.data = found or threads
            return response

    return ChatDataLayer


def build():
    """构造 data layer 实例（agent/app.py 的 `@cl.data_layer` 用）。"""
    return _build_class()()


__all__ = ["DB_PATH", "build", "ensure_schema"]
