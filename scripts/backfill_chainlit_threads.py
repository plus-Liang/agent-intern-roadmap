# -*- coding: utf-8 -*-
"""把 `chat_history.db` 里已有的会话补进 Chainlit 的 threads 表（**只需跑一次**）。

为什么要有这一步：接上 data layer 之前，整个应用只有一条按 user_id 固定 id 的
会话（`user:<user_id>`），它从来没在 Chainlit 的 threads 表里出现过。接上之后
侧边栏读的是那张表，于是「以前聊过的那些轮次」不会自己冒出来 —— 这个脚本负责
补登记，让老会话也能出现在侧边栏里、点名切回去。

**它只写 chainlit.db（threads 表），读 chat_history.db，一个字都不改自建库。**

    python -m scripts.backfill_chainlit_threads            # 补登记
    python -m scripts.backfill_chainlit_threads --dry-run  # 只看要写什么
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import chat_history                                    # noqa: E402
from agent import data_layer                                      # noqa: E402


def _chainlit_user_id(conn: sqlite3.Connection, identifier: str) -> str:
    """Chainlit 自己的 users.id（uuid）——侧边栏是按这个列线程的。

    `POST /chat/project/threads` 会把当前用户的 uuid 塞进 filter.userId
    （server.py:983-989），而 `get_all_user_threads` 是
    `WHERE t."userId" = :user_id OR t."id" = :thread_id`（sql_alchemy.py:680）。
    所以**这条会话要出现在"我的会话"列表里，userId 必须写对**。

    没登录过（或没开 CHAT_AUTH_ENABLED）时 Chainlit 的 users 表是空的，
    那就退一步用业务 user_id 当 userId：列表按 id 取单条仍能命中，
    只是要等到登录一次之后"我的全部会话"才有归属。
    """
    row = conn.execute(
        'SELECT "id" FROM users WHERE "identifier" = ?', (identifier,)
    ).fetchone()
    return str(row[0]) if row else identifier


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def backfill(dry_run: bool = False, bind_user: str = "") -> int:
    data_layer.ensure_schema()                 # 表不在就先建出来

    if not chat_history.DB_PATH.exists():
        print(f"[跳过] 自建历史库不存在：{chat_history.DB_PATH}")
        return 0

    chat_history.init_db()
    # 把库里所有会话一并捞出来（chat_history.list_threads 只按单一用户查，会漏人）
    with sqlite3.connect(str(chat_history.DB_PATH)) as src:
        src.row_factory = sqlite3.Row
        rows = src.execute(
            "SELECT thread_id, user_id, title, created_at, updated_at "
            "FROM chat_threads ORDER BY updated_at DESC"
        ).fetchall()

    if not rows:
        print("[跳过] 自建历史库里没有任何会话")
        return 0

    with sqlite3.connect(str(data_layer.DB_PATH)) as conn:
        written = 0
        for row in rows:
            thread_id = str(row["thread_id"])
            # --bind-user：登录过之后把老会话挂到这个 Chainlit 账号名下
            user_identifier = bind_user or str(row["user_id"] or chat_history.DEFAULT_USER_ID)
            user_id = _chainlit_user_id(conn, user_identifier)
            if bind_user and user_id == bind_user:
                print(f"[警告] Chainlit users 表里没有 {bind_user!r}（先登录一次再跑）")
            name = (row["title"] or "").strip() or "历史会话"
            created_at = row["created_at"] or _now()
            updated_at = row["updated_at"] or created_at
            metadata = json.dumps({"backfilled": True, "updatedAt": updated_at})

            print(
                f"  {thread_id:24s} user={user_identifier:8s} "
                f"userId={user_id:38s} name={name!r}"
            )
            if dry_run:
                continue
            conn.execute(
                'INSERT INTO threads ("id","createdAt","name","userId",'
                '"userIdentifier","metadata") VALUES (?,?,?,?,?,?) '
                'ON CONFLICT ("id") DO UPDATE SET "name"=excluded."name", '
                '"userId"=excluded."userId", "userIdentifier"=excluded."userIdentifier"',
                (thread_id, created_at, name, user_id, user_identifier, metadata),
            )
            written += 1
        if not dry_run:
            conn.commit()

    label = "将写入" if dry_run else "已写入"
    print(f"[完成] {label} {len(rows) if dry_run else written} 条会话 → {data_layer.DB_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="补登记 Chainlit 侧边栏的历史会话")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写库")
    parser.add_argument("--bind-user", default="",
                        help="把这些老会话挂到某个 Chainlit 账号名下（如 admin）；"
                             "该账号要先登录过一次，Chainlit 的 users 表里才有它")
    args = parser.parse_args()
    return backfill(dry_run=args.dry_run, bind_user=args.bind_user)


if __name__ == "__main__":
    raise SystemExit(main())
