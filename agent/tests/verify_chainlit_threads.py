# -*- coding: utf-8 -*-
"""Chainlit data layer 端到端探针（需求 3：侧边栏历史会话列表 / 搜索 / 切换回放）。

跑之前先在**另一个端口**把服务起起来（探针只发 HTTP + socket.io，不起服务）：

    set CHAT_AUTH_ENABLED=true & set CHAT_ADMIN_USER=admin ^
     & set CHAT_ADMIN_PASSWORD=e2e-pass-123456 ^
     & set CHAT_HISTORY_DB=<临时库> & set CHAINLIT_DB=<临时库>
    python -m uvicorn main:app --host 127.0.0.1 --port 8010

然后：

    python agent/tests/verify_chainlit_threads.py

环境变量：
    CHAT_BASE   默认 http://127.0.0.1:8010
    E2E_USER / E2E_PASSWORD   默认 admin / e2e-pass-123456

它验的是**用户能看到的那件事**，不是内部函数：
  1. 侧边栏列表接口（POST /chat/project/threads）能列出已有会话；
  2. 按标题搜能搜到（官方实现只搜步骤正文，我们补了标题匹配）；
  3. GET /chat/project/thread/{id} 能取回那条会话的内容（切换后回放的数据源）；
  4. 带 threadId 重新握手能恢复历史会话（走的是 on_chat_resume）；
  5. 新会话发一条消息后，库里出现新的 thread 行（新建对话 → 侧边栏出现）。
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import httpx
import socketio

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASE = os.getenv("CHAT_BASE", "http://127.0.0.1:8010").rstrip("/")
CHAT = f"{BASE}/chat"
USER = os.getenv("E2E_USER", "admin")
PASSWORD = os.getenv("E2E_PASSWORD", "e2e-pass-123456")

PASS: list[str] = []
FAIL: list[tuple[str, str]] = []
NOTES: list[str] = []


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f"  → {detail}" if detail else ""))
    except Exception as exc:                                   # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {label}  → {type(exc).__name__}: {exc}")


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def chainlit_db() -> Path:
    from agent import data_layer
    return Path(data_layer.DB_PATH)


def q(sql: str, args: tuple = ()) -> list:
    with sqlite3.connect(str(chainlit_db())) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def threads(client: httpx.Client, search: str | None = None) -> list:
    body = {"pagination": {"first": 20, "cursor": None},
            "filter": {"feedback": None, "userId": None, "search": search}}
    resp = client.post(f"{CHAT}/project/threads", json=body)
    if resp.status_code != 200:
        raise AssertionError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json().get("data") or []


async def send_message(base: httpx.AsyncClient, sio, session_id: str, text: str,
                       timeout: float = 240.0) -> str:
    """发一条真实消息，等助手回复 / task_end，返回收到的新消息数描述。"""
    done = asyncio.Event()
    seen: list = []

    @sio.on("new_message")
    async def on_new_message(data):                            # noqa: ANN001
        seen.append(data)
        if isinstance(data, dict) and data.get("type") == "assistant_message" \
                and (data.get("output") or "").strip():
            done.set()

    @sio.on("task_end")
    async def on_task_end(_data=None):                         # noqa: ANN001
        done.set()

    await sio.emit("client_message", {"message": {
        "id": str(uuid.uuid4()), "type": "user_message", "output": text,
        "createdAt": "2026-09-27T00:00:00+00:00",
    }})
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        NOTES.append(f"等待助手回复超时（{timeout:.0f}s）：{text[:20]}")
    return f"{len(seen)} 条新消息"


async def connect(auth_extra: dict | None = None, cookies: dict | None = None):
    """连 websocket。注意：开了认证时**必须带上登录 cookie**，
    否则 socket.py:158 会 `raise ConnectionRefusedError("authentication failed")`。"""
    sio = socketio.AsyncClient(reconnection=False)
    session_id = str(uuid.uuid4())
    auth = {"sessionId": session_id, "threadId": None, "clientType": "webapp",
            "userEnv": "{}", "chatProfile": None}
    auth.update(auth_extra or {})
    header = {"Cookie": "; ".join(f"{k}={v}" for k, v in (cookies or {}).items())} \
        if cookies else None
    await sio.connect(CHAT, socketio_path="/chat/ws/socket.io", auth=auth,
                      transports=["websocket"], wait_timeout=20, headers=header)
    await sio.emit("connection_successful")
    await asyncio.sleep(1.0)
    return sio, session_id


async def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                      # noqa: BLE001
            pass

    print(f"目标 {BASE}  用户 {USER}")
    print(f"直连检查的库：{chainlit_db()}")
    print("（这个库必须和服务端 CHAINLIT_DB 指向同一个文件，否则直连那几项会假失败）")
    transport = httpx.AsyncClient(timeout=120.0)
    client = httpx.Client(timeout=120.0)

    # ---- 登录：data layer 生效后，未登录连线程列表都拿不到 ----
    section("0. 登录（拿 cookie）")
    resp = client.post(f"{CHAT}/login", data={"username": USER, "password": PASSWORD})
    print(f"  POST /chat/login → HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  登录失败：{resp.text[:300]}")
        await transport.aclose()
        client.close()
        return 1
    cookies = {k: v for k, v in client.cookies.items()}
    transport.cookies.update(cookies)

    # ---- 1. 侧边栏列表 ----
    section("1. 侧边栏历史会话列表（POST /chat/project/threads）")
    listing: list = []

    def _list() -> str:
        nonlocal listing
        listing = threads(client)
        if not listing:
            raise AssertionError("列表为空 —— 侧边栏会一条历史都看不到")
        return " / ".join(f"{(t.get('name') or '?')[:14]}({t['id'][:12]})" for t in listing[:4])

    check("列表非空", _list)

    # ---- 2. 按标题搜索 ----
    section("2. 搜索会话标题")
    keyword = ""
    for t in listing:
        for token in (t.get("name") or "").split():
            if len(token) >= 2:
                keyword = token
                break
        if keyword:
            break

    def _search() -> str:
        if not keyword:
            raise AssertionError("列表里没有可用来搜的标题词")
        hits = threads(client, search=keyword)
        if not hits:
            raise AssertionError(f"搜 {keyword!r} 一条都没有")
        return f"{keyword!r} → {len(hits)} 条：{hits[0].get('name')!r}"

    check("按标题能搜到", _search)

    def _search_miss() -> str:
        hits = threads(client, search="这个词肯定不存在zzz")
        if hits:
            raise AssertionError(f"不存在词却搜出 {len(hits)} 条")
        return "无关键词 → 0 条"

    check("无关词搜不到", _search_miss)

    # ---- 3. 单条会话内容（切换后回放的数据源） ----
    section("3. 取单条会话内容（GET /chat/project/thread/{id}）")
    if not listing:
        print("  跳过：列表为空（先看第 4 步新建的会话能不能被列出来）")
        target_id = ""
    else:
        target_id = listing[0]["id"]

    if target_id:
        def _thread_body() -> str:
            resp = client.get(f"{CHAT}/project/thread/{target_id}")
            if resp.status_code != 200:
                raise AssertionError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            body = resp.json()
            return f"steps={len(body.get('steps') or [])} name={body.get('name')!r}"

        check("按 id 取回会话", _thread_body)

    # ---- 4. 新建对话 ----
    section("4. 新建对话 → 发送一条消息")
    sio, session_id = await connect(cookies=cookies)
    before = {row["id"] for row in q('SELECT "id" FROM threads')}
    try:
        detail = await send_message(transport, sio, session_id, "帮我找广州的 Agent 岗位")
        print(f"  助手侧：{detail}")
    finally:
        await sio.disconnect()

    # 线程行是 Chainlit 在 `flush_thread_queues` 里写的，可能晚于 task_end 一点点，
    # 所以这里等一会儿再判（不是 sleep 糊过去 —— 等不到就真的 FAIL）。
    new_rows: list = []
    for _ in range(20):
        new_rows = [row for row in q('SELECT * FROM threads') if row["id"] not in before]
        if new_rows:
            break
        await asyncio.sleep(0.5)

    def _new_thread() -> str:
        if not new_rows:
            raise AssertionError("库里没有出现新的 thread 行")
        row = new_rows[0]
        if not row.get("userId"):
            raise AssertionError(f"新线程 userId 为空，侧边栏不会列出它：{row}")
        if not row.get("name"):
            raise AssertionError(f"新线程没有标题：{row}")
        return f"{row['name'][:20]!r} userId={row['userId'][:12]}…"

    check("新会话写进 threads 表且有归属", _new_thread)

    def _new_visible() -> str:
        if not new_rows:
            raise AssertionError("没有新线程可查")
        names = [t.get("name") for t in threads(client)]
        if new_rows[0]["name"] not in names:
            raise AssertionError(f"新标题不在侧边栏列表里：{new_rows[0]['name']!r}")
        return f"列表里出现 {(new_rows[0]['name'] or '')[:20]!r}"

    check("新会话出现在侧边栏列表里", _new_visible)

    def _user_mirrored() -> str:
        rows = q('SELECT "identifier" FROM users')
        if not any(r["identifier"] == USER for r in rows):
            raise AssertionError(f"Chainlit users 表里没有 {USER}: {rows}")
        return f"users={[r['identifier'] for r in rows]}"

    check("登录账号镜像进 Chainlit users 表", _user_mirrored)

    # ---- 5. 切换回历史会话 ----
    section("5. 带 threadId 重连 = 切换回那条会话")
    resume_id = new_rows[0]["id"] if new_rows else target_id

    def _resume_http() -> str:
        resp = client.get(f"{CHAT}/project/thread/{resume_id}")
        if resp.status_code != 200:
            raise AssertionError(f"HTTP {resp.status_code}")
        return f"steps={len(resp.json().get('steps') or [])}"

    check("会话内容可取（回放数据源）", _resume_http)

    async def _resume_ws() -> str:
        sio2 = socketio.AsyncClient(reconnection=False)
        seen: list = []

        @sio2.on("new_message")
        async def _on_msg(data):                               # noqa: ANN001
            if isinstance(data, dict):
                seen.append(data.get("output") or "")

        @sio2.on("first_interaction")
        async def _on_first(_data=None):                       # noqa: ANN001
            seen.append("[first_interaction]")

        auth = {"sessionId": str(uuid.uuid4()), "threadId": resume_id,
                "clientType": "webapp", "userEnv": "{}", "chatProfile": None}
        try:
            await sio2.connect(CHAT, socketio_path="/chat/ws/socket.io", auth=auth,
                               transports=["websocket"], wait_timeout=20,
                               headers={"Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items())})
        except Exception as exc:                               # noqa: BLE001
            return f"__REFUSED__ {type(exc).__name__}: {exc}"
        await sio2.emit("connection_successful")
        await asyncio.sleep(4.0)
        await sio2.disconnect()
        joined = "\n".join(seen)
        if "已恢复" not in joined:
            return f"__NORESUME__ 握手过了但没走恢复（收到 {len(seen)} 条：{joined[:120]!r}）"
        return f"握手成功且收到恢复回执：{joined[:60]!r}"

    detail = await _resume_ws()
    if detail.startswith("__"):
        check("切换回历史会话（websocket 带 threadId → on_chat_resume）",
              lambda: (_ for _ in ()).throw(AssertionError(detail)))
    else:
        check("切换回历史会话（websocket 带 threadId → on_chat_resume）", lambda: detail)

    await transport.aclose()
    client.close()

    print()
    print("=" * 74)
    print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
    for label, err in FAIL:
        print(f"  [FAIL] {label}  → {err}")
    for note in NOTES:
        print(f"  [NOTE] {note}")
    print("=" * 74)
    return 0 if PASS and not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
