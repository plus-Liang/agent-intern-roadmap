# -*- coding: utf-8 -*-
"""真测：长任务的进度推送能不能把「60s 静默 → websocket 超时」修掉。

做法（全部走**真实链路**，只换库文件，不 mock 任何东西）：
  1. 用临时库起一个真实的 Chainlit 服务（`chainlit run agent/app.py`，端口 8011）；
  2. socket.io 客户端用 **websocket** 连上去，像前端一样发消息；
  3. 逐条记录服务端推回来的消息（带时间戳），算出**最长静默间隔**；
  4. 先跑精确搜索（应 1 条进度都没有、6s 内返回），再跑匹配打分
     （应有多步进度、最长静默 < 20s、全程不断连、结果能收到）。

跑法（在项目根目录）：
    python scripts/verify_progress_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import socketio

REPO = Path(__file__).resolve().parents[1]
PORT = int(os.getenv("E2E_PORT", "8011"))
BASE = f"http://127.0.0.1:{PORT}"
CHAT = f"{BASE}/chat"
TMP = REPO / "evaluation" / "_artifacts" / "progress_e2e"
MATCH_Q = "帮我匹配墨泊可士的 AI Agent 岗位"
EXACT_Q = "帮我找广州的 Python 岗位"

RESULT: dict = {"started_at": datetime.now().isoformat(timespec="seconds"),
                "match": {}, "exact": {}, "notes": []}


def seed_resume() -> str:
    """把一份**真实简历**放进临时库（不然匹配打分会因为"没有简历"短路）。"""
    os.environ["APP_DB_PATH"] = str(TMP / "applications.db")
    os.environ["RESUME_ROOT"] = str(TMP / "resumes")
    os.environ["USER_PROFILE_DIR"] = str(TMP / "profiles")
    src_dir = REPO / "agent" / "data" / "resumes" / "admin"
    files = sorted([p for p in src_dir.glob("*.json") if p.is_file()],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"简历库里没有可用样本：{src_dir}")
    # 挑**有内容**的一份（技能 / 项目 / 实习里至少有一个）：空壳简历会让打分短路
    picked, content = None, None
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw = raw.get("content", raw)
        if raw.get("skills") or raw.get("projects") or raw.get("experience"):
            picked, content = path, raw
            break
    if picked is None:
        raise SystemExit(f"简历库里没有一份带技能/项目/实习的样本：{src_dir}")
    sys.path.insert(0, str(REPO))
    from agent import storage
    storage.init_db()
    resume_id = storage.save_resume(content.get("name") or "进度真测简历", content)
    storage.set_default_resume(resume_id)
    return (f"{picked.name} → {resume_id}（技能 {len(content.get('skills') or [])}、"
            f"项目 {len(content.get('projects') or [])}、"
            f"实习 {len(content.get('experience') or [])}）")


def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def start_server(log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.update({
        "APP_DB_PATH": str(TMP / "applications.db"),
        "RESUME_ROOT": str(TMP / "resumes"),
        "USER_PROFILE_DIR": str(TMP / "profiles"),
        "CHAT_HISTORY_DB": str(TMP / "chat_history.db"),
        "CHAINLIT_DB": str(TMP / "chainlit.db"),
        "TOKEN_DB_PATH": str(TMP / "token_usage.db"),
        "CHAT_AUTH_ENABLED": "false",
        "AGENT_ENGINE": "langgraph",
        "PYTHONIOENCODING": "utf-8",
        "SEARCH_CACHE_TTL": "0",
    })
    out = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "chainlit", "run", "agent/app.py",
         "--host", "127.0.0.1", "--port", str(PORT), "--no-cache", "-h"],
        cwd=str(REPO), env=env, stdout=out, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )


def wait_ready(proc: subprocess.Popen, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"服务提前退出（code={proc.returncode}），看日志 {proc.args}")
        try:
            resp = httpx.get(f"{CHAT}/", timeout=3.0)
            if resp.status_code < 500:
                return
        except Exception:                                      # noqa: BLE001
            time.sleep(1.0)
    raise SystemExit("服务在 90s 内没有就绪")


async def run_turn(sio, text: str, timeout: float) -> dict:
    """发一条消息，收集服务端推回的所有消息 + 时间戳（直到 task_end）。"""
    marks: list[dict] = []
    task_end = asyncio.Event()
    started = {"at": 0.0}

    @sio.on("new_message")
    async def _on_message(data):                               # noqa: ANN001
        if not isinstance(data, dict):
            return
        output = str(data.get("output") or "")
        marks.append({"t": time.monotonic(), "type": str(data.get("type") or ""),
                      "text": output})

    @sio.on("task_start")
    async def _on_start(_data=None):                           # noqa: ANN001
        if not started["at"]:
            started["at"] = time.monotonic()

    @sio.on("task_end")
    async def _on_end(_data=None):                             # noqa: ANN001
        task_end.set()

    if not started["at"]:
        started["at"] = time.monotonic()
    await sio.emit("client_message", {"message": {
        "id": str(uuid.uuid4()), "type": "user_message", "output": text,
        "createdAt": datetime.now().isoformat(),
    }})
    timed_out = False
    try:
        await asyncio.wait_for(task_end.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
    await asyncio.sleep(0.5)

    t0 = started["at"]
    steps = [m for m in marks if m["text"].strip()]
    progress = [m for m in steps if m["text"].strip().startswith("⏳")]
    answer = [m for m in steps if m["type"] == "assistant_message"
              and not m["text"].strip().startswith("⏳")]
    points = [t0] + [m["t"] for m in steps]
    gaps = [round(b - a, 1) for a, b in zip(points, points[1:])]
    last = points[-1]
    return {
        "question": text,
        "elapsed_s": round(last - t0, 1),
        "timed_out": timed_out,
        "progress": [m["text"].strip() for m in progress],
        "progress_offsets_s": [round(m["t"] - t0, 1) for m in progress],
        "max_silence_s": round(max(gaps) if gaps else (last - t0), 1),
        "answer_head": (answer[-1]["text"][:120] if answer else "（没有回答）"),
        "answer_len": (len(answer[-1]["text"]) if answer else 0),
        "message_count": len(steps),
    }


async def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                      # noqa: BLE001
            pass

    if not port_free(PORT):
        raise SystemExit(f"端口 {PORT} 已被占用，先关掉再跑")
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 铺一份真实简历到临时库：{seed_resume()}")
    log_path = TMP / "server.log"
    proc = start_server(log_path)
    print(f"[2/4] 起服务 {BASE}（日志 {log_path}）")
    try:
        wait_ready(proc)
        print("[3/4] 服务就绪，websocket 连上去")

        sio = socketio.AsyncClient(reconnection=False)
        disconnected = {"reason": None, "phase": "连接中"}

        @sio.event
        async def disconnect(reason=None):                     # noqa: ANN001
            disconnected["reason"] = reason
            print(f"  [socket] 断开：{reason!r}（当时处于：{disconnected['phase']}）")

        session_id = str(uuid.uuid4())
        # socket.io 挂在根路径 `/ws/socket.io`（不带 /chat 前缀，已用 HTTP 探过）
        await sio.connect(BASE, socketio_path="/ws/socket.io",
                          auth={"sessionId": session_id, "threadId": None,
                                "clientType": "webapp", "userEnv": "{}",
                                "chatProfile": None},
                          headers={"Origin": BASE, "Referer": f"{BASE}/"},
                          transports=["websocket"], wait_timeout=30)
        await sio.emit("connection_successful")
        await asyncio.sleep(2.0)

        print(f"[4/4] 真测开始\n  ① 精确搜索：{EXACT_Q}")
        disconnected["phase"] = "精确搜索进行中"
        RESULT["exact"] = await run_turn(sio, EXACT_Q, timeout=120.0)
        print(f"     耗时 {RESULT['exact']['elapsed_s']}s，"
              f"进度 {len(RESULT['exact']['progress'])} 条")

        print(f"  ② 匹配打分：{MATCH_Q}")
        disconnected["phase"] = "匹配打分进行中"
        RESULT["match"] = await run_turn(sio, MATCH_Q, timeout=300.0)
        disconnected["phase"] = "两轮都已结束"
        print(f"     耗时 {RESULT['match']['elapsed_s']}s，"
              f"进度 {len(RESULT['match']['progress'])} 条，"
              f"最长静默 {RESULT['match']['max_silence_s']}s")
        # 断连判定必须在**客户端主动 disconnect 之前**取：否则自己那次断开会把
        # 事件回调写进去，变成"假失败"（实测踩过）。所以在两轮都跑完的那一刻就快照。
        reason = disconnected["reason"]
        RESULT["disconnect_reason"] = reason
        await sio.disconnect()
        RESULT["pass"] = {
            "match_has_multi_progress": len(RESULT["match"]["progress"]) >= 3,
            "match_no_long_silence": RESULT["match"]["max_silence_s"] < 60,
            "match_not_disconnected": reason is None,
            "match_finished_without_timeout": not RESULT["match"]["timed_out"],
            "match_answer_received": RESULT["match"]["answer_len"] > 50,
            "exact_no_progress": len(RESULT["exact"]["progress"]) == 0,
            "exact_fast": RESULT["exact"]["elapsed_s"] <= 30,
        }
        if reason is not None:
            RESULT["notes"].append(f"websocket 断开原因：{reason}（判为失败）")
    finally:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False)
        except Exception:                                      # noqa: BLE001
            proc.kill()

    out = REPO / "evaluation" / "results" / \
        f"progress_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== 结果 ====")
    for key, ok in RESULT["pass"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {key}")
    print("\n匹配打分进度：")
    for text, offset in zip(RESULT["match"]["progress"],
                            RESULT["match"]["progress_offsets_s"]):
        print(f"  +{offset:>5.1f}s  {text}")
    print(f"\n报告：{out}")
    print("回答开头：" + re.sub(r"\s+", " ", RESULT['match']['answer_head']))
    return 0 if all(RESULT["pass"].values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
