"""一键启动：FastAPI（含 Chainlit /chat，8000）+ Streamlit Dashboard（8501）。

用法：
    python start.py

Ctrl+C 会同时停止两个子进程。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SERVICES = [
    {
        "name": "FastAPI + Chainlit",
        "cmd": [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0", "--port", "8000",
        ],
        "url": "http://localhost:8000",
    },
    {
        "name": "Streamlit Dashboard",
        "cmd": [
            sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "dashboard" / "app.py"),
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        "url": "http://localhost:8501",
    },
]


def _terminate_all(procs: list[subprocess.Popen]) -> None:
    """先 terminate，5 秒内没退出就 kill。"""
    alive = [p for p in procs if p.poll() is None]
    for p in alive:
        p.terminate()

    deadline = time.time() + 5
    for p in alive:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass

    for p in alive:
        if p.poll() is None:
            p.kill()

    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            pass


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 62)
    print("  Agent 求职助手 —— 一键启动")
    print("=" * 62)

    procs: list[subprocess.Popen] = []
    for svc in SERVICES:
        print(f"  启动 {svc['name']} ...")
        procs.append(subprocess.Popen(svc["cmd"], cwd=str(BASE_DIR)))

    print("-" * 62)
    print("  首页 / Chainlit Agent : http://localhost:8000   （/chat、/docs）")
    print("  Streamlit Dashboard   : http://localhost:8501")
    print("  按 Ctrl+C 同时停止两个服务")
    print("=" * 62)

    exit_code = 0
    stopped_early = False
    try:
        while True:
            time.sleep(0.5)
            for svc, proc in zip(SERVICES, procs):
                rc = proc.poll()
                if rc is not None:
                    print(f"\n[!] {svc['name']} 已退出（exit code {rc}），正在停止其余服务 ...")
                    exit_code = rc or 0
                    stopped_early = True
                    break
            if stopped_early:
                break
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止服务 ...")
    finally:
        _terminate_all(procs)

    print("已全部停止。")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
