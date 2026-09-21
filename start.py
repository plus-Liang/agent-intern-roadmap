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

#: Streamlit 的启动前缀。
#:
#: 不能用裸 ``python -m streamlit``：``-m`` 会把 cwd（也就是项目根）预先塞进
#: 子进程的 ``sys.path``，而 Streamlit 启动时又把自己脚本所在目录
#: （``dashboard/``）插到 ``sys.path[0]``。于是 ``dashboard/shared.py`` 会抢在
#: ``shared/`` 包前面被当成顶层模块 ``shared`` 导入，接着
#: ``from shared.llm_client import ...`` 就报
#: ``ModuleNotFoundError: No module named 'shared.llm_client'; 'shared' is not a package``。
#: 手动 ``streamlit run dashboard/app.py`` 之所以正常，是因为控制台脚本不会把
#: cwd 放进 ``sys.path``，``dashboard/app.py`` 开头的
#: 「根目录不在 sys.path 里才 insert(0)」引导才能把项目根顶到最前面。
#:
#: ``-P``（Python 3.11+ 等价于 PYTHONSAFEPATH）正好只做一件事：不给子进程塞
#: cwd，于是引导逻辑重新生效。同理这里**不要**设 ``PYTHONPATH=项目根`` ——
#: 一旦项目根「已存在于 sys.path」，引导条件为假，就又会退回到上面那个
#: 被 dashboard/ 抢先的坏顺序。
#:
#: 注意 uvicorn 那行**不能**加 ``-P``：它需要 cwd（项目根）在 sys.path 里才能
#: 找到 ``main:app``；而 dashboard/ 不参与它的导入，不存在遮蔽问题。
_STREAMLIT = (
    [sys.executable, "-P", "-m", "streamlit"]
    if sys.version_info >= (3, 11)
    else [sys.executable, "-m", "streamlit"]
)

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
            *_STREAMLIT, "run", str(BASE_DIR / "dashboard" / "app.py"),
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
