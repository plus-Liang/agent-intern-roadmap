"""统一入口：FastAPI（REST API）+ Chainlit（/chat）。

启动：
    python main.py                # 等价于 uvicorn main:app --host 0.0.0.0 --port 8000

现在所有入口都挂在同一个 8000 端口上：
    /        teal-ink 风格首页（三张入口卡片）
    /api/*   REST 接口（api/router.py，先占位）
    /chat    Chainlit Agent（agent/app.py）
    /docs    FastAPI 自动生成的 API 文档

注意：请在本文件所在目录（项目根）下启动，Chainlit 需要读到 .chainlit/ 和 chainlit.md。
"""
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api.router import router

app = FastAPI(title="Agent 求职助手")

# REST 接口
app.include_router(router, prefix="/api")


# ---------------------------------------------------------------------------
# 首页（teal-ink，与 dashboard/styles.py 的 CSS 变量同一套配色）
# ---------------------------------------------------------------------------
LANDING_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent 求职助手</title>
<style>
  :root {
    --brand: #0b626c;
    --bg: #fbfbfa;
    --bg-soft: #f5f4f1;
    --fg: #1a1a18;
    --fg-soft: #6f6e6a;
    --border: #e6e5e1;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
            "Hiragino Sans GB", "Microsoft YaHei", Roboto, Helvetica, Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 56px 24px;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg);
    color: var(--fg);
    font-family: var(--sans);
    -webkit-font-smoothing: antialiased;
  }
  .wrap { width: 100%; max-width: 940px; }
  .brand {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
    color: var(--brand); font-weight: 600; margin-bottom: 18px;
  }
  .brand::before {
    content: ""; width: 22px; height: 2px; border-radius: 2px; background: var(--brand);
  }
  h1 { margin: 0 0 12px; font-size: 32px; font-weight: 650; letter-spacing: -.01em; }
  .sub { margin: 0 0 36px; font-size: 15px; line-height: 1.7; color: var(--fg-soft); }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  a.card {
    display: block; padding: 22px 20px 20px;
    background: #ffffff; border: 1px solid var(--border); border-radius: 8px;
    color: inherit; text-decoration: none;
    transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
  }
  a.card:hover {
    border-color: var(--brand);
    box-shadow: 0 2px 12px rgba(11, 98, 108, .10);
    transform: translateY(-1px);
  }
  .ico {
    width: 38px; height: 38px; margin-bottom: 16px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    background: #ecf6f8; color: var(--brand);
  }
  .t { margin: 0 0 6px; font-size: 15px; font-weight: 600; }
  .d { margin: 0; font-size: 13px; line-height: 1.65; color: var(--fg-soft); }
  .go {
    margin-top: 16px; font-size: 12px; color: var(--brand);
    font-family: var(--sans);
  }
  .go code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: var(--bg-soft); border: 1px solid var(--border);
    border-radius: 4px; padding: 1px 6px; color: var(--brand);
  }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } h1 { font-size: 26px; } }
</style>
</head>
<body>
  <div class="wrap">
    <div class="brand">Agent Intern Roadmap</div>
    <h1>Agent 求职助手</h1>
    <p class="sub">岗位搜索 / 投递追踪 / 匹配打分 / Token 成本 / 简历管理</p>

    <div class="grid">
      <a class="card" href="/chat">
        <div class="ico">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20 15a2 2 0 0 1-2 2H8l-4 3V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v9Z"/>
          </svg>
        </div>
        <p class="t">对话式 Agent</p>
        <p class="d">用自然语言搜岗位、看匹配打分、做模拟面试</p>
        <p class="go"><code>/chat</code></p>
      </a>

      <a class="card" href="http://localhost:8501">
        <div class="ico">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 20h20"/>
            <path d="M4 20v-9"/>
            <path d="M10 20V5"/>
            <path d="M16 20v-6"/>
          </svg>
        </div>
        <p class="t">Dashboard</p>
        <p class="d">投递追踪、Token 成本与简历库的可视化看板</p>
        <p class="go"><code>localhost:8501</code></p>
      </a>

      <a class="card" href="/docs">
        <div class="ico">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 6 4 12l4 6"/>
            <path d="M16 6l4 6-4 6"/>
          </svg>
        </div>
        <p class="t">API 文档</p>
        <p class="d">REST 接口清单与在线调试（Swagger UI）</p>
        <p class="go"><code>/docs</code></p>
      </a>
    </div>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """项目首页：三个入口卡片（对话 Agent / Dashboard / API 文档）。"""
    return HTMLResponse(LANDING_HTML)


# Chainlit 挂到 /chat（Dashboard 的 Streamlit 迁移下一步再做）
mount_chainlit(app=app, target=str(BASE_DIR / "agent" / "app.py"), path="/chat")


if __name__ == "__main__":
    # 这里刻意不用 uvicorn.run(app, ...)：
    #   uvicorn.run 内部自己 asyncio.run(...) 建事件循环，在 Windows 上与
    #   mount_chainlit 注册的 signal handler / 事件循环钩子容易打架，表现就是
    #   `python main.py` 启动后没有任何输出、看着像静默退出
    #   （`uvicorn main:app` 走的就是 Config + Server.run() 这条路，所以正常）。
    # 显式 Config + Server 让 uvicorn 自己接管事件循环与信号处理，和 CLI 一致。
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")

    # 顺带修掉被吞掉的启动横幅：
    #   项目里（经 shared/、dashboard/）会 import streamlit，Streamlit 的 logger
    #   会把 "uvicorn.error" 的 propagate 设成 False 并挂上自己的 handler；
    #   而 uvicorn.Config 的 dictConfig 又会把 "uvicorn.error" 的 handler 清空，
    #   结果这个 logger 既不自己输出、也不往 "uvicorn" 冒泡 —— 端口提示全被吞。
    logging.getLogger("uvicorn.error").propagate = True

    server = uvicorn.Server(config)
    server.run()
