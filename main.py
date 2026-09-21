"""统一入口：FastAPI（REST API）+ Chainlit（/chat）。

启动：
    python main.py                # 等价于 uvicorn main:app --host 0.0.0.0 --port 8000

现在只有两个入口，都挂在同一个 8000 端口上：
    /api/*   REST 接口（api/router.py，先占位）
    /chat    Chainlit Agent（agent/app.py）
    /docs    FastAPI 自动生成的 API 文档

注意：请在本文件所在目录（项目根）下启动，Chainlit 需要读到 .chainlit/ 和 chainlit.md。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from chainlit.utils import mount_chainlit
from fastapi import FastAPI

from api.router import router

app = FastAPI(title="Agent 求职助手")

# REST 接口
app.include_router(router, prefix="/api")

# Chainlit 挂到 /chat（Dashboard 的 Streamlit 迁移下一步再做）
mount_chainlit(app=app, target=str(BASE_DIR / "agent" / "app.py"), path="/chat")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
