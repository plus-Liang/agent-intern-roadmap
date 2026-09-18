import os
from pathlib import Path
from dotenv import load_dotenv

# 仓库根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 加载根目录的 .env
load_dotenv(ROOT_DIR / ".env")

# 兼容 Streamlit Cloud：云端没有 .env，改用 st.secrets 注入环境变量。
# 本地无 streamlit 或 secrets 不可用时静默跳过，.env 流程不受影响。
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key in ["ARK_API_KEY", "ARK_CHAT_MODEL", "ARK_EMBEDDING_MODEL",
                    "USE_RERANKER", "APP_DB_PATH"]:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass

# API 配置
ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_CHAT_MODEL = os.getenv("ARK_CHAT_MODEL", "deepseek-v4-flash-ga-260731")
ARK_EMBEDDING_MODEL = os.getenv(
    "ARK_EMBEDDING_MODEL", "doubao-embedding-vision-251215"
)

# 路径配置
DATA_DIR = ROOT_DIR / "rag" / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
LOG_DIR = ROOT_DIR / "logs"
PLANS_DIR = ROOT_DIR / "plans"


def check_config():
    """启动时校验关键配置"""
    from shared.errors import ConfigError
    if not ARK_API_KEY:
        raise ConfigError("未找到 ARK_API_KEY，请检查 .env 文件")
