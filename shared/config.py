import os
from pathlib import Path
from dotenv import load_dotenv

# 仓库根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 加载根目录的 .env
load_dotenv(ROOT_DIR / ".env")

# ========== 对话模型：智谱官方 ==========
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZHIPU_CHAT_MODEL = os.getenv("ZHIPU_CHAT_MODEL", "glm-5.3-flash")

# ========== Embedding：本地 fastembed ==========
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ========== 应用配置 ==========
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"

# 路径配置
DATA_DIR = ROOT_DIR / "rag" / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
LOG_DIR = ROOT_DIR / "logs"
PLANS_DIR = ROOT_DIR / "plans"


def check_config():
    """启动时校验关键配置"""
    from shared.errors import ConfigError
    if not ZHIPU_API_KEY:
        raise ConfigError("未找到 ZHIPU_API_KEY，请检查 .env 文件")
