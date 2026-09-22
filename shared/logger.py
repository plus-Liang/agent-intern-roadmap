import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logger(name: str = "agent_assistant") -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(
        LOG_DIR / "app.log", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


# 模块级 logger：直接 `from shared.logger import log_event` 的调用方不用自己
# 再 setup 一遍，handler 复用同一个文件（logs/app.log），不会重复挂 handler。
logger = setup_logger()


def log_event(trace_id, event_type, **kwargs):
    """输出 JSON 格式的结构化日志"""
    import json
    import time
    record = {
        "trace_id": trace_id,
        "event": event_type,
        "timestamp": time.time(),
        **kwargs,
    }
    logger.info(json.dumps(record, ensure_ascii=False))