import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from errors import ConfigError, APIError

load_dotenv()

MODEL_NAME = os.getenv("ARK_MODEL", "ep-你的接入点ID")
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

def _get_client() -> OpenAI:
    """每次调用时取 key，缺失就抛 ConfigError"""
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ConfigError("未找到 ARK_API_KEY，请检查 .env 文件")
    return OpenAI(api_key=api_key, base_url=BASE_URL)

def chat(messages: list, retries: int = 3) -> str:
    """调用模型，失败时重试"""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=90
            )
            return response.choices[0].message.content
        except ConfigError:
            raise  # 配置错误不重试，直接抛给上层
        except Exception as e:
            last_error = e
            print(f"[第{attempt}次尝试失败] {e}")
            if attempt < retries:
                wait = attempt * 2
                print(f"等待 {wait} 秒后重试...")
                time.sleep(wait)
    raise APIError(f"API 调用失败，已重试 {retries} 次：{last_error}")

def chat_stream(messages: list):
    """流式调用模型，逐字返回"""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        timeout=90,
        stream=True
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content