import time
from openai import OpenAI
from shared.config import ARK_API_KEY, ARK_BASE_URL, ARK_CHAT_MODEL
from shared.errors import ConfigError, APIError

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not ARK_API_KEY:
            raise ConfigError("未找到 ARK_API_KEY，请检查 .env 文件")
        _client = OpenAI(api_key=ARK_API_KEY, base_url=ARK_BASE_URL)
    return _client


def chat(messages: list, model: str = None, retries: int = 3) -> str:
    """非流式调用，失败重试"""
    client = _get_client()
    model = model or ARK_CHAT_MODEL
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=90,
            )
            return response.choices[0].message.content
        except ConfigError:
            raise
        except Exception as e:
            last_error = e
            print(f"[第{attempt}次尝试失败] {e}")
            if attempt < retries:
                wait = attempt * 2
                print(f"等待 {wait} 秒后重试...")
                time.sleep(wait)

    raise APIError(f"API 调用失败，已重试 {retries} 次：{last_error}")


def chat_stream(messages: list, model: str = None):
    """流式调用，逐字返回"""
    client = _get_client()
    model = model or ARK_CHAT_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=90,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content