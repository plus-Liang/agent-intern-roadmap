import time
from openai import OpenAI
from shared.config import ZHIPU_API_KEY, ZHIPU_BASE_URL, ZHIPU_CHAT_MODEL
from shared.errors import ConfigError, APIError
from shared import token_tracker

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not ZHIPU_API_KEY:
            raise ConfigError("未找到 ZHIPU_API_KEY，请检查 .env 文件")
        _client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)
    return _client


def _record_usage(response=None, model: str = "unknown", source: str = "unknown",
                  usage=None, request_id=None) -> None:
    """把一次调用的 token 用量记进 token_usage 表。

    - usage 为 None（流式接口通常不给 usage）时记 0，并给 source 打上
      ":stream_no_usage" 后缀，避免把「没拿到」当成「真的用了 0」；
    - request_id 优先用调用方给的（流式从 chunk.id 取），否则取响应对象自带 id；
    - 追踪是旁路功能，任何异常都吞掉，绝不能影响正常对话。
    """
    try:
        if usage is None and response is not None:
            usage = getattr(response, "usage", None)
        if request_id is None and response is not None:
            request_id = getattr(response, "id", None)
        if usage is None:
            token_tracker.record_usage(
                model, 0, 0,
                f"{source}{token_tracker.STREAM_NO_USAGE_SUFFIX}",
                request_id,
            )
            return
        token_tracker.record_usage(
            model,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            source,
            request_id,
        )
    except Exception as e:                      # noqa: BLE001 - 记账失败不影响主流程
        print(f"[token] 用量记录失败（忽略）：{type(e).__name__}: {e}")


def chat(messages: list, model: str = None, retries: int = 3, source: str = "unknown") -> str:
    """非流式调用，失败重试。

    source: 调用方标记（如 "react_agent"），用于 token 用量按来源聚合，默认 "unknown"。
    """
    client = _get_client()
    model = model or ZHIPU_CHAT_MODEL
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=90,
            )
            _record_usage(response, model, source)
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


def _is_unsupported_param_error(e: Exception) -> bool:
    """判断异常是否属于「网关不认 stream_options」这类参数错误。

    只有这一类才值得回退重发；网络超时、鉴权失败等照旧直接抛出，
    免得白白多打一次 API。
    """
    if isinstance(e, TypeError):
        # 旧版 SDK 不认识 stream_options 这个关键字参数
        return True
    status = getattr(e, "status_code", None)
    if status in (400, 422):
        return True
    text = str(e).lower()
    return "stream_options" in text or "include_usage" in text


def chat_stream(messages: list, model: str = None, source: str = "unknown"):
    """流式调用，逐字返回。

    source: 调用方标记，用于 token 用量按来源聚合，默认 "unknown"。

    关于用量：默认带上 stream_options={"include_usage": True}，让流式响应在
    收尾 chunk 里带回真实 usage——智谱这类 OpenAI 兼容网关必须显式开启，
    否则全程 usage 都是 None，只能记 0 并标记 stream_no_usage。
    若该接口不认这个参数，会回退成不带参数的原始调用并打印警告。
    注意记账发生在生成器结束之后，所以调用方必须把生成器跑完；
    中途 break / 抛异常时会走 finally，同样落一条记录。
    """
    client = _get_client()
    model = model or ZHIPU_CHAT_MODEL

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=90,
            stream=True,
            # 没有它就拿不到 usage，token 只能记 0
            stream_options={"include_usage": True},
        )
    except Exception as e:                      # noqa: BLE001 - 只回落参数类错误
        if not _is_unsupported_param_error(e):
            raise
        print(f"[token] 该接口不认 stream_options（{type(e).__name__}: {e}），"
              f"回退为普通流式调用，本次用量只能记 0")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=90,
            stream=True,
        )
    usage = None
    request_id = None
    try:
        for chunk in response:
            if request_id is None:
                request_id = getattr(chunk, "id", None)
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:                 # 带 usage 的收尾 chunk 可能没有 choices
                continue
            delta = choices[0].delta
            if delta and delta.content:
                yield delta.content
    finally:
        # response 本身是流对象、没有 usage/id，用量和 id 都从 chunk 里捡
        _record_usage(model=model, source=source, usage=usage, request_id=request_id)
