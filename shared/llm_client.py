"""智谱对话客户端：用 httpx 直连 HTTP 接口（不再用 urllib / requests）。

- 请求体交给 `httpx.Client(json=payload)` 序列化，Content-Type 由 httpx 自动
  设置，不自己拼 header，也不碰任何隐式编码。
- `trust_env=False`：完全忽略环境变量（HTTP_PROXY / HTTPS_PROXY / NO_PROXY ...），
  避免代理相关的环境变量被当成 header 参与编码。
- 云端真正的报错源是 **日志打印**：进程 stdout 编码退化（ascii / latin-1）时，
  `print` 一句带中文的日志就会抛 UnicodeEncodeError，把原始异常整个盖掉。
  所以本模块所有日志都走 _safe_print，异常对象一律先过 _safe_str。
"""
import json
import os
import sys
import time

import httpx
from shared.config import ZHIPU_API_KEY, ZHIPU_BASE_URL, ZHIPU_CHAT_MODEL
from shared.errors import ConfigError, APIError
from shared import token_tracker

CHAT_URL = f"{ZHIPU_BASE_URL}/chat/completions"

TIMEOUT = 90


def _safe_print(*args) -> None:
    """打印日志：stdout 编码退化时也不抛异常。

    云端 stdout 可能是 ascii / latin-1，直接 print 中文会抛
    UnicodeEncodeError: 'latin-1' codec can't encode characters in position ...
    这会盖掉真正的 API 错误，所以失败时退化为全 ASCII 的转义输出。
    """
    text = " ".join(str(a) for a in args)
    try:
        print(text)
    except Exception:                        # noqa: BLE001 - 只可能是编码问题
        try:
            sys.stderr.write(text.encode("ascii", "backslashreplace").decode("ascii") + "\n")
        except Exception:                    # noqa: BLE001 - 日志失败绝不影响主流程
            pass


def _safe_str(obj) -> str:
    """把任意对象转成字符串，永不让 str() 自己抛异常。

    异常对象走 repr（带引号、转义），并截断到 200 字符，
    避免异常信息本身在被格式化/打印时再崩一次。
    """
    try:
        if isinstance(obj, BaseException):
            return f"{type(obj).__name__}: " + repr(obj)[:200]
        return str(obj)[:200]
    except Exception:                        # noqa: BLE001
        try:
            return f"<{type(obj).__name__}>"
        except Exception:                    # noqa: BLE001
            return "<unknown>"


def _headers():
    if not ZHIPU_API_KEY:
        raise ConfigError("未找到 ZHIPU_API_KEY，请检查 .env 文件")
    # 只传 Authorization：Content-Type 由 httpx 按 json= 自动带上
    return {"Authorization": f"Bearer {ZHIPU_API_KEY}"}


def _log_proxy_diag() -> None:
    """诊断：只打印代理环境变量的「有没有」，不打印值（值里可能含中文/凭据）。"""
    _safe_print("[diag] proxy_env: "
                f"HTTP_PROXY={bool(os.environ.get('HTTP_PROXY'))}, "
                f"HTTPS_PROXY={bool(os.environ.get('HTTPS_PROXY'))}, "
                f"NO_PROXY={bool(os.environ.get('NO_PROXY'))}")


def _usage_field(usage, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return usage.get(name) or 0
    return getattr(usage, name, 0) or 0


def _record_usage(model: str = "unknown", source: str = "unknown",
                  usage=None, request_id=None) -> None:
    """把一次调用的 token 用量记进 token_usage 表。

    - usage 为 None（流式接口通常不给 usage）时记 0，并给 source 打上
      ":stream_no_usage" 后缀，避免把「没拿到」当成「真的用了 0」；
    - 追踪是旁路功能，任何异常都吞掉，绝不能影响正常对话。
    """
    try:
        if usage is None:
            token_tracker.record_usage(
                model, 0, 0,
                f"{source}{token_tracker.STREAM_NO_USAGE_SUFFIX}",
                request_id,
            )
            return
        token_tracker.record_usage(
            model,
            _usage_field(usage, "prompt_tokens"),
            _usage_field(usage, "completion_tokens"),
            source,
            request_id,
        )
    except Exception as e:                      # noqa: BLE001 - 记账失败不影响主流程
        _safe_print(f"[token] 用量记录失败（忽略）：{_safe_str(e)}")


def _error_detail(e: Exception) -> str:
    """HTTPStatusError 的响应体里通常写着网关的真实原因（如模型不存在）。"""
    resp = getattr(e, "response", None)
    if resp is None:
        return ""
    try:
        return (resp.text or "")[:500]
    except Exception:                            # noqa: BLE001
        return ""


def _log_failure(attempt: int, e: Exception) -> None:
    detail = _error_detail(e)
    _safe_print(f"[第{attempt}次尝试失败] {type(e).__name__}: {_safe_str(e)}"
                + (f" | 响应体: {detail}" if detail else ""))


def chat(messages: list, model: str = None, retries: int = 3, source: str = "unknown") -> str:
    """非流式调用，失败重试。

    source: 调用方标记（如 "react_agent"），用于 token 用量按来源聚合，默认 "unknown"。
    """
    model = model or ZHIPU_CHAT_MODEL
    headers = _headers()                    # 顺便校验 Key，缺了直接抛 ConfigError
    payload = {"model": model, "messages": messages, "stream": False}
    last_err = None

    _log_proxy_diag()

    for attempt in range(1, retries + 1):
        try:
            # httpx 用 json= 自己序列化 UTF-8 body，也自己设 Content-Type
            # trust_env=False：不读代理等环境变量，避免它们参与请求编码
            with httpx.Client(timeout=TIMEOUT, trust_env=False) as client:
                resp = client.post(CHAT_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            _record_usage(model=model, source=source,
                          usage=data.get("usage"), request_id=data.get("id"))
            return data["choices"][0]["message"]["content"]
        except ConfigError:
            raise
        except Exception as e:
            last_err = e
            _log_failure(attempt, e)
            if attempt < retries:
                wait = attempt * 2
                _safe_print(f"等待 {wait} 秒后重试...")
                time.sleep(wait)

    raise APIError(f"API 调用失败，已重试 {retries} 次：{_safe_str(last_err)}")


def chat_stream(messages: list, model: str = None, source: str = "unknown"):
    """流式调用，逐字返回（httpx 按行读 SSE）。

    source: 调用方标记，用于 token 用量按来源聚合，默认 "unknown"。
    网关若在收尾 chunk 里带 usage 就记，没带则记 0 并标记 stream_no_usage。
    记账发生在生成器结束之后，调用方 break / 抛异常同样会落一条。
    """
    model = model or ZHIPU_CHAT_MODEL
    headers = _headers()
    payload = {"model": model, "messages": messages, "stream": True}

    usage = None
    request_id = None
    try:
        # trust_env=False：同 chat()，不看代理环境变量
        with httpx.Client(timeout=TIMEOUT, trust_env=False) as client:
            with client.stream("POST", CHAT_URL, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                try:
                    # SSE 响应头常不带 charset，显式按 UTF-8 解码，避免中文乱码
                    resp.encoding = "utf-8"
                except Exception:                # noqa: BLE001 - 赋值失败也无妨
                    pass
                for raw in resp.iter_lines():
                    line = raw.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if request_id is None:
                        request_id = chunk.get("id")
                    if chunk.get("usage") is not None:
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:             # 带 usage 的收尾 chunk 可能没有 choices
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        yield content
    finally:
        _record_usage(model=model, source=source, usage=usage, request_id=request_id)
