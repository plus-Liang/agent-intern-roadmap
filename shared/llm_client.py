"""智谱对话客户端：只用标准库 urllib.request 直连 HTTP 接口。

- 不再依赖任何第三方 HTTP 库（openai SDK / requests 都去掉了）：请求体显式
  `json.dumps(..., ensure_ascii=False).encode("utf-8")`，响应显式
  `decode("utf-8")`，全程不经过任何隐式编码。
- 云端真正的报错源是 **日志打印**：进程 stdout 编码退化（ascii / latin-1）时，
  `print` 一句带中文的日志就会抛 UnicodeEncodeError，把原始异常整个盖掉。
  所以本模块所有日志都走 _safe_print，编码再差也不会中断主流程。
"""
import json
import sys
import time
import urllib.error
import urllib.request

from shared.config import ZHIPU_API_KEY, ZHIPU_BASE_URL, ZHIPU_CHAT_MODEL
from shared.errors import ConfigError, APIError
from shared import token_tracker

CHAT_URL = f"{ZHIPU_BASE_URL}/chat/completions"


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


def _headers():
    if not ZHIPU_API_KEY:
        raise ConfigError("未找到 ZHIPU_API_KEY，请检查 .env 文件")
    return {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
    }


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
        _safe_print(f"[token] 用量记录失败（忽略）：{type(e).__name__}: {e}")


def _encode(payload: dict) -> bytes:
    """关键：显式 UTF-8，中文原样进 body，不依赖任何默认编码。"""
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _error_detail(e: Exception) -> str:
    """HTTPError 的响应体里通常写着网关的真实原因（如 model 不存在）。"""
    if not isinstance(e, urllib.error.HTTPError):
        return ""
    try:
        return e.read().decode("utf-8", "replace")[:500]
    except Exception:                            # noqa: BLE001
        return ""


def chat(messages: list, model: str = None, retries: int = 3, source: str = "unknown") -> str:
    """非流式调用，失败重试。

    source: 调用方标记（如 "react_agent"），用于 token 用量按来源聚合，默认 "unknown"。
    """
    model = model or ZHIPU_CHAT_MODEL
    headers = _headers()                    # 顺便校验 Key，缺了直接抛 ConfigError
    body = _encode({"model": model, "messages": messages, "stream": False})
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(CHAT_URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            _record_usage(model=model, source=source,
                          usage=data.get("usage"), request_id=data.get("id"))
            return data["choices"][0]["message"]["content"]
        except ConfigError:
            raise
        except Exception as e:
            last_err = e
            detail = _error_detail(e)
            # 失败原因（含网关响应体）保留一行，方便云端排障；[DEBUG]/逐次 traceback 噪音已清理
            _safe_print(f"[第{attempt}次尝试失败] {type(e).__name__}: {e}"
                        + (f" | 响应体: {detail}" if detail else ""))
            if attempt < retries:
                wait = attempt * 2
                _safe_print(f"等待 {wait} 秒后重试...")
                time.sleep(wait)

    raise APIError(f"API 调用失败，已重试 {retries} 次：{last_err}")


def chat_stream(messages: list, model: str = None, source: str = "unknown"):
    """流式调用，逐字返回（直接按行读 SSE，标准库无第三方编码逻辑）。

    source: 调用方标记，用于 token 用量按来源聚合，默认 "unknown"。
    网关若在收尾 chunk 里带 usage 就记，没带则记 0 并标记 stream_no_usage。
    记账发生在生成器结束之后，调用方 break / 抛异常同样会落一条。
    """
    model = model or ZHIPU_CHAT_MODEL
    headers = _headers()
    body = _encode({"model": model, "messages": messages, "stream": True})

    usage = None
    request_id = None
    try:
        req = urllib.request.Request(CHAT_URL, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=90) as resp:
            for raw in resp:
                line = raw.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data.decode("utf-8"))
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
