"""智谱对话客户端：用 httpx 直连 HTTP 接口（不再用 urllib / requests）。

- 请求体交给 `httpx.Client(json=payload)` 序列化，Content-Type 由 httpx 自动
  设置，不自己拼 header，也不碰任何隐式编码。
- `trust_env=False`：完全忽略环境变量（HTTP_PROXY / HTTPS_PROXY / NO_PROXY ...），
  避免代理相关的环境变量被当成 header 参与编码。
- 云端真正的报错源是 **日志打印**：进程 stdout 编码退化（ascii / latin-1）时，
  `print` 一句带中文的日志就会抛 UnicodeEncodeError，把原始异常整个盖掉。
  所以本模块所有日志都走 _safe_print，异常对象一律先过 _safe_str。

成本控制（第 1 道闸门）：`chat()` / `chat_stream()` 会带上 `max_tokens`
（默认取 shared.limits.default_max_tokens()，即 LLM_MAX_TOKENS，调用方可覆盖）
与 `reasoning_effort`（默认取 shared.limits.llm_default_reasoning_effort()，
即 LLM_DEFAULT_REASONING_EFFORT，调用方可覆盖）——
**两个参数是一起兜底的**：思考模型的 max_tokens 同时卡住思考与正文，
只兜额度不兜档位照样会被思考吃光。并读取 `choices[0].finish_reason == "length"`
识别「输出被截断」—— 截断不是正常答案，在 shared.limits 里打个标记让
react_agent 按解析失败处理。

换模型自适配（reasoning_effort）：思考档位参数不是所有 OpenAI 兼容网关都认
（DeepSeek / Moonshot / OpenAI 等多数会直接 400 invalid_request_error）。
这里做了一个自适应兜底 —— 一旦报错文本同时命中「reasoning_effort」和
「不支持 / unknown / invalid」字样，就自动去掉该参数重试一次，并把
「这个模型不吃 reasoning_effort」写进进程内缓存，本进程后续请求不再注入。
调用方（react_agent 等）无需感知，换模型只改 .env 即可。
"""
import json
import os
import sys
import time

import httpx
from shared.config import ZHIPU_API_KEY, ZHIPU_BASE_URL, ZHIPU_CHAT_MODEL
from shared.errors import ConfigError, APIError
from shared import limits
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


# 进程内缓存：已确认「不吃 reasoning_effort」的模型（键 = 小写模型名）。
# 命中后 _build_payload 直接不注入该参数，避免每次请求都先吃一个 400 再重试。
_EFFORT_UNSUPPORTED_MODELS = set()

# 「参数被拒」的特征词：必须先命中 reasoning_effort，再配以下任意一个。
# 中英文都收，网关的措辞不完全统一（智谱/部分代理会回中文）。
_EFFORT_REJECT_HINTS = (
    "not supported", "unsupported", "unknown", "invalid",
    "不支持", "未知", "无效",
)


def _effort_key(model) -> str:
    """缓存键：模型名（大小写不敏感），空模型名不缓存。"""
    return (model or "").strip().lower()


def reasoning_effort_unsupported(model: str = None) -> bool:
    """该模型是否已被判定「不认识 reasoning_effort」（本进程内）。"""
    return _effort_key(model) in _EFFORT_UNSUPPORTED_MODELS


def is_reasoning_effort_rejected(e: Exception) -> bool:
    """判断这次失败是不是「网关不认 reasoning_effort」导致的。

    必须**同时**满足：报错文本提到 reasoning_effort，且带不支持/未知/非法字样。
    这样不会把模型不存在、额度不足、网络超时之类的错误误判成参数不兼容。
    """
    text = f"{_safe_str(e)} {_error_detail(e)}".lower()
    if "reasoning_effort" not in text:
        return False
    return any(hint in text for hint in _EFFORT_REJECT_HINTS)


def _note_effort_unsupported(model: str, e: Exception) -> None:
    """降级时记一行 INFO 日志，并把该模型写进进程内缓存。"""
    key = _effort_key(model)
    if key:
        _EFFORT_UNSUPPORTED_MODELS.add(key)
    detail = (_error_detail(e) or _safe_str(e)).replace("\n", " ")
    _safe_print(f"[llm] INFO 模型 {model} 不支持 reasoning_effort"
                f"（{detail[:120]}），已自动去掉该参数重试；"
                f"本进程后续对它的请求不再注入")


def _resolve_max_tokens(max_tokens) -> int:
    """定出本次调用的 max_tokens：显式参数优先，否则取闸门默认值。

    返回 0（或更小）表示**不注入**该字段 —— 这样总开关关闭时
    payload 与改造前逐字节一致。
    """
    if max_tokens is None:
        return limits.default_max_tokens()
    try:
        return max(0, int(max_tokens))
    except (TypeError, ValueError):
        return limits.default_max_tokens()


def _resolve_reasoning_effort(reasoning_effort) -> str:
    """定出本次调用的思考档位：显式参数优先，否则取**全局**默认档位。

    与 `_resolve_max_tokens` 同构，因为它们是同一个 bug 的两个变量：
    思考模型下 `max_tokens` 同时卡住思考（`reasoning_content`）与正文 ——
    只兜额度、不兜档位，思考照样能把额度吃光、正文变空。

    与 max_tokens 的唯一区别：**显式传空串 = 调用方主动要求不注入**，
    此时不回退到全局默认（保留「这个调用方就要默认档」的表达能力），
    只有传 `None`（完全没表态）才回退。返回空串 = 不注入该字段。
    """
    if reasoning_effort is None:
        return limits.llm_default_reasoning_effort()
    return str(reasoning_effort).strip().lower()


def _build_payload(messages: list, model: str, stream: bool, max_tokens,
                   reasoning_effort=None) -> dict:
    """拼请求体：max_tokens 为 0 时不带这个键（保持旧 payload 形状）。

    reasoning_effort: 思考档位（low / high / max）。glm-5.3-flash 是思考模型，
        `max_tokens` 同时卡住思考（reasoning_content）与正文，抽取类任务用 low
        才不会被思考吃光额度。不传取全局默认 `LLM_DEFAULT_REASONING_EFFORT`
        （默认 low）；空值不注入 —— payload 形状与老版本一致。

    换模型自适配：该模型已被判定不吃 reasoning_effort 时（见
    reasoning_effort_unsupported），这里直接不注入，省掉一次注定失败的请求。
    """
    payload = {"model": model, "messages": messages, "stream": stream}
    limit = _resolve_max_tokens(max_tokens)
    if limit > 0:
        payload["max_tokens"] = limit
    effort = _resolve_reasoning_effort(reasoning_effort)
    if effort and not reasoning_effort_unsupported(model):
        payload["reasoning_effort"] = effort
    return payload


def _usage_field(usage, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return usage.get(name) or 0
    return getattr(usage, name, 0) or 0


def _record_usage(model: str = "unknown", source: str = "unknown",
                  usage=None, request_id=None) -> None:
    """把一次调用的 token 用量记进 token_usage 表，并累加本次请求的预算。

    - usage 为 None（流式接口通常不给 usage）时记 0，并给 source 打上
      ":stream_no_usage" 后缀，避免把「没拿到」当成「真的用了 0」；
    - user_id 不在这里传：token_tracker.record_usage 自己读当前用户上下文；
    - 单次预算累加放在这里（见 shared/limits.add_run_tokens）：这是**唯一的**
      用量入口，累加一次就够，react_agent 那边只做判断；
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
        prompt = _usage_field(usage, "prompt_tokens")
        completion = _usage_field(usage, "completion_tokens")
        token_tracker.record_usage(model, prompt, completion, source, request_id)
        limits.add_run_tokens(prompt + completion)
    except Exception as e:                      # noqa: BLE001 - 记账失败不影响主流程
        _safe_print(f"[token] 用量记录失败（忽略）：{_safe_str(e)}")


def _note_finish_reason(finish_reason) -> bool:
    """finish_reason == "length" 说明输出被 max_tokens 掐断了。

    只打一行日志 + 打标记：真正的处理（当解析失败重来）在 react_agent。
    返回「是否截断」，调用方据此在正文为空时给出更准确的说明 ——
    思考模型被截断时 content 是空串，只报 JSON 解析错误会把人带偏。
    """
    if finish_reason == "length":
        limits.mark_truncated()
        _safe_print("[llm] 输出触顶（finish_reason=length），本轮回答被 max_tokens 截断")
        return True
    return False


def _warn_if_empty_truncated(truncated: bool, content) -> None:
    """触顶且正文为空：额度全被思考（reasoning_content）吃掉了。"""
    if truncated and not (content or "").strip():
        _safe_print("[llm] 输出触顶且正文为空：max_tokens 被思考（reasoning_content）吃光，"
                    "请调大额度，或把 reasoning_effort 调低（如 low）")


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


def chat(messages: list, model: str = None, retries: int = 3, source: str = "unknown",
         max_tokens: int = None, reasoning_effort: str = None) -> str:
    """非流式调用，失败重试。

    source: 调用方标记（如 "react_agent"），用于 token 用量按来源聚合，默认 "unknown"。
    max_tokens: 单次输出上限；不传取 LLM_MAX_TOKENS（**全局默认 4096**）。
        输出被截断时 finish_reason 会是 "length"，这里打标记、react_agent 当解析
        失败处理（截断的 JSON 静默变成「格式错误」非常难排查）。
    reasoning_effort: 思考档位（low / high / max），不传取全局默认
        LLM_DEFAULT_REASONING_EFFORT（默认 low）；显式传空串 = 不注入。
        思考模型下 max_tokens **同时**卡思考与正文，长思考会把额度吃光、正文变空 ——
        额度和档位必须一起兜底，只调其中一个都治不住。
    """
    model = model or ZHIPU_CHAT_MODEL
    headers = _headers()                    # 顺便校验 Key，缺了直接抛 ConfigError
    payload = _build_payload(messages, model, False, max_tokens, reasoning_effort)
    last_err = None
    downgraded = False                      # 参数退让只做一次

    _log_proxy_diag()

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            # httpx 用 json= 自己序列化 UTF-8 body，也自己设 Content-Type
            # trust_env=False：不读代理等环境变量，避免它们参与请求编码
            with httpx.Client(timeout=TIMEOUT, trust_env=False) as client:
                resp = client.post(CHAT_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            truncated = _note_finish_reason(
                choices[0].get("finish_reason") if choices else None)
            _record_usage(model=model, source=source,
                          usage=data.get("usage"), request_id=data.get("id"))
            content = data["choices"][0]["message"]["content"]
            _warn_if_empty_truncated(truncated, content)
            return content
        except ConfigError:
            raise
        except Exception as e:
            # 换模型自适配：网关不认 reasoning_effort → 去掉参数立刻重试一次。
            # 这一次不计进重试预算（预算留给真正的网络/服务端错误），也不 sleep。
            if (not downgraded and "reasoning_effort" in payload
                    and is_reasoning_effort_rejected(e)):
                downgraded = True
                _note_effort_unsupported(model, e)
                payload.pop("reasoning_effort", None)
                attempt -= 1
                continue
            last_err = e
            _log_failure(attempt, e)
            if attempt < retries:
                wait = attempt * 2
                _safe_print(f"等待 {wait} 秒后重试...")
                time.sleep(wait)

    raise APIError(f"API 调用失败，已重试 {retries} 次：{_safe_str(last_err)}")


def chat_stream(messages: list, model: str = None, source: str = "unknown",
                max_tokens: int = None, reasoning_effort: str = None):
    """流式调用，逐字返回（httpx 按行读 SSE）。

    source: 调用方标记，用于 token 用量按来源聚合，默认 "unknown"。
    max_tokens: 同 chat()，不传取 LLM_MAX_TOKENS。
    网关若在收尾 chunk 里带 usage 就记，没带则记 0 并标记 stream_no_usage。
    记账发生在生成器结束之后，调用方 break / 抛异常同样会落一条。
    换模型自适配同 chat()：网关不认 reasoning_effort 就自动去掉重试一次
    （只在还没吐出任何内容时降级，避免同一段正文被发两遍）。
    """
    model = model or ZHIPU_CHAT_MODEL
    headers = _headers()
    payload = _build_payload(messages, model, True, max_tokens, reasoning_effort)

    usage = None
    request_id = None
    downgraded = False                      # 参数退让只做一次
    yielded_any = False
    try:
        while True:
            try:
                # trust_env=False：同 chat()，不看代理环境变量
                with httpx.Client(timeout=TIMEOUT, trust_env=False) as client:
                    with client.stream("POST", CHAT_URL, json=payload,
                                       headers=headers) as resp:
                        resp.raise_for_status()
                        try:
                            # SSE 响应头常不带 charset，显式按 UTF-8 解码，避免中文乱码
                            resp.encoding = "utf-8"
                        except Exception:        # noqa: BLE001 - 赋值失败也无妨
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
                            if not choices:     # 带 usage 的收尾 chunk 可能没有 choices
                                continue
                            if choices[0].get("finish_reason"):
                                _note_finish_reason(choices[0].get("finish_reason"))
                            content = (choices[0].get("delta") or {}).get("content")
                            if content:
                                yielded_any = True
                                yield content
                return
            except Exception as e:              # noqa: BLE001 - 只处理参数退让，其余原样抛
                if (downgraded or yielded_any or "reasoning_effort" not in payload
                        or not is_reasoning_effort_rejected(e)):
                    raise
                downgraded = True
                _note_effort_unsupported(model, e)
                payload.pop("reasoning_effort", None)
                usage = None
                request_id = None
    finally:
        _record_usage(model=model, source=source, usage=usage, request_id=request_id)
