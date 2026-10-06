"""换模型自适配单测：网关不认 reasoning_effort 时自动降级 + 进程内缓存。

不需要真网、不需要 pytest，直接跑：

    python agent/tests/test_llm_client_compat.py

覆盖：
1. 第一次请求带 reasoning_effort 被 400 拒 → 自动去掉参数重试一次，调用成功；
2. 降级会写进进程内缓存 → 同一模型第二次调用**直接不注入**，不再吃 400；
3. 降级那一次不占重试预算（retries=1 也能成功）；
4. 不相干的报错（模型不存在 / 无额度）不会被误判成参数不兼容；
5. 网关接受 reasoning_effort 时（智谱 glm-4.5-air）一切照旧，不降级；
6. 流式 chat_stream 同样能降级，且不会把正文重复吐两遍；
7. reasoning_effort 为空时 payload 形状与改造前逐字一致。
"""
import io
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 和 test_limits.py 同一套路：环境变量必须在 import shared.* 之前钉死，
# 否则 shared.config 的 load_dotenv 会把手动的临时路径覆盖回去。
_TMP_DIR = Path(os.environ.get(
    "LLM_COMPAT_TEST_DIR", str(Path(__file__).resolve().parent / "_tmp_llm_compat")))
_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TOKEN_DB_PATH"] = str(_TMP_DIR / f"tokens_{os.getpid()}.db")
os.environ.setdefault("ZHIPU_API_KEY", "test-key")
os.environ["RATE_LIMIT_ENABLED"] = "true"

import httpx                                        # noqa: E402
from shared import llm_client as LC                 # noqa: E402
from shared.errors import APIError                  # noqa: E402

PASS = []
FAIL = []

# 典型的「参数不兼容」400：OpenAI 风格 + 中文网关风格各来一份。
REJECT_BODY = json.dumps({
    "error": {
        "message": "Unrecognized request argument supplied: reasoning_effort",
        "type": "invalid_request_error",
        "code": "unknown_parameter",
    }
}, ensure_ascii=False)

REJECT_BODY_CN = json.dumps({
    "error": {"message": "不支持的参数 reasoning_effort", "code": "invalid_parameter"}
}, ensure_ascii=False)

# 典型的「跟参数无关」的错误：这段绝不能触发降级。
OTHER_BODY = json.dumps({
    "error": {"message": "model not found or no permission", "code": "model_not_found"}
}, ensure_ascii=False)


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f" → {detail}" if detail else ""))
    except Exception as exc:                        # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {label} → {type(exc).__name__}: {exc}")


def section(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def _fail(msg):
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# 假 httpx.Client：带 reasoning_effort 的请求回 400，其余回正常响应
# ---------------------------------------------------------------------------

def _ok_body(content="好的"):
    return {
        "id": "req-ok",
        "choices": [{"message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


class _StatusResponse:
    """网关的 4xx：raise_for_status() 抛真的 httpx.HTTPStatusError。"""

    def __init__(self, status, body):
        self.status_code = status
        self.text = body

    def raise_for_status(self):
        req = httpx.Request("POST", LC.CHAT_URL)
        resp = httpx.Response(self.status_code, request=req, text=self.text)
        raise httpx.HTTPStatusError(
            f"Client error '{self.status_code}' for url '{LC.CHAT_URL}'",
            request=req, response=resp)

    def json(self):
        return json.loads(self.text)


class _OkResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _StreamResponse:
    """极简 SSE：两个 delta + 一个带 usage 的收尾 chunk。"""

    encoding = "utf-8"

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield ('data: {"id":"req-s","choices":[{"delta":{"content":"你"},'
               '"finish_reason":null}]}')
        yield ('data: {"id":"req-s","choices":[{"delta":{"content":"好"},'
               '"finish_reason":"stop"}],'
               '"usage":{"prompt_tokens":3,"completion_tokens":2}}')
        yield "data: [DONE]"


class _FakeClient:
    calls = []                  # 每次请求的 payload
    reject_effort = True        # True：带 reasoning_effort 就回 400
    reject_body = REJECT_BODY
    ok_payload = None

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _route(self, payload):
        if _FakeClient.reject_effort and "reasoning_effort" in (payload or {}):
            return _StatusResponse(400, _FakeClient.reject_body)
        return _OkResponse(_FakeClient.ok_payload or _ok_body())

    def post(self, url, json=None, headers=None):
        # 必须存浅拷贝：llm_client 降级时会原地 pop 掉 reasoning_effort，
        # 直接存引用会把「历史请求」一起改掉，测出来的东西全是假的。
        _FakeClient.calls.append(dict(json) if json else json)
        return self._route(json)

    @contextmanager
    def stream(self, method, url, json=None, headers=None):
        _FakeClient.calls.append(dict(json) if json else json)
        if _FakeClient.reject_effort and "reasoning_effort" in (json or {}):
            yield _StatusResponse(400, _FakeClient.reject_body)
        else:
            yield _StreamResponse()


@contextmanager
def _fake_http(reject_effort=True, body=REJECT_BODY):
    """装好假客户端 + 清空进程内缓存，退出时还原。"""
    real_client = LC.httpx.Client
    _FakeClient.calls = []
    _FakeClient.reject_effort = reject_effort
    _FakeClient.reject_body = body
    _FakeClient.ok_payload = _ok_body()
    LC.httpx.Client = _FakeClient
    LC._EFFORT_UNSUPPORTED_MODELS.clear()
    try:
        yield _FakeClient
    finally:
        LC.httpx.Client = real_client
        LC._EFFORT_UNSUPPORTED_MODELS.clear()


def _has_effort(call):
    return "reasoning_effort" in (call or {})


# ---------------------------------------------------------------------------
section("1. 降级 + 缓存")
# ---------------------------------------------------------------------------

def _downgrade_and_cache():
    """400 拒参 → 去掉参数重试一次成功 → 缓存后第二次不再注入。"""
    with _fake_http() as fake:
        buf = io.StringIO()
        with redirect_stdout(buf):
            first = LC.chat([{"role": "user", "content": "hi"}], model="deepseek-chat",
                            reasoning_effort="low", source="test")
        log = buf.getvalue()
        if first != "好的":
            _fail(f"降级后没拿到正文：{first!r}")
        if len(fake.calls) != 2:
            _fail(f"该是「400 + 去掉参数重试」共 2 次请求，实际 {len(fake.calls)}")
        if not _has_effort(fake.calls[0]):
            _fail("第一次请求就该带 reasoning_effort（默认行为不能变）")
        if _has_effort(fake.calls[1]):
            _fail("重试必须去掉 reasoning_effort")
        if fake.calls[0].get("reasoning_effort") != "low":
            _fail(f"档位被改掉了：{fake.calls[0].get('reasoning_effort')}")
        if "[llm] INFO" not in log or "reasoning_effort" not in log:
            _fail(f"该打一行 INFO 日志，实际输出：{log!r}")
        if not LC.reasoning_effort_unsupported("deepseek-chat"):
            _fail("降级后没写进进程内缓存")
        # 第二次：同一个模型，应直接不带参数，一次成功
        with redirect_stdout(io.StringIO()):
            second = LC.chat([{"role": "user", "content": "hi"}], model="deepseek-chat",
                             reasoning_effort="low", source="test")
        if second != "好的":
            _fail(f"第二次调用失败：{second!r}")
        if len(fake.calls) != 3:
            _fail(f"第二次该只发 1 个请求，实际新增 {len(fake.calls) - 2}")
        if _has_effort(fake.calls[2]):
            _fail("缓存命中后仍注入了 reasoning_effort")
    return "400 → 去参重试成功 → 缓存命中（第 2 次 1 个请求成形）"


def _downgrade_cn_message():
    """中文网关的「不支持的参数」也要能认出来。"""
    with _fake_http(body=REJECT_BODY_CN) as fake:
        with redirect_stdout(io.StringIO()):
            out = LC.chat([{"role": "user", "content": "hi"}], model="glm-alt",
                          reasoning_effort="low", source="test")
        if out != "好的" or len(fake.calls) != 2:
            _fail(f"中文报错没触发降级：out={out!r} calls={len(fake.calls)}")
    return "中文「不支持的参数」同样命中"


def _downgrade_free_of_retry_budget():
    """降级那次不算重试预算：retries=1 也得能成功。"""
    with _fake_http() as fake:
        with redirect_stdout(io.StringIO()):
            out = LC.chat([{"role": "user", "content": "hi"}], model="moonshot-v1-8k",
                          reasoning_effort="low", retries=1, source="test")
        if out != "好的":
            _fail(f"retries=1 时没能降级成功：{out!r}")
        if len(fake.calls) != 2:
            _fail(f"请求次数不对：{len(fake.calls)}")
    return "retries=1 仍能降级成功（请求 2 次）"


# ---------------------------------------------------------------------------
section("2. 该降级的降，不该降的不碰")
# ---------------------------------------------------------------------------

def _unrelated_error_not_mistaken():
    """模型不存在之类的报错不能被误判成参数不兼容。"""
    with _fake_http(body=OTHER_BODY) as fake:
        try:
            with redirect_stdout(io.StringIO()):
                LC.chat([{"role": "user", "content": "hi"}], model="ghost-model",
                        reasoning_effort="low", retries=2, source="test")
        except APIError:
            pass
        else:
            _fail("模型不存在时应当抛 APIError")
        if fake.calls and not all(_has_effort(c) for c in fake.calls):
            _fail("不相干的错误竟然触发了去参重试")
        if LC.reasoning_effort_unsupported("ghost-model"):
            _fail("不相干的错误被写进了缓存")
    return "模型不存在：照常重试，缓存不受污染"


def _accepted_model_keeps_effort():
    """网关认这个参数（智谱 glm-4.5-air）→ 不降级、不改 payload。"""
    with _fake_http(reject_effort=False) as fake:
        with redirect_stdout(io.StringIO()):
            out = LC.chat([{"role": "user", "content": "hi"}], model="glm-4.5-air",
                          reasoning_effort="low", source="test")
        if out != "好的":
            _fail(f"正常模型调用失败：{out!r}")
        if len(fake.calls) != 1:
            _fail(f"不该多发请求：{len(fake.calls)}")
        if fake.calls[0].get("reasoning_effort") != "low":
            _fail(f"档位丢了：{fake.calls[0]}")
        if LC.reasoning_effort_unsupported("glm-4.5-air"):
            _fail("正常模型被误记进缓存")
    return "glm-4.5-air：1 次请求、档位保留、无降级"


def _build_payload_known_unsupported():
    """已进缓存的模型，_build_payload 直接不注入该键。"""
    payload = LC._build_payload([{"role": "user", "content": "x"}], "deepseek-chat",
                                False, 128, "low")
    LC._EFFORT_UNSUPPORTED_MODELS.add("deepseek-chat")
    try:
        payload = LC._build_payload([{"role": "user", "content": "x"}], "deepseek-chat",
                                    False, 128, "low")
    finally:
        LC._EFFORT_UNSUPPORTED_MODELS.clear()
    if "reasoning_effort" in payload:
        _fail(f"缓存命中后仍注入：{sorted(payload)}")
    if payload.get("max_tokens") != 128:
        _fail("max_tokens 被连累了")
    return "缓存命中：payload 无 reasoning_effort，max_tokens 保留"


def _empty_effort_payload_unchanged():
    """reasoning_effort 为空 → payload 形状与改造前逐字一致。"""
    messages = [{"role": "user", "content": "你好"}]
    payload = LC._build_payload(messages, "any-model", False, 0, "")
    if payload != {"model": "any-model", "messages": messages, "stream": False}:
        _fail(f"payload 形状变了：{sorted(payload)}")
    return f"payload keys={sorted(payload)}"


def _reject_detector():
    """识别函数本身：关键词不全时不认。"""
    class _E400(httpx.HTTPStatusError):
        pass

    req = httpx.Request("POST", LC.CHAT_URL)

    def _mk(body):
        resp = httpx.Response(400, request=req, text=body)
        return httpx.HTTPStatusError("400", request=req, response=resp)

    cases = [
        (REJECT_BODY, True),
        (REJECT_BODY_CN, True),
        (OTHER_BODY, False),
        ('{"error":{"message":"reasoning_effort is fine"}}', False),
        ('{"error":{"message":"invalid request"}}', False),
    ]
    for body, want in cases:
        got = LC.is_reasoning_effort_rejected(_mk(body))
        if got != want:
            _fail(f"误判：{body[:60]} → {got}，期望 {want}")
    return f"{len(cases)} 个样本判定全部正确"


# ---------------------------------------------------------------------------
section("3. 流式调用")
# ---------------------------------------------------------------------------

def _stream_downgrade():
    with _fake_http() as fake:
        with redirect_stdout(io.StringIO()):
            pieces = list(LC.chat_stream([{"role": "user", "content": "hi"}],
                                         model="deepseek-chat", reasoning_effort="low",
                                         source="test"))
        if "".join(pieces) != "你好":
            _fail(f"流式内容不对：{pieces!r}")
        if len(pieces) != 2:
            _fail(f"正文被重复吐了：{pieces!r}")
        if len(fake.calls) != 2:
            _fail(f"该降级重连一次，实际请求 {len(fake.calls)}")
        if not _has_effort(fake.calls[0]) or _has_effort(fake.calls[1]):
            _fail(f"两次请求的参数不对：{[sorted(c) for c in fake.calls]}")
        if not LC.reasoning_effort_unsupported("deepseek-chat"):
            _fail("流式降级没写缓存")
    return "流式：降级重连一次，正文只吐一遍"


# ---------------------------------------------------------------------------
section("4. 跑用例")
# ---------------------------------------------------------------------------

check("400 拒参 → 去掉参数重试 → 进程内缓存", _downgrade_and_cache)
check("中文「不支持的参数」同样识别", _downgrade_cn_message)
check("降级不占重试预算", _downgrade_free_of_retry_budget)
check("不相干报错不误判、不污染缓存", _unrelated_error_not_mistaken)
check("支持的模型照旧（智谱不降级）", _accepted_model_keeps_effort)
check("缓存命中后 payload 不注入", _build_payload_known_unsupported)
check("effort 为空时 payload 逐字不变", _empty_effort_payload_unchanged)
check("识别函数的关键词判据", _reject_detector)
check("chat_stream 降级不重复正文", _stream_downgrade)

print()
print("=" * 74)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
for label, detail in FAIL:
    print(f"  [FAIL] {label} → {detail}")
print("=" * 74)
sys.exit(1 if FAIL else 0)
