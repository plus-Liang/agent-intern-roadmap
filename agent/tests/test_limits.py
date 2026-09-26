# -*- coding: utf-8 -*-
"""限流 + 成本控制离线回归测试。

跑法：
    python agent/tests/test_limits.py

全程离线：不联网、不调真实 LLM、不碰真实用量库。
- 用量库指向仓库内 `agent/tests/_tmp_limits/token_usage_<pid>.db`
  （用仓库内临时目录而不是 %TEMP%：受限沙箱里进程对 %TEMP% 常常没有写权限）；
- LLM 调用用假的 `httpx.Client` 顶替，只回一个可编程的响应体；
- 频率闸门用**显式单调时钟**，测试不真的 sleep。

覆盖（对应实施方案 Q4 的测试清单）：
1. 令牌桶：补充 / 超限 / 恢复 / 多用户互不影响；
2. 日额度边界：`>=` 判定、跨天归零、按用户隔离、全局口径；
3. 单次预算熔断：累加器 + react_agent 降级收尾（不崩）；
4. `max_tokens` 透传 + 截断识别 + react_agent 不把截断当正常答案；
5. `RATE_LIMIT_ENABLED=false` 时四道闸门全部关闭（payload 与改造前逐字一致）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 临时目录：默认仓库内 agent/tests/_tmp_limits/，可用 LIMITS_TEST_DIR 覆盖
_TMP_DIR = Path(os.getenv("LIMITS_TEST_DIR",
                          str(Path(__file__).resolve().parent / "_tmp_limits")))
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_TMP_DB = _TMP_DIR / f"token_usage_{os.getpid()}.db"
# 必须在 import shared.token_tracker **之前**设好：DB_PATH 在模块导入时固化
os.environ["TOKEN_DB_PATH"] = str(_TMP_DB)
os.environ.setdefault("ZHIPU_API_KEY", "test-key")

# 其余落库路径也全部重定向到临时目录 —— 第 8 节要 import agent.app，
# 而它在导入时就会 init 一堆库；不重定向就会写真实 agent/data。
os.environ["APP_DB_PATH"] = str(_TMP_DIR / f"app_{os.getpid()}.db")
os.environ["CHAT_HISTORY_DB"] = str(_TMP_DIR / f"chat_history_{os.getpid()}.db")
os.environ["USER_PROFILE_DIR"] = str(_TMP_DIR / "profiles")
os.environ["RESUME_ROOT"] = str(_TMP_DIR / "resumes")

# 阈值显式钉住（与 .env.example 的默认值一致），避免宿主机 .env 干扰
os.environ["RATE_LIMIT_ENABLED"] = "true"
os.environ["RATE_PER_MIN"] = "6"
os.environ["RATE_BURST"] = "3"
os.environ["LLM_MAX_TOKENS"] = "1024"
os.environ["RUN_TOKEN_BUDGET"] = "30000"
os.environ["DAILY_TOKENS_PER_USER"] = "200000"
os.environ["GLOBAL_DAILY_TOKENS"] = "2000000"
os.environ["MAX_CONCURRENCY"] = "3"

from agent import react_agent as RA                    # noqa: E402
from shared import limits as L                         # noqa: E402
from shared import llm_client as LC                    # noqa: E402
from shared import token_tracker as T                  # noqa: E402
from shared.user_context import user_scope             # noqa: E402

# token_tracker.DB_PATH 在导入时按 TOKEN_DB_PATH 固化；再显式指一次，
# 保证任何导入顺序下都不会落到真实 logs/token_usage.db
T.DB_PATH = _TMP_DB

PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f" → {detail}" if detail else ""))
    except Exception as exc:                                   # noqa: BLE001
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
# 假 httpx.Client：记录 payload，回一个预设响应
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    calls: list = []
    response: dict = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None):
        _FakeClient.calls.append({"url": url, "payload": json, "headers": headers})
        return _FakeResponse(dict(_FakeClient.response))


@contextmanager
def _fake_http(response: dict):
    real = LC.httpx.Client
    _FakeClient.calls = []
    _FakeClient.response = response
    LC.httpx.Client = _FakeClient
    try:
        yield _FakeClient
    finally:
        LC.httpx.Client = real


def _resp(content='{"thought":"t","final_answer":"ok"}', finish_reason="stop",
          usage=None):
    return {
        "id": "req-1",
        "choices": [{"message": {"role": "assistant", "content": content},
                     "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _insert_raw(user_id, tokens, timestamp):
    """直接写一行用量（用来造跨天 / 历史数据）。"""
    T.init_token_db()
    conn = T._get_conn()
    conn.execute(
        """INSERT INTO token_usage
        (timestamp, model, prompt_tokens, completion_tokens, total_tokens,
         source, request_id, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (timestamp, "test-model", tokens, 0, tokens, "test", None, user_id),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
section("1. 库隔离与存量迁移")
# ---------------------------------------------------------------------------


def _t_db_isolated():
    if str(T.DB_PATH) != str(_TMP_DB):
        _fail(f"用量库没指向临时库：{T.DB_PATH}")
    if not str(T.DB_PATH).endswith(f"token_usage_{os.getpid()}.db"):
        _fail("临时库文件名不符")
    return str(T.DB_PATH.name)


def _legacy_migration():
    """老库（没有 user_id 列）→ init 后补列 + 补索引，存量行归 local。"""
    legacy = _TMP_DIR / f"legacy_{os.getpid()}.db"
    if legacy.exists():
        legacy.unlink()
    conn = sqlite3.connect(str(legacy))
    conn.execute("""
        CREATE TABLE token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, model TEXT, prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
            source TEXT, request_id TEXT)
    """)
    conn.execute(
        "INSERT INTO token_usage (timestamp, model, total_tokens, source) "
        "VALUES ('2026-01-01 00:00:00', 'm', 7, 's')"
    )
    conn.commit()
    conn.close()

    old = T.DB_PATH
    T.DB_PATH = legacy
    try:
        T.init_token_db()
        conn = T._get_conn()
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(token_usage)")}
        indexes = {r["name"] for r in conn.execute("PRAGMA index_list(token_usage)")}
        row = conn.execute("SELECT user_id FROM token_usage").fetchone()
        conn.close()
        if "user_id" not in cols:
            _fail("迁移后仍没有 user_id 列")
        if row["user_id"] != "local":
            _fail(f"存量行没归到 local：{row['user_id']}")
        if "idx_token_usage_user_time" not in indexes:
            _fail("缺少 (user_id, timestamp) 复合索引")
        return "补列 + 归 local + 复合索引齐备"
    finally:
        T.DB_PATH = old


check("用量库指向仓库内临时库", _t_db_isolated)
check("老库迁移：补 user_id 列 + 归 local + 建索引", _legacy_migration)

# ---------------------------------------------------------------------------
section("2. 用量归属（user_id 维度）")
# ---------------------------------------------------------------------------


def _record_default_user():
    uid = "rec_default"
    with user_scope(uid):
        T.record_usage("m", 100, 50, source="test")
    total = T.usage_today(uid)
    if total != 150:
        _fail(f"默认归属当前用户失败：{total}")
    return f"{uid} → {total}"


def _record_explicit_user():
    T.record_usage("m", 1, 2, source="test", user_id="rec_explicit")
    if T.usage_today("rec_explicit") != 3:
        _fail("显式 user_id 没生效")
    return "显式 user_id 优先于上下文"


def _usage_today_isolated():
    T.record_usage("m", 10, 0, source="test", user_id="iso_a")
    T.record_usage("m", 20, 0, source="test", user_id="iso_b")
    if T.usage_today("iso_a") != 10 or T.usage_today("iso_b") != 20:
        _fail(f"用户之间串了：a={T.usage_today('iso_a')} b={T.usage_today('iso_b')}")
    return "a=10 b=20"


def _usage_today_cross_day():
    """昨天的行不算今天：日额度必须每天归零。"""
    uid = "cross_day"
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    _insert_raw(uid, 999, yesterday)
    _insert_raw(uid, 5, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    today = T.usage_today(uid)
    if today != 5:
        _fail(f"跨天没归零：{today}（应只算今天的 5）")
    return f"昨天 999 不计入，今天 {today}"


def _usage_today_global():
    before = T.usage_today(None)
    T.record_usage("m", 7, 0, source="test", user_id="glob_probe")
    after = T.usage_today(None)
    if after - before != 7:
        _fail(f"全局口径不对：{before} → {after}")
    return f"全局 {before} → {after}"


def _by_user_aggregate():
    data = T.query_usage(1)
    if "by_user" not in data:
        _fail("query_usage 没有 by_user")
    if not data["by_user"].get("iso_a"):
        _fail("by_user 里没有 iso_a")
    return f"by_user keys={len(data['by_user'])}"


check("record_usage 默认取当前用户（user_scope）", _record_default_user)
check("record_usage 显式 user_id 优先", _record_explicit_user)
check("usage_today 按用户隔离", _usage_today_isolated)
check("usage_today 跨天归零", _usage_today_cross_day)
check("usage_today(None) 是全局口径", _usage_today_global)
check("query_usage 增加 by_user", _by_user_aggregate)

# ---------------------------------------------------------------------------
section("3. 闸门 3：单用户频率（令牌桶 6/60s，桶容量 3）")
# ---------------------------------------------------------------------------


def _burst_three():
    L.reset_rate_limits()
    seq = [L.check_rate("burst", now=1000.0)[0] for _ in range(7)]
    if seq[:3] != [True, True, True]:
        _fail(f"前 3 次（桶容量）应当放行：{seq}")
    if seq[6] is not False:
        _fail(f"连发第 7 次应当被拒：{seq}")
    if any(seq[3:]):
        _fail(f"桶空之后应当全拒：{seq}")
    return f"7 连发 → {['通过' if v else '拒绝' for v in seq]}"


def _refill_recovery():
    L.reset_rate_limits()
    for _ in range(3):
        L.check_rate("refill", now=0.0)
    denied, retry = L.check_rate("refill", now=0.0)
    if denied:
        _fail("第 4 次不该放行")
    if not (9.0 < retry <= 10.5):
        _fail(f"6/分钟下建议等待应≈10s，实际 {retry}")
    allowed, _ = L.check_rate("refill", now=10.0)
    if not allowed:
        _fail("10 秒后（补满 1 个令牌）应当放行")
    return f"第 4 次被拒（等待 {retry:.1f}s），10s 后恢复"


def _users_independent():
    L.reset_rate_limits()
    for _ in range(3):
        L.check_rate("user_x", now=0.0)
    if L.check_rate("user_x", now=0.0)[0]:
        _fail("user_x 应当被限")
    if not L.check_rate("user_y", now=0.0)[0]:
        _fail("user_y 不该被 user_x 影响")
    return "user_x 被限不影响 user_y"


check("连发 7 次：第 7 次被拒（桶容量 3）", _burst_three)
check("令牌补充后恢复", _refill_recovery)
check("多用户互不影响", _users_independent)

# ---------------------------------------------------------------------------
section("4. 闸门 4：日额度判定 + 告警")
# ---------------------------------------------------------------------------


def _verdict_ok():
    if L.quota_verdict(10, 100, 10, 1000) != "ok":
        _fail("未超限应当 ok")
    return "ok"


def _verdict_user():
    if L.quota_verdict(100, 100, 0, 1000) != "user":
        _fail("等于上限也算超（>=）")
    if L.quota_verdict(101, 100, 0, 1000) != "user":
        _fail("超过上限应当 user")
    return "used >= user_limit → user"


def _verdict_global():
    if L.quota_verdict(10, 100, 1000, 1000) != "global":
        _fail("全局触顶应当 global")
    return "global_used >= global_limit → global"


def _verdict_unlimited():
    if L.quota_verdict(10 ** 9, 0, 10 ** 9, 0) != "ok":
        _fail("0 = 不限")
    return "0 表示不限"


def _quota_warning_wired():
    src = (REPO / "agent" / "app.py").read_text(encoding="utf-8")
    if "quota_exceeded" not in src:
        _fail("app.py 没有记 quota_exceeded 告警")
    if "token_tracker.usage_today" not in src:
        _fail("app.py 没有查 usage_today")
    if "limits.quota_verdict" not in src:
        _fail("app.py 没有用 quota_verdict")
    return "告警 + 日额度查询都已接线"


check("未超限 → ok", _verdict_ok)
check("单用户超限 → user（>= 判定）", _verdict_user)
check("全局超限 → global", _verdict_global)
check("限额 0 = 不限", _verdict_unlimited)
check("日额度触顶会记告警日志", _quota_warning_wired)

# ---------------------------------------------------------------------------
section("5. 闸门 2：单次预算熔断（降级，不拒绝）")
# ---------------------------------------------------------------------------


def _budget_accumulator():
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    L.start_run_budget(limit=100)
    L.add_run_tokens(60)
    if L.run_tokens_used() != 60:
        _fail(f"累加不对：{L.run_tokens_used()}")
    if L.run_budget_status()["exceeded"]:
        _fail("60/100 不该触顶")
    L.add_run_tokens(50)
    status = L.run_budget_status()
    if not status["exceeded"] or status["used"] != 110:
        _fail(f"110/100 应当触顶：{status}")
    L.reset_run_budget()
    return "60 → 110/100 触顶，reset 后清零"


def _budget_without_start():
    L.reset_run_budget()
    if L.add_run_tokens(999) != 0 or L.run_tokens_used() != 0:
        _fail("没开记账时 add 应当是 no-op")
    return "没记账时 no-op"


def _agent_budget_stop():
    """预算触顶时 react_agent **降级收尾**：返回话术、保留 steps、不抛异常。"""
    os.environ["RUN_TOKEN_BUDGET"] = "10"
    real = RA.chat

    def fake(messages, **kwargs):
        L.add_run_tokens(20)                    # 模拟 llm_client 记账
        return '{"thought":"t","action":"no_such_tool_xyz","action_input":{}}'

    RA.chat = fake
    try:
        result = RA.run("测试预算熔断", resume_data={}, verbose=False)
    finally:
        RA.chat = real
        os.environ["RUN_TOKEN_BUDGET"] = "30000"

    if result["answer"] != RA.BUDGET_STOP_ANSWER:
        _fail(f"应当降级收尾，实际：{result['answer']!r}")
    if not result["steps"] or result["steps"][-1]["type"] != "budget_stop":
        _fail(f"steps 里没有 budget_stop：{result['steps']}")
    if result["steps"][-1]["limit_tokens"] != 10:
        _fail("budget_stop 没带上限")
    return f"降级收尾，steps={len(result['steps'])} 条（未崩溃）"


check("预算累加器 / 触顶判定", _budget_accumulator)
check("未开记账时累加是 no-op", _budget_without_start)
check("react_agent 超限降级收尾（不崩）", _agent_budget_stop)

# ---------------------------------------------------------------------------
section("6. 闸门 1：max_tokens 透传 + 截断识别")
# ---------------------------------------------------------------------------


def _max_tokens_default():
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    L.start_run_budget(limit=100000)
    with _fake_http(_resp()) as fake:
        content = LC.chat([{"role": "user", "content": "hi"}], source="t")
    payload = fake.calls[0]["payload"]
    if payload.get("max_tokens") != 1024:
        _fail(f"默认没注入 1024：{payload.get('max_tokens')}")
    if content != '{"thought":"t","final_answer":"ok"}':
        _fail("返回值不对")
    if L.run_tokens_used() != 15:
        _fail(f"用量没累加进预算：{L.run_tokens_used()}")
    L.reset_run_budget()
    return "payload.max_tokens=1024，用量 15 已入预算"


def _max_tokens_override():
    with _fake_http(_resp()) as fake:
        LC.chat([{"role": "user", "content": "hi"}], source="t", max_tokens=64)
    if fake.calls[0]["payload"].get("max_tokens") != 64:
        _fail("调用方覆盖没生效")
    return "max_tokens=64"


def _truncation_marked():
    L.reset_truncated()
    with _fake_http(_resp(finish_reason="length")) as fake:
        LC.chat([{"role": "user", "content": "hi"}], source="t")
    if not L.consume_truncated():
        _fail("finish_reason=length 没被识别")
    if L.consume_truncated():
        _fail("截断标记应当读一次就复位")
    if not fake.calls:
        _fail("没有发出请求")
    return "length 被识别，且读一次即复位"


def _agent_truncation_not_answer():
    """react_agent 不把截断输出当正常答案：重来一轮，最终用第二轮的回答。"""
    os.environ["RUN_TOKEN_BUDGET"] = "0"        # 关闭预算，只测截断
    real = RA.chat
    calls = {"n": 0}

    def fake(messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            L.mark_truncated()                  # 模拟 llm_client 的识别
            return '{"thought":"t","final_answer":"不该被采纳的半截答案"}'
        return '{"thought":"t","final_answer":"真正的答案"}'

    RA.chat = fake
    try:
        result = RA.run("测试截断", resume_data={}, verbose=False)
    finally:
        RA.chat = real
        os.environ["RUN_TOKEN_BUDGET"] = "30000"

    if result["answer"] != "真正的答案":
        _fail(f"把截断当正常答案了：{result['answer']!r}")
    if calls["n"] != 2:
        _fail(f"应当在截断后重来一轮，实际调了 {calls['n']} 次")
    L.reset_truncated()
    return "截断轮被丢弃，重发请求后取到真答案"


check("chat() 默认注入 max_tokens=1024 并累加用量", _max_tokens_default)
check("调用方可覆盖 max_tokens", _max_tokens_override)
check("finish_reason=length 被识别为截断", _truncation_marked)
check("react_agent 不把截断当正常答案", _agent_truncation_not_answer)

# ---------------------------------------------------------------------------
section("7. RATE_LIMIT_ENABLED=false：四道闸门全关，行为与改造前逐字一致")
# ---------------------------------------------------------------------------


def _disabled_allows_all():
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    try:
        L.reset_rate_limits()
        seq = [L.check_rate("off_user", now=0.0)[0] for _ in range(10)]
        if not all(seq):
            _fail(f"开关关闭时不该拒绝：{seq}")
        if L.default_max_tokens() != 0:
            _fail("开关关闭时不该注入 max_tokens")
        L.start_run_budget(limit=1)
        L.add_run_tokens(100)
        if L.run_budget_status()["exceeded"]:
            _fail("开关关闭时不该熔断")
        L.reset_run_budget()
        return "连发 10 次全放行、不熔断、不注入 max_tokens"
    finally:
        os.environ["RATE_LIMIT_ENABLED"] = "true"


def _disabled_payload_unchanged():
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    try:
        messages = [{"role": "user", "content": "hi"}]
        with _fake_http(_resp()) as fake:
            LC.chat(list(messages), source="t")
        payload = dict(fake.calls[0]["payload"])
        payload.pop("model", None)
        if payload != {"messages": messages, "stream": False}:
            _fail(f"payload 与改造前不一致：{sorted(payload)}")
        return f"payload keys={sorted(payload)}"
    finally:
        os.environ["RATE_LIMIT_ENABLED"] = "true"


def _disabled_gate_short_circuit():
    src = (REPO / "agent" / "app.py").read_text(encoding="utf-8")
    marker = "if not limits.rate_limit_enabled():\n        return False"
    if marker not in src:
        _fail("入口闸门没有在总开关关闭时短路")
    return "app.py 闸门入口：开关关闭直接放行"


check("开关关闭：频率闸门恒放行、不熔断", _disabled_allows_all)
check("开关关闭：chat payload 与改造前逐字一致", _disabled_payload_unchanged)
check("开关关闭：入口闸门短路", _disabled_gate_short_circuit)

# ---------------------------------------------------------------------------
section("8. 入口闸门端到端（真跑 agent/app.py 的 _deny_if_throttled）")
# ---------------------------------------------------------------------------

import asyncio                                          # noqa: E402
import agent.app as APP                                 # noqa: E402


class _FakeCL:
    """只替掉 cl.Message：把回话内容记下来，不碰 Chainlit 会话。"""

    sent: list = []

    class Message:
        def __init__(self, content=""):
            self.content = content

        async def send(self):
            _FakeCL.sent.append(self.content)


def _run_gate(user_id):
    """在指定用户上下文里跑一次入口闸门，返回 (是否拒绝, 回话列表, 告警列表)。"""
    warnings = []
    real_cl, real_warn = APP.cl, APP.log_event
    APP.cl = _FakeCL
    APP.log_event = lambda trace, event, **kw: warnings.append((event, kw))
    _FakeCL.sent = []
    try:
        with user_scope(user_id):
            denied = asyncio.run(APP._deny_if_throttled("test"))
    finally:
        APP.cl, APP.log_event = real_cl, real_warn
    return denied, list(_FakeCL.sent), warnings


def _gate_allows_normal():
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    L.reset_rate_limits()
    denied, sent, _ = _run_gate("gate_ok")
    if denied:
        _fail(f"正常请求不该被拒：{sent}")
    return "放行，无回话"


def _gate_rate_rejects_and_no_agent():
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    L.reset_rate_limits()
    results = [_run_gate("gate_rate") for _ in range(7)]
    denied = [r[0] for r in results]
    if denied[6] is not True:
        _fail(f"连发第 7 次应当被拒：{denied}")
    if results[6][1] and "太频繁" not in results[6][1][0]:
        _fail(f"拒绝话术不对：{results[6][1]}")
    if not results[6][1]:
        _fail("被拒时应当回一句话")
    return f"7 连发 → {['过' if not d else '拒' for d in denied]}，被拒时已回话"


def _gate_daily_rejects_with_warning():
    """把单用户日额度调到 1，用量已经远超 → 拒绝 + 告警日志。"""
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    os.environ["DAILY_TOKENS_PER_USER"] = "1"
    L.reset_rate_limits()
    try:
        T.record_usage("m", 500, 0, source="test", user_id="gate_daily")
        denied, sent, warnings = _run_gate("gate_daily")
        if not denied:
            _fail("日额度超限应当被拒")
        if not warnings or warnings[0][0] != "quota_exceeded":
            _fail(f"没有记 quota_exceeded 告警：{warnings}")
        if warnings[0][1].get("scope") != "user":
            _fail(f"告警口径不对：{warnings[0][1]}")
        if not sent or "今日额度已用完" not in sent[0]:
            _fail(f"拒绝话术不对：{sent}")
        return f"拒绝（{warnings[0][1]['used_tokens']}/{warnings[0][1]['limit']}）+ 告警"
    finally:
        os.environ["DAILY_TOKENS_PER_USER"] = "200000"


def _gate_global_rejects():
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    os.environ["DAILY_TOKENS_PER_USER"] = "0"           # 只看全局
    os.environ["GLOBAL_DAILY_TOKENS"] = "1"
    L.reset_rate_limits()
    try:
        denied, sent, warnings = _run_gate("gate_global")
        if not denied or warnings[0][1].get("scope") != "global":
            _fail(f"全局额度触顶应当拒绝并告警：{denied} {warnings}")
        return "全局触顶 → 拒绝 + global 告警"
    finally:
        os.environ["DAILY_TOKENS_PER_USER"] = "200000"
        os.environ["GLOBAL_DAILY_TOKENS"] = "2000000"


def _gate_disabled_allows():
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    L.reset_rate_limits()
    try:
        results = [_run_gate("gate_off") for _ in range(7)]
        if any(r[0] for r in results):
            _fail("开关关闭时入口闸门不该拒绝任何请求")
        return "7 连发全放行（与改造前一致）"
    finally:
        os.environ["RATE_LIMIT_ENABLED"] = "true"


check("入口闸门：正常请求放行", _gate_allows_normal)
check("入口闸门：连发 7 次第 7 次被拒（不落库不调 Agent）", _gate_rate_rejects_and_no_agent)
check("入口闸门：日 token 超限被拒 + 告警", _gate_daily_rejects_with_warning)
check("入口闸门：全局日 token 超限被拒", _gate_global_rejects)
check("入口闸门：开关关闭时全放行", _gate_disabled_allows)

# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
for label, detail in FAIL:
    print(f"  [FAIL] {label} → {detail}")
print("=" * 74)
sys.exit(1 if FAIL else 0)
