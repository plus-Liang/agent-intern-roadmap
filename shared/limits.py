# -*- coding: utf-8 -*-
"""限流 + 成本控制：令牌桶、单次预算累加器、阈值读取。

这一层**刻意不碰任何 IO**（不读库、不发请求），只做纯逻辑，所以可以直接单测：
- `TokenBucket` / `check_rate()`：单用户频率（进程内内存桶）；
- `start_run_budget()` / `add_run_tokens()` / `run_budget_status()`：
  单次请求的 token 预算（ContextVar 累加器，随一次请求生命周期自动回收，无需清理）；
- `default_max_tokens()`：单次输出上限的默认值（真正的透传在 shared/llm_client.py）；
- 各阈值 getter：一律现读环境变量，改完立刻生效，测试也好覆盖。

日 token 额度**不在这一层**：它的真相源必须是 `token_usage` 表（重启要保留、
多进程要一致，内存计数器两条都不满足），查询走 `shared/token_tracker.usage_today()`，
比较逻辑在 `agent/app.py` 的入口闸门里。

总开关 `RATE_LIMIT_ENABLED` 默认 **true**（与 CHAT_AUTH_ENABLED 默认 false 不同：
限流是纯保护，不开认证时也应当生效）。置 false 时四道闸门**全部**关闭 ——
包括 chat() 默认注入的 max_tokens 与单次预算，保证「关掉开关 == 改造前的行为」。

多进程注意：进程内桶在上多 worker 后会退化成 N 倍限额（每个 worker 一份桶）。
上 worker 之前必须把频率桶换成 Redis / 落库滑动窗口（见部署待办）。
"""
from __future__ import annotations

import contextvars
import os
import threading
import time

from shared.user_context import normalize_user_id

# ========== 阈值（全部走环境变量，默认值保守；0 一律表示「不限」） ==========


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        print(f"[限流] {name}={raw!r} 非法，改用默认值 {default}")
        return default
    return max(minimum, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def rate_limit_enabled() -> bool:
    """限流 + 成本控制总开关，默认 true。false 时四道闸门全部短路。"""
    return _env_bool("RATE_LIMIT_ENABLED", True)


def rate_per_min() -> int:
    """单用户每分钟补充的请求数（令牌桶补充速率）。"""
    return _env_int("RATE_PER_MIN", 6, 1)


def rate_burst() -> int:
    """令牌桶容量：允许的小突发（连发几次不被拒）。"""
    return _env_int("RATE_BURST", 3, 1)


def llm_max_tokens() -> int:
    """单次 LLM 输出的 token 上限；0 = 不注入该参数（不限）。"""
    return _env_int("LLM_MAX_TOKENS", 1024, 0)


def default_max_tokens() -> int:
    """chat() 未显式传 max_tokens 时用的默认值；总开关关闭时为 0（不注入）。"""
    return llm_max_tokens() if rate_limit_enabled() else 0


def resume_max_tokens() -> int:
    """简历解析（结构化抽取）的单次输出上限，默认 4096。

    为什么要单独给：模型是**思考模型**，`max_tokens` 同时卡住思考
    （`reasoning_content`）与正文。实测这份 1631 字简历，默认 1024 档
    思考就要 3139 字符以上 → 1024 全被思考吃光、`content` 为空，
    前端看到的是 `Expecting value: line 1 column 1 (char 0)`。
    4096 仍不够（实测 finish_reason=length、content 空），8192 才出正文；
    配合 `reasoning_effort=low` 后 ~500 token 就够，4096 是留余量。
    总开关关闭时为 0（不注入），与改造前行为一致。
    """
    return _env_int("RESUME_LLM_MAX_TOKENS", 4096, 0) if rate_limit_enabled() else 0


def resume_reasoning_effort() -> str:
    """简历解析的思考档位（low / high / max；空串 = 不注入）。

    实测 `low` 把思考从 3.1k 字符压到 ~100 字符，总输出降到 486~533 token，
    抽取到的技能 / 项目条数与默认档位一致 —— 纯抽取任务不需要长时间思考。
    """
    if not rate_limit_enabled():
        return ""
    raw = os.getenv("RESUME_LLM_REASONING_EFFORT", "low")
    return str(raw).strip().lower()


def match_max_tokens() -> int:
    """简历-JD 匹配（match_resume 工具）的单次输出上限，默认 4096。

    为什么要单独给：与简历解析同源 —— `glm-5.3-flash` 是思考模型，
    `max_tokens` 同时卡住思考（`reasoning_content`）与正文。**工具内部
    的 chat() 不经过 ReAct 的分档逻辑**，不显式传就是全局默认 1024。
    实测（真实 PDF 简历 + 出问题的两个 JD，各跑 2 遍）：默认档单次就要
    23.8~24.3s、finish_reason 100% 是 length —— 要么正文被砍成
    `Unterminated string starting at: line 19 column 5`，要么正文 0 字
    （1024 全被思考吃光）。4096 只是上限、不花额外 token，留够余量。
    总开关关闭时为 0（不注入），与改造前行为一致。
    """
    return _env_int("MATCH_LLM_MAX_TOKENS", 4096, 0) if rate_limit_enabled() else 0


def match_reasoning_effort() -> str:
    """简历-JD 匹配的思考档位（low / high / max；空串 = 不注入）。

    默认 `low`，与简历解析、ReAct 同一档：实测同一份简历 + 同一个 JD，
    默认档 23.8~24.3s 且必触顶，`low` + 4096 降到 3.3~6.8s 且 JSON 合法，
    打分维度（skills/experience/education/location）与差距/亮点条数不变
    —— 打分是固定形状的结构化输出，长思考的边际收益远小于它挤掉正文的代价。
    """
    if not rate_limit_enabled():
        return ""
    raw = os.getenv("MATCH_LLM_REASONING_EFFORT", "low")
    return str(raw).strip().lower()


def cover_letter_max_tokens() -> int:
    """投递包（自荐信 + 简历定制）的单次输出上限，默认 4096。

    为什么要单独给：与简历解析 / 匹配同源 —— `glm-5.3-flash` 是思考模型，
    `max_tokens` 同时卡住思考（`reasoning_content`）与正文，且**工具内部的
    chat() 不经过 ReAct 的分档逻辑**，不显式传就是全局默认 1024。
    实测（真实简历 + 真实岗位 466364，容器内）：
    - 自荐信：默认 1024 档 `finish_reason=length`、正文只有 168 字（连称呼都没写完）；
      4096 + `low` 一次出全、`finish_reason=stop`；
    - 简历定制（返回 JSON）：默认 1024 档正文为空、`json.loads` 报
      `Expecting value: line 1 column 1 (char 0)`，连试 3 次都失败 → 投递包被迫用原简历。
    4096 只是上限、不花额外 token。总开关关闭时为 0（不注入），与改造前一致。
    """
    return _env_int("COVER_LETTER_LLM_MAX_TOKENS", 4096, 0) if rate_limit_enabled() else 0


def cover_letter_reasoning_effort() -> str:
    """投递包（自荐信 + 简历定制）的思考档位（low / high / max；空串 = 不注入）。

    默认 `low`，与简历解析 / 匹配同档：实测同一提示下 `low` 把 1024 档从
    「触顶截断」变成「一次写完整封」，行文与呼应的技能点没有变差 ——
    自荐信与简历定制都是短文本重写，长思考的边际收益远小于它挤掉正文的代价。
    """
    if not rate_limit_enabled():
        return ""
    raw = os.getenv("COVER_LETTER_LLM_REASONING_EFFORT", "low")
    return str(raw).strip().lower()


def react_long_max_tokens() -> int:
    """ReAct 循环里**可能输出长文本**的那一轮的单次输出上限，默认 8192。

    为什么单独一档：`glm-5.3-flash` 是思考模型，`max_tokens` 同时卡住思考
    （`reasoning_content`）与正文 —— 默认 1024 下「查看我的完整简历」这类
    要复述整份简历的轮次会被思考吃光额度（finish_reason=length、正文为空）。

    为什么不是全局提高：`max_tokens` 只是**上限**，模型说完就停（finish=stop），
    上限本身不花 token —— 真正花 token 的是生成出来的内容。所以这里给足
    （8192 这一档在上一轮简历解析里实测够用），日常短工具调用轮仍走 1024 那一档。
    总开关关闭时为 0（不注入），与改造前行为一致。
    """
    return _env_int("REACT_LLM_LONG_MAX_TOKENS", 8192, 0) if rate_limit_enabled() else 0


def react_reasoning_effort() -> str:
    """ReAct 每轮的思考档位（low / high / max；空串 = 不注入）。

    默认 `low`：与简历解析同一档 —— 实测把思考从 3.1k 字符压到 ~100 字符，
    抽取/复述类任务质量不降。ReAct 每轮输出的是固定形状的 JSON，长思考的
    边际收益小，却会把 1024 的额度吃光、把正文挤没（本轮的根因）。
    """
    if not rate_limit_enabled():
        return ""
    raw = os.getenv("REACT_LLM_REASONING_EFFORT", "low")
    return str(raw).strip().lower()


def run_token_budget() -> int:
    """单次请求（一轮对话）累计 token 上限；0 = 不熔断。"""
    return _env_int("RUN_TOKEN_BUDGET", 30000, 0)


def daily_tokens_per_user() -> int:
    """单用户日 token 上限；0 = 不限。"""
    return _env_int("DAILY_TOKENS_PER_USER", 200000, 0)


def global_daily_tokens() -> int:
    """全局日 token 上限（防单账号吃光共享额度）；0 = 不限。"""
    return _env_int("GLOBAL_DAILY_TOKENS", 2000000, 0)


def max_concurrency() -> int:
    """同时在跑的 Agent 请求数上限（asyncio.Semaphore）。"""
    return _env_int("MAX_CONCURRENCY", 3, 1)


def quota_verdict(used: int, user_limit: int,
                  global_used: int, global_limit: int) -> str:
    """第 4 道闸门的判定（纯逻辑，查询在 agent/app.py 里做）。

    返回 "user" / "global" / "ok"；限额为 0 表示不限。
    判定是 `>=`：用量刚好等于上限时也算超（否则永远差最后一个 token 才拦）。
    单用户优先报（对用户更有信息量：是他自己用完了，而不是整站用完了）。
    """
    if user_limit > 0 and used >= user_limit:
        return "user"
    if global_limit > 0 and global_used >= global_limit:
        return "global"
    return "ok"


# ========== 第 3 道闸门：单用户频率（内存令牌桶） ==========


class TokenBucket:
    """进程内令牌桶：`dict[key] -> (tokens, last_ts)` + 一把锁。

    零 IO、约 40 行、可直接单测。`now` 可显式传入（秒，单调时钟），
    这样测试不用真的 sleep。
    """

    def __init__(self, rate_per_min: int, burst: int, clock=None):
        self.rate = max(1, int(rate_per_min))
        self.capacity = max(1, int(burst))
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._state: dict = {}

    def _refill(self, key, now: float):
        """按距上次补充的时间折算令牌；新用户直接给满桶。"""
        tokens, last = self._state.get(key, (float(self.capacity), now))
        elapsed = max(0.0, now - last)
        tokens = min(float(self.capacity), tokens + elapsed * self.rate / 60.0)
        return tokens

    def allow(self, key, now: float = None):
        """返回 (是否放行, 建议等待秒数)；未放行时也会把补充结果写回。"""
        now = self._clock() if now is None else float(now)
        with self._lock:
            tokens = self._refill(key, now)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return True, 0.0
            self._state[key] = (tokens, now)
            return False, (1.0 - tokens) * 60.0 / self.rate

    def reset(self, key=None) -> None:
        """清空桶状态（测试/运维用）：key 为 None 时全清。"""
        with self._lock:
            if key is None:
                self._state.clear()
            else:
                self._state.pop(key, None)


_BUCKETS: dict = {}
_BUCKETS_LOCK = threading.Lock()


def _bucket(rate: int, burst: int) -> TokenBucket:
    """按 (速率, 容量) 缓存桶：参数改了就拿一个新桶，旧桶自然作废。"""
    key = (rate, burst)
    with _BUCKETS_LOCK:
        bucket = _BUCKETS.get(key)
        if bucket is None:
            bucket = TokenBucket(rate, burst)
            _BUCKETS[key] = bucket
        return bucket


def check_rate(user_id, now: float = None):
    """第 3 道闸门：返回 (是否放行, 建议等待秒数)。总开关关闭时恒放行。"""
    if not rate_limit_enabled():
        return True, 0.0
    return _bucket(rate_per_min(), rate_burst()).allow(
        normalize_user_id(user_id), now
    )


def reset_rate_limits() -> None:
    """清空全部频率桶（测试 / 阈值变更后使用）。"""
    with _BUCKETS_LOCK:
        _BUCKETS.clear()


# ========== 第 2 道闸门：单次请求的 token 预算（ContextVar 累加器） ==========
#
# 为什么用 ContextVar：一次请求就是一个独立上下文，累加器跟着请求生灭，
# 不需要任何清理逻辑，也不会串到别的用户身上。

_budget: contextvars.ContextVar = contextvars.ContextVar("run_token_budget", default=None)


def start_run_budget(limit: int = None) -> int:
    """开始一次请求的预算记账，返回生效的上限（0 = 不限）。"""
    value = run_token_budget() if limit is None else max(0, int(limit))
    _budget.set({"limit": value, "used": 0})
    return value


def reset_run_budget() -> None:
    """结束 / 清空本次请求的预算记账。"""
    _budget.set(None)


def add_run_tokens(count) -> int:
    """累加本次请求消耗的 token（由 shared/llm_client._record_usage 调用）。"""
    state = _budget.get()
    if state is None:
        return 0
    try:
        delta = max(0, int(count or 0))
    except (TypeError, ValueError):
        delta = 0
    state["used"] = int(state.get("used", 0)) + delta
    return state["used"]


def run_tokens_used() -> int:
    """本次请求已消耗的 token（没开记账时返回 0）。"""
    state = _budget.get()
    return int(state.get("used", 0)) if state else 0


def run_budget_status() -> dict:
    """第 2 道闸门的状态：{"enabled", "used", "limit", "exceeded"}。

    `exceeded` 只表示「该降级收尾了」，**不表示拒绝请求**（见 react_agent）。
    """
    state = _budget.get()
    limit = int(state.get("limit", 0)) if state else run_token_budget()
    used = int(state.get("used", 0)) if state else 0
    enabled = rate_limit_enabled() and limit > 0
    return {
        "enabled": enabled,
        "used": used,
        "limit": limit,
        "exceeded": bool(enabled and used >= limit),
    }


# ========== 第 1 道闸门：输出触顶（max_tokens）标记 ==========
#
# llm_client 读到 finish_reason == "length" 就在这里打个标记，
# react_agent 每轮 chat() 之后取一次：截断的输出不是正常答案，
# 不能静默当成「JSON 格式错误」（那样很难排查），要单独识别出来。

_truncated: contextvars.ContextVar = contextvars.ContextVar("llm_truncated", default=False)


def mark_truncated() -> None:
    _truncated.set(True)


def consume_truncated() -> bool:
    """读一次「上一次调用是否被截断」，读完自动复位。"""
    value = bool(_truncated.get())
    if value:
        _truncated.set(False)
    return value


def reset_truncated() -> None:
    _truncated.set(False)
