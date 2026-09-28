# -*- coding: utf-8 -*-
"""当前用户上下文（ContextVar）。

为什么需要它：多用户要按用户隔离投递记录 / 简历版本 / 长期偏好 / 会话历史，
但 storage、user_profile、tools_registry 这些函数都在很深的地方被调用
（工具还跑在子线程里），把 user_id 当参数一路透传会改掉几十个函数签名。
用 ContextVar 记「当前是谁」，调用方只在自己边界 set 一次，深处直接 get。

有两个地方 ContextVar 不会自动生效，必须显式处理：
1. **子线程**：`ThreadPoolExecutor.submit` 不复制调用方的上下文
   （见 `tools_registry.call_tool` 里的 `copy_context()`）；
2. **回调之间**：Chainlit 每条消息是独立 task，所以在每个回调入口 set 一次，
   覆盖该轮的全部同步调用链。

默认值 `local`：没有登录态的脚本 / Dashboard / 调度器 / 单测拿到的就是它 ——
行为与多用户改造之前完全一致，存量数据也都归在它名下。

`thread_id` 与 `user_id` 是**两个独立的 ContextVar**：user 决定「数据归谁」，
thread 决定「数据来自哪条会话」。工具里要跨会话记住一条东西（save_memory）时，
归属仍旧按 user 走（一个用户的记忆跨会话共享），thread 只作为来源标记写进表里，
方便日后按会话回溯。thread 的兜底值是空串：脚本 / 单测 / 调度器没有会话概念，
和「没绑过」是同一件事，不编造一个假 thread_id。
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager

# 无认证 / 后台任务 / 脚本 / 测试时的兜底用户，也是存量数据的归属者
DEFAULT_USER_ID = "local"

_current_user: contextvars.ContextVar = contextvars.ContextVar(
    "current_user_id", default=DEFAULT_USER_ID
)


def normalize_user_id(value) -> str:
    """归一：None / 空白 → DEFAULT_USER_ID，其余转成去空白的字符串。"""
    text = "" if value is None else str(value).strip()
    return text or DEFAULT_USER_ID


def get_current_user() -> str:
    """当前用户 id（从没 set 过时返回 DEFAULT_USER_ID）。"""
    return normalize_user_id(_current_user.get())


def set_current_user(user_id) -> contextvars.Token:
    """设置当前用户，返回可用于复原的 token。"""
    return _current_user.set(normalize_user_id(user_id))


def reset_current_user(token) -> None:
    """复原到 set 之前；token 失效等异常一律吞掉，不影响主流程。"""
    try:
        _current_user.reset(token)
    except (ValueError, LookupError, RuntimeError):
        pass


@contextmanager
def user_scope(user_id):
    """`with user_scope("alice"): ...` —— 出作用域自动复原（脚本 / 测试用）。"""
    token = set_current_user(user_id)
    try:
        yield get_current_user()
    finally:
        reset_current_user(token)


# --------------------------------------------------------------------------
# 当前会话（thread）上下文
# --------------------------------------------------------------------------

# 没有会话概念时的兜底值：空串表示「不知道来自哪条会话」，
# 与 user 的 `local` 不同 —— user 必须有归属（数据要落谁名下），
# thread 只作来源标记，没有就是没有，不编造。
DEFAULT_THREAD_ID = ""

_current_thread: contextvars.ContextVar = contextvars.ContextVar(
    "current_thread_id", default=DEFAULT_THREAD_ID
)


def normalize_thread_id(value) -> str:
    """归一：None / 空白 → 空串，其余转成去空白的字符串。"""
    text = "" if value is None else str(value).strip()
    return text


def get_current_thread() -> str:
    """当前 thread_id（从没 set 过时返回空串）。"""
    return normalize_thread_id(_current_thread.get())


def set_current_thread(thread_id) -> contextvars.Token:
    """设置当前会话，返回可用于复原的 token。"""
    return _current_thread.set(normalize_thread_id(thread_id))


def reset_current_thread(token) -> None:
    """复原到 set 之前；token 失效等异常一律吞掉，不影响主流程。"""
    try:
        _current_thread.reset(token)
    except (ValueError, LookupError, RuntimeError):
        pass


@contextmanager
def thread_scope(thread_id):
    """`with thread_scope("uuid-xxx"): ...` —— 出作用域自动复原（脚本 / 测试用）。"""
    token = set_current_thread(thread_id)
    try:
        yield get_current_thread()
    finally:
        reset_current_thread(token)
