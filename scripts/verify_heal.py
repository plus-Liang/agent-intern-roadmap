# -*- coding: utf-8 -*-
"""验证 shixiseng 崩溃自愈逻辑（不需要浏览器，用假 context/page）。"""
import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent.scrapers import shixiseng as sx  # noqa: E402

PASS, FAIL = [], []


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f" -> {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {label} -> {type(exc).__name__}: {exc}")


class DeadPage:
    async def goto(self, *a, **k):
        raise RuntimeError("Target page, context or browser has been closed")

    async def wait_for_selector(self, *a, **k):
        raise RuntimeError("closed")

    async def wait_for_timeout(self, *a, **k):
        raise RuntimeError("Target page, context or browser has been closed")


class DeadContext:
    async def new_page(self):
        return DeadPage()

    async def close(self):
        pass


class AlivePage:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class AliveContext:
    def __init__(self):
        self.made = []

    async def new_page(self):
        p = AlivePage()
        self.made.append(p)
        return p


class DeadNewPageContext:
    """new_page 直接失败 —— 模拟浏览器已死。"""

    async def new_page(self):
        raise RuntimeError("Target page, context or browser has been closed")


print("=" * 70)
print("1. 详情页全军覆没 -> BrowserGoneError（恢复逻辑的触发器）")
print("=" * 70)


def t_all_failed_raises():
    items = [{"url": f"https://x/{i}", "job_id": f"j{i}"} for i in range(3)]
    try:
        asyncio.run(sx._fetch_details_concurrent(
            DeadContext(), items, max_concurrency=3))
    except sx.BrowserGoneError as exc:
        return f"3/3 全失败 -> BrowserGoneError（{str(exc)[:60]}）"
    raise AssertionError("全失败却没有抛 BrowserGoneError")


def t_single_item_no_raise():
    """只有 1 条时不判定（单独一条失败说明不了浏览器状态）。"""
    items = [{"url": "https://x/1", "job_id": "j1"}]
    out = asyncio.run(sx._fetch_details_concurrent(
        DeadContext(), items, max_concurrency=1))
    assert out == [{}], out
    return "1 条失败 -> 只记空 dict、不误判浏览器已死"


check("3/3 全失败抛 BrowserGoneError", t_all_failed_raises)
check("单条失败不误判", t_single_item_no_raise)

print()
print("=" * 70)
print("2. 探活 _context_alive / _ensure_healthy_context")
print("=" * 70)


def t_alive():
    sc = sx.ShixisengScraper(headless=True)
    ctx = AliveContext()
    sc._context = ctx
    ok = asyncio.run(sc._context_alive())
    assert ok is True, ok
    assert ctx.made and ctx.made[0].closed, "探活应开一个 page 并关掉它"
    return "活着的 context -> True，且探测页已回收"


def t_dead():
    sc = sx.ShixisengScraper(headless=True)
    sc._context = DeadNewPageContext()
    ok = asyncio.run(sc._context_alive())
    assert ok is False, ok
    return "new_page 失败 -> 判定已死（不抛异常）"


def t_none():
    sc = sx.ShixisengScraper(headless=True)
    assert asyncio.run(sc._context_alive()) is False
    return "context 为 None -> False"


def t_ensure_healthy_rebuilds():
    """死 context 时：先 teardown，再交给 _ensure_context 重建。"""
    sc = sx.ShixisengScraper(headless=True)
    sc._context = DeadNewPageContext()
    torn = []
    calls = []

    async def fake_teardown():
        torn.append(1)
        sc._context = None

    async def fake_ensure():
        calls.append(1)
        return "NEW_CONTEXT"

    sc._teardown = fake_teardown
    sc._ensure_context = fake_ensure
    got = asyncio.run(sc._ensure_healthy_context())
    assert got == "NEW_CONTEXT", got
    assert torn and calls, f"torn={torn} calls={calls}"
    return "死 context -> 先 teardown 再重建，返回新 context"


check("_context_alive：活着", t_alive)
check("_context_alive：已死", t_dead)
check("_context_alive：None", t_none)
check("_ensure_healthy_context 触发重建", t_ensure_healthy_rebuilds)

print()
print("=" * 70)
print(f"合计：{len(PASS)} 通过 / {len(FAIL)} 失败")
for label, err in FAIL:
    print(f"  x {label} -> {err}")
print("=" * 70)
raise SystemExit(1 if FAIL else 0)
