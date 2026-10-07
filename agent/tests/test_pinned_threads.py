# -*- coding: utf-8 -*-
"""会话置顶（需求：对话置顶）离线回归测试。

跑法：
    python agent/tests/test_pinned_threads.py

覆盖：
  1. 库隔离（绝不写真实 agent/data/chainlit.db）
  2. ensure_schema 补出 `pinnedAt` 列，且幂等
  3. pin_thread / unpin_thread 基本语义（重复置顶刷新时间、未置顶返回 False、
     不存在的 thread 返回 False）
  4. 置顶数上限 MAX_PINNED
  5. **排序**：置顶在前（按 pinned_at DESC）→ 未置顶按 updatedAt DESC。
     直接调 ChatDataLayer.get_all_user_threads（Chainlit 侧边栏列表与恢复历史
     的唯一出口），断言返回顺序，而不是只断言 SQL 片段。
  6. 单条查询（thread_id=...）不因排序被打乱

不联网、不调 LLM、不连真实 Chainlit 服务。临时库放仓库内
agent/tests/_tmp_pin/（受限环境 %TEMP% 可能不可写）。
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_TMP_DIR = Path(os.getenv("PIN_TEST_DIR",
                          str(Path(__file__).resolve().parent / "_tmp_pin")))
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_TMP_DB = _TMP_DIR / f"chainlit_{os.getpid()}.db"
os.environ["CHAINLIT_DB"] = str(_TMP_DB)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DB.as_posix()}"

from agent import data_layer as DL                     # noqa: E402

# 双保险：模块级 DB_PATH 覆盖成临时库（别碰真库）
DL.DB_PATH = _TMP_DB

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


def eq(got, want, what=""):
    if got != want:
        raise AssertionError(f"{what}: got={got!r} want={want!r}")
    return f"{what}={got!r}"


# --------------------------------------------------------------------------
# 准备临时库：3 条会话 A/B/C，各自带一条 step（updatedAt 由 MAX(step.createdAt) 决定）
# --------------------------------------------------------------------------
USER = "local"
THREADS = {"A": "thread-A", "B": "thread-B", "C": "thread-C"}
# 越靠后 = 越新。默认顺序（未置顶）应为 C → B → A
_UPDATED = {"A": "2026-01-01T00:00:01.000Z",
            "B": "2026-01-01T00:00:02.000Z",
            "C": "2026-01-01T00:00:03.000Z"}


def seed():
    DL.ensure_schema()
    with sqlite3.connect(str(_TMP_DB)) as conn:
        for key, tid in THREADS.items():
            conn.execute(
                'INSERT OR REPLACE INTO threads '
                '("id","createdAt","name","userId","userIdentifier") '
                "VALUES (?,?,?,?,?)",
                (tid, _UPDATED[key], f"会话{key}", USER, USER),
            )
            conn.execute('DELETE FROM steps WHERE "threadId" = ?', (tid,))
            conn.execute(
                'INSERT INTO steps ("id","name","type","threadId","isError","createdAt") '
                "VALUES (?,?,?,?,?,?)",
                (f"step-{tid}", "on_message", "assistant_message", tid, "0",
                 _UPDATED[key]),
            )


def layer():
    return DL.build()


# 复用同一个 data layer + 事件循环：每次 asyncio.run 都会新建 aiosqlite 连接池，
# 旧连接池没人 dispose 就占着 Windows 文件句柄，临时库删不掉。
_LAYER = None
_LOOP = asyncio.new_event_loop()


def _data_layer():
    global _LAYER
    if _LAYER is None:
        _LAYER = layer()
    return _LAYER


def run(coro):
    return _LOOP.run_until_complete(coro)


def list_order(thread_id=None, user_id=USER):
    """返回 get_all_user_threads 的顺序（就是侧边栏最终顺序）。"""
    kwargs = {"thread_id": thread_id}
    if user_id is not None:
        kwargs["user_id"] = user_id
    data = run(_data_layer().get_all_user_threads(**kwargs))
    return [t["id"] for t in (data or [])]


def check_async(label, coro_fn):
    check(label, lambda: run(coro_fn()))


# --------------------------------------------------------------------------
print("== 1. 库隔离 / 建表 ==")


def t_seed():
    seed()
    with sqlite3.connect(str(_TMP_DB)) as conn:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("threads")')]
    return f" cols={cols}"


def t_isolated():
    return eq(str(DL.DB_PATH), str(_TMP_DB), "DB_PATH")


def t_idempotent():
    DL.ensure_schema()
    DL.ensure_schema()
    with sqlite3.connect(str(_TMP_DB)) as conn:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("threads")')]
    return eq(cols.count("pinnedAt"), 1, "pinnedAt 列数")


check("临时库建表（pinnedAt 已补列）", t_seed)
check("库隔离（不碰真库）", t_isolated)
check("ensure_schema 幂等", t_idempotent)

print("== 2. pin / unpin 基本语义 ==")


def t_default_order():
    return eq(list_order(), [THREADS["C"], THREADS["B"], THREADS["A"]],
              "未置顶按 updatedAt DESC")


def t_pin_a():
    ok = DL.pin_thread(THREADS["A"], USER)
    return eq(ok, True, "pin_thread")


def t_pin_a_order():
    return eq(list_order(), [THREADS["A"], THREADS["C"], THREADS["B"]],
              "置顶 A 在最上")


def t_pin_missing():
    return eq(DL.pin_thread("no-such-thread", USER), False, "不存在的 thread")


def t_unpin_a():
    return eq(DL.unpin_thread(THREADS["A"]), True, "unpin_thread")


def t_unpin_twice():
    return eq(DL.unpin_thread(THREADS["A"]), False, "重复 unpin")


def t_after_unpin():
    return eq(list_order(), [THREADS["C"], THREADS["B"], THREADS["A"]],
              "取消置顶后回到 updatedAt DESC")


check("默认顺序 = updatedAt DESC", t_default_order)
check("pin_thread 返回 True", t_pin_a)
check("置顶后排最上", t_pin_a_order)
check("pin 不存在的 thread = False", t_pin_missing)
check("unpin_thread 返回 True", t_unpin_a)
check("重复 unpin = False", t_unpin_twice)
check("取消置顶后复位", t_after_unpin)

print("== 3. 置顶倒序 / 上限 / 单条查询 ==")


def t_two_pinned():
    """先置顶 B，再置顶 A → A 在前（pinned_at 倒序），C 掉到第 3。"""
    DL.pin_thread(THREADS["B"], USER)
    DL.pin_thread(THREADS["A"], USER)
    return eq(list_order(), [THREADS["A"], THREADS["B"], THREADS["C"]],
              "置顶按 pinned_at DESC")


def t_repin_refresh():
    """重复置顶 = 刷新时间 → B 再到最前。"""
    DL.pin_thread(THREADS["B"], USER)
    return eq(list_order(), [THREADS["B"], THREADS["A"], THREADS["C"]],
              "重复置顶刷新顺序")


def t_single_thread():
    """`thread_id` 单条查询（不带 user_id）—— 恢复历史的入口，顺序不该被动过。"""
    order = list_order(thread_id=THREADS["C"], user_id=None)
    return eq(order, [THREADS["C"]], "thread_id 单条查询")


def t_single_thread_keeps_pin_flag():
    """单条查询也要把 pinnedAt 带回（前端/调试能看出置顶状态）。"""
    data = run(_data_layer().get_all_user_threads(thread_id=THREADS["A"]))
    stamped = (data or [{}])[0].get("pinnedAt")
    if not stamped:
        raise AssertionError(f"pinnedAt 没带回来：{stamped!r}")
    return f"pinnedAt={stamped}"


def t_limit():
    """塞满上限后再置顶一条应被拒（且不破坏已有顺序）。"""
    before = list_order()
    with sqlite3.connect(str(_TMP_DB)) as conn:
        for i in range(DL.MAX_PINNED):
            conn.execute(
                'INSERT OR REPLACE INTO threads '
                '("id","createdAt","name","userId","pinnedAt") VALUES (?,?,?,?,?)',
                (f"filler-{i}", _UPDATED["A"], f"填充{i}", USER,
                 "2025-01-01T00:00:00.000000Z"),
            )
    ok = DL.pin_thread(THREADS["C"], USER)
    if ok is not False:
        raise AssertionError(f"超上限仍置顶成功 ok={ok!r}")
    with sqlite3.connect(str(_TMP_DB)) as conn:
        conn.execute('DELETE FROM threads WHERE "id" LIKE \'filler-%\'')
    return eq(list_order(), before, "上限拒绝且顺序不变")


def t_is_pinned():
    if not DL.is_pinned(THREADS["A"]):
        raise AssertionError("A 应为置顶")
    return eq(DL.is_pinned(THREADS["C"]), False, "is_pinned(C)")


def t_app_command_text():
    """app.py 里确实挂了 /pin、/unpin 两个命令分支。"""
    src = (REPO / "agent" / "app.py").read_text(encoding="utf-8")
    for token in ('content in ("/pin", "/unpin")',
                  "data_layer.pin_thread",
                  "data_layer.unpin_thread"):
        if token not in src:
            raise AssertionError(f"app.py 缺 {token}")
    return "app.py 命令分支齐备"


check("置顶两个按 pinned_at DESC", t_two_pinned)
check("重复置顶刷新到最前", t_repin_refresh)
check("thread_id 单条查询不被排序打乱", t_single_thread)
check("单条查询带回 pinnedAt", t_single_thread_keeps_pin_flag)
check(f"置顶上限 MAX_PINNED={DL.MAX_PINNED}", t_limit)
check("is_pinned 判定", t_is_pinned)
check("app.py 命令入口存在", t_app_command_text)

print("== 4. 清理 ==")


def t_cleanup():
    """关掉连接池 + 事件循环再删 —— aiosqlite 句柄还开着时 Windows 删不掉文件。"""
    import gc

    eng = getattr(_LAYER, "engine", None)
    if eng is not None:
        run(eng.dispose())
    _LOOP.close()
    gc.collect()
    left = []
    for path in sorted(_TMP_DIR.glob("chainlit_*.db")):
        try:
            path.unlink()
        except OSError:
            left.append(path.name)
    if left:
        raise AssertionError(f"仍有临时库没删掉：{left}")
    return f"{_TMP_DIR.name} 已清空"


check("删除临时库", t_cleanup)

print()
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
if FAIL:
    for label, err in FAIL:
        print(f"  - {label}: {err}")
sys.exit(1 if FAIL else 0)
