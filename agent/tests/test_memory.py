# -*- coding: utf-8 -*-
"""长期记忆（需求 2）离线回归测试：user_memories 表 + save_memory / list_memories 工具。

跑法：
    python agent/tests/test_memory.py

覆盖：
  1. 库隔离（绝不写真实 agent/data/chat_history.db）
  2. 建表 / 索引 / 幂等
  3. save_memory 规则（空内容不落库、超长截断、kind 归一）
  4. list_memories（时间倒序、kind 过滤、limit 收敛）
  5. 用户隔离（一个用户的记忆别人看不到）
  6. thread 上下文（地基：ContextVar 自动带上 + 子线程 copy_context 也能带进去）
  7. 工具注册（save_memory / list_memories 在 TOOLS 里，静态前缀有【主动保存】）
  8. 缺口②：面试收尾 —— 状态清空但评价留档（含 app._record_interview_memory）

不联网、不调 LLM。临时库放仓库内 agent/tests/_tmp_memory/。
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 临时目录默认放仓库内（理由同 test_chat_history：受限环境 %TEMP% 可能不可写）
_TMP_DIR = Path(os.getenv("CHAT_HISTORY_TEST_DIR",
                          str(Path(__file__).resolve().parent / "_tmp_memory")))
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_TMP_DB = _TMP_DIR / f"memory_{os.getpid()}.db"
os.environ["CHAT_HISTORY_DB"] = str(_TMP_DB)
# app.py 在导入时就会检查认证配置：开着认证又没设密码会直接 SystemExit
os.environ["CHAT_AUTH_ENABLED"] = "false"

from agent import chat_history as CH            # noqa: E402
from shared.user_context import (               # noqa: E402
    get_current_thread,
    set_current_user,
    thread_scope,
    user_scope,
)

CH.DB_PATH = _TMP_DB

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


def _count(user_id: str, kind: str = "") -> int:
    return len(CH.list_memories(kind=kind or None, limit=CH.MEMORY_LIST_MAX,
                                user_id=user_id))


# ---------------------------------------------------------------------------
section("1. 库隔离（绝不能写到真实库）")


def t_isolation():
    if Path(CH.DB_PATH) != _TMP_DB:
        _fail(f"临时库没生效：{CH.DB_PATH}")
    parts = [p.lower() for p in Path(CH.DB_PATH).parts]
    if "data" in parts and "agent" in parts:
        _fail(f"测试库指向了真实 agent/data：{CH.DB_PATH}")
    return f"{Path(CH.DB_PATH).name}"


# ---------------------------------------------------------------------------
section("2. 建表 / 索引 / 幂等")


def t_table():
    CH.init_db()
    CH.init_db()                                   # 幂等：再跑一次不报错
    with CH._get_conn() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(user_memories)")}
        index = {r["name"] for r in conn.execute("PRAGMA index_list(user_memories)")}
    need = {"id", "user_id", "thread_id", "kind", "content", "created_at"}
    if not need <= cols:
        _fail(f"user_memories 缺字段：{sorted(need - cols)}")
    if "idx_memories_user" not in index:
        _fail("缺索引 idx_memories_user（按用户 + 类型分页查询会全表扫）")
    # 只增不改：没有 updated_at / 没有 UPDATE 语句
    if "updated_at" in cols:
        _fail("记忆表不该是可变记录（只增不改）")
    src = (REPO / "agent" / "chat_history.py").read_text(encoding="utf-8")
    if "UPDATE user_memories" in src or "DELETE FROM user_memories" in src:
        _fail("chat_history.py 里出现了改 / 删记忆的语句")
    return f"{len(cols)} 字段 + 索引 + 只增不改"


# ---------------------------------------------------------------------------
section("3. save_memory 规则")


def t_save_rules():
    uid = "mem-save"
    before = _count(uid)

    empty = CH.save_memory("   ", user_id=uid)
    if empty.get("ok") or _count(uid) != before:
        _fail("空内容不该落库")

    ok = CH.save_memory("记住：我只会 Python", user_id=uid)
    if not ok.get("ok") or not ok.get("id"):
        _fail(f"正常内容没落库：{ok}")
    if ok["kind"] != "note" or not ok["created_at"]:
        _fail(f"默认 kind / 时间戳不对：{ok}")

    # kind 归一：大小写 / 标点 → 只留字母数字下划线
    if CH.save_memory("一", kind="Interview-Record!", user_id=uid)["kind"] != "interview_record":
        _fail("kind 没归一成 interview_record")
    if CH.save_memory("二", kind="   ", user_id=uid)["kind"] != "note":
        _fail("空 kind 应回落到 note")
    if len(CH.save_memory("三", kind="x" * 80, user_id=uid)["kind"]) != 32:
        _fail("kind 超长没截断到 32")

    long_text = "长" * (CH.MEMORY_MAX_CHARS + 500)
    cut = CH.save_memory(long_text, user_id=uid)
    if not cut.get("truncated") or len(cut["content"]) != CH.MEMORY_MAX_CHARS:
        _fail(f"超长记忆没截断：{len(cut['content'])} / truncated={cut.get('truncated')}")

    # 多行内容原样保留（面试记录就是多行的）
    multi = CH.save_memory("第一行\n第二行", user_id=uid)
    if multi["content"] != "第一行\n第二行":
        _fail("多行内容被改写了")

    if _count(uid) != before + 6:
        _fail(f"落库条数不对：{_count(uid)}")
    return f"{_count(uid)} 条，截断 / 归一 / 多行 均正确"


def t_list_and_filter():
    uid = "mem-list"
    for text, kind in (("第一条", "note"), ("第二条", "interview_record"), ("第三条", "note")):
        CH.save_memory(text, kind=kind, user_id=uid)

    recent = CH.list_memories(limit=2, user_id=uid)
    if [r["content"] for r in recent] != ["第三条", "第二条"]:
        _fail(f"没按时间倒序：{[r['content'] for r in recent]}")

    only_interview = CH.list_memories(kind="interview_record", user_id=uid)
    if [r["content"] for r in only_interview] != ["第二条"]:
        _fail(f"kind 过滤不对：{only_interview}")

    if len(CH.list_memories(limit=0, user_id=uid)) != 1:
        _fail("limit=0 应收敛到 1，而不是返回 0 条 / 全量")
    if len(CH.list_memories(limit=10 ** 6, user_id=uid)) != 3:
        _fail("limit 上限没收敛")
    if len(CH.list_memories(limit="不是数字", user_id=uid)) != 3:
        _fail("非法 limit 应回落默认值")
    return "倒序 / 过滤 / limit 收敛均正确"


# ---------------------------------------------------------------------------
section("4. 用户隔离")


def t_user_isolation():
    CH.save_memory("甲的私事", user_id="mem-a")
    CH.save_memory("乙的私事", user_id="mem-b")
    a = [r["content"] for r in CH.list_memories(user_id="mem-a")]
    b = [r["content"] for r in CH.list_memories(user_id="mem-b")]
    if a != ["甲的私事"] or b != ["乙的私事"]:
        _fail(f"用户之间串了：a={a} b={b}")

    # 默认归属走 ContextVar（不是硬编码 local）
    with user_scope("mem-c"):
        CH.save_memory("丙的私事")
        got = [r["content"] for r in CH.list_memories()]
    if got != ["丙的私事"]:
        _fail(f"默认用户没走 ContextVar：{got}")
    if CH.list_memories(user_id="mem-c") == CH.list_memories(user_id="mem-a"):
        _fail("不同用户的列表相同，说明没按 user_id 过滤")
    return "三个用户互不可见"


# ---------------------------------------------------------------------------
section("5. thread 上下文（地基）")


def t_thread_context():
    # 没设过就是空串 —— 不编造假 thread_id（脚本 / 调度器场景）
    if get_current_thread() != "":
        _fail(f"未绑定时 thread_id 应为空串，实际 {get_current_thread()!r}")

    with thread_scope("thread-A"):
        rec = CH.save_memory("带来源标记", user_id="mem-thread")
        if rec["thread_id"] != "thread-A":
            _fail(f"thread_id 没自动带上：{rec}")
        inner = get_current_thread()
    if inner != "thread-A" or get_current_thread() != "":
        _fail("thread_scope 出作用域没复原")

    plain = CH.save_memory("没有来源", user_id="mem-thread")
    if plain["thread_id"] != "":
        _fail(f"没绑会话时不该带 thread_id：{plain}")
    return "默认空串 + thread_scope 复原 + 自动带来源"


def t_thread_crosses_subthread():
    """工具跑在 ThreadPoolExecutor 子线程里：靠 call_tool 的 copy_context() 带进去。"""
    from agent import react_agent  # noqa: F401  导入即注册 profile 工具（含 save_memory）
    from agent.tools_registry import call_tool

    set_current_user("mem-sub")
    with thread_scope("thread-sub-1"):
        res = call_tool("save_memory", {"content": "子线程里读上下文", "kind": "note"})
    if not isinstance(res, dict) or not res.get("ok"):
        _fail(f"工具调用失败：{res}")
    if res.get("user_id") != "mem-sub" or res.get("thread_id") != "thread-sub-1":
        _fail(f"子线程里丢了上下文：user={res.get('user_id')} thread={res.get('thread_id')}")

    # list_memories 同样要能读到刚写进去的那条
    with thread_scope("thread-sub-1"), user_scope("mem-sub"):
        listing = call_tool("list_memories", {"kind": "note", "limit": 5})
    if "子线程里读上下文" not in listing.get("text", ""):
        _fail(f"工具读不回：{listing}")

    set_current_user("local")                       # 复原，免得影响后面的用例
    return "copy_context 把 user + thread 一起带进子线程"


# ---------------------------------------------------------------------------
section("6. 工具注册 + 静态前缀")


def t_tools_registered():
    from agent import react_agent as RA
    from agent.tools_registry import TOOLS

    for name in ("save_memory", "list_memories"):
        spec = TOOLS.get(name)
        if not spec:
            _fail(f"{name} 没注册进 TOOLS")
        if not callable(spec.get("func")):
            _fail(f"{name} 缺可调用的 func")
        if spec.get("risk_level") not in ("read", "reversible", "irreversible"):
            _fail(f"{name} risk_level 非法：{spec.get('risk_level')}")
        if not isinstance(spec.get("requires_confirmation"), bool):
            _fail(f"{name} 缺 requires_confirmation")
        if not spec.get("parameters"):
            _fail(f"{name} 没有参数声明（call_tool 会拒绝一切参数）")

    if "【主动保存】" not in RA.STATIC_PREFIX:
        _fail("STATIC_PREFIX 缺【主动保存】节")
    prefix = RA.build_static_prefix()
    for token in ("save_memory", "list_memories", "【主动保存】"):
        if token not in prefix:
            _fail(f"拼好的前缀里缺 {token!r}")
    # 前缀里出现工具名 = 模型看得到这两个工具
    return f"2 个工具 + 【主动保存】节，前缀 {len(prefix)} 字"


def t_tool_returns_readable_text():
    from agent import react_agent as RA

    with user_scope("mem-tool"):
        saved = RA._save_memory_tool("工具层探针", kind="resume_note")
        if not saved.get("ok"):
            _fail(f"_save_memory_tool 失败：{saved}")
        listing = RA._list_memories_tool(kind="resume_note", limit=5)
    if listing.get("count") != 1 or "工具层探针" not in listing["text"]:
        _fail(f"_list_memories_tool 输出不对：{listing}")
    if not listing["items"][0]["kind"] == "resume_note":
        _fail("items 里没有完整记录")

    bad = RA._save_memory_tool("   ")               # 空内容不该炸
    if bad.get("ok"):
        _fail("空内容竟然落库了")
    return "工具返回值可读（text + items）"


# ---------------------------------------------------------------------------
section("7. 缺口②：面试收尾留档")


def t_interview_record():
    uid = "mem-interview"
    rec = CH.save_interview_record("字节跳动", "Agent 开发实习生",
                                   "综合评价：工程基础扎实，表达清楚。",
                                   turns=6, user_id=uid, thread_id="t-int")
    if rec.get("kind") != "interview_record" or not rec.get("ok"):
        _fail(f"面试记录没落库：{rec}")
    for token in ("字节跳动", "Agent 开发实习生", "综合评价", "共 6 轮"):
        if token not in rec["content"]:
            _fail(f"面试记录内容缺 {token!r}：{rec['content']}")

    # 公司 / 岗位都空也要能落一条（用户直接停掉面试的场景）
    bare = CH.save_interview_record("", "", "评价：先这样", user_id=uid)
    if not bare.get("ok"):
        _fail("空公司 / 岗位时不该拒绝落库")

    # 收尾语义：状态清掉、记录还在（这就是缺口②要修的东西）
    tid = "thread-interview-finish"
    session = {"active": True, "company": "腾讯", "title": "后端开发实习生",
               "history": [{"question": "q1", "answer": "a1"}]}
    CH.set_interview(tid, session)
    if not CH.load_interview(tid):
        _fail("面试状态没写进去")
    CH.save_interview_record(session["company"], session["title"], "评价：不错", turns=1,
                             user_id=uid, thread_id=tid)
    CH.set_interview(tid, {**session, "active": False})
    if CH.load_interview(tid) is not None:
        _fail("面试结束后状态应被清空")
    texts = [r["content"] for r in CH.list_memories(kind="interview_record", user_id=uid)]
    if not any("腾讯" in t and "评价：不错" in t for t in texts):
        _fail(f"面试结束后记录丢了（缺口②没修好）：{texts}")
    return f"{len(texts)} 条面试记录：状态清空但评价留档"


def t_app_wiring():
    src = (REPO / "agent" / "app.py").read_text(encoding="utf-8")
    for token in ("_record_interview_memory(", "save_interview_record",
                  "_save_resume_to_library", "storage.save_resume",
                  "set_current_thread", "get_current_thread"):
        if token not in src:
            _fail(f"app.py 缺少接入点：{token!r}")

    # app 层收尾函数真跑一遍（_thread_id() 在无 socket 时回落旧固定会话，不影响断言）
    import agent.app as APP

    with user_scope("mem-app"), thread_scope("mem-app-thread"):
        session = {"active": True, "company": "美团", "title": "数据分析实习生",
                   "history": [{"question": "q", "answer": "a"}]}
        res = APP._record_interview_memory(session, "综合评价：数据敏感度不错")
    if not res.get("ok"):
        _fail(f"_record_interview_memory 失败：{res}")
    if res.get("user_id") != "mem-app":
        _fail(f"归属用户不对：{res.get('user_id')}")
    items = CH.list_memories(kind="interview_record", user_id="mem-app")
    if not items or "数据敏感度不错" not in items[0]["content"]:
        _fail(f"app 层没把评价落库：{items}")
    # 空会话不该落空记录
    empty = APP._record_interview_memory({}, "")
    if empty.get("ok"):
        _fail("空会话不该落库")
    return "app._record_interview_memory 落库 + 空会话拒写"


# ---------------------------------------------------------------------------
check("临时库隔离", t_isolation)
check("user_memories 建表 / 索引 / 幂等", t_table)
check("save_memory 规则", t_save_rules)
check("list_memories 倒序 / 过滤 / limit", t_list_and_filter)
check("用户隔离", t_user_isolation)
check("thread ContextVar", t_thread_context)
check("thread 跨子线程（copy_context）", t_thread_crosses_subthread)
check("工具注册 + 静态前缀", t_tools_registered)
check("工具返回值", t_tool_returns_readable_text)
check("缺口②面试收尾留档", t_interview_record)
check("app.py 接线 + app 层收尾", t_app_wiring)


# ---------------------------------------------------------------------------
section("结果")
# ---------------------------------------------------------------------------
print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
for label, err in FAIL:
    print(f"  FAIL {label} → {err}")
if FAIL:
    sys.exit(1)
print("全部通过")
