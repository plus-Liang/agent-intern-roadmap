# -*- coding: utf-8 -*-
"""对话历史落库（agent/chat_history.py）离线回归测试。

跑法：
    python agent/tests/test_chat_history.py

**全程跑在仓库内 `agent/tests/_tmp_chat_history/` 下的临时库**，
不碰真实 `agent/data/chat_history.db`、不联网、不调 LLM。
（用仓库内临时目录而不是系统 %TEMP%：部分沙箱/受限环境里进程对 %TEMP% 无写权限，
sqlite 会直接报 "unable to open database file"；放在仓库里任何环境都能跑。）
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 临时目录默认放仓库内 agent/tests/_tmp_chat_history/。
# 受限沙箱/CI 里仓库目录可能不可写，此时用 CHAT_HISTORY_TEST_DIR 指到可写位置
# （例如 %TEMP%），测试本身不需要改代码。
_TMP_DIR = Path(os.getenv("CHAT_HISTORY_TEST_DIR",
                          str(Path(__file__).resolve().parent / "_tmp_chat_history")))
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_TMP_DB = _TMP_DIR / f"chat_history_{os.getpid()}.db"
os.environ["CHAT_HISTORY_DB"] = str(_TMP_DB)

from agent import chat_history as CH            # noqa: E402

# chat_history.DB_PATH 在模块导入时按 CHAT_HISTORY_DB 固化；
# 这里再显式指一次，保证任何导入顺序下都不会落到真实库。
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


def _fresh(thread_id: str):
    """每个用例用独立 thread_id，避免相互污染（同一个临时库里跑）"""
    CH.clear_thread(thread_id)
    return thread_id


# ---------------------------------------------------------------------------
section("1. 库隔离（绝不能写到真实库）")
# ---------------------------------------------------------------------------
def t_isolated():
    real = CH.ROOT_DIR / "agent" / "data" / "chat_history.db"
    if Path(CH.DB_PATH) == real:
        _fail(f"DB_PATH 指向真实库：{CH.DB_PATH}（测试会污染运行时数据）")
    if Path(CH.DB_PATH).parent != _TMP_DIR:
        _fail(f"DB_PATH 不在测试临时目录：{CH.DB_PATH}")
    if Path(CH.DB_PATH).name == "chat_history.db":
        _fail("测试库文件名与真实库同名，容易看错")
    return f"临时库 {Path(CH.DB_PATH).name}"


def t_init_idempotent():
    CH.init_db()
    before = Path(CH.DB_PATH).stat().st_size if Path(CH.DB_PATH).exists() else 0
    CH.init_db()
    CH.init_db()
    after = Path(CH.DB_PATH).stat().st_size if Path(CH.DB_PATH).exists() else 0
    if after != before:
        _fail(f"重复 init_db 改变了库大小：{before} → {after}")
    return "init_db 幂等（连调 3 次库大小不变）"


check("测试跑在临时库，不碰真实 chat_history.db", t_isolated)
check("init_db 幂等", t_init_idempotent)


# ---------------------------------------------------------------------------
section("2. 建会话 / 会话元信息")
# ---------------------------------------------------------------------------
def t_ensure_thread():
    tid = _fresh("t-ensure")
    row = CH.ensure_thread(tid, title="  广州   Agent 实习 ")
    if row["thread_id"] != tid:
        _fail(f"thread_id 不符：{row['thread_id']}")
    if row["title"] != "广州 Agent 实习":
        _fail(f"标题未压成单行：{row['title']!r}")
    if row["user_id"] != CH.DEFAULT_USER_ID:
        _fail(f"默认 user_id 应为 {CH.DEFAULT_USER_ID}，实际 {row['user_id']}")
    # 再调一次不该把已有标题冲掉
    again = CH.ensure_thread(tid, title="新标题")
    if again["title"] != "广州 Agent 实习":
        _fail(f"重复 ensure 覆盖了标题：{again['title']!r}")
    return "建会话 + 标题归一 + 重复调用不覆盖"


def t_ensure_thread_rejects_empty():
    try:
        CH.ensure_thread("")
    except ValueError:
        return "空 thread_id 抛 ValueError（不写孤儿会话）"
    _fail("空 thread_id 竟然建成功了")


check("ensure_thread 基本契约", t_ensure_thread)
check("ensure_thread 拒绝空 id", t_ensure_thread_rejects_empty)


# ---------------------------------------------------------------------------
section("3. 轮次落库与读取顺序")
# ---------------------------------------------------------------------------
def t_append_and_order():
    tid = _fresh("t-order")
    for i in range(1, 4):
        idx = CH.append_turn(tid, f"问题{i}", f"答案{i}",
                             tool_calls=[{"turn": 1, "action": "search_jobs"}],
                             trace_id=f"trace{i}")
        if idx != i:
            _fail(f"第 {i} 轮返回的 turn_index 是 {idx}")
    history = CH.load_history(tid)
    if [h["turn_index"] for h in history] != [1, 2, 3]:
        _fail(f"轮次顺序错：{[h['turn_index'] for h in history]}")
    if [h["question"] for h in history] != ["问题1", "问题2", "问题3"]:
        _fail(f"问句顺序错：{[h['question'] for h in history]}")
    if history[0]["trace_id"] != "trace1":
        _fail("trace_id 没落库")
    if CH.count_turns(tid) != 3:
        _fail(f"count_turns={CH.count_turns(tid)}")
    return "3 轮顺序与 trace_id 正确"


def t_recent_n_is_the_latest():
    """核心回归：limit=N 必须是**最后** N 轮，不是最早 N 轮"""
    tid = _fresh("t-limit")
    for i in range(1, 6):
        CH.append_turn(tid, f"问题{i}", f"答案{i}")
    last2 = CH.load_history(tid, limit=2)
    if [h["turn_index"] for h in last2] != [4, 5]:
        _fail(f"limit=2 拿到的是 {[h['turn_index'] for h in last2]}，应为 [4, 5]")
    if last2[0]["question"] != "问题4" or last2[1]["question"] != "问题5":
        _fail("limit 截断后内容不对")
    all5 = CH.load_history(tid, limit=0)
    if len(all5) != 5:
        _fail(f"limit=0 应返回全部，实际 {len(all5)}")
    return "limit=2 → [4,5]；limit=0 → 全部 5 轮"


def t_append_rejects_empty_question():
    tid = _fresh("t-empty")
    CH.append_turn(tid, "正常问句", "答案")
    for bad in ("", "   ", None):
        try:
            CH.append_turn(tid, bad, "答案")
        except ValueError:
            continue
        _fail(f"空提问 {bad!r} 竟然落库了")
    if CH.count_turns(tid) != 1:
        _fail(f"空提问污染了库：{CH.count_turns(tid)} 轮")
    return "空提问抛 ValueError，库里仍是 1 轮"


def t_missing_thread_is_safe():
    if CH.load_history("不存在的线程") != []:
        _fail("不存在的会话应返回 []")
    if CH.get_thread("不存在的线程") is not None:
        _fail("不存在的会话应返回 None")
    if CH.count_turns("不存在的线程") != 0:
        _fail("不存在的会话 count 应为 0")
    if CH.clear_thread("不存在的线程") != 0:
        _fail("清空不存在的会话应返回 0")
    return "不存在的 thread_id 一律安静返回空"


check("append_turn 顺序 / trace_id / count", t_append_and_order)
check("load_history(limit=N) 取的是最近 N 轮", t_recent_n_is_the_latest)
check("空提问不落库", t_append_rejects_empty_question)
check("不存在的 thread_id 不抛异常", t_missing_thread_is_safe)


# ---------------------------------------------------------------------------
section("4. 反馈回写 / 快照 / 清空")
# ---------------------------------------------------------------------------
def t_feedback():
    tid = _fresh("t-feedback")
    CH.append_turn(tid, "问1", "答1")
    CH.append_turn(tid, "问2", "答2")
    if CH.set_feedback(tid, 2, "up") is not True:
        _fail("回写第 2 轮失败")
    history = CH.load_history(tid)
    if history[1]["feedback"] != "up":
        _fail(f"第 2 轮 feedback={history[1]['feedback']!r}")
    if history[0]["feedback"] is not None:
        _fail("不该动到第 1 轮")
    if CH.set_feedback(tid, 99, "up") is not False:
        _fail("不存在的轮次应返回 False")
    if CH.set_feedback(tid, None, "up") is not False:
        _fail("turn_index 为 None 应返回 False 而不是抛异常")
    if CH.set_feedback(tid, 1, "down") is not True:
        _fail("覆盖写应成功")
    if CH.load_history(tid)[0]["feedback"] != "down":
        _fail("覆盖写没生效")
    return "按轮次回写 / 不越界 / 覆盖 / 参数兜底"


def t_resume_snapshot():
    tid = _fresh("t-resume")
    if CH.load_resume_snapshot(tid) is not None:
        _fail("没有快照时应返回 None")
    CH.save_resume_snapshot(tid, {"name": "张三", "skills": ["Python", "RAG"]})
    snap = CH.load_resume_snapshot(tid)
    if snap["name"] != "张三" or snap["skills"] != ["Python", "RAG"]:
        _fail(f"快照读回不对：{snap}")
    CH.save_resume_snapshot(tid, None)
    if CH.load_resume_snapshot(tid) is not None:
        _fail("传 None 应清空快照")
    return "存 / 读 / 清空 都正确"


def t_interview_snapshot():
    tid = _fresh("t-interview")
    if CH.load_interview(tid) is not None:
        _fail("没有面试时应返回 None")
    CH.set_interview(tid, {"active": True, "company": "阶跃星辰", "asked": ["自我介绍"]})
    got = CH.load_interview(tid)
    if not got or got["company"] != "阶跃星辰":
        _fail(f"面试快照读回不对：{got}")
    CH.set_interview(tid, {"active": False, "company": "阶跃星辰"})
    if CH.load_interview(tid) is not None:
        _fail("已结束（active=False）应清空")
    CH.set_interview(tid, None)
    if CH.load_interview(tid) is not None:
        _fail("传 None 应清空")
    return "进行中才保留，结束即清"


def t_clear_thread_only_touches_one():
    keep = _fresh("t-keep")
    drop = _fresh("t-drop")
    CH.append_turn(keep, "保留会话的问", "答")
    CH.append_turn(drop, "待删会话的问", "答")
    CH.append_turn(drop, "待删会话的问2", "答")
    CH.save_resume_snapshot(drop, {"name": "会被清掉"})
    deleted = CH.clear_thread(drop)
    if deleted != 2:
        _fail(f"应删 2 轮，实际 {deleted}")
    if CH.get_thread(drop) is not None:
        _fail("会话元信息没被清掉")
    if CH.count_turns(keep) != 1:
        _fail("误伤了别的会话")
    return "只清目标会话，别的会话不受影响"


def t_list_threads():
    tid = _fresh("t-list")
    CH.append_turn(tid, "列会话用的问", "答")
    rows = CH.list_threads(limit=50)
    mine = [r for r in rows if r["thread_id"] == tid]
    if not mine:
        _fail("list_threads 没列出刚建的会话")
    if mine[0]["turns"] != 1:
        _fail(f"turns 统计错：{mine[0]['turns']}")
    if all(r["user_id"] != CH.DEFAULT_USER_ID for r in rows):
        _fail("user_id 维度没落库")
    return f"列出 {len(rows)} 个会话（含 turns 统计与 user_id）"


check("反馈按轮次回写", t_feedback)
check("简历快照存/读/清", t_resume_snapshot)
check("面试快照只在进行中保留", t_interview_snapshot)
check("clear_thread 只清当前会话", t_clear_thread_only_touches_one)
check("list_threads 带 turns 与 user_id", t_list_threads)


# ---------------------------------------------------------------------------
section("5. react_agent 集成契约（不 import，静态解析源码）")
# ---------------------------------------------------------------------------
# 为什么用 ast 而不是直接 import：import agent.react_agent 会连带拉起
# shared.llm_client / tools_registry（jieba、fastembed 等重依赖），
# 在一个纯离线的小单测里不划算，也容易受环境变量影响。
def t_run_signature():
    src = (REPO / "agent" / "react_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    run_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_node = node
            break
    if run_node is None:
        _fail("没找到 run() 定义")
    args = [a.arg for a in run_node.args.args]
    if "history" not in args:
        _fail(f"run() 缺 history 参数：{args}")
    if "return_messages" not in args:
        _fail(f"run() 缺 return_messages 参数：{args}")

    # 默认值必须是 None / False：否则老调用方（eval、Dashboard）行为会变
    defaults = dict(zip(args[-len(run_node.args.defaults):], run_node.args.defaults))
    if not (isinstance(defaults.get("history"), ast.Constant)
            and defaults["history"].value is None):
        _fail(f"history 默认值不是 None：{defaults.get('history')}")
    if not (isinstance(defaults.get("return_messages"), ast.Constant)
            and defaults["return_messages"].value is False):
        _fail(f"return_messages 默认值不是 False：{defaults.get('return_messages')}")

    # 前三个参数的名字与顺序不能变（位置参数调用方依赖它）
    if args[:3] != ["question", "resume_data", "verbose"]:
        _fail(f"前三个参数被改动：{args[:3]}")
    return f"run{tuple(args)}，history/return_messages 默认关闭"


def t_history_helper_exists():
    src = (REPO / "agent" / "react_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "_history_to_messages" not in names:
        _fail("缺 _history_to_messages 辅助函数")
    if "_with_messages" not in names:
        _fail("缺 _with_messages 辅助函数")
    # 只认 question / answer 两个键——不接收状态快照（见 Q2 结论）。
    # 只检查**函数体**（先剥掉 docstring）：文档里本来就会提到「当前状态」这个词，
    # 拿整个函数源码做子串匹配会把自己注释误判成违规。
    func_src = src[src.index("def _history_to_messages"):src.index("def _with_messages")]
    body = func_src.split('"""')
    body = "".join(body[0::2]) if len(body) > 1 else func_src   # 偶数段=docstring 之外
    for forbidden in ("resume_json", "profile", "dynamic_context", "DYNAMIC_CONTEXT"):
        if forbidden in body:
            _fail(f"_history_to_messages 函数体里出现了状态快照字段 {forbidden!r}，会重复注入")
    return "_history_to_messages / _with_messages 均在位，函数体只读 question/answer"


def t_app_wiring():
    src = (REPO / "agent" / "app.py").read_text(encoding="utf-8")
    for token in ("from agent import chat_history", "chat_history.init_db()",
                  "_record_turn(", "/history-clear", "history=history"):
        if token not in src:
            _fail(f"app.py 缺少接入点：{token!r}")
    if "on_chat_resume" in src:
        _fail("本轮不应注册 on_chat_resume（Q1：它依赖 data layer，本轮不做）")
    return "app.py 已接入 chat_history，且未添加 on_chat_resume"


check("run() 签名向后兼容", t_run_signature)
check("历史拼装辅助函数在位且不注入状态快照", t_history_helper_exists)
check("app.py 接入点齐全且无 on_chat_resume", t_app_wiring)


# ---------------------------------------------------------------------------
section("结果")
# ---------------------------------------------------------------------------
print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
for label, err in FAIL:
    print(f"  FAIL {label} → {err}")
if FAIL:
    sys.exit(1)
print("全部通过")
