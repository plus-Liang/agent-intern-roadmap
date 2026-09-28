# -*- coding: utf-8 -*-
"""需求 3 的关键回归：**现有简历 / 面试状态还能读到**。

接 Chainlit data layer 时最容易丢的就是这份状态：会话键从固定的 `"resume"`
改成按 thread 分键（`resume:<thread_id>`），thread_id 也从「按用户固定」变成
Chainlit 真正的那一个。任何一处接错，用户升级后的第一句就是「我的简历没了」。

真实库里通常没有可用的合成数据，所以本脚本：

1. 把真实 `agent/data/chat_history.db` **只读复制**到临时目录（绝不写真实库）；
2. 合成一份简历快照 + 面试状态，走 `chat_history` 的公开 API 做写读回环；
3. 直接调用 `app.py` 的 `_load_resume_for_thread` / `_load_interview_for_thread`，
   验证「新会话读不到自己那份时回退继承 `user:<user_id>`」这条升级安全网真的在跑。

跑法（写临时库，须在能写仓库的权限下跑）：

    python agent/tests/verify_resume_state.py

退出码 0 = 全部通过。
"""
import os
import shutil
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TMP = REPO / "agent" / "tests" / "_tmp_resume_state"
PASS, FAIL = [], []

# 临时库必须在使用 chat_history 之前定好：它 import 时就固定了 DB_PATH
REAL_DB = REPO / "agent" / "data" / "chat_history.db"
TMP.mkdir(parents=True, exist_ok=True)
DB = TMP / "chat_history.db"
if REAL_DB.is_file():
    shutil.copy2(REAL_DB, DB)
os.environ["CHAT_HISTORY_DB"] = str(DB)
# 本脚本只做离线读写，不需要认证；显式关掉，免得 .env 里开着认证却没填密码时
# import agent.app 直接 SystemExit（那是给真实启动用的保护，不该拦测试）。
os.environ["CHAT_AUTH_ENABLED"] = "false"

from agent import chat_history as ch  # noqa: E402


def check(label, fn):
    try:
        note = fn()
        PASS.append(label)
        print(f"  [PASS] {label} -> {note}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((label, f"{type(e).__name__}: {e}"))
        print(f"  [FAIL] {label} -> {type(e).__name__}: {e}")


ch.init_db()

RESUME = {
    "name": "李雷",
    "years": "5 年",
    "city": "广州",
    "skills": ["Python", "LangChain", "RAG"],
    "summary": "5 年后端 / Agent 经验，做过客服问答与知识库检索。",
}
# active 是 chat_history 的约定（chat_history.py:590 / :608）：
# 只持久化「进行中」的面试，active=False 视为已结束 → 清列。
INTERVIEW = {
    "active": True,
    "job_title": "Agent 开发工程师",
    "stage": "技术二面",
    "turn": 7,
    "asked": ["RAG 召回率怎么优化", "Python GIL"],
    "pending": "请介绍一个你主导的 RAG 项目",
}
LEGACY = ch.thread_id_for_user("local")
NEW_TID = "9f2c1a44-resume-state-check"


def t_legacy_id():
    assert LEGACY == "user:local", LEGACY
    return f"旧固定会话 id = {LEGACY}（与接 data layer 之前逐字一致）"


def t_write_read():
    ch.ensure_thread(LEGACY, user_id="local", title="帮我找广州的 Agent 岗位")
    ch.save_resume_snapshot(LEGACY, RESUME)
    ch.set_interview(LEGACY, INTERVIEW)
    got_r = ch.load_resume_snapshot(LEGACY)
    got_i = ch.load_interview(LEGACY)
    assert got_r and got_r.get("name") == "李雷", got_r
    assert got_i and got_i.get("turn") == 7, got_i
    return f"写入后读回：resume.name={got_r['name']} / interview.turn={got_i['turn']}"


def t_db_rows():
    with sqlite3.connect(DB) as conn:
        row = conn.execute(
            "SELECT length(coalesce(resume_json,'')), length(coalesce(interview,'')) "
            "FROM chat_threads WHERE thread_id=?", (LEGACY,)).fetchone()
    assert row and row[0] > 100 and row[1] > 40, row
    return f"落库：resume_json {row[0]} 字节 / interview {row[1]} 字节"


def t_isolated():
    ch.ensure_thread(NEW_TID, user_id="local", title="新对话")
    ch.save_resume_snapshot(NEW_TID, {"name": "韩梅梅", "skills": ["Java"]})
    own = ch.load_resume_snapshot(NEW_TID)
    old = ch.load_resume_snapshot(LEGACY)
    assert own and own["name"] == "韩梅梅", own
    assert old and old["name"] == "李雷", old
    return "新会话写自己的简历快照，旧固定会话那份没被覆盖"


def t_app_fallback():
    """直接跑 app.py 的读取路径（新会话无自己那份 → 继承 user:<uid>）。"""
    from agent import app
    app._user_id = lambda: "local"          # 离线：没有 chainlit socket 上下文
    assert app._user_thread_id() == LEGACY, app._user_thread_id()
    fresh = "brand-new-thread-uuid"
    r = app._load_resume_for_thread(fresh)
    i = app._load_interview_for_thread(fresh)
    assert r and r.get("name") == "李雷", r
    assert i and i.get("turn") == 7, i
    own = app._load_resume_for_thread(NEW_TID)
    assert own and own["name"] == "韩梅梅", own
    return "全新会话继承 user:local 的简历+面试；已有自己那份的会话优先读自己"


def t_inactive_clears():
    ch.set_interview(LEGACY, {**INTERVIEW, "active": False})
    assert ch.load_interview(LEGACY) is None, "已结束的面试不应留在库里"
    ch.set_interview(LEGACY, INTERVIEW)          # 复原，后面的检查还要用
    assert ch.load_interview(LEGACY)["turn"] == 7
    return "面试结束（active=False）→ 清列；重新进行 → 又能读回"


def t_keys_isolated():
    from agent import app
    app._user_id = lambda: "local"
    k1, k2 = app._resume_key(), app._interview_key()
    assert k1.startswith("resume:") and k2.startswith("interview:"), (k1, k2)
    return f"会话键按 thread 隔离：{k1} / {k2}"


print("== 需求3 回归：简历 / 面试状态读写（临时库，真实库只读复制）")
check("旧固定 thread_id 与升级前一致", t_legacy_id)
check("简历快照 + 面试状态 写读回环", t_write_read)
check("两项状态真的落进 chat_history.db", t_db_rows)
check("按 thread 隔离：新会话不覆盖旧会话", t_isolated)
check("app.py 读取路径：新会话继承旧固定会话状态", t_app_fallback)
check("面试结束清列语义未变", t_inactive_clears)
check("user_session 键按 thread 隔离", t_keys_isolated)

print(f"\n结果：通过 {len(PASS)} / 失败 {len(FAIL)}")
for label, err in FAIL:
    print(f"  FAIL {label} -> {err}")
sys.exit(1 if FAIL else 0)
