# -*- coding: utf-8 -*-
"""手动验证：把单用户日额度改成 200，走一遍「额度用完」的四条路径。

跑法：python scripts/verify_quota_ux.py
全程离线：不调 LLM，只读/写仓库内的临时库。
"""
import asyncio
import json
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="quota_ux_")
os.environ.update({
    "APP_DB_PATH": TMP + "/app.db",
    "CHAT_HISTORY_DB": TMP + "/chat.db",
    "TOKEN_DB_PATH": TMP + "/tokens.db",
    "USER_PROFILE_DIR": TMP + "/profiles",
    "RESUME_ROOT": TMP + "/resumes",
    "CHAT_AUTH_ENABLED": "false",
    "ZHIPU_API_KEY": "k",
    "RATE_LIMIT_ENABLED": "true",
    "DAILY_TOKENS_PER_USER": "200",
    "GLOBAL_DAILY_TOKENS": "2000000",
})
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import agent.app as APP                                       # noqa: E402
import agent.tools_registry as TR                             # noqa: E402
from shared import token_tracker as T                         # noqa: E402
from shared.user_context import user_scope                    # noqa: E402

USER = "quota_ux"


class FakeCL:
    sent = []

    class Message:
        def __init__(self, content=""):
            self.content = content

        async def send(self):
            FakeCL.sent.append(self.content)


APP.cl = FakeCL
APP.log_event = lambda *a, **k: None

print("=" * 78)
print("0. 准备：额度 200，已用 205（超 2.5%）")
with user_scope(USER):
    T.record_usage("m", 205, 0, source="verify", user_id=USER)
print("   用量 =", T.usage_today(USER))

print("=" * 78)
print("1. 入口闸门")
with user_scope(USER):
    status = asyncio.run(APP._deny_if_throttled("verify"))
print("   scope =", status.scope, "| bool(硬拒) =", bool(status),
      "| exhausted =", status.exhausted)
print("   话术：")
print("   " + status.quota_message.replace("\n", "\n   "))

print("=" * 78)
print("2. 便宜操作：查投递记录（不跑 Agent、不花 token）")
with user_scope(USER):
    TR._add_tracking("验证公司A", "Agent 开发实习生")
    TR._add_tracking("验证公司B", "后端实习生")
    cheap = asyncio.run(APP._cheap_only_answer("查一下我的投递记录", status))
print("   " + cheap.replace("\n", "\n   "))

print("=" * 78)
print("3. 贵操作降级：匹配打分（只给关键词匹配度）")
with user_scope(USER):
    rows = TR._search_rows("Python", None, 1, False, None)
    job_id = rows[0]["job_id"] if rows else ""
    out = TR.call_tool("match_resume", {
        "job_id": job_id,
        "resume_json": json.dumps({"name": "李四", "skills": ["Python", "Git"],
                                   "city": "北京", "education": "本科"},
                                  ensure_ascii=False)})
print("   " + json.dumps(out, ensure_ascii=False)[:400])

print("=" * 78)
print("4. 贵操作拒绝：生成投递包（含恢复时间，且不产生副作用）")
with user_scope(USER):
    blocked = TR.call_tool("generate_application_package", {"company": "验证公司A"})
print("   " + json.dumps(blocked, ensure_ascii=False))

print("=" * 78)
print("5. 第三层预警：85% 每天只提示一次（100%+ 不再提示，改由闸门说话）")
with user_scope(USER):
    T.record_usage("m", 850 - T.usage_today(USER), 0, source="verify", user_id=USER)
    near = APP.SessionStatus(used=T.usage_today(USER), limit=1000, scope="ok")
    warn1 = APP._take_quota_warn(USER, near)
    warn2 = APP._take_quota_warn(USER, near)
print("   第一次：", warn1 or "（无）")
print("   第二次：", warn2 or "（无，已去重）")
print("=" * 78)
