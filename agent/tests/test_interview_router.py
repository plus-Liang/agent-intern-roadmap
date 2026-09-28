# -*- coding: utf-8 -*-
"""需求 1（自然语言开局模拟面试）的离线回归测试：意图词表 + 实体抽取 + 槽位回问。

跑法：
    python agent/tests/test_interview_router.py

覆盖：
  1. 意图判定：该触发的触发，问句 / 否定句 / 命令 / 无关消息不误触发
  2. 实体抽取：两种语序都要能抠出 (公司, 岗位)，抠不出来返回空串
  3. 回问流程：缺槽位 → 挂 pending_interview:<tid> + 回问；补齐 → 复用 _begin_interview
  4. 不吞消息：无关消息 / 另一个请求 不该被路由吃掉（必须返回 False 走普通 Agent）
  5. app.py 接线：路由调用点必须夹在 /mock-interview 之后、普通 Agent 之前

全程不联网、不调 LLM：`cl` 用桩替换，`_begin_interview` / `_deny_if_throttled` 打桩记录调用。
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# app.py 导入时会检查认证配置：开着认证又没密码会直接 SystemExit
os.environ["CHAT_AUTH_ENABLED"] = "false"

import agent.app as APP                                   # noqa: E402

THREAD = "router-test-thread"

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
# cl 桩：只提供路由真正用到的东西（user_session / Message / context.session.thread_id）
# ---------------------------------------------------------------------------
class _StubUserSession:
    def __init__(self):
        self.store: dict = {}

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        self.store[key] = value


class _StubMessage:
    def __init__(self, content="", **kwargs):
        self.content = content
        self.kwargs = kwargs

    async def send(self):
        CL.sent.append(self.content)
        return self


class _StubCL:
    def __init__(self):
        self.user_session = _StubUserSession()
        self.sent: list[str] = []
        self.started: list[tuple] = []
        self.throttle_calls: list[str] = []
        self.throttled = False
        self.Message = _StubMessage
        self.context = types.SimpleNamespace(
            session=types.SimpleNamespace(thread_id=THREAD))


CL = _StubCL()


async def _fake_begin(company, title):
    CL.started.append((company, title))


async def _fake_deny(kind):
    CL.throttle_calls.append(kind)
    return CL.throttled


_REAL_CL = APP.cl
_REAL_BEGIN = APP._begin_interview
_REAL_DENY = APP._deny_if_throttled
APP.cl = CL
APP._begin_interview = _fake_begin
APP._deny_if_throttled = _fake_deny

PENDING_KEY = f"pending_interview:{THREAD}"


def _route(text: str) -> bool:
    return asyncio.run(APP._route_interview_content(text))


def _reset():
    CL.user_session.store.clear()
    CL.sent.clear()
    CL.started.clear()
    CL.throttle_calls.clear()
    CL.throttled = False


# ---------------------------------------------------------------------------
section("1. 意图判定（_is_interview_intent）")

SHOULD_TRIGGER = (
    "帮我模拟面试字节 Agent 开发实习生",
    "来一场字节的后端开发模拟面试",
    "我想练习一下面试",
    "模拟一次面试",
    "给我来一次面试实战",
    "扮演一次面试官帮我练练",
)

SHOULD_NOT_TRIGGER = (
    "/mock-interview 字节 Agent 开发实习生",   # 命令走命令分支
    "面试怎么准备",                            # 问句：没动词
    "模拟面试常见问题有哪些",                   # 命中 blocker
    "字节的面试题难吗",
    "面试流程是什么",
    "帮我找一下字节的面试经验",
    "先不模拟面试了",                          # 否定句
    "今天天气怎么样",
    "我要投递简历",
    "记一下：我下周要去字节面试",                # 有记忆需求，不是开局
)


def t_intent_positive():
    bad = [t for t in SHOULD_TRIGGER if not APP._is_interview_intent(t)]
    if bad:
        _fail(f"该触发却没触发：{bad}")
    return f"{len(SHOULD_TRIGGER)} 条全部命中"


def t_intent_negative():
    bad = [t for t in SHOULD_NOT_TRIGGER if APP._is_interview_intent(t)]
    if bad:
        _fail(f"不该触发却触发了：{bad}")
    long_text = "帮我模拟面试" + "字节" * 200
    if APP._is_interview_intent(long_text):
        _fail("超长文本（多半是简历正文）不该当成开局意图")
    return f"{len(SHOULD_NOT_TRIGGER)} 条全部放行"


# ---------------------------------------------------------------------------
section("2. 实体抽取（_extract_interview_slots）")

SLOT_CASES = (
    ("帮我模拟面试字节 Agent 开发实习生", ("字节", "Agent 开发实习生")),
    ("来一场字节的后端开发模拟面试", ("字节", "后端开发")),
    ("来一场腾讯的产品经理模拟面试", ("腾讯", "产品经理")),
    ("模拟面试一下字节跳动", ("字节跳动", "")),
    ("帮我模拟面试", ("", "")),
    ("字节跳动 Agent 开发实习生", ("字节跳动", "Agent 开发实习生")),
    ("帮我模拟一下：美团 数据分析实习生", ("美团", "数据分析实习生")),
)


def t_slots():
    bad = []
    for text, want in SLOT_CASES:
        got = APP._extract_interview_slots(text)
        if got != want:
            bad.append(f"{text!r} → {got}（期望 {want}）")
    if bad:
        _fail("；".join(bad))
    return f"{len(SLOT_CASES)} 条语序全部抽对"


def t_strip_fillers():
    if APP._strip_fillers("帮我模拟一下：字节跳动  ") != "字节跳动":
        _fail(f"填充词 / 标点没削干净：{APP._strip_fillers('帮我模拟一下：字节跳动  ')!r}")
    if APP._strip_fillers("") != "":
        _fail("空串不该炸")
    return "首尾填充词 + 中文标点都能削"


def t_slot_reply_shape():
    if not APP._looks_like_slot_reply("字节跳动 Agent 开发实习生"):
        _fail("正常补槽回复应被接受")
    for text in ("帮我找一下北京的岗位", "我还想搜索别的岗位", "投递记录呢",
                 "/mock-interview 字节 开发", "帮我看看简历"):
        if APP._looks_like_slot_reply(text):
            _fail(f"像另一个请求的消息不该当补槽回复：{text!r}")
    if APP._looks_like_slot_reply("字" * 60):
        _fail("超长消息不该当补槽回复")
    return "补槽回复判定正确（另一个请求不吃）"


# ---------------------------------------------------------------------------
section("3. 回问流程（_route_interview_content，cl 用桩）")


def t_full_slots_start():
    _reset()
    handled = _route("帮我模拟面试字节 Agent 开发实习生")
    if not handled:
        _fail("命中意图却没处理")
    if CL.started != [("字节", "Agent 开发实习生")]:
        _fail(f"_begin_interview 参数不对：{CL.started}")
    if CL.user_session.get(PENDING_KEY) is not None:
        _fail("开跑之后不该留下待补槽位")
    if CL.throttle_calls != ["mock-interview"]:
        _fail(f"没走限流闸门：{CL.throttle_calls}")
    if CL.sent:
        _fail(f"不该多发消息：{CL.sent}")
    return "一次说完 → 直接复用 _begin_interview 全链路"


def t_missing_slot_asks_back():
    _reset()
    handled = _route("帮我模拟面试")
    if not handled:
        _fail("缺槽位时也该由路由处理（否则会被 Agent 当闲聊）")
    if CL.started:
        _fail("缺槽位不该开局")
    if not CL.sent or "公司名" not in CL.sent[0] or "岗位名" not in CL.sent[0]:
        _fail(f"回问内容不对：{CL.sent}")
    pending = CL.user_session.get(PENDING_KEY)
    if pending != {"company": "", "title": ""}:
        _fail(f"待补槽位没挂上：{pending}")
    return f"回问并挂 pending：{CL.sent[0][:28]}…"


def t_reply_completes_slots():
    _reset()
    _route("帮我模拟面试")
    handled = _route("字节跳动 Agent 开发实习生")
    if not handled:
        _fail("补槽回复没被处理")
    if CL.started != [("字节跳动", "Agent 开发实习生")]:
        _fail(f"补齐后没开局 / 参数不对：{CL.started}")
    if CL.user_session.get(PENDING_KEY) is not None:
        _fail("补齐之后 pending 该清掉")
    return "缺公司 + 岗位 → 回问一次后正常开局"


def t_partial_slot_merge():
    _reset()
    _route("模拟面试一下字节跳动")                 # 有公司、缺岗位
    if CL.started:
        _fail("缺岗位不该开局")
    if CL.user_session.get(PENDING_KEY) != {"company": "字节跳动", "title": ""}:
        _fail(f"已有槽位没保留：{CL.user_session.get(PENDING_KEY)}")
    if "岗位名" not in CL.sent[0]:
        _fail(f"回问该只问缺的那个：{CL.sent[0]}")
    _route("Agent 开发实习生")                    # 只补岗位
    if CL.started != [("字节跳动", "Agent 开发实习生")]:
        _fail(f"合并已有公司 + 新岗位失败：{CL.started}")
    return "只补一个槽位时与已抽到的合并"


def t_pending_abandoned():
    _reset()
    _route("帮我模拟面试")
    handled = _route("帮我找一下北京的 Agent 岗位")
    if handled:
        _fail("另一个请求被回问吃掉了（必须放行走 Agent）")
    if CL.user_session.get(PENDING_KEY) is not None:
        _fail("放弃回问时 pending 该清掉，否则后面每条消息都被吞")
    if CL.started:
        _fail("不该开局")
    return "答非所问 → 放弃回问 + 放行普通 Agent"


def t_plain_chat_passes_through():
    _reset()
    for text in ("今天天气怎么样", "帮我搜一下广州的 Agent 岗位", "/history-clear"):
        if _route(text):
            _fail(f"无关消息被路由吃掉：{text!r}")
    if CL.started or CL.sent:
        _fail("无关消息不该触发任何动作")
    return "普通对话 / 其它命令一条都没被吞"


def t_throttled_blocks_start():
    _reset()
    CL.throttled = True
    handled = _route("帮我模拟面试字节 Agent 开发实习生")
    if not handled:
        _fail("被限流时也该由路由回答（别丢给 Agent）")
    if CL.started:
        _fail("限流了还是开局了")
    return "限流时不开局但也不漏消息"


# ---------------------------------------------------------------------------
section("4. app.py 接线（插入点回归）")


def t_wiring():
    src = (REPO / "agent" / "app.py").read_text(encoding="utf-8")
    call = "if await _route_interview_content(content):"
    if call not in src:
        _fail("on_message 里没有调用意图路由")
    i_clear = src.find('if content == "/history-clear"')
    i_resume = src.find('if content.startswith("/resume")')
    i_mock = src.find('if content.startswith("/mock-interview")')
    i_route = src.find(call)
    i_agent = src.find("# 正常对话：走 Agent")
    for name, pos in (("history-clear", i_clear), ("resume", i_resume),
                      ("mock-interview", i_mock), ("route", i_route), ("agent", i_agent)):
        if pos < 0:
            _fail(f"找不到锚点：{name}")
    if not (i_clear < i_resume < i_mock < i_route < i_agent):
        _fail(f"路由插入位置不对：{i_clear}/{i_resume}/{i_mock}/{i_route}/{i_agent}")
    # 关键：路由必须在 /resume（含附件）之后，否则用户粘贴的简历正文会被当成开局
    if i_route < i_resume:
        _fail("路由跑到了 /resume 前面：粘贴简历会被误判成模拟面试")
    # _begin_interview 必须复用 /mock-interview 的那条链路，而不是另起一套
    if "_begin_interview(company, title)" not in src:
        _fail("没有复用 _begin_interview")
    return "路由夹在 /mock-interview 之后、普通 Agent 之前，且复用 _begin_interview"


# ---------------------------------------------------------------------------
check("意图：该触发", t_intent_positive)
check("意图：不该触发", t_intent_negative)
check("实体抽取两种语序", t_slots)
check("填充词 / 标点削除", t_strip_fillers)
check("补槽回复判定", t_slot_reply_shape)
check("一次说完直接开局", t_full_slots_start)
check("缺槽位回问 + 挂 pending", t_missing_slot_asks_back)
check("补槽后开局", t_reply_completes_slots)
check("单槽位合并", t_partial_slot_merge)
check("答非所问不吃消息", t_pending_abandoned)
check("普通对话放行", t_plain_chat_passes_through)
check("限流不开局", t_throttled_blocks_start)
check("on_message 接线顺序", t_wiring)


# ---------------------------------------------------------------------------
section("结果")
# ---------------------------------------------------------------------------
APP.cl = _REAL_CL
APP._begin_interview = _REAL_BEGIN
APP._deny_if_throttled = _REAL_DENY

print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
for label, err in FAIL:
    print(f"  FAIL {label} → {err}")
if FAIL:
    sys.exit(1)
print("全部通过")
