"""
投递跟进提醒。

背景：投出去 7 天没动静的岗位，用户往往自己忘了要跟进——这是最容易被忽略、
也最容易补救的一环。本模块只做一件事：把「还在 applied 状态、且投递时间
已经超过 N 天」的记录捞出来，并给出人话版提醒文本。

对外 API：
    check_follow_ups(days=7) -> list[dict]
        每条：{"id", "company", "title", "applied_at", "days_elapsed"}
        按超期天数**从多到少**排序（最该跟进的排最前）。
    format_reminder(items) -> str
        人类可读的提醒文本；没有超期记录时返回一句「无需跟进」。

被谁用：
    * dashboard/app.py 的「投递追踪」Tab（顶部黄色 warning 区）
    * agent/react_agent.py 的 check_reminders 工具（用户问「我该做什么」时）
    两边共用同一份判定逻辑，避免口径不一致（一处说 6 天一处说 7 天）。

依赖：只读 agent.storage 的投递记录，不写任何数据。
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 允许 `python agent/reminder.py` 直接跑自测：把仓库根目录放进 sys.path，
# 否则直接执行脚本时拿不到 agent 包（被 import 时这几行也无害）。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent import storage  # noqa: E402

# 默认超期阈值（天）。可用环境变量 FOLLOW_UP_DAYS 覆盖，便于本地试不同口径。
DEFAULT_FOLLOW_UP_DAYS = 7

# 只关心「投了还没动静」这一种状态；其余状态（viewed/interview/offer/终态）
# 都说明有进展或已结束，不该再催。
PENDING_STATUS = "applied"

# applied_at 理论上统一是 "%Y-%m-%d %H:%M:%S"（storage.now()），
# 但历史数据/手工改过的库可能只有日期，甚至用 "/" 分隔，这里都容忍。
_DT_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%Y-%m-%dT%H:%M:%S",
)


def _parse_dt(value) -> datetime | None:
    """把 applied_at 解析成 datetime；解析不了返回 None（该条不参与提醒）。

    宁可漏提醒也不要因为一条脏数据让整个提醒区报错。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DT_FORMATS:
        # 只裁掉末尾的时区尾巴（如 "2026-01-01 10:00:00+08:00"），让带秒/不带秒
        # 的格式都能按原样匹配——不做「按格式长度切片」，那会把长格式截断成短格式。
        candidate = text[:19].strip() if len(text) > 19 else text
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def _now() -> datetime:
    """当前时间（单独抽出来，测试可以覆盖它固定"今天"）"""
    return datetime.now()


def _days_between(later: datetime, earlier: datetime) -> int:
    """整整几天的差值（按自然天数算，不受具体时刻影响）

    用日期相减而不是秒数相除：投递时间精确到秒，直接除会让
    「7 天前 23:00 投的」在次日早上算出 6 天，阈值判定忽左忽右。
    """
    return (later.date() - earlier.date()).days


def check_follow_ups(days: int = DEFAULT_FOLLOW_UP_DAYS, now: datetime = None) -> list[dict]:
    """找出投递超过 N 天、状态仍是 applied 的记录。

    参数：
        days: 超期阈值（天），默认 7。判定用 `days_elapsed >= days`——
              第 7 天当天就算该跟进，符合「投出去 7 天没动静」的直觉。
        now:  基准时间（测试用；不传取系统当前时间）

    返回：
        [{"id", "company", "title", "applied_at", "days_elapsed"}, ...]
        按 days_elapsed 降序（最久的排最前）；没有超期记录时返回 []。

    注意：状态过滤在 SQL 层做（list_applications(status="applied")），
    但会再核一遍状态字符串，防止库里存了 "Applied" 之类漏掉。
    """
    base = now or _now()
    try:
        threshold = int(days)
    except (TypeError, ValueError):
        threshold = DEFAULT_FOLLOW_UP_DAYS
    if threshold < 0:
        threshold = 0

    items: list[dict] = []
    for record in storage.list_applications(status=PENDING_STATUS):
        status = str(record.get("status") or "").strip().lower()
        if status != PENDING_STATUS:
            continue

        applied_at = record.get("applied_at") or ""
        parsed = _parse_dt(applied_at)
        if parsed is None:
            continue

        elapsed = _days_between(base, parsed)
        if elapsed >= threshold:
            items.append({
                "id": record.get("id", ""),
                "company": record.get("company", ""),
                "title": record.get("title", ""),
                "applied_at": str(applied_at),
                "days_elapsed": elapsed,
            })

    items.sort(key=lambda item: item["days_elapsed"], reverse=True)
    return items


def format_reminder(items: list[dict], days: int = DEFAULT_FOLLOW_UP_DAYS) -> str:
    """把 check_follow_ups 的结果拼成人类可读的提醒文本。

    没有超期记录时返回一句明确的「无需跟进」，而不是空串——
    空串在 Dashboard / Agent 回答里都会变成一片空白，看不出是「没问题」
    还是「功能坏了」。
    """
    items = list(items or [])
    if not items:
        return f"✅ 暂时没有需要跟进的投递（没有「投递超过 {days} 天、状态仍是已投递」的记录）。"

    longest = max(item.get("days_elapsed", 0) for item in items)
    lines = [
        f"⏰ 有 {len(items)} 条投递超过 {days} 天没有动静，建议主动跟进"
        f"（最久的一条已经 {longest} 天）：",
        "",
    ]
    for item in items:
        lines.append(
            "- {company} | {title}｜投递于 {applied_at}（已 {days_elapsed} 天）".format(
                company=item.get("company") or "（未知公司）",
                title=item.get("title") or "（未知岗位）",
                applied_at=item.get("applied_at") or "（无投递时间）",
                days_elapsed=item.get("days_elapsed", 0),
            )
        )
    lines += [
        "",
        "建议做法：给 HR 发一条简短跟进（说明投递的岗位 + 一句话优势 + 询问进度）；",
        "或者用 list_tracking / update_tracking_status 更新这些记录的实际进展。",
    ]
    return "\n".join(lines)


def summary(days: int = DEFAULT_FOLLOW_UP_DAYS) -> dict:
    """一次性拿到结构化结果 + 文本，给 Agent 工具和 Dashboard 复用。

    返回：{"days", "count", "items", "text"}
    """
    items = check_follow_ups(days)
    return {
        "days": days,
        "count": len(items),
        "items": items,
        "text": format_reminder(items, days),
    }


# ---------------------------------------------------------------------------
# 自测用的小工具（只在 __main__ 里调用，不参与正常流程）
# ---------------------------------------------------------------------------
def _make_workdir(prefix: str) -> Path:
    """在系统临时目录下建一个**直属**的临时子目录并返回它。

    为什么不用 tempfile.mkdtemp：某些受限环境（例如带沙箱的执行器）只放开
    系统临时目录本身，不允许在其中**再建**一层随机子目录，mkdtemp 出来的
    目录会不可写、sqlite 直接打不开。这里自己在 tempdir 下拼一个唯一名字，
    路径深度和 tempdir 一致，兼容性更好。
    """
    import tempfile
    import uuid

    root = Path(tempfile.gettempdir())
    path = root / f"{prefix}{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    # 自测：用临时库预置两条记录（3 天前 / 10 天前），阈值 7 天。
    # 只读真实库的话这里没有可验证的数据，所以走 APP_DB_PATH 临时库，
    # 顺带验证「临时库隔离」——真实 applications.db 一个字节都不会被写。
    import os
    import shutil
    from pathlib import Path

    from agent import storage as _storage

    tmp_dir = _make_workdir("reminder_selftest_")
    tmp_db = tmp_dir / "applications.db"
    os.environ["APP_DB_PATH"] = str(tmp_db)
    _storage.DB_PATH = tmp_db          # storage 在 import 时已固化路径，这里直接改写
    _storage.init_db()

    fresh_id = _storage.create_application("三天前公司", "Agent 开发实习生", "selftest", "")
    stale_id = _storage.create_application("十天前公司", "大模型算法实习生", "selftest", "")

    # 回填 applied_at 成过去的时间（create_application 只会写"现在"）
    conn = _storage._get_conn()
    for app_id, days_ago in ((fresh_id, 3), (stale_id, 10)):
        stamp = (_now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "UPDATE applications SET applied_at = ? WHERE id = ?", (stamp, app_id)
        )
    conn.commit()
    conn.close()

    found = check_follow_ups(days=7)
    print(format_reminder(found, days=7))
    print()
    print(f"临时库：{tmp_db}")

    assert [item["company"] for item in found] == ["十天前公司"], f"应只命中 10 天前那条，实际：{found}"
    assert found[0]["days_elapsed"] == 10, found
    assert found[0]["id"] == stale_id, found
    assert check_follow_ups(days=30) == [], "阈值 30 天时不该有超期记录"
    assert format_reminder([]).startswith("✅"), "空列表要给明确的「无需跟进」文案"

    print("\n自测通过：3 天前的不提醒，10 天前的命中 1 条。")

    # 清理临时库（失败时保留现场便于排查）
    for suffix in ("", "-wal", "-shm", "-journal"):
        leftover = Path(str(tmp_db) + suffix)
        if leftover.exists():
            leftover.unlink()
    shutil.rmtree(tmp_dir, ignore_errors=True)
