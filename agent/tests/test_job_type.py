# -*- coding: utf-8 -*-
"""岗位类型（实习 / 正式 / 兼职）离线回归测试。

跑法::

    python agent/tests/test_job_type.py

全程离线：不联网、不调 LLM、不写真实数据（临时目录用
`agent/tests/_tmp_job_type/`，可用 `JOB_TYPE_TEST_DIR` 覆盖）。

覆盖：
1. 分类器 `classify_job_type`：三平台各自的信号强弱（实习僧/牛客平台兜底、
   ncss 按标题、校招池兜底），以及"可转正仍是实习""兼职优先"这类顺序陷阱；
2. 查询侧 `detect_query_type` / `strip_query_type_words`：认类型词、摘类型词；
3. 抓取器打标：三个平台的 `_to_raw_job` / `to_raw_job` 真的带 job_type；
4. 清洗 + 落库：`clean_jobs` 输出带 job_type，`upsert_jobs` 把它写进库；
5. 库层过滤 + 迁移：老库补列回填、`search_jobs(job_type=...)` 只返回该类型、
   **库内每行的 job_type 与分类器逐行一致**（防两套口径漂移）。
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_TMP_DIR = Path(os.getenv("JOB_TYPE_TEST_DIR",
                          str(Path(__file__).resolve().parent / "_tmp_job_type")))
_TMP_DIR.mkdir(parents=True, exist_ok=True)

from shared.job_type import (TYPE_FULLTIME, TYPE_INTERN, TYPE_PARTTIME,
                             classify_job_type, detect_query_type,
                             matches, normalize_job_type,
                             strip_query_type_words)      # noqa: E402

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


# ---------------------------------------------------------------------------
section("1. 分类器：三个平台各自的类型信号")


def _intern_platform_default():
    """实习僧 type=intern / 牛客 recruitType=2：标题没有「实习」也是实习。

    这是本轮最关键的判据：牛客实测只有 51.1% 的实习岗位标题带「实习」，
    按标题判会把近一半真实习岗判丢。
    """
    cases = [
        ("shixiseng", "AI Agent开发（可转正）", TYPE_INTERN),
        ("shixiseng", "Java开发实习生", TYPE_INTERN),
        ("shixiseng", "大模型算法（可转正）", TYPE_INTERN),
        ("niuke", "大模型算法", TYPE_INTERN),
        ("niuke", "算法工程师", TYPE_INTERN),
        ("niuke", "AI智能体工程师（实习）", TYPE_INTERN),
    ]
    bad = [f"{p}/{t} -> {classify_job_type(p, t)!r} 期望 {w!r}"
           for p, t, w in cases if classify_job_type(p, t) != w]
    if bad:
        raise AssertionError("；".join(bad))
    return f"{len(cases)} 例全对（含标题不带「实习」的牛客岗）"


def _ncss_title_based():
    """ncss 没有结构化类型字段 → 标题带「实习」算实习，其余按校招池算正式。"""
    cases = [
        ("AI医疗算法大模型实习生", TYPE_INTERN),
        ("算法工程师", TYPE_FULLTIME),
        ("【广州】算法工程师（博士后）", TYPE_FULLTIME),
        ("某公司社会招聘-Java", TYPE_FULLTIME),
        ("校招-蔚来顾问", TYPE_FULLTIME),
    ]
    bad = [f"{t} -> {classify_job_type('ncss', t)!r} 期望 {w!r}"
           for t, w in cases if classify_job_type("ncss", t) != w]
    if bad:
        raise AssertionError("；".join(bad))
    return f"{len(cases)} 例全对"


def _parttime_wins():
    """兼职是独立类型，且优先级高于平台兜底（实习僧上也有兼职）。"""
    if classify_job_type("shixiseng", "某岗位（兼职）") != TYPE_PARTTIME:
        raise AssertionError("实习僧上的兼职没被识别")
    if classify_job_type("ncss", "兼职助教") != TYPE_PARTTIME:
        raise AssertionError("ncss 上的兼职没被识别")
    return "兼职优先于平台兜底"


def _unknown_platform_stays_unknown():
    """不认识的平台 + 标题没有类型词 → 不硬猜（返回空，按不限处理）。"""
    if classify_job_type("boss", "算法工程师") != "":
        raise AssertionError("未知平台不该硬猜类型")
    if classify_job_type("", "") != "":
        raise AssertionError("空标题不该判出类型")
    return '未知 → ""（调用方按不限处理）'


def _explicit_wins():
    """上游已经打了标就直接采信（抓取器给什么是什么）。"""
    if classify_job_type("ncss", "算法工程师", explicit="实习") != TYPE_INTERN:
        raise AssertionError("显式 job_type 没被采信")
    if classify_job_type("shixiseng", "Java开发实习生", explicit="正式") != TYPE_FULLTIME:
        raise AssertionError("显式 job_type 没被采信（正式）")
    if classify_job_type("shixiseng", "X", explicit="intern") != TYPE_INTERN:
        raise AssertionError("英文 intern 没被归一")
    if classify_job_type("shixiseng", "X", explicit="火星岗") != TYPE_INTERN:
        raise AssertionError("认不出来的显式值应回落到规则判定")
    return "显式值优先，认不出来才回落规则"


def _normalize_variants():
    table = {
        "实习": TYPE_INTERN, "实习生": TYPE_INTERN, "intern": TYPE_INTERN,
        "Internship": TYPE_INTERN, "兼职": TYPE_PARTTIME, "part-time": TYPE_PARTTIME,
        "正式": TYPE_FULLTIME, "全职": TYPE_FULLTIME, "社招": TYPE_FULLTIME,
        "fulltime": TYPE_FULLTIME, "": "", None: "", "火星": "",
    }
    bad = [f"{k!r} -> {normalize_job_type(k)!r} 期望 {v!r}"
           for k, v in table.items() if normalize_job_type(k) != v]
    if bad:
        raise AssertionError("；".join(bad))
    return f"{len(table)} 种写法归一正确"


# ---------------------------------------------------------------------------
section("2. 查询侧：认类型词 / 摘类型词")


def _detect_query_type():
    cases = [
        ("帮我找广州的 agent 的实习岗位", TYPE_INTERN),
        ("帮我找广州的 agent 岗位", ""),
        ("有没有正式岗", TYPE_FULLTIME),
        ("找一份兼职", TYPE_PARTTIME),
        ("有没有校招岗位", TYPE_FULLTIME),
        ("Intern 岗位有吗", TYPE_INTERN),
        ("今天天气真好", ""),
    ]
    bad = [f"{q!r} -> {detect_query_type(q)!r} 期望 {w!r}"
           for q, w in cases if detect_query_type(q) != w]
    if bad:
        raise AssertionError("；".join(bad))
    return f"{len(cases)} 句全对（无关句不误判）"


def _strip_type_words():
    out = strip_query_type_words("帮我找广州的 agent 的实习岗位")
    if "实习" in out:
        raise AssertionError(f"类型词没摘掉：{out!r}")
    if "agent" not in out.lower() or "广州" not in out:
        raise AssertionError(f"不该动其它内容：{out!r}")
    plain = strip_query_type_words("帮我找广州的 agent 岗位")
    if plain != "帮我找广州的 agent 岗位":
        raise AssertionError(f"没有类型词时不应改动：{plain!r}")
    return f"{out.strip()!r}（业务词原样保留）"


def _matches_helper():
    if not matches("", "", "ncss", "算法工程师"):
        raise AssertionError("wanted 为空应一律放行")
    # 行里没有类型时按平台/标题现场推
    if not matches("", TYPE_INTERN, "niuke", "算法工程师"):
        raise AssertionError("牛客未打标的行应按实习放行")
    if matches("", TYPE_INTERN, "ncss", "算法工程师"):
        raise AssertionError("ncss 未打标的正式岗不该被当成实习")
    if matches(TYPE_FULLTIME, TYPE_INTERN, "ncss", "算法工程师"):
        raise AssertionError("已打标的行不该被平台规则覆盖")
    return "空 wanted 放行 + 未打标时按同一份口径现场推"


# ---------------------------------------------------------------------------
section("3. 抓取器打标（真调用 _to_raw_job，不联网）")


def _scrapers_tag():
    from agent.scrapers.ncss import NcssScraper
    from agent.scrapers.niuke import NiukeScraper

    niuke_item = {"data": {"id": 12345, "jobName": "大模型算法", "jobCity": "广州",
                           "salaryMin": 300, "salaryMax": 530,
                           "recommendInternCompany": {"companyName": "某公司"},
                           "ext": "{}"}}
    job = NiukeScraper()._to_raw_job(niuke_item)
    if job is None or job.job_type != TYPE_INTERN:
        raise AssertionError(f"牛客打标失败：{getattr(job, 'job_type', None)!r}")

    ncss_item = {"jobId": "AAAA", "jobName": "算法工程师", "recName": "某公司",
                 "lowMonthPay": 10, "highMonthPay": 14}
    job2 = NcssScraper()._to_raw_job(ncss_item, city="广州")
    if job2 is None or job2.job_type != TYPE_FULLTIME:
        raise AssertionError(f"ncss 打标失败：{getattr(job2, 'job_type', None)!r}")

    ncss_intern = NcssScraper()._to_raw_job(
        {**ncss_item, "jobName": "AI 算法实习生"}, city="广州")
    if ncss_intern is None or ncss_intern.job_type != TYPE_INTERN:
        raise AssertionError(f"ncss 实习打标失败：{getattr(ncss_intern, 'job_type', None)!r}")

    d = job2.to_dict()
    if "job_type" not in d:
        raise AssertionError("RawJob.to_dict() 丢了 job_type")
    return "牛客=实习 / ncss=正式 / ncss实习生=实习，且 to_dict 带该字段"


def _shixiseng_tag():
    from agent.scrapers.shixiseng import ShixisengScraper

    job = ShixisengScraper.to_raw_job({
        "platform": "shixiseng", "job_id": "inn_x", "title": "AI Agent开发（可转正）",
        "company": "墨泊可士", "city": "广州", "salary": "200-300/天",
        "url": "https://www.shixiseng.com/intern/inn_x", "description": "x" * 300,
    })
    if job.job_type != TYPE_INTERN:
        raise AssertionError(f"实习僧打标失败：{job.job_type!r}")
    return "实习僧（type=intern 平台属性）→ 实习"


# ---------------------------------------------------------------------------
section("4. 清洗 + 落库：字段一路不丢")


def _cleaner_carries_type():
    sys.path.insert(0, str(REPO / "rag" / "quality"))
    import datetime
    import importlib
    cleaner = importlib.import_module("cleaner")

    if "job_type" not in cleaner.JOB_FIELDS:
        raise AssertionError("cleaner.JOB_FIELDS 没有 job_type")
    today = datetime.date.today()
    fresh = (today - datetime.timedelta(days=1)).isoformat()
    raw = [
        {"platform": "niuke", "job_id": "n1", "title": "大模型算法", "company": "A",
         "city": "广州", "salary": "300/天", "url": "u", "description": "x" * 300,
         "publish_date": fresh},
        {"platform": "ncss", "job_id": "c1", "title": "算法工程师", "company": "B",
         "city": "广州", "salary": "10-14/月", "url": "u", "description": "y" * 300,
         "publish_date": fresh},
    ]
    out = cleaner.clean_jobs(raw, city="广州")["jobs"]
    if len(out) != 2:
        raise AssertionError(f"清洗把用例数据滤掉了（应留 2 条）：{len(out)} 条")
    kinds = {j["job_id"]: j["job_type"] for j in out}
    if kinds.get("n1") != TYPE_INTERN or kinds.get("c1") != TYPE_FULLTIME:
        raise AssertionError(f"清洗后类型不对：{kinds}")
    # 上游已打标的原样保留，不被规则改写
    raw[0]["job_type"] = "兼职"
    out2 = cleaner.clean_jobs(raw, city="广州")["jobs"]
    if {j["job_id"]: j["job_type"] for j in out2}.get("n1") != TYPE_PARTTIME:
        raise AssertionError("清洗把上游打好的类型改掉了")
    return "clean_jobs 输出带 job_type，且尊重上游已打标的值"


def _db_roundtrip():
    from rag.data import db as D

    db_path = _TMP_DIR / f"jobs_{os.getpid()}.db"
    if db_path.exists():
        db_path.unlink()
    real_path = D.DB_PATH
    D.DB_PATH = db_path
    try:
        stats = D.upsert_jobs([
            {"job_id": "n1", "platform": "niuke", "title": "大模型算法", "city": "广州",
             "description": "x", "publish_date": "2099-01-01"},
            {"job_id": "c1", "platform": "ncss", "title": "算法工程师", "city": "广州",
             "description": "y", "publish_date": "2099-01-01"},
        ])
        if stats["added"] != 2:
            raise AssertionError(f"入库条数不对：{stats}")
        got = D.search_jobs(keyword="", city="广州", limit=0)
        kinds = {r["job_id"]: r["job_type"] for r in got}
        if kinds != {"n1": TYPE_INTERN, "c1": TYPE_FULLTIME}:
            raise AssertionError(f"库内类型不对：{kinds}")
        intern = D.search_jobs(keyword="", city="广州", limit=0, job_type="实习")
        full = D.search_jobs(keyword="", city="广州", limit=0, job_type="正式")
        if [r["job_id"] for r in intern] != ["n1"]:
            raise AssertionError(f"实习过滤不对：{[r['job_id'] for r in intern]}")
        if [r["job_id"] for r in full] != ["c1"]:
            raise AssertionError(f"正式过滤不对：{[r['job_id'] for r in full]}")
        return "入库自动打标 + job_type 过滤命中正确"
    finally:
        D.DB_PATH = real_path


def _migration_on_legacy_db():
    """老库（无 job_type 列）迁移：补列 + 回填 + 过滤立刻可用。"""
    from rag.data import db as D

    db_path = _TMP_DIR / f"legacy_{os.getpid()}.db"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    con.executescript(
        "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, platform TEXT NOT NULL DEFAULT '',"
        " title TEXT NOT NULL DEFAULT '', company TEXT NOT NULL DEFAULT '',"
        " city TEXT NOT NULL DEFAULT '', salary TEXT NOT NULL DEFAULT '',"
        " url TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '',"
        " publish_date TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '',"
        " updated_at TEXT NOT NULL DEFAULT (datetime('now')));"
    )
    con.execute("INSERT INTO jobs (job_id, platform, title, city, description)"
                " VALUES ('old1','shixiseng','Java开发（可转正）','广州','x')")
    con.execute("INSERT INTO jobs (job_id, platform, title, city, description)"
                " VALUES ('old2','ncss','算法工程师','广州','y')")
    con.commit()
    con.close()

    real_path = D.DB_PATH
    D.DB_PATH = db_path
    try:
        stats = D.migrate_db()
        if not stats["column_added"]:
            raise AssertionError(f"没有补列：{stats}")
        if stats["backfilled"] != 2:
            raise AssertionError(f"回填条数不对：{stats}")
        again = D.migrate_db()
        if again["column_added"] or again["backfilled"]:
            raise AssertionError(f"迁移不幂等：{again}")
        cols = sqlite3.connect(str(db_path)).execute("PRAGMA table_info(jobs)").fetchall()
        if "job_type" not in {c[1] for c in cols}:
            raise AssertionError("补列失败")
        kinds = {r["job_id"]: r["job_type"]
                 for r in D.search_jobs(keyword="", city="广州", limit=0)}
        if kinds != {"old1": TYPE_INTERN, "old2": TYPE_FULLTIME}:
            raise AssertionError(f"回填结果不对：{kinds}")
        return "老库补列 + 回填 2 条 + 二次调用零变更"
    finally:
        D.DB_PATH = real_path


def _db_types_match_classifier():
    """防漂移：真库里每一行的 job_type 都必须等于分类器现算的结果。

    这条是**两套口径**之间的闸门：db.py 的 SQL 兜底规则与 shared/job_type.py
    的判定表必须逐条一致，任何一边改了另一边，这里立刻红。
    """
    from rag.data import db as D

    if not D.DB_PATH.exists():
        return "（真库不存在，跳过）"
    con = sqlite3.connect(str(D.DB_PATH))
    rows = con.execute("SELECT job_id, platform, title, job_type FROM jobs").fetchall()
    con.close()
    bad = []
    for job_id, platform, title, stored in rows:
        want = classify_job_type(platform, title)
        if (stored or "") != want:
            bad.append(f"{job_id} [{platform}] {title[:24]!r}: 库内 {stored!r} != 现算 {want!r}")
    if bad:
        raise AssertionError(f"{len(bad)} 行不一致，前 3 条：" + "；".join(bad[:3]))
    return f"{len(rows)} 行库内类型与分类器完全一致"


def _empty_type_unlimited():
    """不传 / 传空 / 传认不出来的值 → 都按不限处理（旧调用方行为不变）。"""
    from rag.data import db as D

    if not D.DB_PATH.exists():
        return "（真库不存在，跳过）"
    base = len(D.search_jobs(keyword="Agent", city="广州", limit=0))
    for value in (None, "", "火星岗"):
        got = len(D.search_jobs(keyword="Agent", city="广州", limit=0, job_type=value))
        if got != base:
            raise AssertionError(f"job_type={value!r} 时 {got} != 不限时的 {base}")
    return f"三种空值都等于不限（{base} 条）"


def _json_fallback_order_stable():
    """兜底 JSON 路：同一天（publish_date 相同）的岗位顺序不随写入顺序漂移。

    Bug 2 的离线复现口径：**同一批数据**打乱顺序写三次 JSON，搜索结果必须逐条一致，
    且同日期组内按 job_id 升序。只按 publish_date 排序时这里会红（顺序跟着输入走）；
    空日期的行必须沉到最后。
    """
    import json
    from agent.tools import job_search as JS

    def row(job_id, date):
        return {"platform": "niuke", "job_id": job_id, "title": f"算法工程师 {job_id}",
                "company": "信投智联", "city": "广州", "salary": "300-500/天",
                "url": "u", "description": "x" * 300, "publish_date": date}

    base = [row("455163", "2026-10-05"), row("455160", "2026-10-05"),
            row("455162", "2026-10-05"), row("455161", "2026-10-05"),
            row("nodate1", "")]
    path = _TMP_DIR / f"cleaned_{os.getpid()}.json"
    original = JS.REAL_JD_PATH
    try:
        orders = []
        for order in (base, list(reversed(base)),
                      [base[2], base[4], base[0], base[3], base[1]]):
            path.write_text(json.dumps(order, ensure_ascii=False), encoding="utf-8")
            JS.REAL_JD_PATH = path                 # ≠ 默认路径 → 强制走 JSON 兜底
            orders.append([j.job_id for j in JS.search_jobs("算法工程师", "广州", 0)])
        if len({tuple(o) for o in orders}) != 1:
            raise AssertionError(f"同日并列顺序不稳定：{orders}")
        if orders[0] != ["455160", "455161", "455162", "455163", "nodate1"]:
            raise AssertionError(f"排序口径不符（期望同日升序 + 空日期沉底）：{orders[0]}")
    finally:
        JS.REAL_JD_PATH = original
        if path.exists():
            path.unlink()
    return "同一批数据三种写入顺序 → 结果完全一致（publish_date DESC, job_id ASC）"


# ---------------------------------------------------------------------------
section("5. 真库类型分布（只读，不写）")


def _real_db_distribution():
    from rag.data import db as D

    if not D.DB_PATH.exists():
        return "（真库不存在，跳过）"
    con = sqlite3.connect(str(D.DB_PATH))
    rows = con.execute("SELECT job_type, COUNT(*) FROM jobs GROUP BY 1").fetchall()
    unknown = con.execute(
        "SELECT COUNT(*) FROM jobs WHERE TRIM(IFNULL(job_type,'')) = ''").fetchone()[0]
    con.close()
    if unknown:
        raise AssertionError(f"还有 {unknown} 行没回填类型（迁移没跑全）")
    return " / ".join(f"{k or '(空)'}={v}" for k, v in rows)


check("分类器：实习专属平台（实习僧/牛客）按平台兜底", _intern_platform_default)
check("分类器：ncss 按标题 + 校招池兜底", _ncss_title_based)
check("分类器：兼职优先于平台兜底", _parttime_wins)
check("分类器：未知平台不硬猜", _unknown_platform_stays_unknown)
check("分类器：显式 job_type 优先", _explicit_wins)
check("分类器：类型写法归一", _normalize_variants)
check("查询：从问题里认类型", _detect_query_type)
check("查询：把类型词从关键词里摘掉", _strip_type_words)
check("查询：matches 空值放行 / 未打标现场推", _matches_helper)
check("抓取器：牛客/ncss 打标 + to_dict 带字段", _scrapers_tag)
check("抓取器：实习僧打标", _shixiseng_tag)
check("清洗：clean_jobs 带类型且尊重上游打标", _cleaner_carries_type)
check("库层：入库自动打标 + 类型过滤", _db_roundtrip)
check("库层：老库迁移补列 + 回填 + 幂等", _migration_on_legacy_db)
check("库层：真库逐行类型与分类器一致（防口径漂移）", _db_types_match_classifier)
check("库层：空/未知类型按不限处理", _empty_type_unlimited)
check("兜底 JSON 路：同日并列顺序稳定（Bug 2）", _json_fallback_order_stable)
check("真库：类型分布（0 行未回填）", _real_db_distribution)

print()
print("=" * 74)
print(f"总计：{len(PASS)} 通过 / {len(FAIL)} 失败")
print("=" * 74)
if FAIL:
    for label, err in FAIL:
        print(f"  [FAIL] {label} → {err}")
    sys.exit(1)
sys.exit(0)
