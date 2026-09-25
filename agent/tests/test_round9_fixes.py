# -*- coding: utf-8 -*-
"""Round 9 回归测试：合并落盘 / 关键词组 / limit / 多城市 / 孤儿清理。

**不联网、不调 embedding API、不碰真实 chroma_db/、不写 rag/data/**。
需要写入的用例一律落在自己的一次性临时目录里，跑完删掉。

跑法：
    python agent/tests/test_round9_fixes.py

为什么把这些用例单独落一份文件（而不是只塞进 scheduler --selftest）：
scheduler 的自测走的是"调度器视角"（关键词解析 + 全链路），
这里补的是"函数契约视角"——merge_jds 的三场景、limit=0 的语义、
cleanup_orphans 的边界、真实文件零改动。两者的失败信息互不遮挡。
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import uuid
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# rag/quality 没有 __init__.py（不是包），按模块名导入 cleaner
_QUALITY_DIR = str(REPO / "rag" / "quality")
if _QUALITY_DIR not in sys.path:
    sys.path.insert(0, _QUALITY_DIR)

import cleaner                                                    # noqa: E402
from agent.scrapers import scheduler as S                          # noqa: E402
from agent.tools.job_search import search_jobs                     # noqa: E402
from rag import vector_store as vs                                 # noqa: E402

TODAY = date.today()
REAL_JSON = REPO / "rag" / "data" / "cleaned_jd.json"
REAL_TXT = REPO / "rag" / "data" / "scraped_jd.txt"

PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f" → {detail}" if detail else ""))
    except Exception as exc:                                      # noqa: BLE001
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
# 真实文件指纹（跑完必须一模一样）
# ---------------------------------------------------------------------------
GUARDED = [
    REAL_JSON,
    REAL_TXT,
    REPO / ".env",
    REPO / "agent" / "data" / "applications.db",
]


def _fingerprint():
    out = {}
    for p in GUARDED:
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    chroma = REPO / "chroma_db"
    if chroma.is_dir():
        # 用内容哈希：只比 size/mtime 不足以证明"没动过"
        for f in sorted(chroma.rglob("*")):
            if f.is_file():
                out[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


BEFORE = _fingerprint()
REAL_JOBS = json.loads(REAL_JSON.read_text(encoding="utf-8"))

# 临时目录放系统 temp；沙箱下不可写时退回仓库内的一次性目录
try:
    _probe = Path(tempfile.gettempdir()) / f"round9_probe_{uuid.uuid4().hex[:8]}"
    _probe.mkdir(parents=True, exist_ok=True)
    _probe.rmdir()
    _TMP_ROOT = Path(tempfile.gettempdir())
except OSError:                                                   # pragma: no cover
    _TMP_ROOT = REPO
WORK = _TMP_ROOT / f"round9_test_{uuid.uuid4().hex[:8]}"
WORK.mkdir(parents=True, exist_ok=True)


def mk(job_id, days_ago, **kw):
    """造一条能过 cleaner 全部过滤的岗位（正文 > 200 字）"""
    job = {
        "platform": "test",
        "job_id": job_id,
        "title": f"Agent 开发实习生{job_id}",
        "company": f"测试公司{job_id}",
        "city": "广州",
        "salary": "200-300/天",
        "url": f"https://example.com/{job_id}",
        "publish_date": (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
        "description": "【岗位职责】参与 Agent 与 RAG 检索链路开发，负责把大模型能力"
                       "落到业务场景。【任职要求】熟悉 Python 与 LLM 应用开发。" * 4,
    }
    job.update(kw)
    return job


# ===========================================================================
section("任务 1：merge_jds 合并落盘")
# ===========================================================================
EXISTING = [mk(f"ex_{i}", i) for i in range(15)]


def t_scene1():
    r = cleaner.merge_jds(EXISTING, [mk("n1", 0), mk("n2", 1), mk("n3", 2)], today=TODAY)
    if len(r["jobs"]) != 18 or r["stats"]["duplicates"] != 0:
        _fail(f"15+3 应 18 条无重复：{r['stats']}")
    return f"15 + 3（不同 id）→ {len(r['jobs'])} 条"


def t_scene2():
    r = cleaner.merge_jds(EXISTING, copy.deepcopy(EXISTING), today=TODAY)
    if len(r["jobs"]) != 15 or r["stats"]["duplicates"] != 15:
        _fail(f"15+15 重复应 15 条：{r['stats']}")
    return f"15 + 15（完全重复）→ {len(r['jobs'])} 条，识别重复 {r['stats']['duplicates']}"


def t_scene3():
    r = cleaner.merge_jds(EXISTING, [mk("n1", 0), mk("n2", 1), mk("s91", 91)], today=TODAY)
    if len(r["jobs"]) != 17 or len(r["archived"]) != 1:
        _fail(f"含 91 天前应 17 条 + 归档 1：{r['stats']}")
    if r["archived"][0]["job_id"] != "s91":
        _fail("归档的应是那条 91 天前的记录")
    return f"15 + 3（含 1 条 91 天前）→ {len(r['jobs'])} 条，归档 {len(r['archived'])} 条"


def t_sort_desc():
    dates = [j["publish_date"] for j in cleaner.merge_jds(EXISTING, [mk("x", 0)], today=TODAY)["jobs"]]
    if dates != sorted(dates, reverse=True):
        _fail(f"未降序：{dates}")
    return f"降序 OK（{dates[0]} → {dates[-1]}）"


def t_empty_date_last():
    jobs = [mk("a", 0), mk("b", 0, publish_date=""), mk("c", 0, publish_date="2026-13-99")]
    r = cleaner.merge_jds(jobs, [], today=TODAY)
    flags = [cleaner._parse_date(j["publish_date"]) is not None for j in r["jobs"]]
    if flags != [True, False, False]:
        _fail(f"空/脏日期应垫底：{flags}")
    if r["archived"]:
        _fail("无法解析的日期不应被归档（保守保留）")
    return "空/脏日期垫底且不误归档"


def t_new_wins():
    r = cleaner.merge_jds([dict(mk("s", 5), title="旧")], [dict(mk("s", 5), title="新")], today=TODAY)
    if len(r["jobs"]) != 1 or r["jobs"][0]["title"] != "新":
        _fail(f"同 id 应以新为准：{r['jobs']}")
    return "同 job_id 以新数据为准"


def t_archive_append():
    ap = WORK / "archive" / cleaner.ARCHIVE_FILENAME
    cleaner.merge_jds([], [mk("a1", 100)], archive_path=ap, today=TODAY)
    cleaner.merge_jds([], [mk("a2", 120)], archive_path=ap, today=TODAY)
    n = len(json.loads(ap.read_text(encoding="utf-8")))
    if n != 2:
        _fail(f"归档被覆盖（期望 2，实际 {n}）")
    return f"归档为追加：两次运行后 {n} 条"


def t_stale_boundary():
    b90 = cleaner.merge_jds([], [mk("d90", 90)], today=TODAY)
    b91 = cleaner.merge_jds([], [mk("d91", 91)], today=TODAY)
    if len(b90["jobs"]) != 1 or b90["archived"]:
        _fail("恰好 90 天应保留")
    if b91["jobs"] or len(b91["archived"]) != 1:
        _fail("91 天应归档")
    return "90 天保留 / 91 天归档"


def t_existing_stale():
    r = cleaner.merge_jds([mk("old", 200)], [mk("fresh", 1)], today=TODAY)
    if len(r["jobs"]) != 1 or len(r["archived"]) != 1:
        _fail(f"已有记录过期也要归档：{r['stats']}")
    return "归档对已有记录同样生效"


def t_no_job_id_dedup():
    a = [{"platform": "p", "job_id": "", "company": "C", "title": "T", "url": "u", "publish_date": ""}]
    r = cleaner.merge_jds(a, copy.deepcopy(a), today=TODAY)
    if len(r["jobs"]) != 1:
        _fail("无 job_id 时应按内容签名兜底去重")
    return "无 job_id 走内容签名兜底去重"


def t_scheduler_reexport():
    if len(S.merge_jds(EXISTING, [mk("x", 0)], today=TODAY)["jobs"]) != 16:
        _fail("scheduler.merge_jds 结果不对")
    return "scheduler.merge_jds 入口可用（15+1→16）"


for _label, _fn in [
    ("场景1：15 + 3（不同 job_id）→ 18", t_scene1),
    ("场景2：15 + 15（完全重复）→ 15", t_scene2),
    ("场景3：15 + 3（含 1 条 91 天前）→ 17 + 归档 1", t_scene3),
    ("排序：publish_date 降序", t_sort_desc),
    ("排序：空/脏日期垫底且不归档", t_empty_date_last),
    ("去重：同 job_id 以新数据为准", t_new_wins),
    ("归档文件是追加不是覆盖", t_archive_append),
    ("过期边界：90 保留 / 91 归档", t_stale_boundary),
    ("已有记录过期同样归档", t_existing_stale),
    ("无 job_id 的兜底去重", t_no_job_id_dedup),
    ("scheduler.merge_jds 入口可用", t_scheduler_reexport),
]:
    check(_label, _fn)


# ===========================================================================
section("任务 1b：落盘是「合并」不是「覆盖」（Round 8 事故回归）")
# ===========================================================================


def t_landing_merge():
    out = WORK / "landing"
    out.mkdir(parents=True, exist_ok=True)
    existing = len(REAL_JOBS)                                     # 真实语料条数（随数据扩充而变）
    shutil.copyfile(REAL_JSON, out / "cleaned_jd.json")           # 真实语料副本
    res = S.merge_and_write_cleaned(
        [mk("brand_1", 0), mk("brand_2", 1), mk("brand_3", 2)], out_dir=out, today=TODAY
    )
    written = json.loads((out / "cleaned_jd.json").read_text(encoding="utf-8"))
    if res["stats"]["existing"] != existing:
        _fail(f"应读到 {existing} 条原有数据：{res['stats']}")
    # 落盘数 = 原有 + 本次新增 − 同城镜像折叠掉的条数。
    # 不能写死成 existing + 3：真实语料里可能存在同 (公司,标题,城市) 且正文
    # 逐字一致的镜像，merge_jds 会把它们折成一条（Round 3 的 fold_mirrors）。
    # 从 stats 取 mirrors_folded，数据再变化这条断言依然成立。
    folded = res["stats"]["mirrors_folded"]
    expected = existing + 3 - folded
    if len(written) != expected:
        _fail(f"应 {expected} 条（{existing}+3-{folded} 折叠），实际 {len(written)}：{res['stats']}")
    return (f"原有 {existing} + 本次 3 − 折叠 {folded} → 落盘 {len(written)} 条"
            f"（旧数据未被冲掉）")


def t_txt_full_merged():
    out = WORK / "landing_txt"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REAL_JSON, out / "cleaned_jd.json")
    res = S.merge_and_write_cleaned([mk("txt_new", 0)], out_dir=out, today=TODAY)
    merged = json.loads((out / "cleaned_jd.json").read_text(encoding="utf-8"))
    S.write_rag_text(merged, out)
    txt = (out / "scraped_jd.txt").read_text(encoding="utf-8")
    blocks = txt.count(S.SEPARATOR) // 2
    # 同上：本次 merge 会折掉 mirrors_folded 条镜像，语料块数要相应扣掉，
    # 从 stats 取而不是写死 len(REAL_JOBS) + 1。
    folded = res["stats"]["mirrors_folded"]
    expected = len(REAL_JOBS) + 1 - folded                        # 真实语料 + 本次新增 1 − 折叠
    if blocks != expected:
        _fail(f"txt 应有 {expected} 个岗位块，实际 {blocks}（折叠 {folded}）")
    if "txt_new" not in txt and REAL_JOBS[0]["company"] not in txt:
        _fail("txt 内容不对")
    return f"txt 为合并后全量：{blocks} 个岗位块（折叠 {folded}）"


check("落盘合并：真实全量数据 + 3 条新抓", t_landing_merge)
check("语料 txt 写合并后全量（不是只写本次）", t_txt_full_merged)


# ===========================================================================
section("任务 2：关键词组 + 中文同义词")
# ===========================================================================


def t_default_groups():
    want = [["Agent", "智能体"], ["大模型", "LLM", "大语言模型"], ["RAG", "检索增强生成"]]
    if S.DEFAULT_KEYWORDS != want:
        _fail(f"DEFAULT_KEYWORDS 不符：{S.DEFAULT_KEYWORDS}")
    return S._format_keyword_groups(S.DEFAULT_KEYWORDS)


def t_flatten():
    flat = S.flatten_keyword_groups(S.DEFAULT_KEYWORDS)
    want = ["Agent", "智能体", "大模型", "LLM", "大语言模型", "RAG", "检索增强生成"]
    if flat != want:
        _fail(f"平铺不符：{flat}")
    return f"平铺 {len(flat)} 个词"


def t_parsers():
    cases = [
        (S.DEFAULT_KEYWORDS, S.DEFAULT_KEYWORDS),
        ("Agent,智能体;RAG,检索增强生成", [["Agent", "智能体"], ["RAG", "检索增强生成"]]),
        ("智能体,Agent", [["智能体", "Agent"]]),
        (["Agent,智能体", "RAG"], [["Agent", "智能体"], ["RAG"]]),
        ([["Agent", "智能体"], ["RAG"]], [["Agent", "智能体"], ["RAG"]]),
        (["Agent", "智能体"], [["Agent"], ["智能体"]]),
        ("智能体", [["智能体"]]),
        (None, []),
    ]
    bad = [f"{v!r}→{S.split_keyword_groups(v)}" for v, want in cases
           if S.split_keyword_groups(v) != want]
    if bad:
        _fail("；".join(bad))
    return f"{len(cases)} 种写法全部正确"


def t_profile_priority():
    cfg = S.resolve_config(profile={"target_keywords": ["Agent", "RAG"]})
    if cfg["keyword_groups"] != [["Agent"], ["RAG"]] or cfg["keywords"] != ["Agent", "RAG"]:
        _fail(f"画像解析不对：{cfg['keyword_groups']}")
    return "target_keywords 优先，平铺字段向后兼容"


def t_env_priority():
    import os
    os.environ["SCHEDULER_KEYWORDS"] = "智能体,Agent;大模型"
    try:
        cfg = S.resolve_config(profile={"target_keywords": ["会被覆盖"]})
    finally:
        os.environ.pop("SCHEDULER_KEYWORDS", None)
    if cfg["keyword_groups"] != [["智能体", "Agent"], ["大模型"]]:
        _fail(f"环境变量分组不对：{cfg['keyword_groups']}")
    return "环境变量优先且支持分号分组"


def t_cleaner_accepts_synonyms():
    missing = []
    for group in S.DEFAULT_KEYWORDS:
        for word in group:
            res = cleaner.clean_jobs([{
                "platform": "t", "job_id": f"syn_{word}", "title": f"{word}实习生",
                "company": "C", "city": "广州", "salary": "1", "url": "u",
                "publish_date": TODAY.strftime("%Y-%m-%d"),
                "description": f"【岗位职责】负责{word}方向的工作。" * 12,
            }], city="广州")
            if not res["jobs"]:
                missing.append(word)
    if missing:
        _fail(f"cleaner 不认这些同义词（岗位会在入库前被丢掉）：{missing}")
    return f"scheduler 组的 {sum(len(g) for g in S.DEFAULT_KEYWORDS)} 个词 cleaner 全部认可"


def t_end_to_end_chinese():
    """纯中文岗位（完全不含 RAG / 大模型 字面）端到端命中。"""
    import agent.tools.job_search as js

    zh = [
        mk("zh_a", 0, title="检索增强生成工程师", company="中文岗科技",
           description="【岗位职责】负责检索增强生成系统建设与向量检索优化。" * 8),
        mk("zh_b", 0, title="大语言模型算法实习生", company="中文岗科技",
           description="【岗位职责】参与大语言模型训练与评测，熟悉大语言模型微调。" * 8),
    ]
    if any("rag" in j["description"].lower() for j in zh):
        _fail("样本不该含 'RAG'，测试无效")
    if any("大模型" in j["description"] for j in zh):
        _fail("样本不该含 '大模型'，测试无效")

    kept = cleaner.clean_jobs(zh, city="广州")["jobs"]
    if len(kept) != 2:
        _fail(f"两条中文岗位都应保留，实际 {len(kept)}")

    out = WORK / "zh_e2e"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cleaned_jd.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    original = js.REAL_JD_PATH
    js.REAL_JD_PATH = out / "cleaned_jd.json"
    try:
        n_zh = len(search_jobs("检索增强生成", city="广州", limit=0))
        n_llm = len(search_jobs("大语言模型", city="广州", limit=0))
        n_en = len(search_jobs("RAG", city="广州", limit=0))
    finally:
        js.REAL_JD_PATH = original

    if n_zh != 1 or n_llm != 1:
        _fail(f"中文关键词应各命中 1 条：{n_zh} / {n_llm}")
    if n_en != 0:
        _fail(f"英文 'RAG' 应 0 命中（实际 {n_en}）→ 这才证明增益来自中文同义词")
    return f"『检索增强生成』{n_zh} 条、『大语言模型』{n_llm} 条，英文 'RAG' {n_en} 条 → 0→1 增益"


for _label, _fn in [
    ("DEFAULT_KEYWORDS 为组结构", t_default_groups),
    ("组 → 平铺关键词（7 个）", t_flatten),
    ("组解析器 8 种写法", t_parsers),
    ("画像 target_keywords 优先", t_profile_priority),
    ("环境变量优先 + 支持分组", t_env_priority),
    ("cleaner 认全部中文同义词", t_cleaner_accepts_synonyms),
    ("端到端：纯中文岗位 0→1 增益", t_end_to_end_chinese),
]:
    check(_label, _fn)


# ===========================================================================
section("任务 3：search_jobs limit")
# ===========================================================================
import inspect                                                    # noqa: E402


def t_default_50():
    default = inspect.signature(search_jobs).parameters["limit"].default
    if default != 50:
        _fail(f"默认 limit 应为 50，实际 {default}")
    return f"默认 limit = {default}"


def t_zero_unlimited():
    n50 = len(search_jobs("Agent", city="广州", limit=50))
    n0 = len(search_jobs("Agent", city="广州", limit=0))
    n_none = len(search_jobs("Agent", city="广州", limit=None))
    if n0 != n50 or n_none != n50:
        _fail(f"limit=0/None 应等于不限：{n0}/{n_none} != {n50}")
    return f"真实数据 limit=50 → {n50} 条；limit=0/None → {n0}/{n_none} 条（不限）"


def t_mechanism_more():
    """真实语料没有任何关键词命中 >10 条，机制用合成 25 条验证。"""
    import agent.tools.job_search as js

    out = WORK / "limit_mech"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cleaned_jd.json").write_text(
        json.dumps([mk(f"lim_{i}", i) for i in range(25)], ensure_ascii=False, indent=2),
        encoding="utf-8")
    original = js.REAL_JD_PATH
    js.REAL_JD_PATH = out / "cleaned_jd.json"
    try:
        n10 = len(search_jobs("Agent", city="广州", limit=10))
        n50 = len(search_jobs("Agent", city="广州", limit=50))
        n0 = len(search_jobs("Agent", city="广州", limit=0))
    finally:
        js.REAL_JD_PATH = original
    if not (n10 == 10 and n50 == 25 and n0 == 25):
        _fail(f"截断机制不对：10→{n10}, 50→{n50}, 0→{n0}")
    return f"25 条命中：limit=10 → {n10} 条，limit=50 → {n50} 条，limit=0 → {n0} 条"


def t_real_data_baseline():
    """如实记录真实语料的命中上限（不预设立场，只为说明 limit=50 的效果）。"""
    rows = []
    for kw in ["Agent", "大模型", "RAG", "Python"]:
        n10 = len(search_jobs(kw, city="广州", limit=10))
        n50 = len(search_jobs(kw, city="广州", limit=50))
        rows.append(f"{kw}:10→{n10}/50→{n50}")
    return " / ".join(rows) + "（真实语料命中 ≤9 条，故 50 与 10 无差）"


def t_limit5_no_regression():
    n = len(search_jobs("Agent", city="广州", limit=5))
    if n != 5:
        _fail(f"limit=5 应精确 5 条，实际 {n}")
    return f"limit=5 仍精确返回 {n} 条"


def t_unlimited_unique():
    jobs = search_jobs("Agent", city="广州", limit=0)
    ids = [j.job_id for j in jobs]
    if len(ids) != len(set(ids)):
        _fail("有重复 job_id")
    return f"不限时 {len(jobs)} 条、job_id 无重复"


for _label, _fn in [
    ("默认 limit = 50", t_default_50),
    ("limit=0 / None 表示不限", t_zero_unlimited),
    ("合成 25 条：limit=50 机械生效", t_mechanism_more),
    ("真实数据 limit 实测（如实记录）", t_real_data_baseline),
    ("limit=5 无回归", t_limit5_no_regression),
    ("不限时结果无重复", t_unlimited_unique),
]:
    check(_label, _fn)


# ===========================================================================
section("任务 4：多城市抓取")
# ===========================================================================
CALLS: list = []


def fake_scraper(keywords, city=None, **kwargs):
    CALLS.append({"city": city, "keywords": list(keywords), **kwargs})
    if city == "广州":
        return [mk("gz_1", 0), mk("gz_2", 1), mk("shared_1", 2, city="全国")]
    if city == "深圳":
        return [mk("sz_1", 0, city="深圳"), mk("sz_2", 1, city="深圳"),
                mk("shared_1", 2, city="全国")]
    return []


def t_all_cities():
    CALLS.clear()
    res = S.scrape_all_cities(["广州", "深圳"], ["Agent", "智能体"], scraper=fake_scraper)
    if [c["city"] for c in CALLS] != ["广州", "深圳"]:
        _fail(f"应逐个城市调用：{[c['city'] for c in CALLS]}")
    if res["per_city"] != {"广州": 3, "深圳": 3}:
        _fail(f"分城市统计不对：{res['per_city']}")
    if len(res["jobs"]) != 5 or res["raw_total"] != 6:
        _fail(f"去重应 6→5：{res['raw_total']} → {len(res['jobs'])}")
    return f"两城市都跑，分城市 {res['per_city']}，去重 6→{len(res['jobs'])}"


def t_keywords_passed():
    CALLS.clear()
    S.scrape_all_cities(["广州"], ["Agent", "智能体"], scraper=fake_scraper)
    if CALLS[0]["keywords"] != ["Agent", "智能体"]:
        _fail(f"关键词未完整传入：{CALLS[0]['keywords']}")
    return f"关键词平铺传入：{CALLS[0]['keywords']}"


def t_per_city_log():
    log_path = WORK / "multi_city.log"
    logger = S.setup_logger(log_path=log_path, verbose=False)
    CALLS.clear()
    S.scrape_all_cities(["广州", "深圳"], ["Agent"], scraper=fake_scraper, logger=logger)
    for h in logger.handlers:
        h.flush()
    text = log_path.read_text(encoding="utf-8")
    missing = [c for c in ("[城市 广州]", "[城市 深圳]") if c not in text]
    if missing:
        _fail(f"缺少每城市日志 {missing}")
    lines = [ln for ln in text.splitlines() if "[城市 " in ln]
    return f"{len(lines)} 行城市日志：" + " ‖ ".join(ln.split("] ", 1)[1] for ln in lines)


def t_city_failure_isolated():
    def flaky(keywords, city=None, **kwargs):
        if city == "深圳":
            raise RuntimeError("模拟深圳失败")
        return [mk("gz_only", 0)]

    res = S.scrape_all_cities(["广州", "深圳"], ["Agent"], scraper=flaky)
    if len(res["jobs"]) != 1 or "深圳" not in res["failed"]:
        _fail(f"单城市失败应隔离：{res['per_city']} / {res['failed']}")
    return "深圳失败被隔离，广州仍抓到 1 条"


def t_end_to_end_multi_city():
    out = WORK / "e2e_multi"
    out.mkdir(parents=True, exist_ok=True)
    real_db, real_embed = vs.DB_PATH, vs.embed_texts
    vs.DB_PATH = str(out / "chroma")
    vs.embed_texts = lambda texts: [[float(len(t) % 7), 1.0, 0.5] for t in texts]
    try:
        import chromadb
        chromadb.PersistentClient(path=str(vs.DB_PATH)).get_or_create_collection(
            name="jd_chunks", metadata={"hnsw:space": "cosine"})
        CALLS.clear()
        cfg = {"keyword_groups": [["Agent", "智能体"]], "keywords": ["Agent", "智能体"],
               "cities": ["广州", "深圳"], "city": "广州", "max_pages": 1,
               "limit_per_keyword": 5, "limit_total": 10}
        result = S.run_daily_job(config=cfg, out_dir=out, state_file=out / "state.json",
                                 logger=S.setup_logger(log_path=out / "job.log", verbose=False),
                                 scraper=fake_scraper)
    finally:
        vs.DB_PATH, vs.embed_texts = real_db, real_embed

    if not result["ok"]:
        _fail(f"任务失败：{result['error']}")
    if sorted(result["per_city"]) != ["广州", "深圳"]:
        _fail(f"per_city 应含两城市：{result['per_city']}")
    # 断言「去重后」的条数要用 cleaned，不能用 scraped：
    # scraped = len(jobs)（平台 × 城市 × 关键词的原始累加，见 scheduler.py:1841），
    # 而 scrape_multi_platform 按 (platform, job_id) 去重、**跨平台刻意不去重**
    # （scheduler.py:926-927 有明确注释），所以 2 个平台就是 10 条，这是既定设计。
    # cleaned 才是跨城聚合去重后的结果，也就是这里想验的「去重后应 5 条」。
    if result["cleaned"] != 5:
        _fail(f"去重后应 5 条，实际 {result['cleaned']}")
    written = json.loads((out / "cleaned_jd.json").read_text(encoding="utf-8"))
    if len(written) != 5:
        _fail(f"落盘应 5 条，实际 {len(written)}")
    # per_city 记的是**原始抓取量**（平台 × 城市 × 关键词，未跨平台去重）：
    # 2 个平台 × 2 个关键词 × 每城 3 条 = 12，不是去重后的 3。
    # 去重后的条数以 result["cleaned"] 为准（上面已断言 = 5）。
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    if state.get("per_city") != {"广州": 12, "深圳": 12}:
        _fail(f"state.per_city 不对：{state.get('per_city')}")
    return (f"两城市都抓到（{[c['city'] for c in CALLS]}），落盘 {len(written)} 条，"
            f"state.per_city={state['per_city']}")


for _label, _fn in [
    ("遍历所有城市（不只第一个）", t_all_cities),
    ("关键词完整传给每个城市", t_keywords_passed),
    ("每个城市一行日志", t_per_city_log),
    ("单城市失败不拖垮整轮", t_city_failure_isolated),
    ("端到端 run_daily_job 多城市", t_end_to_end_multi_city),
]:
    check(_label, _fn)


# ===========================================================================
section("任务 5：cleanup_orphans")
# ===========================================================================


def _fake_collection(n):
    import chromadb
    path = WORK / f"orphan_{n}_{len(PASS)}"
    path.mkdir(parents=True, exist_ok=True)
    col = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
        name="jd_chunks", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"chunk_{i}" for i in range(n)],
        documents=[f"文档 {i}" for i in range(n)],
        embeddings=[[float(i), 1.0, 0.5] for i in range(n)],
        metadatas=[{"company": "C", "title": "T", "city": "广州", "chunk_index": str(i)}
                   for i in range(n)],
    )
    return col


def t_delete_orphans():
    col = _fake_collection(5)
    deleted = vs.cleanup_orphans(col, ["chunk_0", "chunk_1", "chunk_2"])
    remaining = sorted(col.get()["ids"])
    if deleted != 2 or remaining != ["chunk_0", "chunk_1", "chunk_2"]:
        _fail(f"应删 2 留 3：deleted={deleted}, 剩余={remaining}")
    return f"5 条留 3 条 → 删除 {deleted} 条，剩余 {remaining}"


def t_no_orphans():
    col = _fake_collection(3)
    deleted = vs.cleanup_orphans(col, ["chunk_0", "chunk_1", "chunk_2"])
    if deleted != 0 or len(col.get()["ids"]) != 3:
        _fail(f"无孤儿应删 0：{deleted}")
    return "无孤儿时返回 0 且不动数据"


def t_empty_valid_ids():
    col = _fake_collection(4)
    for empty in ([], set(), None):
        if vs.cleanup_orphans(col, empty) != 0:
            _fail(f"valid_ids={empty!r} 应保守返回 0")
    if len(col.get()["ids"]) != 4:
        _fail("保守策略下不该删数据")
    return "valid_ids 为空 → 返回 0，一条都不删（防误清库）"


def t_set_and_unknown():
    col = _fake_collection(3)
    deleted = vs.cleanup_orphans(col, {"chunk_0", "不存在的id"})
    if deleted != 2:
        _fail(f"应删 2，实际 {deleted}")
    return f"接受 set，忽略未知 id（删 {deleted} 条）"


def t_batching():
    col = _fake_collection(7)
    orig = vs.ORPHAN_DELETE_BATCH
    vs.ORPHAN_DELETE_BATCH = 2
    try:
        deleted = vs.cleanup_orphans(col, ["chunk_0"])
    finally:
        vs.ORPHAN_DELETE_BATCH = orig
    if deleted != 6 or sorted(col.get()["ids"]) != ["chunk_0"]:
        _fail(f"分批删除不对：{deleted}")
    return f"分批删除（batch=2）仍正确删 {deleted} 条"


def t_not_auto_called():
    src = (REPO / "rag" / "vector_store.py").read_text(encoding="utf-8")
    if "def cleanup_orphans" not in src:
        _fail("cleanup_orphans 未定义")
    calls = [ln for ln in src.splitlines()
             if "cleanup_orphans(" in ln and not ln.strip().startswith("def ")
             and "cleanup_orphans：" not in ln]
    if calls:
        _fail(f"不应自动调用，但有调用点：{calls}")
    return "函数已提供，全模块无自动调用点"


for _label, _fn in [
    ("删除孤儿并返回数量", t_delete_orphans),
    ("无孤儿时返回 0", t_no_orphans),
    ("valid_ids 为空时不误删整库", t_empty_valid_ids),
    ("接受 set / 忽略未知 id", t_set_and_unknown),
    ("分批删除", t_batching),
    ("只提供函数、不自动调用", t_not_auto_called),
]:
    check(_label, _fn)


# ===========================================================================
section("收尾：真实文件零改动")
# ===========================================================================
AFTER = _fingerprint()


def t_guard():
    changed = [k for k in set(BEFORE) | set(AFTER) if BEFORE.get(k) != AFTER.get(k)]
    if changed:
        _fail(f"以下受保护文件被改动：{changed}")
    return f"{len(GUARDED)} 个受保护文件 + chroma_db/（内容哈希）全部一致"


check("cleaned_jd.json / scraped_jd.txt / .env / applications.db / chroma_db 未变", t_guard)

print()
print("=" * 74)
print(f"总计：{len(PASS)} 通过 / {len(FAIL)} 失败")
for _label, _err in FAIL:
    print(f"  ✗ {_label} → {_err}")
print("=" * 74)
shutil.rmtree(WORK, ignore_errors=True)
raise SystemExit(1 if FAIL else 0)
