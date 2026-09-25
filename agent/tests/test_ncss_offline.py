# -*- coding: utf-8 -*-
"""ncss 抓取器（agent/scrapers/ncss.py）离线回归测试。

跑法：
    python agent/tests/test_ncss_offline.py

**不联网、不读真实数据、不写 rag/data/**——这里测的全是纯函数与离线契约
（城市映射 / 字段映射 / 限速 / 未收录城市不瞎猜 / 落库所需字段齐备）。
真实 HTTP 行为的验证在本轮验证脚本里做（北京 × 3 词小批量）。

为什么要单独一个文件：
`niuke.py` 当年只靠 `scheduler --selftest` 覆盖，结果接口字段口径的变化
（薪资单位、镜像折叠）在自测里看不出来。ncss 的字段映射同样有"单位是千元/月"
这种容易写错的口径，值得一组不依赖网络的固定断言。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.scrapers.base import PlatformScraper, RawJob       # noqa: E402
from agent.scrapers import ncss as N                            # noqa: E402

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
section("1. 类契约（能否被调度器直接登记）")
# ---------------------------------------------------------------------------
def t_is_subclass():
    if not issubclass(N.NcssScraper, PlatformScraper):
        _fail("NcssScraper 不是 PlatformScraper 子类")
    return "继承 PlatformScraper"


def t_platform_name():
    if N.NcssScraper.platform_name != "ncss":
        _fail(f"platform_name={N.NcssScraper.platform_name!r}，应为 'ncss'")
    if N.NcssScraper().platform_name != "ncss":
        _fail("实例 platform_name 不是 'ncss'")
    return "platform_name='ncss'（类+实例）"


def t_search_is_coroutine():
    sc = N.NcssScraper()
    if not asyncio.iscoroutinefunction(sc.search):
        _fail("search 不是 async（基类要求 async def）")
    return "search 是 async，签名 (keyword, city=None, limit=20)"


def t_swallows_unknown_kwargs():
    # 调度器会传 headless 等公共参数，不能被 **kwargs 缺失打挂
    sc = N.NcssScraper(headless=True, detail_concurrency=4, interval=0)
    if sc.interval != 0:
        _fail("显式 interval 未生效")
    return "**kwargs 吞掉调度器公共参数"


check("NcssScraper 继承 PlatformScraper", t_is_subclass)
check("platform_name == 'ncss'", t_platform_name)
check("search 是 async", t_search_is_coroutine)
check("吞掉未知公共参数", t_swallows_unknown_kwargs)

# ---------------------------------------------------------------------------
section("2. 城市 → areaCode 映射（6 城，不认识就报告）")
# ---------------------------------------------------------------------------
EXPECTED = {"广州": "440100", "深圳": "440300", "北京": "110000",
            "上海": "310000", "杭州": "330100", "成都": "510100"}


def t_six_cities():
    if N.CITY_AREA_CODES != EXPECTED:
        _fail(f"映射表不符：{N.CITY_AREA_CODES}")
    return f"{len(EXPECTED)} 城：{N.CITY_AREA_CODES}"


def t_resolve_known():
    for city, ac in EXPECTED.items():
        got = N.resolve_area_code(city)
        if got != ac:
            _fail(f"{city} -> {got!r}，应为 {ac!r}")
    # 带"市"后缀也要认
    if N.resolve_area_code("广州市") != "440100":
        _fail("'广州市' 未被规整到 '广州'")
    return "6 城全部命中；'广州市' 正确规整"


def t_resolve_unknown_is_none():
    for city in ["武汉", "南京", "西安", "纽约", "  武汉  "]:
        got = N.resolve_area_code(city)
        if got is not None:
            _fail(f"未收录城市 {city!r} 应返回 None，实际 {got!r}（不许瞎猜 areaCode）")
    return "武汉/南京/西安/纽约 -> None"


def t_resolve_nationwide():
    for city in [None, "", "全国", "不限", "all"]:
        if N.resolve_area_code(city) is not None:
            _fail(f"{city!r} 应表示不限城市（None）")
    return "None/''/'全国'/'不限'/'all' -> None（不限城市）"


def t_unknown_city_returns_empty_offline():
    """未收录城市必须**不发请求**就返回空列表（离线可验证）。"""
    sc = N.NcssScraper(interval=0)
    sc._request = lambda params: _fail(f"未收录城市不该发请求：{params}")
    jobs = asyncio.run(sc.search("法务", city="武汉", limit=20))
    if jobs:
        _fail(f"未收录城市应返回 []，实际 {len(jobs)} 条")
    return "search('法务', city='武汉') -> []，且零请求"


check("6 城映射表与实测值一致", t_six_cities)
check("已知城市解析正确", t_resolve_known)
check("未收录城市 -> None（不瞎猜）", t_resolve_unknown_is_none)
check("不限城市 -> None", t_resolve_nationwide)
check("未收录城市 search 返回 [] 且不发请求", t_unknown_city_returns_empty_offline)

# ---------------------------------------------------------------------------
section("3. 薪资映射（单位：千元/月）")
# ---------------------------------------------------------------------------
def t_salary_normal():
    if N._fmt_salary(0.2, 0.3) != "0.2-0.3千元/月":
        _fail(N._fmt_salary(0.2, 0.3))
    if N._fmt_salary(4.0, 7.0) != "4-7千元/月":
        _fail(N._fmt_salary(4.0, 7.0))
    if N._fmt_salary(4.5, 4.5) != "4.5千元/月":
        _fail(N._fmt_salary(4.5, 4.5))
    return "0.2-0.3 / 4-7 / 4.5 三种形态正确"


def t_salary_degenerate():
    if N._fmt_salary(0, 0) != "":
        _fail("两边都为 0 应返回空串（不瞎猜）")
    if N._fmt_salary(None, None) != "":
        _fail("两边都缺失应返回空串")
    if N._fmt_salary(0, 5.0) != "5千元/月":
        _fail("只有 high 有值时应用 high")
    if N._fmt_salary(3.0, 0) != "3千元/月":
        _fail("只有 low 有值时应用 low")
    if N._fmt_salary("abc", None) != "":
        _fail("脏数据应返回空串")
    return "0/0、缺失、单边有值、脏数据全部安全"


def t_salary_unit_is_thousands():
    """单位必须是千元/月（实测口径），不能写成 '元/月' 或 '/天'。"""
    out = N._fmt_salary(0.2, 0.3)
    if "千元/月" not in out:
        _fail(f"单位口径错：{out}")
    return f"{out!r} 带 '千元/月' 后缀"


check("薪资常规形态", t_salary_normal)
check("薪资退化/脏数据形态", t_salary_degenerate)
check("单位口径是千元/月", t_salary_unit_is_thousands)

# ---------------------------------------------------------------------------
section("4. 时间戳 / 关键词 / 正文清洗")
# ---------------------------------------------------------------------------
def t_date():
    # 1790258416000 ms = 2026-09-24 14:00:16 UTC（与 niuke.py 同口径，用 UTC 出日期）
    if N._fmt_date(1790258416000) != "2026-09-24":
        _fail(f"毫秒时间戳解析错：{N._fmt_date(1790258416000)}")
    # 站点发布时间戳都落在整点（实测 19:00 / 22:00 / 04:00 UTC），跨时区不会串日
    if N._fmt_date(1790287200000) != "2026-09-24":
        _fail(f"整点时间戳解析错：{N._fmt_date(1790287200000)}")
    for bad in [None, "", 0, -1, "abc"]:
        if N._fmt_date(bad) != "":
            _fail(f"{bad!r} 应返回空串，实际 {N._fmt_date(bad)!r}")
    return "毫秒 -> YYYY-MM-DD；空/脏数据 -> ''"


def t_keyword_match():
    if not N._match_keyword("Java开发工程师", ["java"]):
        _fail("大小写不敏感匹配失败")
    if N._match_keyword("法务专员", ["java"]):
        _fail("不相关标题不该命中")
    if not N._match_keyword("任意标题", []):
        _fail("空关键词应全通过")
    if not N._match_keyword("大模型算法工程师", ["大模型", "算法"]):
        _fail("多关键词 AND 失败")
    return "大小写不敏感 / AND / 空关键词全通过"


def t_platform_prefix_stripped():
    for raw in ["ncss:算法", "ncss 算法", "24365:算法", "ncSS：算法"]:
        got = N._PLATFORM_PREFIX.sub("", raw).strip()
        if got != "算法":
            _fail(f"{raw!r} 剥离后应为 '算法'，实际 {got!r}")
    return "ncss:/ncss /24365: 前缀都能剥掉"


def t_html_to_text():
    html = "一、岗位职责<br/>1、做A；</p><div>2、做B；</div><p>&nbsp;任职要求：本科</p>"
    text = N._html_to_text(html)
    for frag in ["一、岗位职责", "1、做A；", "2、做B；", "任职要求：本科"]:
        if frag not in text:
            _fail(f"正文缺片段 {frag!r}：{text!r}")
    if "<" in text or ">" in text:
        _fail(f"HTML 标签没清干净：{text!r}")
    if "\n" not in text:
        _fail("块级标签应转成换行（保留段落结构）")
    return f"{len(text)} 字，标签清零、段落保留"


check("publishDate 毫秒时间戳", t_date)
check("关键词过滤语义", t_keyword_match)
check("平台前缀剥离", t_platform_prefix_stripped)
check("详情页 HTML -> 纯文本", t_html_to_text)

# ---------------------------------------------------------------------------
section("5. 单条 item -> RawJob 字段映射（含 areaCodeName 陷阱）")
# ---------------------------------------------------------------------------
SAMPLE_ITEM = {
    "jobName": "电控工程师",
    "highMonthPay": 0.3,
    "lowMonthPay": 0.2,
    "publishDate": 1790258416000,
    "recName": "深圳品图视觉科技有限公司",
    "areaCodeName": "广东",          # ← 省级名称，不能拿它当 city
    "jobId": "SiQBi8Jg8BSccqHiB28qTR",
}


def t_to_raw_job():
    sc = N.NcssScraper()
    job = sc._to_raw_job(SAMPLE_ITEM, city="深圳")
    assert isinstance(job, RawJob)
    if job.platform != "ncss":
        _fail(f"platform={job.platform!r}")
    if job.job_id != "SiQBi8Jg8BSccqHiB28qTR":
        _fail(f"job_id={job.job_id!r}")
    if job.title != "电控工程师":
        _fail(f"title={job.title!r}")
    if job.company != "深圳品图视觉科技有限公司":
        _fail(f"company={job.company!r}")
    if job.city != "深圳":
        _fail(f"city={job.city!r}（必须是请求城市，不是 areaCodeName='广东'）")
    if job.salary != "0.2-0.3千元/月":
        _fail(f"salary={job.salary!r}")
    if job.publish_date != "2026-09-24":
        _fail(f"publish_date={job.publish_date!r}")
    if job.url != "https://www.ncss.cn/student/jobs/detail/SiQBi8Jg8BSccqHiB28qTR.html":
        _fail(f"url={job.url!r}")
    if job.description != "":
        _fail("description 本轮必须留空")
    return "9 字段全部正确（city 取请求城市，非 areaCodeName）"


def t_to_raw_job_bad():
    sc = N.NcssScraper()
    for bad in [None, {}, {"jobName": "x"}, {"jobId": ""}, "not-a-dict"]:
        if sc._to_raw_job(bad, city="深圳") is not None:
            _fail(f"{bad!r} 应返回 None（单条失败只跳过）")
    return "缺 jobId / 非 dict -> None"


def t_required_fields_present():
    """落库所需字段必须齐（rag/data/db.py 与 cleaner 的口径）。"""
    required = ["platform", "job_id", "title", "company", "city", "salary",
                "url", "description", "publish_date"]
    job = N.NcssScraper()._to_raw_job(SAMPLE_ITEM, city="深圳")
    d = job.to_dict()
    missing = [f for f in required if f not in d]
    if missing:
        _fail(f"缺字段：{missing}")
    return f"{len(required)} 字段齐备：{required}"


check("单条 item 字段映射", t_to_raw_job)
check("坏 item 安全跳过", t_to_raw_job_bad)
check("落库字段齐备", t_required_fields_present)

# ---------------------------------------------------------------------------
section("6. 同城镜像折叠")
# ---------------------------------------------------------------------------
def t_dedup():
    sc = N.NcssScraper()
    a = sc._to_raw_job({**SAMPLE_ITEM, "jobId": "AAAA"}, city="深圳")
    b = sc._to_raw_job({**SAMPLE_ITEM, "jobId": "BBBB"}, city="深圳")   # 同公司同标题同城
    c = sc._to_raw_job({**SAMPLE_ITEM, "jobId": "CCCC"}, city="广州")   # 跨城：必须保留
    out = N._dedup_mirrors([a, b, c])
    if len(out) != 2:
        _fail(f"应折成 2 条（同城折 1、跨城保留），实际 {len(out)}")
    if out[0].job_id != "AAAA":
        _fail(f"代表行应取 id 最小的原记录，实际 {out[0].job_id!r}")
    cities = sorted(j.city for j in out)
    if cities != ["广州", "深圳"]:
        _fail(f"跨城岗位被误折：{cities}")
    return "同城折 1 条（保 id 最小）+ 跨城保留"


def t_dedup_keeps_best_salary():
    sc = N.NcssScraper()
    poor = sc._to_raw_job({**SAMPLE_ITEM, "jobId": "AAAA", "lowMonthPay": 0, "highMonthPay": 0}, "深圳")
    rich = sc._to_raw_job({**SAMPLE_ITEM, "jobId": "BBBB", "lowMonthPay": 3.0, "highMonthPay": 5.0}, "深圳")
    out = N._dedup_mirrors([poor, rich])
    if len(out) != 1 or out[0].salary != "3-5千元/月":
        _fail(f"折叠应合并更完整的薪资，实际 {[j.salary for j in out]}")
    return "折叠时合并更完整的薪资"


check("同城镜像折叠 + 跨城保留", t_dedup)
check("折叠时合并薪资", t_dedup_keeps_best_salary)

# ---------------------------------------------------------------------------
section("7. 限速开关（离线，不发请求）")
# ---------------------------------------------------------------------------
def t_rate_limit_off():
    async def run():
        sc = N.NcssScraper(interval=0)
        sc._request = lambda params: {"flag": True, "data": {"list": []}}
        return await sc.search("法务", city="北京", limit=20)
    import time
    t0 = time.monotonic()
    asyncio.run(run())
    dt = time.monotonic() - t0
    if dt > 0.5:
        _fail(f"interval=0 时应立即返回，实际 {dt:.2f}s")
    return f"interval=0 -> {dt * 1000:.1f}ms（测试可关限速）"


def t_rate_limit_default():
    if N.DEFAULT_INTERVAL != 1.0:
        _fail(f"DEFAULT_INTERVAL={N.DEFAULT_INTERVAL!r}，应为 1.0（≤1 req/s）")
    return "DEFAULT_INTERVAL == 1.0"


def t_pagination_cap():
    sc = N.NcssScraper()
    if sc.max_pages != 5:
        _fail(f"max_pages={sc.max_pages}，接口上限是 5")
    sc2 = N.NcssScraper(max_pages=99)
    if sc2.max_pages != 5:
        _fail("max_pages 必须被夹到 5（offset>=6 接口直接 flag=false）")
    return "每组合最多 5 页 = 100 条"


def t_no_write_to_disk():
    """抓取器不得写盘：确认模块里没有 open(...)/写文件调用。"""
    src = (REPO / "agent" / "scrapers" / "ncss.py").read_text(encoding="utf-8")
    for bad in ["open(", "Path(", ".write_text", "json.dump", "os.makedirs"]:
        # 允许出现在注释里，这里只做粗筛，命中就人工确认
        if bad in src.split('"""')[-1]:
            _fail(f"ncss.py 正文出现疑似写盘调用：{bad}")
    return "无 open/write/dump（只读抓取）"


check("interval=0 可关限速", t_rate_limit_off)
check("默认限速 1.0s", t_rate_limit_default)
check("分页上限夹到 5", t_pagination_cap)
check("模块不写盘", t_no_write_to_disk)

# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"总计：{len(PASS)} 通过 / {len(FAIL)} 失败")
for _label, _err in FAIL:
    print(f"  ✗ {_label} → {_err}")
print("=" * 74)
raise SystemExit(1 if FAIL else 0)
