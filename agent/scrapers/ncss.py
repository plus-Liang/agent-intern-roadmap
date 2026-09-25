# -*- coding: utf-8 -*-
"""国家大学生就业服务平台（ncss.cn）岗位抓取器 —— 纯 HTTP 匿名接口实现。

为什么是纯 HTTP + 匿名
----------------------
ncss.cn（教育部「国家大学生就业服务平台」，原 24365 平台）的岗位列表页背后是
一个**公开的 JSON 接口**，不需要登录、不需要 cookie、不需要浏览器渲染：

    GET https://www.ncss.cn/student/jobs/jobslist/ajax/
        ?jobType=            # 留空 = 不限岗位类型（默认）；"03" = 只要实习
        &areaCode=110000     # 地区代码
        &jobName=算法         # 关键词（接口自带，服务端子串匹配）
        &offset=1            # 页码，从 1 开始；>5 被拒
        &limit=20            # 每页条数；>20 被服务端夹到 20

    成功：{"flag": true, "global": [], "errors": [],
           "data": {"list": [ {...岗位...} ],
                    "pagenation": {"count": 200, "total": 10, "limit": 20, "offset": 1}}}
    翻页越界（offset > 5）：{"flag": false, ...}（data 缺失）

所以本模块完全不依赖 playwright，只用 requests，与 `niuke.py` 同属"接口型"抓取器，
与 `shixiseng.py` 的浏览器方案互补。

实测口径（2026 年摸查，见本轮验证报告）
--------------------------------------
    * 匿名可重复调用：同一参数多次请求都返回 flag=true，无 cookie 会话要求；
    * 翻页真实上限：offset 1..5 可用（5 × 20 = 100 条/组合），offset >= 6 返回 flag=false；
    * limit > 20 被服务端夹到 20，故本模块固定每页 20；
    * 候选池耗尽时返回 flag=true + 空 list（不是报错），例如北京「临床研究」为 0 条；
    * 无结果是真的没数据，不是被限流（连续请求 30+ 次没有出现一次风控）。

城市与 areaCode 的关系（重要，别踩坑）
--------------------------------------
接口吃的是 **6 位地区代码**，地级市级别的代码是有效的且互不重叠：

    440100 广州 / 440300 深圳    —— 同属广东省，但两者返回的 jobId 交集为 0
                                    （实测 20 ∩ 20 = 0），确实是两个城市的独立结果集。

但返回体里的 `areaCodeName` 是**省级名称**：搜深圳拿到的记录 `areaCodeName = "广东"`。
所以本模块的 `city` 一律取**调用方请求的城市名**（`_norm_city` 规整过），
而不是 `areaCodeName`；若直接用 `areaCodeName`，广州和深圳的岗位在库里都会
变成 `city="广东"`，看板/检索的六城维度会直接塌掉。

description 的处理（本轮默认已开启）
------------------------------------
`description` 来自匿名可访问的详情页：`GET /student/jobs/{jobId}/detail.html`，
HTML 里 `<div class="jobdetail-box"><div class="mainContent ...">` 内含完整
「岗位职责 + 任职要求」（实测约 80KB/页，正常页面都能解析出正文）。

本轮决策把 `fetch_description` 默认从 False 改为 **True**：列表接口本身不带正文，
而 RAG 需要正文。代价是**每个岗位多 1 次详情请求**，受同一 1 req/s 限速约束 ——
实测 6 组合样本：平均 14.7 条/组合、23.0 秒/组合，其中详情请求占绝大部分
（纯列表页只要 2 次请求 ≈ 2 秒）。

想省时间可显式关掉（`NcssScraper(fetch_description=False)`），此时列表字段照常，
只有 `description` 为空串。

用法::

    import asyncio
    from agent.scrapers.ncss import NcssScraper

    jobs = asyncio.run(NcssScraper().search("算法工程师", city="北京", limit=100))

边界情况
--------
    * 未知城市：**不猜 areaCode**，直接返回空列表并打印一条可读告警；
    * 请求失败（网络 / 超时 / 非 200 / JSON 解析失败 / flag=false）重试 2 次，timeout=30；
    * 进程内全局限速 1 秒/请求（≤1 req/s），由 `interval` 控制；
    * 单条岗位解析失败只跳过这一条（try/except 包在循环体内）；
    * 任何异常都吞掉并返回已抓到的部分（基类约定：不抛异常）。
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from agent.scrapers.base import PlatformScraper, RawJob

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
API_URL = "https://www.ncss.cn/student/jobs/jobslist/ajax/"
DETAIL_URL_TMPL = "https://www.ncss.cn/student/jobs/detail/{job_id}.html"
# 匿名可访问的详情 HTML（含 JD 正文），仅在 fetch_description=True 时使用
DETAIL_HTML_TMPL = "https://www.ncss.cn/student/jobs/{job_id}/detail.html"
PLATFORM = "ncss"

JOB_TYPE_INTERN = "03"   # 03 = 实习；保留常量，供"只要实习"的调用方显式传入
# jobType 默认**留空**：实测 "03" 过滤过狠（约丢 30% 数据，北京「算法工程师」
# 26 条 -> 0 条），留空 = 不限岗位类型。请求里表现为 jobType=，服务端按不限处理。
DEFAULT_JOB_TYPE = ""
# 详情页正文默认抓（RAG 需要正文）；关掉可省每个岗位 1 次请求（见模块 docstring）
DEFAULT_FETCH_DESCRIPTION = True
PAGE_SIZE = 20           # 接口每页固定 20（传更大的值会被服务端夹回 20）
MAX_OFFSET = 5           # 实测 offset 1..5 可用，>=6 返回 flag=false（故单组合最多 100 条）

TIMEOUT = 30             # 秒
MAX_RETRIES = 2          # 失败后重试 2 次（总计最多 3 次尝试），与 niuke.py 同口径
RETRY_BACKOFF = 1.5      # 重试间隔基数（秒），第 n 次重试等 RETRY_BACKOFF * n

DEFAULT_INTERVAL = 1.0   # 进程内全局限速：两次请求之间至少间隔 1 秒（≤1 req/s）

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
REFERER = "https://www.ncss.cn/student/jobs/joblist.html"

# 城市 -> areaCode。只收录**实测过**的 6 城；找不到映射就报告，绝不瞎猜。
# 顺序与 config/scraping.yaml 的城市池一致，方便对账。
CITY_AREA_CODES: dict[str, str] = {
    "广州": "440100",   # 广东省（city 级代码，与深圳结果集不重叠）
    "深圳": "440300",
    "北京": "110000",   # 直辖市，省级代码 == 城市代码
    "上海": "310000",
    "杭州": "330100",   # 浙江（330000 是省级，实测只有 3 条，用 330100）
    "成都": "510100",   # 四川（510000 省级实测 0 条，用 510100）
}

# 城市名后缀规整（"广州市" -> "广州"），与 niuke.py 同口径
_CITY_SUFFIX = re.compile(r"(市|特别行政区)$")

# 入参 keyword 允许带平台前缀（如 "ncss:算法" / "24365 算法"），送接口前必须剥掉，
# 否则它会被当成标题里必须出现的词，把所有结果都过滤没。
_PLATFORM_PREFIX = re.compile(r"^\s*(?:ncss|24365|国家大学生就业服务平台)\s*(?:[:：]|\s)\s*", re.IGNORECASE)

# JD 正文提取：详情页里正文所在容器（见模块 docstring「description 的处理」）
_DETAIL_BOX = re.compile(
    r'<div class="jobdetail-box">\s*<div class="mainContent[^"]*">(.*?)</div>',
    re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")

# ---------------------------------------------------------------------------
# 进程内全局限速器
# ---------------------------------------------------------------------------
# 为什么是模块级而不是实例级：调度器会为每个平台建实例，但"1 req/s"是对站点的
# 承诺，应当跨实例生效（同一进程里两个 NcssScraper 也不该把速率翻倍）。
_RATE_LOCK = asyncio.Lock()
_LAST_REQUEST_AT = 0.0


async def _rate_limit(interval: float) -> None:
    """确保两次 HTTP 请求之间至少间隔 `interval` 秒（进程内全局）。"""
    global _LAST_REQUEST_AT
    if interval <= 0:
        return
    async with _RATE_LOCK:
        now = time.monotonic()
        wait = interval - (now - _LAST_REQUEST_AT)
        if wait > 0:
            await asyncio.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()


# ---------------------------------------------------------------------------
# 纯函数：归一化 / 字段映射
# ---------------------------------------------------------------------------
def _norm_city(value: Any) -> str:
    """城市归一：去后缀（"广州市" -> "广州"）、去空白。"""
    return _CITY_SUFFIX.sub("", str(value or "").strip())


def resolve_area_code(city: Optional[str]) -> Optional[str]:
    """城市名 -> areaCode；未收录的城市返回 None（调用方负责报告，不许瞎猜）。

    `None` / `""` / "全国" / "不限" 都表示"不限城市"，返回接口默认值 None。
    """
    if city is None:
        return None
    text = str(city).strip()
    if not text or text in {"全国", "不限", "all", "ALL"}:
        return None
    return CITY_AREA_CODES.get(_norm_city(text))


def _money(value: Any) -> Optional[float]:
    """把 highMonthPay / lowMonthPay 转成正数 float；缺失或 <=0 返回 None。"""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


def _trim(num: float) -> str:
    """4.0 -> "4"，4.5 -> "4.5"（去掉无意义的小数尾巴）。"""
    text = f"{num:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _fmt_salary(low: Any, high: Any) -> str:
    """lowMonthPay / highMonthPay -> 可读薪资原文。

    ⚠️ 单位口径：接口字段名带 "MonthPay"，实测取值为**千元/月**
    （例：low=0.2 / high=0.3 的深圳「电控工程师」，是 200-300 元/月；
    low=4.0 / high=7.0 的四川「算法工程师实习生」，是 4000-7000 元/月）。

    形态：
        * low=0.2, high=0.3  -> "0.2-0.3千元/月"
        * low=4,   high=4    -> "4千元/月"
        * 只有一边有值        -> 用那一边（不瞎补另一边）
        * 两边都缺失/为 0     -> ""（不瞎猜）
    """
    lo = _money(low)
    hi = _money(high)
    if lo is None and hi is None:
        return ""
    if lo is None:
        lo = hi
    if hi is None:
        hi = lo
    assert lo is not None and hi is not None
    if lo == hi:
        return f"{_trim(lo)}千元/月"
    return f"{_trim(lo)}-{_trim(hi)}千元/月"


def _fmt_date(value: Any) -> str:
    """毫秒时间戳 -> "YYYY-MM-DD"；拿不到或非法则返回 ""。"""
    if not value:
        return ""
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return ""
    if ms <= 0:
        return ""
    # 兼容误传的「秒级」时间戳（< 1e11 视为秒）
    if ms < 100_000_000_000:
        ms *= 1000
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def _match_keyword(title: str, keywords: list[str]) -> bool:
    """客户端兜底过滤：所有关键词都出现在标题里才算命中（AND，大小写不敏感）。

    接口自带的 `jobName` 已是服务端子串匹配，正常情况下这一步不会过滤掉任何东西；
    它存在只为兜住「接口语义变化 / 客户端拼词」两种意外。关键词为空 -> 全通过。
    """
    if not keywords:
        return True
    lowered = (title or "").lower()
    return all(kw in lowered for kw in keywords)


def _norm_text(value: Any) -> str:
    """公司名/标题归一：去首尾空白、内部空白折叠、转小写（用于比对）。"""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _salary_value(text: str) -> float:
    """从 "0.3千元/月" 这类原文里取第一个数字，用于比较两条镜像的薪资完整度。"""
    match = re.search(r"\d+(?:\.\d+)?", str(text or ""))
    return float(match.group()) if match else -1.0


def _sort_key(job: RawJob) -> tuple[int, str]:
    """确定性排序键：id 短的排前，同长按字典序（ncss 的 jobId 是定长 base62，无数字 id）。"""
    text = str(job.job_id)
    return (len(text), text)


def _mirror_key(job: RawJob) -> tuple[str, str, str]:
    """同城镜像判定键：(公司, 标题, 城市)。

    ncss 也会把同一岗位投放成多条 id 不同的记录（实测深圳「机械装调工程师（实习生）」
    两条同名同公司）。折叠只做**同城**，跨城市投放保留——搜"广州"仍要能看到广州的岗位。
    与 niuke.py 同口径，保证多平台数据在清洗端表现一致。
    """
    return (_norm_text(job.company), _norm_text(job.title), _norm_city(job.city))


def _dedup_mirrors(jobs: list[RawJob]) -> list[RawJob]:
    """把同 (公司, 标题, 城市) 的镜像折叠成一条，返回保序去重后的列表。

    代表行 = 组内**原始 id 排序最小**的那条（确定性，与翻页顺序无关），
    并把组内信息量最大的 description / 薪资合并上去。刻意不改写 job_id：
    只用站点自己的 id，所以同一岗位重复抓取 id 恒定。
    """
    groups: dict[tuple[str, str, str], list[RawJob]] = {}
    for job in jobs:
        groups.setdefault(_mirror_key(job), []).append(job)

    out: list[RawJob] = []
    for group in groups.values():
        if len(group) == 1:
            out.append(group[0])
            continue
        ordered = sorted(group, key=_sort_key)
        rep = ordered[0]
        if not (rep.description or "").strip():
            richest = max(ordered, key=lambda j: len(j.description or ""))
            if (richest.description or "").strip():
                rep = replace(rep, description=richest.description)
        if _salary_value(rep.salary) < max(_salary_value(j.salary) for j in ordered):
            best_pay = max(ordered, key=lambda j: _salary_value(j.salary))
            rep = replace(rep, salary=best_pay.salary)
        out.append(rep)
    return out


def _html_to_text(html: str) -> str:
    """详情页正文 HTML -> 纯文本：去标签、把块级标签变段落、压缩空白。"""
    text = re.sub(r"(?i)<br\s*/?>", "\n", html or "")
    text = re.sub(r"(?i)</(p|div|li|tr|h\d)>", "\n", text)
    text = _TAG.sub("", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    lines = [re.sub(r"[ \t\u3000]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


# ---------------------------------------------------------------------------
# 抓取器
# ---------------------------------------------------------------------------
class NcssScraper(PlatformScraper):
    """国家大学生就业服务平台（ncss.cn）岗位抓取器。

    公开 JSON 接口，无浏览器、无登录、无 cookie。
    """

    platform_name = PLATFORM

    def __init__(
        self,
        interval: float = DEFAULT_INTERVAL,
        timeout: int = TIMEOUT,
        max_pages: int = MAX_OFFSET,
        job_type: str = DEFAULT_JOB_TYPE,
        fetch_description: bool = DEFAULT_FETCH_DESCRIPTION,
        **kwargs: Any,  # 吞掉调度器传来的公共参数（headless 等），避免被打挂
    ) -> None:
        """
        参数：
            interval:          进程内全局请求间隔（秒），默认 1.0（≤1 req/s）
            timeout:           单次请求超时（秒）
            max_pages:         单组合最多翻几页（接口上限 5）
            job_type:          岗位类型，默认 "" = 不限；传 "03" 只要实习
            fetch_description: 是否为每条岗位额外抓详情页正文（默认 True，见模块 docstring）
        """
        self.interval = max(0.0, float(interval))
        self.timeout = int(timeout or TIMEOUT)
        self.max_pages = max(1, min(int(max_pages or MAX_OFFSET), MAX_OFFSET))
        self.job_type = str(job_type or "")
        self.fetch_description = bool(fetch_description)
        self._session: Optional[requests.Session] = None
        # 最近一次 search 的逐页统计，供验证脚本/日志取用（不参与数据口径）
        self.last_trace: list[dict[str, Any]] = []

    # -- 内部：HTTP --------------------------------------------------------
    def _get_session(self) -> requests.Session:
        """惰性建 Session（复用 TCP 连接 + 固定 UA / Referer / XHR 头）。"""
        if self._session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": UA,
                    "Accept": "application/json, text/plain, */*",
                    # 接口是 /ajax/ 端点，缺 X-Requested-With 会被站点网关拦成 HTML
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": REFERER,
                }
            )
            self._session = session
        return self._session

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        """同步 GET + 重试（与 niuke.py 同模式）；彻底失败返回 {}。

        注意：本方法只做"重试内退避"，**不做限速**——限速由 `_rate_limit()`
        在 await 侧统一管，避免同步阻塞把事件循环卡住。
        """
        session = self._get_session()
        last_err: Optional[Exception] = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = session.get(API_URL, params=params, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                return resp.json() or {}
            except Exception as exc:  # 网络 / 超时 / JSON 解析失败
                last_err = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
        print(f"[ncss] 请求失败（params={params}，已重试 {MAX_RETRIES} 次）：{last_err}")
        return {}

    def _fetch_detail(self, job_id: str) -> str:
        """按 jobId 抓详情页正文（仅 fetch_description=True 时调用）。失败返回 ""。"""
        session = self._get_session()
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = session.get(DETAIL_HTML_TMPL.format(job_id=job_id), timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                match = _DETAIL_BOX.search(resp.text)
                return _html_to_text(match.group(1)) if match else ""
            except Exception:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
        return ""

    # -- 内部：解析 --------------------------------------------------------
    def _to_raw_job(self, item: dict[str, Any], city: str = "") -> Optional[RawJob]:
        """单条 item -> RawJob；解析失败返回 None（只跳过这一条）。

        `city` 是**调用方请求的城市名**，不是返回体里的 `areaCodeName`：
        后者是省级名称（搜深圳会得到 "广东"），直接用会让六城维度塌掉。
        """
        if not isinstance(item, dict):
            return None
        job_id = item.get("jobId")
        if not job_id:
            return None
        return RawJob(
            platform=PLATFORM,
            job_id=str(job_id),
            title=str(item.get("jobName") or "").strip(),
            company=str(item.get("recName") or "").strip(),
            city=city,
            salary=_fmt_salary(item.get("lowMonthPay"), item.get("highMonthPay")),
            url=DETAIL_URL_TMPL.format(job_id=job_id),
            description="",  # 占位：fetch_description=True 时由 _attach_descriptions 补齐
            publish_date=_fmt_date(item.get("publishDate") or item.get("updateDate")),
        )

    # -- 公开：search ------------------------------------------------------
    async def search(
        self, keyword: str, city: Optional[str] = None, limit: int = 20
    ) -> list[RawJob]:
        """按关键词 + 城市抓取 ncss 实习岗位。

        流程：
            1) city -> areaCode（未收录城市：打印告警并返回空列表，不许瞎猜）；
            2) offset 1..max_pages（默认 5），每页 limit=20，累计到 limit 或 100 条即停；
            3) 客户端按 jobName 兜底过滤关键词（接口已服务端过滤，正常不丢数据）；
            4) 同城镜像折叠（见 `_dedup_mirrors`）。

        任何异常都被吞掉并返回已抓到的部分（基类约定：不抛异常）。
        """
        keyword = _PLATFORM_PREFIX.sub("", str(keyword or "")).strip()
        keywords = [k.lower() for k in keyword.split() if k]
        limit = max(1, int(limit or 20))
        wanted_city = _norm_city(city)
        self.last_trace = []

        area_code = resolve_area_code(city)
        if wanted_city and area_code is None:
            # 关键设计：未知城市**不猜** areaCode，宁可少数据也不要污染城市维度
            print(
                f"[ncss] 未收录城市 {city!r}，无法映射 areaCode，已跳过"
                f"（已支持：{'、'.join(CITY_AREA_CODES)}）"
            )
            return []

        # 城市名写进 city 字段；city 为 None/全国 时退化为空串（表示不限城市）
        city_name = wanted_city if area_code else ""

        results: list[RawJob] = []
        seen: set[str] = set()
        pages = max(1, min((limit + PAGE_SIZE - 1) // PAGE_SIZE, self.max_pages))

        try:
            for offset in range(1, pages + 1):
                params: dict[str, Any] = {
                    "jobType": self.job_type,
                    "offset": offset,
                    "limit": PAGE_SIZE,
                }
                if area_code:
                    params["areaCode"] = area_code
                if keyword:
                    params["jobName"] = keyword

                await _rate_limit(self.interval)
                payload = self._request(params)
                data = payload.get("data") if isinstance(payload, dict) else None
                pagenation = data.get("pagenation") if isinstance(data, dict) else None
                items = data.get("list") if isinstance(data, dict) else None
                items = items if isinstance(items, list) else []

                self.last_trace.append(
                    {
                        "offset": offset,
                        "flag": payload.get("flag") if isinstance(payload, dict) else None,
                        "n": len(items),
                        "pagenation": pagenation,
                    }
                )

                if not items:
                    # flag=false（翻页越界）或候选池耗尽：两种情况都停止翻页
                    break

                for item in items:
                    try:
                        job = self._to_raw_job(item, city_name)
                    except Exception as exc:  # 单条解析失败 -> 跳过
                        print(f"[ncss] 单条岗位解析失败，已跳过：{exc}")
                        continue
                    if job is None or job.job_id in seen:
                        continue
                    if not _match_keyword(job.title, keywords):
                        continue
                    seen.add(job.job_id)
                    results.append(job)

                if len(results) >= limit:
                    break

            if self.fetch_description:
                results = await self._attach_descriptions(results)
        except Exception as exc:  # 兜底：任何意外都不该拖垮整轮抓取
            print(f"[ncss] 抓取异常，返回已抓到的 {len(results)} 条：{exc}")
            return _dedup_mirrors(results)[:limit]

        return _dedup_mirrors(results)[:limit]

    async def _attach_descriptions(self, jobs: list[RawJob]) -> list[RawJob]:
        """为每条岗位补抓详情页正文（fetch_description=True 时调用）。受同一限速约束。"""
        out: list[RawJob] = []
        for job in jobs:
            await _rate_limit(self.interval)
            text = self._fetch_detail(job.job_id)
            out.append(replace(job, description=text) if text else job)
        return out

    async def close(self) -> None:
        """关闭 HTTP 会话。"""
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None
