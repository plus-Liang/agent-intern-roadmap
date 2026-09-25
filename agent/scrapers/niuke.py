# -*- coding: utf-8 -*-
"""牛客网（nowcoder.com）岗位抓取器 —— 纯 HTTP JSON 接口实现。

为什么是纯 HTTP
---------------
牛客网的岗位广场有一个**公开的 JSON 接口**，不需要登录、不需要浏览器渲染：

    POST https://www.nowcoder.com/np-api/u/job/square-search
    Content-Type: application/x-www-form-urlencoded

    表单：requestFrom=1 & page=N & pageSize=20 & recruitType=2
         & pageSource=5001 & jobCity=城市 & careerJobId=11006
         & visitorId=<随机 UUID>

响应形如::

    {"code": 0,
     "data": {"totalCount": 74, "totalPage": 4, "currentPage": 1,
              "datas": [{"data": {"id": 467829, "jobName": "...",
                                  "jobCity": "广州",
                                  "salaryMin": 300, "salaryMax": 530,
                                  "ext": "{\\"requirements\\":\\"...\\",\\"infos\\":\\"...\\"}",
                                  "recommendInternCompany": {"companyName": "美团金融"},
                                  "createTime": 1789559060000,
                                  "refreshTime": 1790154322000}}]}}

所以本模块完全不依赖 playwright / 浏览器，只用 requests（项目已有依赖），
抓取速度快、资源占用低，与 shixiseng.py 的浏览器方案形成互补。

关键词怎么处理（实测接口行为）
------------------------------
该接口**没有关键词参数**（它是一个"职位广场"枚举接口，不是搜索接口）。
因此本模块的策略是：

    * 按 city 分页拉取候选人（每页 20 条，最多 5 页 = 100 条）；
    * 在**本地**用 keyword 过滤 `jobName`（大小写不敏感、多关键词按空格拆开取交集）；
    * 再对**同城镜像**做一次内容去重（见 `_dedup_mirrors`）。

这样虽然多拉了一点数据，但请求数仍然是常数级（<=5 次/城市），
而且过滤逻辑完全可控、不会因为站点搜索接口变动而失效。

站内镜像为什么必须单独处理
--------------------------
牛客会把「同公司 + 同标题」的一个岗位按城市拆成多条记录：id 各不相同
（常常只差 1）、description 完全一致。它们不是本模块重复抓取造成的，
所以按 `job_id` 去重（以及 rag/data/db.py 的 `ON CONFLICT(job_id)`）
一条都拦不住。本模块因此在返回前按 (公司, 标题, 城市) 折叠同城镜像，
跨城市的投放**保留**——搜「广州」仍应看到广州的岗位。

边界情况
--------
    * 请求失败（网络 / 超时 / 非 200 / JSON 解析失败）自动重试 2 次，timeout=30；
    * 翻页到 totalPage 或 5 页即停；
    * 单条岗位解析失败只跳过这一条，不影响其他（try/except 包在循环体内）。

用法::

    import asyncio
    from agent.scrapers.niuke import NiukeScraper

    jobs = asyncio.run(NiukeScraper().search("大模型", city="广州", limit=20))
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from agent.scrapers.base import PlatformScraper, RawJob

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
API_URL = "https://www.nowcoder.com/np-api/u/job/square-search"
DETAIL_URL_TMPL = "https://www.nowcoder.com/jobs/detail/{job_id}"
PLATFORM = "niuke"

PAGE_SIZE = 20          # 接口固定每页 20 条
MAX_PAGES = 5           # 单次 search 最多翻 5 页（=100 条候选）
RECRUIT_TYPE_INTERN = 2  # 2 = 实习
PAGE_SOURCE = 5001
CAREER_JOB_ID = 11006   # 实习职位分类 ID
REQUEST_FROM = 1

# 入参 keyword 允许带平台前缀（如 "niuke:大模型"），过滤前必须剥掉，
# 否则它会被当成标题里必须出现的词，把所有结果都过滤没。
# 只认「平台名 + 冒号」或「平台名 + 空格」：不加 {1,2} 这类量词，
# 否则 "牛客大模型" 会被误剥成 "大模型"。
_PLATFORM_PREFIX = re.compile(r"^\s*(?:niuke|牛客)\s*(?:[:：]|\s)\s*", re.IGNORECASE)

TIMEOUT = 30            # 秒
MAX_RETRIES = 2         # 失败后重试 2 次（总计最多 3 次尝试）
RETRY_BACKOFF = 1.5     # 重试间隔基数（秒），第 n 次重试等 RETRY_BACKOFF * n

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# 全站会话级 visitorId：同一进程内所有请求复用同一个（模拟一个访客）。
VISITOR_ID = str(uuid.uuid4())

# 中文城市名 -> 牛客 jobCity 参数取值。牛客一般直接吃中文城市名，
# 这里只做「去掉常见后缀」的规整（"广州市" -> "广州"），不做映射表硬编码，
# 避免城市列表过期；未知城市原样透传。
_CITY_SUFFIX = re.compile(r"(市|特别行政区)$")


def _normalize_city(city: Optional[str]) -> str:
    """把调用方的城市名规整成接口能吃的形态；None/空 -> "" （全国）。"""
    if not city:
        return ""
    city = str(city).strip()
    if city in {"全国", "不限", "all", "ALL"}:
        return ""
    return _CITY_SUFFIX.sub("", city)


def _fmt_salary(data: dict[str, Any]) -> str:
    """把 salaryMin / salaryMax / salaryMonth 拼成可读薪资原文。

    牛客实习薪资的常见形态：
        * salaryMin=300, salaryMax=530        -> "300-530/天"
        * salaryMin=300, salaryMax=300        -> "300/天"
        * salaryMin=0,   salaryMax=0 / 缺失   -> ""（不瞎猜）
        * 带 salaryMonth 的月薪形态           -> "3000-5000/月"

    数值单位无法百分百确定（题面给的样例是 300-530/天），所以：
    有 salaryMonth 且不是"天"时按月，否则按天。
    """
    try:
        lo = int(data.get("salaryMin") or 0)
        hi = int(data.get("salaryMax") or 0)
    except (TypeError, ValueError):
        return ""

    if lo <= 0 and hi <= 0:
        return ""
    if lo <= 0:
        lo = hi
    if hi <= 0:
        hi = lo

    month = data.get("salaryMonth")
    unit = "/月" if month not in (None, "", 0, "0") else "/天"
    if lo == hi:
        return f"{lo}{unit}"
    return f"{lo}-{hi}{unit}"


def _parse_ext(ext: Any) -> str:
    """解析 ext 字段（本身是一段 JSON 字符串），拼出 JD 正文。

    ext 形如::

        "{\\"requirements\\":\\"任职要求...\\",\\"infos\\":\\"岗位职责...\\"}"

    也有可能是已经是 dict（接口版本差异）、或者空 / 非法 JSON——
    这些情况都返回 "" 或尽力而为的结果，绝不抛异常。
    """
    if not ext:
        return ""
    payload = ext
    if isinstance(ext, str):
        try:
            payload = json.loads(ext)
        except (ValueError, TypeError):
            # 不是 JSON：当成纯文本正文用（有些岗位 ext 直接就是一段描述）
            return ext.strip()
    if not isinstance(payload, dict):
        return str(payload).strip()

    # 顺序：先职责（infos）后要求（requirements），符合 JD 阅读习惯。
    parts: list[str] = []
    for key in ("infos", "requirements", "jobDesc", "description"):
        value = payload.get(key)
        if not value:
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text = text.strip()
        if text and text not in parts:
            parts.append(text)
    return "\n\n".join(parts)


def _fmt_date(*timestamps: Any) -> str:
    """毫秒时间戳 -> "YYYY-MM-DD"；拿不到或非法则返回 ""。

    优先用 refreshTime（最近刷新，反映岗位是否还在招），缺失时回落到 createTime。
    """
    for ts in timestamps:
        if not ts:
            continue
        try:
            ms = int(ts)
        except (TypeError, ValueError):
            continue
        if ms <= 0:
            continue
        # 兼容误传的「秒级」时间戳（< 1e11 视为秒）
        if ms < 100_000_000_000:
            ms *= 1000
        try:
            return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            continue
    return ""


def _match_keyword(title: str, keywords: list[str]) -> bool:
    """本地关键词过滤：所有关键词都出现在标题里才算命中（AND，大小写不敏感）。

    关键词为空 -> 全部通过（等价于"这个城市的所有实习岗位"）。
    """
    if not keywords:
        return True
    lowered = (title or "").lower()
    return all(kw in lowered for kw in keywords)


def _norm_text(value: Any) -> str:
    """公司名/标题归一：去首尾空白、内部空白折叠、转小写（用于比对）。"""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _norm_city(value: Any) -> str:
    """城市归一：去后缀（"广州市"->"广州"）、去空白。"""
    return _CITY_SUFFIX.sub("", str(value or "").strip())


def _mirror_key(job: RawJob) -> tuple[str, str, str]:
    """同城镜像判定键：(公司, 标题, 城市)。

    为什么按这个键折叠
    ------------------
    牛客一个岗位常被拆成多条记录：同公司、同标题、同 JD，只是 id 不同
    （id 往往只差 1——站点按城市配额批量生成的镜像投放）。实测审计里的
    "哔哩哔哩《大模型算法工程师》6 条"就是 6 个不同 id、description 完全
    一致、分布在 5 个城市。

    因此这里把**同城**镜像折叠成一条（保留各地投放，不丢城市维度）；
    跨城市的同岗位**不折叠**——搜"广州"仍要能看到广州的岗位。
    """
    return (_norm_text(job.company), _norm_text(job.title), _norm_city(job.city))


def _salary_value(text: str) -> int:
    """从 "300-520/天" 这类原文里取第一个数字，用于比较两条镜像的薪资完整度。"""
    match = re.search(r"\d+", str(text or ""))
    return int(match.group()) if match else -1


def _dedup_mirrors(jobs: list[RawJob]) -> list[RawJob]:
    """把同 (公司, 标题, 城市) 的镜像折叠成一条，返回保序去重后的列表。

    代表行 = 组内**原始 id 最小**的那条，并把组内信息量最大的
    description / 薪资合并上去。刻意**不改写 job_id**：

    * 只用站点自己的 id，没有本地 hash，所以同一岗位重复抓取 id 恒定；
    * 代表行的选取只依赖组内最小原始 id，与翻页顺序无关，不会因为接口
      返回顺序变化就换一个 id（按"JD 最长"选代表行会引入这种不稳定）；
    * 镜像的 city 相同、只有 id 不同，所以折叠后每个
      (公司, 标题, 城市) 在库里只留一条可检索记录。
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
        if any((j.description or "").strip() for j in ordered) and not (rep.description or "").strip():
            richest = max(ordered, key=lambda j: len(j.description or ""))
            rep = replace(rep, description=richest.description)
        if _salary_value(rep.salary) < max(_salary_value(j.salary) for j in ordered):
            best_pay = max(ordered, key=lambda j: _salary_value(j.salary))
            rep = replace(rep, salary=best_pay.salary)
        out.append(rep)
    return out


def _sort_key(job: RawJob) -> tuple[int, int, str]:
    """确定性排序键：数字 id 按数值升序排前，非数字 id 排后。"""
    text = str(job.job_id)
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


class NiukeScraper(PlatformScraper):
    """牛客网岗位抓取器（公开 JSON 接口，无浏览器、无登录）。"""

    platform_name = PLATFORM

    def __init__(
        self,
        max_pages: int = MAX_PAGES,
        timeout: int = TIMEOUT,
        career_job_id: int = CAREER_JOB_ID,
        recruit_type: int = RECRUIT_TYPE_INTERN,
        **kwargs: Any,  # 吞掉调度器传来的公共参数（headless 等），避免被打挂
    ) -> None:
        self.max_pages = max(1, int(max_pages or MAX_PAGES))
        self.timeout = int(timeout or TIMEOUT)
        self.career_job_id = career_job_id
        self.recruit_type = recruit_type
        self.visitor_id = VISITOR_ID  # 会话级固定
        self._session: Optional[requests.Session] = None

    # -- 内部：HTTP --------------------------------------------------------
    def _get_session(self) -> requests.Session:
        """惰性建 Session（复用 TCP 连接 + 固定 UA / visitorId）。"""
        if self._session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": UA,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json, text/plain, */*",
                    "Origin": "https://www.nowcoder.com",
                    "Referer": "https://www.nowcoder.com/jobs/intern/square",
                }
            )
            self._session = session
        return self._session

    def _fetch_page(self, page: int, city: str) -> dict[str, Any]:
        """拉取一页岗位，失败重试 MAX_RETRIES 次。

        返回解析后的 JSON dict；彻底失败返回 {}（调用方按"这一页没数据"处理）。
        """
        form = {
            "requestFrom": REQUEST_FROM,
            "page": page,
            "pageSize": PAGE_SIZE,
            "recruitType": self.recruit_type,
            "pageSource": PAGE_SOURCE,
            "jobCity": city,
            "careerJobId": self.career_job_id,
            "visitorId": self.visitor_id,
        }
        session = self._get_session()
        last_err: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = session.post(API_URL, data=form, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                return resp.json() or {}
            except Exception as exc:  # 网络 / 超时 / JSON 解析失败
                last_err = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
        print(f"[niuke] 第 {page} 页请求失败（已重试 {MAX_RETRIES} 次）：{last_err}")
        return {}

    # -- 内部：解析 --------------------------------------------------------
    def _to_raw_job(self, item: dict[str, Any]) -> Optional[RawJob]:
        """单条 item -> RawJob；解析失败返回 None（只跳过这一条）。"""
        data = item.get("data") if isinstance(item, dict) else None
        if not isinstance(data, dict):
            return None

        job_id = data.get("id")
        if job_id is None:
            return None

        title = str(data.get("jobName") or "").strip()
        company_info = data.get("recommendInternCompany") or {}
        if not isinstance(company_info, dict):
            company_info = {}
        company = str(company_info.get("companyName") or "").strip()

        return RawJob(
            platform=PLATFORM,
            job_id=str(job_id),
            title=title,
            company=company,
            city=str(data.get("jobCity") or "").strip(),
            salary=_fmt_salary(data),
            url=DETAIL_URL_TMPL.format(job_id=job_id),
            description=_parse_ext(data.get("ext")),
            publish_date=_fmt_date(data.get("refreshTime"), data.get("createTime")),
        )

    # -- 公开：search ------------------------------------------------------
    async def search(
        self, keyword: str, city: Optional[str] = None, limit: int = 20
    ) -> list[RawJob]:
        """按关键词 + 城市抓取牛客实习岗位。

        接口不支持关键词参数，故按城市分页拉取后在**本地过滤 jobName**。
        然后再对**同城镜像**做一次内容去重（见 _dedup_mirrors）：牛客会把
        同公司同标题的岗位按城市拆成多条 id 不同的记录，只按 job_id 去重
        会把它们当成不同岗位全部留下。

        任何异常都被吞掉并返回已抓到的部分（基类约定：不抛异常）。
        """
        keyword = _PLATFORM_PREFIX.sub("", str(keyword or ""))
        keywords = [k.lower() for k in keyword.split() if k]
        job_city = _normalize_city(city)
        limit = max(1, int(limit or 20))

        results: list[RawJob] = []
        seen: set[str] = set()

        try:
            first = self._fetch_page(1, job_city)
            payload = first.get("data") if isinstance(first, dict) else None
            if not isinstance(payload, dict):
                return results

            total_page = int(payload.get("totalPage") or 1)
            pages = min(self.max_pages, max(1, total_page))

            for page in range(1, pages + 1):
                # 第 1 页已经拉过，直接复用；其余页现拉。
                page_payload = payload if page == 1 else None
                if page_payload is None:
                    raw = self._fetch_page(page, job_city)
                    page_payload = raw.get("data") if isinstance(raw, dict) else None
                    if not isinstance(page_payload, dict):
                        break  # 这一页彻底失败，停止翻页（已抓到的照常返回）

                datas = page_payload.get("datas") or []
                if not isinstance(datas, list) or not datas:
                    break

                for item in datas:
                    try:
                        job = self._to_raw_job(item)
                    except Exception as exc:  # 单条解析失败 -> 跳过
                        print(f"[niuke] 单条岗位解析失败，已跳过：{exc}")
                        continue
                    if job is None or job.job_id in seen:
                        continue
                    if not _match_keyword(job.title, keywords):
                        continue
                    seen.add(job.job_id)
                    results.append(job)
                    # 注意：这里不能在 limit 处提前 return——镜像去重发生在
                    # 全部候选收集完之后，提前截断会让删掉重复后的条数明显变少。
        except Exception as exc:  # 兜底：任何意外都不该拖垮整轮抓取
            print(f"[niuke] 抓取异常，返回已抓到的 {len(results)} 条：{exc}")
            return results[:limit]

        return _dedup_mirrors(results)[:limit]

    async def close(self) -> None:
        """关闭 HTTP 会话。"""
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None
