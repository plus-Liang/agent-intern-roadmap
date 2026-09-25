"""离线 mock 抓取器：不联网、不依赖 playwright，用来几秒钟跑通整条链路。

为什么需要它
------------
调并发参数时，每次验证都要跑一次约 40 分钟的真实抓取，代价太高。有了它：

    SCHEDULER_USE_MOCK=1 SCHEDULER_FAST_MODE=1 python -m agent.scrapers.scheduler --once

调度器会把平台切成 "mock"，用假数据把
「遍历平台 → 城市 × 关键词 → 去重 → 清洗 → 落库 → 切块 → 入库 → 落盘」
整条流程跑一遍，**不访问任何网络**，几秒钟出结果。

与"注入式假抓取器"的区别
------------------------
scheduler 里的 `scraper=` 参数是给单元测试用的**函数式替身**；本模块是走
`scraper_registry()` 的**正规平台**（名字 "mock"），所以调度逻辑一行都不用改，
走的代码路径与真实平台完全相同。

安全护栏
--------
假数据绝不能污染真实数据：mock 模式生效时，scheduler 会自动跳过
落库（jobs.db）、落盘（cleaned_jd.json / scraped_jd.txt）与向量入库，
只跑抓取 + 清洗 + 切块。详见 scheduler.run_daily_job 里的 mock_active 分支。
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any, Optional

from agent.scrapers.base import PlatformScraper, RawJob

PLATFORM = "mock"

# 每个「城市 × 关键词」返回几条：3~5 条足够验证去重 / 清洗 / 切块，
# 又不至于把日志刷满。
DEFAULT_COUNT = 4

# 假岗位模板。字段刻意贴近真实抓取结果：
#   * title / description 含关键词，方便穿过 cleaner 的相关性过滤；
#   * description 超过 DEFAULT_MIN_DESC_LEN(200)，方便穿过"正文太短"过滤；
#   * publish_date 取"今天"，方便穿过时效过滤。
# 这样 mock 跑出来的岗位能真正走完清洗 → 落库 → 切块，而不是在第一步就被全部丢掉。
_SAMPLES = [
    {
        "title": "AI Agent 开发实习生",
        "company": "Mock 科技有限公司",
        "salary": "200-300/天",
        "city": "广州",
        "days": "5 天/周",
        "body": (
            "负责面向招聘场景的 AI Agent 设计与落地：拆解业务问题、设计工具调用链路、"
            "编写提示词并做效果评测。要求熟悉 Python 与主流大模型 API，理解 RAG 检索增强"
            "生成的基本流程（切块、向量化、召回、重排），有 LangChain 或同类框架经验优先。"
            "本岗位为 mock 数据，仅用于离线跑通抓取到入库的整条链路。"
        ),
    },
    {
        "title": "大模型应用开发实习生",
        "company": "Mock 智能研究院",
        "salary": "250-400/天",
        "city": "北京",
        "days": "4 天/周",
        "body": (
            "参与大模型（LLM）应用层研发：提示词工程、函数调用、多轮对话状态管理，"
            "以及检索增强生成链路的调优。要求有扎实的 Python 基础，了解 embedding 与"
            "向量数据库的基本用法，能读懂论文并复现实验。本岗位为 mock 数据，"
            "仅用于离线验证调度器端到端流程，不代表真实招聘需求。"
        ),
    },
    {
        "title": "RAG 检索算法实习生",
        "company": "Mock 数据实验室",
        "salary": "300-450/天",
        "city": "深圳",
        "days": "5 天/周",
        "body": (
            "围绕 RAG（检索增强生成）做召回与重排优化：清洗语料、设计切块策略、"
            "对比不同 embedding 模型的召回率，搭建离线评测集并输出实验报告。"
            "要求熟悉 Python，了解 BM25、向量检索、rerank 的基本原理，有评测经验加分。"
            "本岗位为 mock 数据，仅用于离线跑通整条流程。"
        ),
    },
    {
        "title": "智能体平台后端实习生",
        "company": "Mock 云原生科技",
        "salary": "180-260/天",
        "city": "上海",
        "days": "3 天/周",
        "body": (
            "负责智能体平台的后端服务开发：任务编排、工具注册、调用日志与限流。"
            "要求熟悉 FastAPI 或 Flask，了解异步编程与并发控制（信号量、任务队列），"
            "能写清晰的单测。本岗位为 mock 数据，仅用于离线验证调度器端到端流程，"
            "不会写入任何真实数据库或语料文件。"
        ),
    },
]


# 附加在每条正文后面的"长尾说明"：body 本身只有 130~160 字，会被 cleaner 的
# 「正文太短」（min_desc_len 默认 200）整批滤掉，那样 chunk / 入库两步就永远跑不到。
# 这里刻意把它写长，让假岗位能真正穿过清洗链路，下游步骤才被验证到。
_MOCK_NOTICE = (
    "说明：本条为 MockScraper 生成的离线假数据，不来自任何真实招聘网站，不含真实公司"
    "信息，也不对应任何真实岗位。它存在的意义是：在没有网络、不消耗抓取额度、不触发"
    "风控的前提下，把「遍历平台 → 城市 × 关键词组合 → 去重 → 清洗 → 落库 → 切块 → "
    "向量入库 → 落盘」这整条调度链路完整跑通一遍，让改代码、调并发参数之后的回归验证"
    "从约 40 分钟的整轮抓取压缩到几秒钟。因此这段描述会被刻意写长，以便穿过清洗阶段"
    "的「正文太短」过滤，使切块与入库步骤也真正被执行到；同时 mock 模式下的所有写操作"
    "（jobs.db、cleaned_jd.json、向量库）都会被调度器护栏跳过，假数据不会污染任何真实"
    "数据。"
)


def _kv(keyword: str, city: Optional[str], index: int) -> str:
    """把关键词/城市压成 URL 安全的小写短串，用于拼 mock 的 url 与 job_id。"""
    raw = f"{keyword}-{city or 'any'}-{index}".lower()
    return "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in raw)


class MockScraper(PlatformScraper):
    """假数据抓取器：接口与真实平台一致，但不联网。

    实现的是 `PlatformScraper` 的同一套契约：
        * `search(keyword, city=..., limit=...)` → list[RawJob]
        * `search_multi(pairs, limit=..., return_groups=...)` → 批量（调度器批量路径）
        * `close()` → 空操作（没有浏览器要关）

    `**options` 吞掉调度器传进来的一切通用参数（headless / max_pages /
    detail_concurrency …）——mock 不翻页、不开浏览器，所以这些参数一律忽略，
    但**必须能收下**，否则 `_make_scraper()` 的参数过滤会把它挡下来。
    """

    platform_name = PLATFORM

    def __init__(self, **options: Any) -> None:
        # 调度器传进来的通用连接参数：收下但不使用（mock 没有浏览器/翻页概念）
        self.options = dict(options or {})
        self.timings: dict = {}          # 没有真实分页计时，给个空账本（调度器会跳过）
        self.last_query: dict | None = None

    # -- 核心：一次「关键词 + 城市」的假抓取 ---------------------------------
    async def search(self, keyword: str, city: str = None, limit: int = 20) -> list:
        """返回 3~5 条假岗位（不联网、不 sleep 真实时长）。"""
        # 让出一次事件循环：保持 async 语义，也让并发调用真的交错（便于测并发代码）
        await asyncio.sleep(0)

        count = DEFAULT_COUNT
        try:
            if limit and int(limit) > 0:
                count = max(1, min(DEFAULT_COUNT, int(limit)))
        except (TypeError, ValueError):
            count = DEFAULT_COUNT

        self.last_query = {"keyword": keyword, "city": city, "limit": limit, "count": count}
        today = date.today()
        jobs: list[RawJob] = []
        for index in range(count):
            sample = _SAMPLES[index % len(_SAMPLES)]
            slug = _kv(keyword, city, index + 1)
            jobs.append(RawJob(
                platform=PLATFORM,
                job_id=f"mock-{slug}",
                title=f"[MOCK] {sample['title']}",
                company=sample["company"],
                # 城市用调用方请求的城市：否则多城市抓取会被城市过滤整批丢掉
                city=(city or sample["city"]),
                salary=sample["salary"],
                url=f"https://example.invalid/mock/{slug}",
                description=(
                    f"{keyword}｜{sample['body']}"
                    f"（工作节奏：{sample['days']}；本条为第 {index + 1} 条 mock 数据）"
                    f"{_MOCK_NOTICE}"
                ),
                # 每条错开一天，顺便验证清洗的时效过滤不会误杀新数据
                publish_date=(today - timedelta(days=index)).isoformat(),
            ))
        return jobs

    # -- 批量路径：与真实抓取器的 search_multi 同形，调度器直接走并发分支 -----
    async def search_multi(
        self,
        pairs: list,
        limit: int = 20,
        concurrency: Optional[int] = None,
        return_groups: bool = False,
        progress_cb: Optional[Any] = None,
        chunk_size: Optional[int] = None,
    ) -> list:
        """按 [(keyword, city), ...] 返回结果。

        return_groups=False → 平铺的 RawJob 列表；
        return_groups=True  → 与 pairs 等长的
                              [{"keyword", "city", "jobs": [...], "error": ""}, ...]
        （与 ShixisengScraper.search_multi 的返回形状一致，调度器无需分支）。
        progress_cb / chunk_size 只为对齐签名（mock 不分块，但照样回调进度）。
        """
        del concurrency          # mock 不并发，参数只为对齐签名
        del chunk_size           # mock 无长等待，不需要分块
        import time as _time
        _t0 = _time.monotonic()
        combos: list[tuple[str, Optional[str]]] = []
        for item in pairs or []:
            if isinstance(item, (list, tuple)):
                keyword = str(item[0]).strip() if item else ""
                city = item[1] if len(item) > 1 else None
            else:
                keyword, city = str(item or "").strip(), None
            if keyword:
                combos.append((keyword, city))

        groups = []
        for keyword, city in combos:
            try:
                jobs = await self.search(keyword, city=city, limit=limit)
                groups.append({"keyword": keyword, "city": city, "jobs": jobs, "error": ""})
            except Exception as exc:                     # noqa: BLE001 - 单组合失败不拖垮整批
                groups.append({
                    "keyword": keyword, "city": city, "jobs": [],
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if progress_cb is not None:
                last = groups[-1]
                try:
                    progress_cb(
                        len(groups), len(combos),
                        f"{keyword} @ {city or '不限'}",
                        len(last["jobs"]), last["error"],
                        _time.monotonic() - _t0,
                    )
                except Exception:                        # noqa: BLE001 - 回调不拖垮抓取
                    pass

        if return_groups:
            return groups
        flat: list = []
        for group in groups:
            flat.extend(group["jobs"])
        return flat

    async def close(self) -> None:
        """没有浏览器/连接要关：显式实现，语义与真实抓取器一致（可 await）。"""
        await asyncio.sleep(0)
        self.last_query = None


__all__ = ["MockScraper", "PLATFORM", "DEFAULT_COUNT"]
