# -*- coding: utf-8 -*-
"""多平台岗位抓取器的抽象层。

为什么需要它
------------
`shixiseng.py` 原先是一份"写死的实习僧逻辑"（函数式接口
`search_shixiseng()` / `search_multi_keywords()`，返回 `agent.tools.job_search.Job`）。
要接第二个平台（牛客网 / 应届生 / BOSS），调度器就得为每个平台写一套分支——
平台越多，分支越多，最后没人敢动。

这里把"平台"抽成一个接口：

    * `RawJob`          —— 平台无关的岗位原始数据（统一字段、统一语义）
    * `PlatformScraper` —— 每个平台实现一个子类，只负责"给关键词+城市，还我一堆 RawJob"

调度器只依赖这两个东西，新增平台 = 新增一个子类 + 在注册表里登记一行，
不需要改调度逻辑。

字段口径（与 `rag/data/db.py` 的 jobs 表、`agent/tools/job_search.py` 的 Job 对齐）
-----------------------------------------------------------------------------
    platform      平台标识，如 "shixiseng"（**多平台数据的区分依据**）
    job_id        平台内唯一的岗位 ID（跨平台可能重名，故去重要用 platform+job_id）
    title         岗位名（明文）
    company       公司名
    city          城市
    salary        薪资原文（如 "200-300/天"）
    url           岗位详情页链接
    description   JD 正文（岗位职责 + 任职要求），保留 \\n 段落结构
    publish_date  发布时间 "YYYY-MM-DD"（拿不到为 ""）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class RawJob:
    """平台无关的岗位原始数据。

    这是各平台抓取器的**统一产出**：抓取器负责把平台自己的字段（可能叫
    name / position / 职位名称……）映射到这套字段上，调用方只认这套字段。
    """

    platform: str
    job_id: str
    title: str
    company: str
    city: str
    salary: str
    url: str
    description: str = ""
    publish_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转成纯 dict。

        `rag/data/db.py::upsert_jobs()` 与清洗模块收的都是 dict，
        转换只留这一处，避免每个调用方各写一遍字段名。
        """
        return asdict(self)


class PlatformScraper(ABC):
    """岗位抓取器抽象基类。

    子类必须：
        1) 设置类属性 `platform_name`（用于入库时区分平台、用于日志）；
        2) 实现 `search()`。

    `close()` 提供默认空实现：抓取器若有长驻资源（浏览器上下文、连接池）
    再覆写它。
    """

    platform_name: str = "unknown"

    @abstractmethod
    async def search(
        self, keyword: str, city: str, limit: int = 20
    ) -> list[RawJob]:
        """搜索岗位列表（含详情）。

        参数：
            keyword: 搜索关键词，如 "Agent"
            city:    城市，如 "广州"；None 表示不限城市
            limit:   返回条数上限（平台内部若支持翻页，由子类自行决定怎么用）

        返回：
            list[RawJob]；一条都抓不到时返回空列表，**不要抛异常**
            （单个关键词/城市失败不应该拖垮整轮抓取）。
        """

    async def close(self) -> None:
        """清理资源（默认什么都不做）。"""
        return None
