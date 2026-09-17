"""
岗位搜索工具。
设计原则：接口稳定，实现可换。

- 接口：search_jobs(keyword, city, limit)
- 实现：先 mock，后接真实平台
- 未来可扩展：多平台适配器
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Job:
    """统一的岗位数据结构。

    publish_date 由具体抓取器填充（实习僧取详情页的刷新时间，格式 YYYY-MM-DD）；
    mock 数据与拿不到日期的平台保持 ""。
    """
    platform: str        # 平台名：shixiseng / boss / ...
    job_id: str          # 平台内唯一 ID
    title: str           # 岗位名
    company: str         # 公司名
    city: str            # 城市
    salary: str          # 薪资（字符串，保留原始格式）
    url: str             # 详情链接
    tags: list[str] = None  # 标签
    description: str = ""   # 描述（可选）
    publish_date: str = ""  # 发布时间 "YYYY-MM-DD"，拿不到填 ""


def search_jobs(
    keyword: str,
    city: Optional[str] = None,
    limit: int = 10,
    platform: str = "mock",
) -> list[Job]:
    """
    搜索岗位。

    参数：
        keyword: 搜索关键词，如 "Agent 开发"
        city: 城市过滤，如 "北京"，None 表示不限
        limit: 返回数量上限
        platform: 平台，先支持 "mock"，后续加 "shixiseng"

    返回：Job 列表
    """
    if platform == "mock":
        return _mock_search(keyword, city, limit)
    elif platform == "shixiseng":
        return _fetch_from_shixiseng(keyword, city, limit)
    else:
        raise ValueError(f"不支持的平台：{platform}")


def _mock_search(keyword: str, city: Optional[str], limit: int) -> list[Job]:
    """Mock 实现：返回硬编码的示例数据"""
    sample_data = [
        Job(
            platform="mock",
            job_id="mock_001",
            title="Agent 开发实习生",
            company="阶跃星辰",
            city="北京",
            salary="500-1000/天",
            url="https://example.com/job/001",
            tags=["LLM", "Agent", "Python"],
            description="参与工业级 LLM 预训练数据体系构建。",
        ),
        Job(
            platform="mock",
            job_id="mock_002",
            title="大模型算法实习生",
            company="腾讯",
            city="北京",
            salary="300-500/天",
            url="https://example.com/job/002",
            tags=["RAG", "多智能体", "Agent 架构"],
            description="面向供应链业务，搭建人工智能中心。",
        ),
        Job(
            platform="mock",
            job_id="mock_003",
            title="VLM Agent 实习生",
            company="索尼（中国）",
            city="深圳",
            salary="200-250/天",
            url="https://example.com/job/003",
            tags=["VLM", "多模态", "Agent"],
            description="基于视觉-语言模型的智能体系统研发。",
        ),
    ]

    # 简单过滤：按 city
    results = sample_data
    if city:
        results = [j for j in results if j.city == city]

    return results[:limit]


def _fetch_from_shixiseng(keyword: str, city: Optional[str], limit: int) -> list[Job]:
    """真实抓取实现（待开发）"""
    raise NotImplementedError("实习僧抓取待实现")


if __name__ == "__main__":
    # 测试
    jobs = search_jobs("Agent 开发", city="北京", limit=5)
    for j in jobs:
        print(f"[{j.company}] {j.title} | {j.city} | {j.salary}")
        print(f"  URL: {j.url}")
        print(f"  标签: {', '.join(j.tags) if j.tags else '无'}")
        print()