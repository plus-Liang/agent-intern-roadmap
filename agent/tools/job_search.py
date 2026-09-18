"""
岗位搜索工具。
设计原则：接口稳定，实现可换。

- 接口：search_jobs(keyword, city, limit)
- 实现：优先读本地真实数据 rag/data/cleaned_jd.json，读不到时回退硬编码 mock
- 未来可扩展：多平台适配器
"""
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# 真实数据路径：<repo_root>/rag/data/cleaned_jd.json
# 本文件位于 <repo_root>/agent/tools/job_search.py，parents[2] 即仓库根目录
REAL_JD_PATH = Path(__file__).resolve().parents[2] / "rag" / "data" / "cleaned_jd.json"


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
        keyword: 搜索关键词，如 "Agent 开发"。大小写不敏感；
                 含空格时按多个词处理，要求全部命中 title 或 description。
        city: 城市过滤，如 "广州"，None/"" 表示不限
        limit: 返回数量上限
        platform: "mock"（本地真实数据，读不到时回退硬编码 mock）；
                  别名 "agent" 等价于 "mock"；"shixiseng" 为在线抓取（待实现）

    返回：Job 列表（无命中返回空列表）
    """
    if platform in ("mock", "agent"):
        return _mock_search(keyword, city, limit)
    elif platform == "shixiseng":
        return _fetch_from_shixiseng(keyword, city, limit)
    else:
        raise ValueError(f"不支持的平台：{platform}")


def _load_real_jobs() -> list[Job]:
    """从 rag/data/cleaned_jd.json 读取全部真实岗位，并映射成 Job。

    返回空列表表示"真实数据不可用"（文件不存在 / JSON 损坏 / 结构不是 list），
    调用方据此回退到硬编码 mock 数据。
    """
    try:
        with REAL_JD_PATH.open(encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[job_search] 读取真实数据失败，回退 mock：{REAL_JD_PATH}（{exc}）", file=sys.stderr)
        return []

    if not isinstance(raw, list):
        print(f"[job_search] 真实数据格式异常（期望 list）：{REAL_JD_PATH}", file=sys.stderr)
        return []

    jobs: list[Job] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("job_id"):
            continue
        jobs.append(
            Job(
                platform=item.get("platform") or "shixiseng",
                job_id=str(item["job_id"]),
                title=item.get("title") or "",
                company=item.get("company") or "",
                city=item.get("city") or "",
                salary=item.get("salary") or "",
                url=item.get("url") or "",
                # 真实数据里暂无 tags 字段，保持 None（调用方用 `j.tags or []` 兜底）
                tags=list(item["tags"]) if item.get("tags") else None,
                description=item.get("description") or "",
                publish_date=item.get("publish_date") or "",
            )
        )
    return jobs


def _normalize_city(city: Optional[str]) -> str:
    """规范化城市名：去空白、去掉结尾的“市”。"""
    text = (city or "").strip()
    if text.endswith("市"):
        text = text[:-1]
    return text


def _match_keyword(job: Job, keyword: Optional[str]) -> bool:
    """关键词匹配：大小写不敏感，命中 title 或 description 即可。

    keyword 含空格时按多词处理（如 "Agent 开发"），要求所有词都命中，
    这样 "AI Agent开发" 这类没有空格分隔的标题也能被搜到。
    """
    target = (keyword or "").strip().lower()
    if not target:
        return True
    haystack = f"{job.title}\n{job.description}".lower()
    return all(term in haystack for term in target.split())


def _match_city(job: Job, city: Optional[str]) -> bool:
    """城市匹配：city 为空表示不限；否则按规范化后的城市名比较。"""
    target = _normalize_city(city)
    if not target:
        return True
    actual = _normalize_city(job.city)
    if not actual:
        return False
    return target == actual or target in actual or actual in target


def _mock_search(keyword: str, city: Optional[str], limit: int) -> list[Job]:
    """默认实现：优先用 rag/data/cleaned_jd.json 的真实数据，读不到才回退硬编码 mock。"""
    real_jobs = _load_real_jobs()
    if real_jobs:
        results = [
            j for j in real_jobs
            if _match_keyword(j, keyword) and _match_city(j, city)
        ]
        return results[:limit]

    # ---- fallback：真实数据不可用时使用硬编码示例（保留原数据，语义与真实数据一致）----
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

    results = [
        j for j in sample_data
        if _match_keyword(j, keyword) and _match_city(j, city)
    ]
    return results[:limit]


def _fetch_from_shixiseng(keyword: str, city: Optional[str], limit: int) -> list[Job]:
    """真实抓取实现（待开发）"""
    raise NotImplementedError("实习僧抓取待实现")


if __name__ == "__main__":
    # 便于 `python agent/tools/job_search.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.tools.job_detail import get_job_detail

    def _show(jobs: list[Job]) -> None:
        if not jobs:
            print("  （空）")
        for j in jobs:
            print(f"  [{j.company}] {j.title} | {j.city} | {j.salary} | id={j.job_id}")

    print(f"数据源：{REAL_JD_PATH}（exists={REAL_JD_PATH.exists()}）")

    print("\n== 1) search_jobs('Agent', city='广州') ==")
    jobs_agent = search_jobs("Agent", city="广州", limit=5)
    _show(jobs_agent)
    assert jobs_agent, "应返回真实数据里的 Agent 岗位"
    assert all("agent" in f"{j.title}\n{j.description}".lower() for j in jobs_agent), "结果必须命中 keyword"

    print("\n== 2) search_jobs('Python', city='广州') ==")
    jobs_py = search_jobs("Python", city="广州", limit=5)
    _show(jobs_py)
    assert jobs_py, "应返回含 Python 的岗位"
    assert all("python" in f"{j.title}\n{j.description}".lower() for j in jobs_py), "结果必须命中 keyword"

    print("\n== 3) search_jobs('不存在', city='广州') ==")
    jobs_none = search_jobs("不存在", city="广州", limit=5)
    _show(jobs_none)
    assert jobs_none == [], "无关关键词应返回空列表"

    target_id = jobs_agent[0].job_id if jobs_agent else "inn_78xqcaa6aktp"
    print(f"\n== 4) get_job_detail('agent', '{target_id}') ==")
    detail = get_job_detail("agent", target_id)
    print(f"  [{detail.company}] {detail.title} | {detail.city} | {detail.salary}")
    print(f"  url: {detail.url}")
    print(f"  description {len(detail.description)} 字 | requirements {len(detail.requirements)} 字")
    print(f"  description 预览：{detail.description[:80]}...")
    assert detail.job_id == target_id, "job_id 应原样返回"
    assert detail.title and detail.company and detail.city and detail.salary, "基础字段不应为空"
    assert detail.description, "description 不应为空"

    print("\n== 5) 未知 job_id 应抛 ValueError ==")
    try:
        get_job_detail("agent", "__no_such_job_id__")
    except ValueError as exc:
        print(f"  正确抛出 ValueError：{exc}")
    else:
        raise AssertionError("未知 job_id 应抛 ValueError")

    print("\n[OK] 全部验证通过")
