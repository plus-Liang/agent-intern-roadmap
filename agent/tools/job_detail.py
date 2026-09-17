"""
岗位详情工具。
根据 job_id 获取完整 JD。

接口：get_job_detail(platform, job_id) -> JobDetail
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JobDetail:
    """岗位完整信息"""
    platform: str
    job_id: str
    title: str
    company: str
    city: str
    salary: str
    url: str
    description: str = ""              # 岗位职责
    requirements: str = ""             # 任职要求
    bonus: str = ""                    # 加分项
    tags: list[str] = field(default_factory=list)
    education: str = ""                # 学历要求
    days_per_week: str = ""            # 每周出勤
    duration: str = ""                 # 实习时长


def get_job_detail(platform: str, job_id: str) -> JobDetail:
    """
    获取岗位详情。

    参数：
        platform: 平台名
        job_id: 岗位 ID

    返回：JobDetail
    """
    if platform == "mock":
        return _mock_detail(job_id)
    elif platform == "shixiseng":
        return _fetch_from_shixiseng(job_id)
    else:
        raise ValueError(f"不支持的平台：{platform}")


def _mock_detail(job_id: str) -> JobDetail:
    """Mock 实现：返回硬编码详情"""
    mock_db = {
        "mock_001": JobDetail(
            platform="mock",
            job_id="mock_001",
            title="LLM 预训练数据算法工程师（实习）",
            company="阶跃星辰",
            city="北京",
            salary="500-1000/天",
            url="https://example.com/job/001",
            description=(
                "你将参与工业级 LLM 预训练数据体系的构建与优化，包括但不限于：\n"
                "- 从 Web、Book、Code 等多源数据中进行数据挖掘与质量分析\n"
                "- 利用大规模 CPU/GPU 集群优化数据管线\n"
                "- 参与数据策略制定与消融实验设计"
            ),
            requirements=(
                "- 计算机、数学、数据科学等相关专业，本科及以上学历\n"
                "- CS/Coding 功底扎实，熟悉 Python，具备深度学习/强化学习基础知识\n"
                "- 对数据科学有兴趣，有较好的数据 sense"
            ),
            bonus="有参与 LLM 预训练数据相关工作的经验优先",
            tags=["LLM", "Python", "深度学习"],
            education="本科",
            days_per_week="5天/周",
            duration="6个月",
        ),
        "mock_002": JobDetail(
            platform="mock",
            job_id="mock_002",
            title="大模型算法",
            company="腾讯",
            city="北京",
            salary="300-500/天",
            url="https://example.com/job/002",
            description=(
                "1. 面向供应链业务，负责人工智能中心的设计、开发与部署\n"
                "2. 搭建人工智能中心，赋能各类业务功能\n"
                "3. 审阅业务需求文档，开发自动化解决方案"
            ),
            requirements=(
                "1. 计算机科学、计算机工程、软件工程或相关专业本科及以上学历\n"
                "2. 有人工智能或Python软件开发相关工作经验\n"
                "3. 掌握检索增强生成(RAG)、多智能体协作(MCP/A2A)等技术\n"
                "4. 熟悉主流大模型及其微调方法"
            ),
            tags=["RAG", "Agent", "Python"],
            education="本科",
            days_per_week="5天/周",
            duration="3个月",
        ),
        "mock_003": JobDetail(
            platform="mock",
            job_id="mock_003",
            title="VLM Agent实习生",
            company="索尼（中国）",
            city="深圳",
            salary="200-250/天",
            url="https://example.com/job/003",
            description=(
                "1. 参与基于视觉-语言模型（VLM）的智能体系统研发\n"
                "2. 协助开发 Agent 交互的 Web 前端界面\n"
                "3. 实现与测试 Agent 的核心逻辑"
            ),
            requirements=(
                "1. 计算机科学、人工智能等相关专业在读的本科高年级或硕士研究生\n"
                "2. 了解 LLM、VLM 及 Agent 基本原理\n"
                "3. 熟练掌握 Python，具备良好的代码结构"
            ),
            tags=["VLM", "Agent", "Python"],
            education="本科",
            days_per_week="3天/周",
            duration="3个月",
        ),
    }

    if job_id not in mock_db:
        raise ValueError(f"未找到岗位：{job_id}")
    return mock_db[job_id]


def _fetch_from_shixiseng(job_id: str) -> JobDetail:
    """真实抓取（待开发）"""
    raise NotImplementedError("实习僧详情抓取待实现")


if __name__ == "__main__":
    detail = get_job_detail("mock", "mock_001")
    print(f"【{detail.company}】{detail.title}")
    print(f"城市：{detail.city} | 薪资：{detail.salary} | 学历：{detail.education}")
    print(f"\n岗位职责：\n{detail.description}")
    print(f"\n任职要求：\n{detail.requirements}")