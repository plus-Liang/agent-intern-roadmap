"""
岗位详情工具。
根据 job_id 获取完整 JD。

接口：get_job_detail(platform, job_id) -> JobDetail
实现：优先查 SQLite（rag/data/jobs.db，主键查询），
      jobs.db 建不起来时回退 rag/data/cleaned_jd.json，再不行回退硬编码 mock
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# 本文件位于 <repo_root>/agent/tools/job_detail.py，parents[2] 即仓库根目录
_REPO_ROOT = Path(__file__).resolve().parents[2]

# 兜底 JSON 数据路径：<repo_root>/rag/data/cleaned_jd.json
REAL_JD_PATH = _REPO_ROOT / "rag" / "data" / "cleaned_jd.json"

# 模块初始值：REAL_JD_PATH 被外部改写（单测 / 临时数据源）时就知道该走 JSON 而不是 DB
_DEFAULT_JD_PATH = REAL_JD_PATH


def _db_module():
    """懒导入 rag.data.db（顺带保证仓库根在 sys.path 上）。

    放函数里而不是模块顶层：`python agent/tools/job_detail.py` 直接运行时 sys.path
    里没有仓库根，顶层导入会炸；懒导入也让模块导入保持无副作用。
    """
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from rag.data import db as _db
    return _db


# JD 正文里的分节标记：真实数据把职责和要求写在一段 description 里，用这些标记切分
_REQUIREMENT_MARKERS = (
    "【任职要求】",
    "【岗位要求】",
    "【任职资格】",
    "任职资格（学历、目标院校、语言、技能、性格等要求）",
    "任职要求",
    "岗位要求",
    "任职资格",
)
_BONUS_MARKERS = ("【加分项】", "加分项")
# description 开头的容器标记，去掉只是排版清理
_DESCRIPTION_PREFIXES = ("【职位描述】", "【岗位描述】", "【工作职责】", "【职位信息】", "【岗位职责】")


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
        platform: "mock"（本地真实数据，查不到时回退硬编码 mock）；
                  别名 "agent" 等价于 "mock"；"shixiseng" 为在线抓取（待实现）
        job_id: 岗位 ID

    返回：JobDetail

    异常：未找到该 job_id 时抛 ValueError
    """
    if platform in ("mock", "agent"):
        return _mock_detail(job_id)
    elif platform == "shixiseng":
        return _fetch_from_shixiseng(job_id)
    else:
        raise ValueError(f"不支持的平台：{platform}")


def _split_description(text: str) -> tuple[str, str, str]:
    """把一整段 JD 切成 (岗位职责/描述, 任职要求, 加分项)。

    只做原文切片，不改写、不新增内容；找不到分节标记时，
    整段文本留在 description，requirements / bonus 为空字符串。
    """
    text = text or ""
    if not text.strip():
        return "", "", ""

    def _first_marker(markers: tuple[str, ...]) -> tuple[int, str]:
        """返回出现位置最靠前的标记（同一位置优先取最长的那个，避免残留 "】"）。"""
        hits = [(text.find(m), m) for m in markers]
        hits = [(i, m) for i, m in hits if i != -1]
        if not hits:
            return -1, ""
        position = min(i for i, _ in hits)
        return position, max((m for i, m in hits if i == position), key=len)

    req_index, req_marker = _first_marker(_REQUIREMENT_MARKERS)

    if req_index == -1:
        description = text.strip()
        for prefix in _DESCRIPTION_PREFIXES:
            if description.startswith(prefix):
                description = description[len(prefix):].strip()
                break
        return description, "", ""

    description = text[:req_index].strip()
    for prefix in _DESCRIPTION_PREFIXES:
        if description.startswith(prefix):
            description = description[len(prefix):].strip()
            break
    rest = text[req_index + len(req_marker):].strip()

    bonus_index, bonus_marker = _first_marker(_BONUS_MARKERS)
    # 加分项必须在 requirements 段内出现才算
    bonus = ""
    if bonus_index != -1 and bonus_index >= req_index + len(req_marker):
        bonus = text[bonus_index + len(bonus_marker):].strip()
        rest = text[req_index + len(req_marker):bonus_index].strip()

    return description or text.strip(), rest, bonus


def _detail_from_row(row: dict) -> JobDetail:
    """把 SQLite 一行（dict）映射成 JobDetail（description 仍按分节标记切分）。"""
    description, requirements, bonus = _split_description(row.get("description") or "")
    return JobDetail(
        platform=row.get("platform") or "shixiseng",
        job_id=str(row.get("job_id") or ""),
        title=row.get("title") or "",
        company=row.get("company") or "",
        city=row.get("city") or "",
        salary=row.get("salary") or "",
        url=row.get("url") or "",
        description=description,
        requirements=requirements,
        bonus=bonus,
        tags=list(row["tags"]) if row.get("tags") else [],
    )


def _query_db(job_id: str) -> tuple[bool, Optional[JobDetail]]:
    """按 job_id 单条查 SQLite。返回 (库是否可用, 命中的详情)。

    库不可用（jobs.db 建不起来 / REAL_JD_PATH 被显式改写）时 available=False，
    调用方据此回退 JSON；available=True 但 detail 为 None 表示库里确实没这条。
    """
    if REAL_JD_PATH != _DEFAULT_JD_PATH:
        return False, None
    try:
        db = _db_module()
        if not db.ensure_db():
            return False, None
        row = db.get_job(job_id)
    except Exception as exc:  # noqa: BLE001 —— 数据源坏了不该让详情查询整体挂掉
        print(f"[job_detail] SQLite 不可用，回退 JSON：{exc}", file=sys.stderr)
        return False, None
    return True, (_detail_from_row(row) if row else None)


def _load_real_details() -> dict[str, JobDetail]:
    """返回 {job_id: JobDetail}：优先 SQLite，不可用时回退 cleaned_jd.json。

    空 dict 表示「真实数据不可用」，调用方据此回退硬编码 mock。
    这个函数会把全量详情读进内存，只适合自检/演示；按 id 查详情走 _query_db
    （主键查询，O(log n)）。
    """
    details = _load_details_from_db()
    if details is not None:
        return details
    return _load_details_from_json()


def _load_details_from_db() -> Optional[dict[str, JobDetail]]:
    """全量读 SQLite；None 表示「库不可用」。"""
    if REAL_JD_PATH != _DEFAULT_JD_PATH:
        return None
    try:
        db = _db_module()
        if not db.ensure_db():
            return None
        rows = db.get_all_jobs()
    except Exception as exc:  # noqa: BLE001
        print(f"[job_detail] SQLite 不可用，回退 JSON：{exc}", file=sys.stderr)
        return None
    return {
        str(row["job_id"]): _detail_from_row(row)
        for row in rows if row.get("job_id")
    }


def _load_details_from_json() -> dict[str, JobDetail]:
    """从 rag/data/cleaned_jd.json 读取所有岗位，返回 {job_id: JobDetail}。

    返回空 dict 表示"真实数据不可用"（文件不存在 / JSON 损坏 / 结构不是 list），
    调用方据此回退到硬编码 mock 数据。
    """
    try:
        with REAL_JD_PATH.open(encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[job_detail] 读取真实数据失败，回退 mock：{REAL_JD_PATH}（{exc}）", file=sys.stderr)
        return {}

    if not isinstance(raw, list):
        print(f"[job_detail] 真实数据格式异常（期望 list）：{REAL_JD_PATH}", file=sys.stderr)
        return {}

    details: dict[str, JobDetail] = {}
    for item in raw:
        if not isinstance(item, dict) or not item.get("job_id"):
            continue
        description, requirements, bonus = _split_description(item.get("description") or "")
        job_id = str(item["job_id"])
        details[job_id] = JobDetail(
            platform=item.get("platform") or "shixiseng",
            job_id=job_id,
            title=item.get("title") or "",
            company=item.get("company") or "",
            city=item.get("city") or "",
            salary=item.get("salary") or "",
            url=item.get("url") or "",
            description=description,
            requirements=requirements,
            bonus=bonus,
            tags=list(item["tags"]) if item.get("tags") else [],
        )
    return details


def _mock_detail(job_id: str) -> JobDetail:
    """默认实现：优先查 SQLite，其次 cleaned_jd.json，查不到再回退硬编码 mock。"""
    available, detail = _query_db(job_id)
    if detail is not None:
        return detail

    if not available:
        real_db = _load_real_details()
        if job_id in real_db:
            return real_db[job_id]

    # ---- fallback：真实数据不可用（或该 id 属于旧 mock 数据）时使用硬编码详情 ----
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
    db_path = _db_module().DB_PATH
    real_db = _load_real_details()
    print(f"数据源：SQLite {db_path}（exists={db_path.exists()}）")
    print(f"兜底 JSON：{REAL_JD_PATH}（exists={REAL_JD_PATH.exists()}）")
    print(f"可读岗位：{len(real_db)} 条")

    # 优先取真实数据里的岗位，真实数据不可用时回退到 mock_001
    target_id = next(iter(real_db), "mock_001")
    detail = get_job_detail("agent", target_id)
    print(f"\n【{detail.company}】{detail.title}")
    print(f"platform：{detail.platform} | id：{detail.job_id}")
    print(f"城市：{detail.city} | 薪资：{detail.salary} | 学历：{detail.education or '（数据未提供）'}")
    print(f"URL：{detail.url}")
    print(f"\n岗位职责（{len(detail.description)} 字）：\n{detail.description[:300]}")
    print(f"\n任职要求（{len(detail.requirements)} 字）：\n{detail.requirements[:300]}")
    if detail.bonus:
        print(f"\n加分项（{len(detail.bonus)} 字）：\n{detail.bonus[:300]}")

    try:
        get_job_detail("agent", "__no_such_job_id__")
    except ValueError as exc:
        print(f"\n[OK] 未知 job_id 正确抛 ValueError：{exc}")
    else:
        raise AssertionError("未知 job_id 应抛 ValueError")
