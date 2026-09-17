"""
工具注册中心。
"""
import json
from agent.tools.job_search import search_jobs
from agent.tools.job_detail import get_job_detail
from agent.tools.resume_match import match_resume_to_jd, Resume
from agent import storage


def _search(keyword, city=None, limit=5):
    return [
        {
            "job_id": j.job_id,
            "title": j.title,
            "company": j.company,
            "city": j.city,
            "salary": j.salary,
            "tags": j.tags or [],
        }
        for j in search_jobs(keyword, city, limit, platform="mock")
    ]


def _detail(job_id):
    d = get_job_detail("mock", job_id)
    return {
        "job_id": d.job_id,
        "title": d.title,
        "company": d.company,
        "city": d.city,
        "salary": d.salary,
        "description": d.description,
        "requirements": d.requirements,
        "education": d.education,
    }


def _match(job_id, resume_json):
    if isinstance(resume_json, str):
        resume_data = json.loads(resume_json)
    else:
        resume_data = resume_json
    resume = Resume(
        name=resume_data.get("name", "匿名"),
        skills=resume_data.get("skills", []),
        experience=resume_data.get("experience", []),
        projects=resume_data.get("projects", []),
        education=resume_data.get("education", ""),
        city=resume_data.get("city", ""),
    )
    detail = get_job_detail("mock", job_id)
    result = match_resume_to_jd(resume, detail)
    return {
        "score": result.score,
        "dimensions": result.dimensions,
        "gaps": result.gaps,
        "highlights": result.highlights,
    }


def _add_tracking(company, title, platform="mock", url=""):
    """添加投递记录"""
    app_id = storage.create_application(company, title, platform, url)
    return {"id": app_id, "company": company, "title": title, "status": "applied"}


def _list_tracking(status=None):
    """查询投递记录"""
    apps = storage.list_applications(status)
    return [
        {
            "id": a["id"],
            "company": a["company"],
            "title": a["title"],
            "status": a["status"],
            "applied_at": a["applied_at"],
        }
        for a in apps
    ]


TOOLS = {
    "search_jobs": {
        "description": "搜索实习岗位，返回岗位列表。",
        "parameters": {
            "keyword": "搜索关键词",
            "city": "城市（可选）",
            "limit": "数量，默认 5",
        },
        "func": _search,
    },
    "get_job_detail": {
        "description": "获取岗位完整 JD 详情。",
        "parameters": {"job_id": "岗位 ID"},
        "func": _detail,
    },
    "match_resume": {
        "description": "简历与岗位匹配打分。",
        "parameters": {
            "job_id": "岗位 ID",
            "resume_json": "简历 JSON 字符串或对象",
        },
        "func": _match,
    },
    "add_tracking": {
        "description": "把岗位添加到投递追踪系统。",
        "parameters": {
            "company": "公司名",
            "title": "岗位名",
            "url": "岗位链接（可选）",
        },
        "func": _add_tracking,
    },
    "list_tracking": {
        "description": "查询投递追踪记录。",
        "parameters": {
            "status": "状态过滤（可选），如 applied/viewed/interview",
        },
        "func": _list_tracking,
    },
}


def list_tools_description() -> str:
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"### {name}")
        lines.append(f"作用：{info['description']}")
        lines.append("参数：")
        for k, v in info["parameters"].items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def call_tool(name: str, args: dict):
    if name not in TOOLS:
        raise ValueError(f"未知工具：{name}")
    return TOOLS[name]["func"](**args)