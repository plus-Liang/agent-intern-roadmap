"""
简历定制。
根据 JD 调整简历表达，突出相关经历，不改变事实。
"""
import json
import re
from dataclasses import asdict
from shared.llm_client import chat
from agent.tools.resume_match import Resume


TAILOR_PROMPT = """你是简历优化专家。根据岗位 JD 调整简历的表达，让简历更贴合该岗位。

【原始简历】
姓名：{name}
技能：{skills}
实习经历：{experience}
项目经历：{projects}
教育：{education}
城市：{city}

【目标岗位 JD】
公司：{company}
岗位：{title}
任职要求：
{requirements}

【任务】
1. 重排技能顺序：JD 要求的技能放前面。
2. 重写实习/项目描述：用 JD 的语言表达，突出与岗位相关的部分。
3. **绝对不能编造不存在的事实**：可以重排、重述，但不能新增经历、项目、技能。
4. 输出「定制后简历」+「修改说明」。

输出 JSON（只输出 JSON）：
{{
  "tailored": {{
    "name": "姓名",
    "skills": ["重排后的技能"],
    "experience": [
      {{"company": "公司", "role": "岗位", "months": 数字, "description": "重写后的描述"}}
    ],
    "projects": [
      {{"name": "项目", "tech": ["技术"], "desc": "重写后的描述"}}
    ],
    "education": "学历",
    "city": "城市"
  }},
  "changes": [
    "改动1",
    "改动2"
  ],
  "warnings": [
    "如果发现原文有与JD不匹配但无法美化的事实，在这里说明"
  ]
}}
"""


def tailor_resume(resume: Resume, job_detail) -> dict:
    """根据 JD 定制简历"""
    prompt = TAILOR_PROMPT.format(
        name=resume.name,
        skills=json.dumps(resume.skills, ensure_ascii=False),
        experience=json.dumps(resume.experience, ensure_ascii=False),
        projects=json.dumps(resume.projects, ensure_ascii=False),
        education=resume.education,
        city=resume.city,
        company=job_detail.company,
        title=job_detail.title,
        requirements=job_detail.requirements,
    )

    for attempt in range(1, 4):
        try:
            raw = chat([{"role": "user", "content": prompt}])
            data = _parse_json(raw)
            return data
        except Exception as e:
            print(f"[第{attempt}次失败] {e}")
            if attempt == 3:
                raise RuntimeError(f"简历定制失败：{e}")

    raise RuntimeError("简历定制失败")


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    return json.loads(text)


def format_result(result: dict) -> str:
    """把定制结果格式化为 Markdown"""
    tailored = result["tailored"]
    lines = []

    lines.append(f"# {tailored['name']} 的定制简历\n")
    lines.append(f"**教育**：{tailored['education']}  |  **城市**：{tailored['city']}\n")

    lines.append("## 技能")
    lines.append("、".join(tailored["skills"]) + "\n")

    lines.append("## 实习经历")
    for exp in tailored["experience"]:
        lines.append(f"### {exp['company']} | {exp['role']}（{exp.get('months', 0)} 个月）")
        lines.append(f"{exp['description']}\n")

    lines.append("## 项目经历")
    for p in tailored["projects"]:
        lines.append(f"### {p['name']}")
        lines.append(f"技术：{', '.join(p['tech'])}")
        lines.append(f"{p['desc']}\n")

    lines.append("---\n")
    lines.append("## 修改说明")
    for c in result.get("changes", []):
        lines.append(f"- {c}")

    if result.get("warnings"):
        lines.append("\n## 提醒")
        for w in result["warnings"]:
            lines.append(f"- ⚠️ {w}")

    return "\n".join(lines)


if __name__ == "__main__":
    from agent.tools.resume_match import Resume
    from agent.tools.job_detail import get_job_detail

    resume = Resume(
        name="张三",
        skills=["Python", "Go", "JavaScript", "LangChain", "FastAPI", "Chroma", "Git", "Docker"],
        experience=[
            {"company": "某创业公司", "role": "后端开发实习生", "months": 3,
             "description": "负责 RESTful API 开发，使用 FastAPI + PostgreSQL。"}
        ],
        projects=[
            {"name": "JD 知识库问答系统", "tech": ["RAG", "Chroma", "Chainlit"],
             "desc": "基于 RAG 的岗位信息问答系统，支持混合检索和引用溯源。"}
        ],
        education="本科",
        city="北京",
    )

    detail = get_job_detail("mock", "mock_002")  # 腾讯大模型算法

    print(f"目标岗位：{detail.company} | {detail.title}\n")
    result = tailor_resume(resume, detail)
    print(format_result(result))