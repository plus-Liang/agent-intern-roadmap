"""
简历定制。
根据 JD 调整简历表达，突出相关经历，不改变事实。
"""
import json
import re
from dataclasses import asdict
from shared.llm_client import chat
from shared import limits
from agent.tools.resume_match import Resume


TAILOR_PROMPT = """你是简历优化专家。根据岗位 JD 调整简历的表达，让简历更贴合该岗位。

【原始简历】
姓名：{name}
技能：{skills}
教育经历：{educations}
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
3. **只写客观事实，不写自我评价**：每条描述 = 做了什么（技术 / 方法 / 动作）+ 可量化结果。
   严禁主观自夸句式，例如「体现了较强的效果调优能力」「展现了出色的学习方法与架构抽象
   能力」「具备独立完成工作并持续改进方案的能力」「注重系统安全性与工程规范」
   「出于对…的浓厚兴趣」「具备良好的学习能力」。这类句子一律不许出现 —— 宁可少写一句，
   也不许替候选人自夸。
4. **学校、专业、起止时间照抄，不许丢也不许改**：educations 每条都要带
   school / major / degree / start / end；experience、projects 的 start / end
   原样带回（原文没有就给空字符串）。
5. **绝对不能编造不存在的事实**：可以重排、重述，但不能新增经历、项目、技能、时间。
6. 输出「定制后简历」+「修改说明」。

输出 JSON（只输出 JSON）：
{{
  "tailored": {{
    "name": "姓名",
    "skills": ["重排后的技能"],
    "educations": [
      {{"school": "学校", "major": "专业", "degree": "学历",
        "start": "2025.09", "end": "至今"}}
    ],
    "experience": [
      {{"company": "公司", "role": "岗位", "start": "2025.06", "end": "2025.09",
        "months": 数字, "description": "重写后的描述"}}
    ],
    "projects": [
      {{"name": "项目", "tech": ["技术"], "start": "2025.10", "end": "2026.01",
        "desc": "重写后的描述"}}
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
        educations=json.dumps(getattr(resume, "educations", []), ensure_ascii=False),
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
            # 必须显式带额度与思考档：chat() 不传就是全局默认 1024，思考模型
            # （glm-5.3-flash）的思考与正文共用 max_tokens —— 实测默认档
            # finish_reason=length 且正文为空，json.loads 报
            # `Expecting value: line 1 column 1 (char 0)`，投递包只能退回原简历
            # （见 shared/limits.cover_letter_max_tokens 的实测记录）。
            raw = chat(
                [{"role": "user", "content": prompt}],
                source="tailor_resume",
                max_tokens=limits.cover_letter_max_tokens(),
                reasoning_effort=limits.cover_letter_reasoning_effort(),
            )
            data = _parse_json(raw)
            return _sanitize_result(data)
        except Exception as e:
            print(f"[第{attempt}次失败] {e}")
            if attempt == 3:
                raise RuntimeError(f"简历定制失败：{e}")

    raise RuntimeError("简历定制失败")


# 主观自夸句式（问题 2）：prompt 之外再用确定性后处理兜一层，
# 保证产物里不会残留「体现了…能力 / 展现了…精神」这类自我评价。
_PRAISE_CLAUSE = re.compile(
    r"[，,、；;]?\s*(?:这|也|既|还)?\s*(?:充分|有效|较好|很好地|进一步)?\s*"
    r"(?:体现|展现|展示|彰显|证明|说明|反映|表现出?)(?:了|出)?[^，。；;！？\n]{0,30}?"
    r"(?:能力|素养|精神|意识|习惯|潜力|态度|热情|水平|功底|思维)"
    r"|[，,、；;]\s*(?:具备|拥有|注重|善于|擅长)[^，。；;！？\n]{0,25}?"
    r"(?:能力|素养|精神|意识|习惯|态度|潜力|热情|水平|功底)"
)
_PRAISE_SENTENCE = re.compile(
    r"^(?:这|也|既)?(?:充分|有效|较好|进一步)?"
    r"(?:体现|展现|展示|彰显|证明|说明|反映|表现出?)(?:了|出)"
    r"|^(?:较强|出色|优秀|良好|扎实|深厚|很强)的[^，。；;！？\n]{0,20}"
    r"(?:能力|素养|精神|意识|功底|水平)"
    r"|^(?:具备|拥有|注重)[^，。；;！？\n]{0,20}"
    r"(?:能力|素养|精神|意识|习惯|态度|潜力)"
    r"|^出于对[^，。；;！？\n]{0,25}的(?:浓厚|强烈)(?:兴趣|热情)"
)


def strip_self_praise(text: str) -> str:
    """删掉描述里的主观自我评价，只留「做了什么 + 量化结果」。

    两步：① 句内自夸小句（连同前导逗号）整段切掉；
    ② 以自夸动词开头的整句直接丢弃（"具备了…能力" 这类没有事实的句子）。
    最后清理悬空标点；原文全是自夸时返回空串 —— 宁可少写，也不自夸。
    """
    if not text:
        return ""
    cleaned = _PRAISE_CLAUSE.sub("", str(text))

    kept = []
    for sentence in re.split(r"(?<=[。！？；;])", cleaned):
        core = sentence.strip(" ，,、；;。！？\n")
        if not core:
            continue
        if _PRAISE_SENTENCE.match(core):
            continue
        kept.append(sentence)
    cleaned = "".join(kept)

    cleaned = re.sub(r"[，,、；;]{2,}", "，", cleaned)
    cleaned = re.sub(r"[，,、；;]+\s*(?=[。！？；;])", "", cleaned)
    cleaned = re.sub(r"^[，,、；;\s]+", "", cleaned)
    cleaned = re.sub(r"[，,、；;\s]+$", "", cleaned)
    return cleaned.strip()


def _sanitize_result(result: dict) -> dict:
    """对定制结果做确定性清洗（问题 2）：描述里不许留自我评价。"""
    tailored = (result or {}).get("tailored")
    if not isinstance(tailored, dict):
        return result
    for exp in (tailored.get("experience") or []):
        if isinstance(exp, dict) and exp.get("description"):
            exp["description"] = strip_self_praise(exp["description"])
    for proj in (tailored.get("projects") or []):
        if isinstance(proj, dict) and proj.get("desc"):
            proj["desc"] = strip_self_praise(proj["desc"])
    return result


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

    lines.append("## 教育经历")
    for edu in (tailored.get("educations") or []):
        period = f"{edu.get('start', '')}-{edu.get('end', '')}" if edu.get("start") else ""
        part = " | ".join(str(x) for x in (
            edu.get("school", ""),
            f"{edu.get('major', '')}（{edu.get('degree', '')}）" if edu.get("degree") else edu.get("major", ""),
            period or "",
        ) if str(x).strip())
        lines.append(part or str(edu))
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