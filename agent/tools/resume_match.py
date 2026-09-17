"""
简历-JD 匹配工具。
输出多维度打分 + 差距分析，可解释。

接口：match_resume_to_jd(resume, job_detail) -> MatchResult
"""
from dataclasses import dataclass, field
import json
import time
from shared.llm_client import chat


@dataclass
class Resume:
    """结构化简历"""
    name: str
    skills: list[str] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)  # [{"company": ..., "role": ..., "months": ...}]
    projects: list[dict] = field(default_factory=list)    # [{"name": ..., "tech": [...], "desc": ...}]
    education: str = ""                                    # "本科" / "硕士"
    city: str = ""


@dataclass
class MatchResult:
    """匹配结果"""
    score: int                              # 总分 0-100
    dimensions: dict                        # 各维度分数
    gaps: list[str] = field(default_factory=list)      # 差距列表
    highlights: list[str] = field(default_factory=list)  # 匹配亮点


def match_resume_to_jd(resume: Resume, job_detail) -> MatchResult:
    """
    简历-JD 匹配打分。

    维度：
    - 技能匹配（40分）：JD 要求的技能，简历覆盖多少
    - 经历匹配（30分）：实习/项目经历是否相关
    - 学历匹配（15分）：学历是否达标
    - 城市匹配（15分）：城市是否一致

    返回：MatchResult
    """
    prompt = f"""你是简历-JD 匹配专家。分析下面的简历和岗位 JD，输出匹配打分。

【简历】
姓名：{resume.name}
技能：{', '.join(resume.skills)}
教育：{resume.education}
城市：{resume.city}
实习经历：{json.dumps(resume.experience, ensure_ascii=False)}
项目经历：{json.dumps(resume.projects, ensure_ascii=False)}

【岗位 JD】
公司：{job_detail.company}
岗位：{job_detail.title}
城市：{job_detail.city}
学历要求：{job_detail.education}
任职要求：
{job_detail.requirements}

【打分规则】
- 技能匹配（0-40）：JD 要求的技能，简历覆盖多少
- 经历匹配（0-30）：实习/项目经历与岗位的相关性
- 学历匹配（0-15）：学历是否达标（超标或达标满分，差一档扣分）
- 城市匹配（0-15）：城市一致满分，不一致按情况扣分

输出 JSON（只输出 JSON，不要其他内容）：
{{
  "score": 总分,
  "dimensions": {{
    "skills": 分数,
    "experience": 分数,
    "education": 分数,
    "location": 分数
  }},
  "gaps": ["差距1", "差距2"],
  "highlights": ["亮点1", "亮点2"]
}}
"""
    for attempt in range(1, 4):
        try:
            content = chat([{"role": "user", "content": prompt}])
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            data = json.loads(content)
            return MatchResult(
                score=data["score"],
                dimensions=data["dimensions"],
                gaps=data.get("gaps", []),
                highlights=data.get("highlights", []),
            )
        except Exception as e:
            print(f"[第{attempt}次失败] {e}")
            if attempt < 3:
                time.sleep(attempt * 2)

    raise RuntimeError("匹配打分失败")


if __name__ == "__main__":
    from agent.tools.job_detail import get_job_detail

    # 示例简历
    resume = Resume(
        name="张三",
        skills=["Python", "RAG", "LangChain", "Git"],
        experience=[
            {"company": "某创业公司", "role": "后端实习生", "months": 3}
        ],
        projects=[
            {"name": "JD 知识库问答", "tech": ["RAG", "Chroma", "Chainlit"], "desc": "基于 RAG 的岗位信息问答系统"}
        ],
        education="本科",
        city="北京",
    )

    # 对 3 个岗位分别打分
    for job_id in ["mock_001", "mock_002", "mock_003"]:
        detail = get_job_detail("mock", job_id)
        print(f"\n{'='*60}")
        print(f"岗位：{detail.company} | {detail.title} | {detail.city}")
        print(f"{'='*60}")
        result = match_resume_to_jd(resume, detail)
        print(f"总分：{result.score}/100")
        print(f"维度：{result.dimensions}")
        print(f"差距：{result.gaps}")
        print(f"亮点：{result.highlights}")