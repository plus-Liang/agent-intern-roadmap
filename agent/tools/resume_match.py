"""
简历-JD 匹配工具。
输出多维度打分 + 差距分析，可解释。

接口：match_resume_to_jd(resume, job_detail) -> MatchResult
"""
from dataclasses import dataclass, field
import json
import time
from shared.llm_client import chat
from shared.limits import match_max_tokens, match_reasoning_effort


@dataclass
class Resume:
    """结构化简历"""
    name: str
    skills: list[str] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)  # [{"company": ..., "role": ..., "months": ...}]
    projects: list[dict] = field(default_factory=list)    # [{"name": ..., "tech": [...], "desc": ...}]
    education: str = ""                                    # "本科" / "硕士"（学历档位）
    city: str = ""
    educations: list[dict] = field(default_factory=list)   # [{"school","major","degree","start","end"}]


@dataclass
class MatchResult:
    """匹配结果"""
    score: int                              # 总分 0-100
    dimensions: dict                        # 各维度分数
    gaps: list[str] = field(default_factory=list)      # 差距列表（必须来自 JD 原文）
    highlights: list[str] = field(default_factory=list)  # 匹配亮点
    suggestions: list[str] = field(default_factory=list)  # 通用建议（JD 没提、仅供提醒）


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
教育经历：{json.dumps(resume.educations, ensure_ascii=False)}
教育：{resume.education}
城市：{resume.city}
实习经历：{json.dumps(resume.experience, ensure_ascii=False)}
项目经历：{json.dumps(resume.projects, ensure_ascii=False)}

【岗位 JD】
公司：{job_detail.company}
岗位：{job_detail.title}
城市：{job_detail.city}
薪资：{job_detail.salary}
学历要求：{job_detail.education}
岗位职责 / 描述：
{job_detail.description or "（无）"}
任职要求：
{job_detail.requirements or "（无）"}
加分项：
{job_detail.bonus or "（无）"}

【打分规则】
- 技能匹配（0-40）：JD 原文（上面的「岗位职责 / 描述」＋「任职要求」＋「加分项」）
  里出现的技能 / 工具 / 框架，简历覆盖多少。**不要因为「任职要求」字段为空就给 0 分** ——
  很多平台的 JD 把要求写在同一段描述里。
- 经历匹配（0-30）：实习/项目经历与岗位的相关性
- 学历匹配（0-15）：学历是否达标（超标或达标满分，差一档扣分）
- 城市匹配（0-15）：城市一致满分，不一致按情况扣分

【gaps 的硬约束（违反即不合格）】
- 上面的「岗位职责 / 描述」「任职要求」「加分项」合起来就是【岗位 JD】原文，
  gaps 每一项都必须来自这里**明确写出的要求**（技能 / 工具 / 框架 / 语言 / 职责 /
  学历 / 城市 / 年限），并且能在 JD 原文里找到对应的词。
- **严禁**引入 JD 里没有提到的技术栈、工具、框架或能力项：JD 没写 TensorFlow / PyTorch /
  部署 / 云服务，就不许把它们当成差距（"行业里通常都要"不算理由）。
- JD 确实要求、简历里找不到证据 → **必须如实写进 gaps**（有几项写几条，不要因为
  拿不准就留空）；只有每条要求都能在简历里找到证据时，gaps 才可以是空列表。
- JD 没提、但你仍想提醒候选人的通用建议 → 写进 "suggestions"，**不要**放进 gaps。

输出 JSON（只输出 JSON，不要其他内容）：
{{
  "score": 总分,
  "dimensions": {{
    "skills": 分数,
    "experience": 分数,
    "education": 分数,
    "location": 分数
  }},
  "gaps": ["来自 JD 原文的差距1", "来自 JD 原文的差距2"],
  "highlights": ["亮点1", "亮点2"],
  "suggestions": ["JD 未提及、仅作通用提醒的建议"]
}}
"""
    for attempt in range(1, 4):
        try:
            # 这里**必须**显式带上额度与思考档：chat() 不传就是全局默认 1024，
            # 而 glm-5.3-flash 是思考模型，思考（reasoning_content）与正文共用
            # max_tokens —— 真实 PDF 简历下思考必超 1024，正文要么被截断成
            # `Unterminated string starting at: line 19 column 5`，要么整段为空
            # （术语见 shared/limits.resume_max_tokens 的注释）。
            # 实测（真实简历 + 真实 JD）：默认档 23.8~24.3s 且 100% 触顶，
            # 4096 + low 降到 3.3~6.8s 且 JSON 合法，故单开一档 MATCH_LLM_*。
            content = chat(
                [{"role": "user", "content": prompt}],
                source="resume_match",
                max_tokens=match_max_tokens(),
                reasoning_effort=match_reasoning_effort(),
            )
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
                suggestions=data.get("suggestions", []),
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
