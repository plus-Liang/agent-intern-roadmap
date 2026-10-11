"""
简历-JD 匹配工具。
输出多维度打分 + 差距分析，可解释。

接口：match_resume_to_jd(resume, job_detail) -> MatchResult
"""
from dataclasses import dataclass, field
import json
import re
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
            # 仍然显式带上额度与思考档（不依赖全局兜底）：chat() 不传现在虽是
            # 4096 + low，但这里要的是「匹配这一档」的明确契约。
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


def _keywords(text) -> set:
    """把一段文本切成可比较的关键片段（英文/数字整词 + 中文 2 字滑窗）。

    刻意不引依赖、不做同义扩展：这是**降级路**的兜底匹配度，宁可少给分，
    也不要因为"猜"同义词而虚高（虚高的分数比不报分数更糟）。
    """
    raw = str(text or "")
    terms = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}", raw)}
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", raw):
        for i in range(len(run) - 1):
            terms.add(run[i:i + 2])
    return terms


_DEGREE_RANK = {"专科": 1, "大专": 1, "本科": 2, "学士": 2, "硕士": 3, "研究生": 3, "博士": 4}


def _degree_rank(text) -> int:
    """从文本里读学历档位（读不到返回 0）。"""
    for key, rank in _DEGREE_RANK.items():
        if key in str(text or ""):
            return rank
    return 0


def keyword_match(resume: Resume, job_detail) -> MatchResult:
    """额度不足时的降级打分：**纯关键词/规则**，不调 LLM。

    为什么必须有这条兜底路：额度用完后老的入口闸门是**全拒** —— 用户连
    「这份简历跟这个岗位像不像」都问不了。降级成关键词匹配度至少保住
    「能不能投」的判断，且明确标注这是降级结果（score 前说明 + degraded 字段）。

    维度口径与 LLM 路一致（技能 40 / 经历 30 / 学历 15 / 城市 15），
    但每项都用可解释的字面依据：
    - 技能：JD 原文里的技能片段被简历覆盖的比例（没有可判定项时按 0 计，
      并在 gaps 里说明是降级结果，不假装满分）；
    - 经历：简历经历/项目的字面片段与 JD 的重合度；
    - 学历：按档位比较（JD 没写要求 = 达标）；
    - 城市：完全一致满分，JD 未写城市按满分（不因数据缺失扣分）。
    """
    jd_text = " ".join(str(getattr(job_detail, f, "") or "") for f in
                       ("company", "title", "city", "salary", "education",
                        "description", "requirements", "bonus"))
    jd_text += " " + " ".join(str(t) for t in (getattr(job_detail, "tags", None) or []))
    jd_terms = _keywords(jd_text) - {"岗位", "职责", "要求", "工作", "公司", "我们", "以及", "相关"}

    skills = [s for s in (resume.skills or []) if str(s).strip()]
    hit = [s for s in skills if str(s).lower() in jd_text.lower()]
    missing = [s for s in skills if s not in hit]
    # 覆盖率只看「有交集的技能」：简历里的技能 JD 一个字没提就不进分母，
    # 否则「简历写了 20 个技能、JD 只要求 3 个」会被算成 15% 的假低分。
    coverage = len(hit) + len(missing)
    skill_score = min(40, int(round(40 * len(hit) / coverage))) if coverage else 0

    exp_text = json.dumps(
        {"experience": resume.experience, "projects": resume.projects},
        ensure_ascii=False)
    exp_terms = _keywords(exp_text)
    overlap = len(jd_terms & exp_terms)
    exp_score = min(30, int(round(30 * overlap / max(1, min(len(jd_terms), 40)))))

    jd_rank, resume_rank = _degree_rank(getattr(job_detail, "education", "")), \
        _degree_rank(resume.education)
    if jd_rank == 0 or resume_rank == 0:
        edu_score = 15
    elif resume_rank >= jd_rank:
        edu_score = 15
    else:
        edu_score = 8

    jd_city = str(getattr(job_detail, "city", "") or "")
    resume_city = str(resume.city or "")
    city_score = 15 if (not jd_city or not resume_city
                        or resume_city[:2] in jd_city or jd_city[:2] in resume_city) else 6

    dimensions = {"skills": skill_score, "experience": exp_score,
                  "education": edu_score, "location": city_score}
    score = max(0, min(100, sum(dimensions.values())))
    # gaps 只列「JD 里明确写过、简历技能表里没有」的技术词，按字面取，
    # 不做同义扩展（降级路的差距必须能在原文里指出来）。
    gaps = [t for t in sorted(jd_terms)
            if re.search(r"[a-z]", t) and t not in {str(s).lower() for s in skills}][:5]
    return MatchResult(
        score=score,
        dimensions=dimensions,
        gaps=gaps,
        highlights=[f"命中技能：{', '.join(hit[:6])}"] if hit else [],
        suggestions=["本条为降级结果：只做关键词比对，未经过模型分析。"],
    )


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
