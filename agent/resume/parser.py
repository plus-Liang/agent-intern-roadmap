"""
简历解析。
输入 PDF / TXT / 纯文本，输出结构化 Resume 对象。
"""
import json
import re
from pathlib import Path
from shared import limits
from shared.llm_client import chat
from agent.tools.resume_match import Resume


EXTRACT_PROMPT = """你是简历解析专家。从下面的简历文本中提取结构化信息。

简历文本：
{resume_text}

请输出 JSON（只输出 JSON，不要其他内容）：
{{
  "name": "姓名",
  "skills": ["技能1", "技能2"],
  "educations": [
    {{"school": "学校名", "major": "专业", "degree": "本科" 或 "硕士" 或 "博士",
      "start": "2025.09", "end": "至今"}}
  ],
  "experience": [
    {{"company": "公司名", "role": "岗位", "start": "2025.06", "end": "2025.09",
      "months": 数字, "description": "工作描述"}}
  ],
  "projects": [
    {{"name": "项目名", "tech": ["技术1"], "start": "2025.10", "end": "2026.01",
      "desc": "项目描述"}}
  ],
  "education": "本科" 或 "硕士" 或 "博士",
  "city": "当前城市",
  "email": "邮箱",
  "phone": "手机号"
}}

规则：
1. 如果某字段在简历中找不到，填空字符串或空数组。
2. skills 去重、标准化（如"会写Python" → "Python"）。
3. experience 里的 months 是实习月数，找不到就填 0。
4. **educations 必须逐条完整照抄学校名、专业、学历层次、起止时间**，并保留原顺序。
   简历里写了学校/专业/时间就绝不能省略 —— "上海大学 | 人工智能（硕士研究生）|
   2025.09-至今" 必须录成 school="上海大学"、major="人工智能"、
   degree="硕士研究生"、start="2025.09"、end="至今"；多条教育经历就多项。
5. **时间范围必须原样保留**：experience / projects 的 start（开始）与 end（结束）
   一律来自简历原文（"2025 年 10 月 - 2026 年 1 月" → start="2025.10"、
   end="2026.01"），统一成 YYYY.MM；还在进行中的写 "至今"。原文没写时间才留空。
6. education 只填最高学历档位（本科/硕士/博士），它和 educations 不能互相替代：
   educations 是明细，education 是档位，两个都要输出。
"""


def parse_text(text: str) -> Resume:
    """从纯文本解析简历

    ⚠️ 必须显式给 max_tokens + reasoning_effort：模型是思考模型，
    `max_tokens` 同时卡住思考（reasoning_content）与正文。走默认 1024 时
    思考就能吃掉全部额度 → content 是空串 → json.loads 报
    `Expecting value: line 1 column 1 (char 0)`，看起来像「模型没按格式输出」，
    其实输出被掐断了（实测这份 1631 字简历：1024 空正文，4096 仍空，
    8192 才出正文；配 reasoning_effort=low 后 ~500 token 就够）。
    """
    prompt = EXTRACT_PROMPT.format(resume_text=text)
    budget = limits.resume_max_tokens()
    effort = limits.resume_reasoning_effort()

    for attempt in range(1, 4):
        try:
            raw = chat([{"role": "user", "content": prompt}],
                       source="resume_parse",
                       max_tokens=budget, reasoning_effort=effort)
        except Exception as e:
            print(f"[第{attempt}次失败] {e}")
            if attempt == 3:
                raise RuntimeError(f"简历解析失败：{e}")
            continue

        # 触顶 = 输出不完整（可能正文为空）：同样的提示再重试必然同样触顶，
        # 直接报清楚，别白等 3 轮、也别把它伪装成 JSON 格式错误
        if limits.consume_truncated():
            raise RuntimeError(
                "简历解析输出被 max_tokens 截断（思考模型把额度吃完了，正文为空）。"
                "请调大 RESUME_LLM_MAX_TOKENS，或把 RESUME_LLM_REASONING_EFFORT 设为 low。")

        try:
            data = _parse_json(raw)
            data = _ensure_educations(data, text)
            return _dict_to_resume(data)
        except Exception as e:
            print(f"[第{attempt}次失败] {e}")
            if attempt == 3:
                raise RuntimeError(f"简历解析失败：{e}")

    raise RuntimeError("简历解析失败")


def parse_file(path: str) -> Resume:
    """从文件解析简历，自动识别 PDF / TXT"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(p)
    elif suffix in (".txt", ".md"):
        text = p.read_text(encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型：{suffix}")

    return parse_text(text)


def _read_pdf(path: Path) -> str:
    """读 PDF 文本"""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("请先安装 pypdf：pip install pypdf")

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _parse_json(text: str) -> dict:
    """解析 LLM 的 JSON 输出"""
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    return json.loads(text)


def _clean_educations(raw) -> list:
    """规整教育经历明细：丢掉空项，字段名统一成 school/major/degree/start/end。"""
    items = []
    for edu in (raw or []):
        if not isinstance(edu, dict):
            continue
        entry = {
            "school": str(edu.get("school") or edu.get("学校") or "").strip(),
            "major": str(edu.get("major") or edu.get("专业") or "").strip(),
            "degree": str(edu.get("degree") or edu.get("学历") or "").strip(),
            "start": str(edu.get("start") or edu.get("开始") or "").strip(),
            "end": str(edu.get("end") or edu.get("结束") or "").strip(),
        }
        if any(entry.values()):
            items.append(entry)
    return items


# --------------------------------------------------------------------------
# 教育经历确定性兜底（问题 1）
# --------------------------------------------------------------------------
# LLM 抽取是非确定性的：同一段简历文本两次解析，可能一次带 educations、
# 一次只剩 education="硕士"（实测库里的 f9bbd037 就是这么落库的），
# 结果投递包整段「教育经历」消失，而下游 _backfill_from_original 的源
# （已被解析坏的 orig）本来就是空的，兜底无从触发。
# 学校名/专业/入学时间是简历里最不能丢的客观字段 —— 这里用纯正则从原始
# 文本再捞一遍：LLM 没给出带 school 的明细时，强制回填。

_SCHOOL_RE = re.compile(r"[\u4e00-\u9fa5]{2,15}(?:大学|学院|学校|研究院|科学院)")
_DEGREE_RE = re.compile(
    r"博士研究生|博士后|博士|硕士研究生|硕士|学士|本科|大专|专科|中专|高中")
_RANGE_RE = re.compile(
    r"(\d{4}\s*[.\-/年]\s*\d{1,2})\s*[-–—~～至到]\s*"
    r"(\d{4}\s*[.\-/年]\s*\d{1,2}|至今|现在|今|[Pp]resent)")
_EDU_HEAD_RE = re.compile(r"教育")
_NEXT_SECTION_RE = re.compile(r"技能|实习|工作|项目|证书|荣誉|奖励|自我评价|校园|获奖")
_EDGE_STRIP = r"^[\s|｜/、,，.。:：()（）\-–—]+|[\s|｜/、,，.。:：()（）\-–—]+$"


def _norm_ym(text) -> str:
    """「2025 年 9 月」/「2025-09」→「2025.09」；认不出来就返回原文。"""
    match = re.search(r"(\d{4})\s*[.\-/年]\s*(\d{1,2})", str(text or ""))
    if not match:
        return str(text or "").strip()
    return f"{match.group(1)}.{int(match.group(2)):02d}"


def _edu_section_lines(text: str) -> list:
    """取「教育背景 / 教育经历」小节的行；没有小节头就用全部行。"""
    lines = [line.strip() for line in str(text or "").splitlines()]
    start = next((i for i, line in enumerate(lines) if _EDU_HEAD_RE.search(line)), None)
    if start is None:
        return lines
    picked = []
    for line in lines[start:]:
        if picked and line and _NEXT_SECTION_RE.search(line) and not _SCHOOL_RE.search(line):
            break
        picked.append(line)
    return picked or lines


def extract_educations(text: str) -> list:
    """从简历原文里确定性抽取教育明细（school/major/degree/start/end）。"""
    lines = _edu_section_lines(text)
    items = []
    for index, line in enumerate(lines):
        school_match = _SCHOOL_RE.search(line)
        if not school_match:
            continue
        # 时间可能写在下一行（「上海大学 人工智能」/「2025.09-至今」两行式）
        window = line
        if not _RANGE_RE.search(line) and index + 1 < len(lines):
            window = f"{line} {lines[index + 1]}"
        range_match = _RANGE_RE.search(window)
        degree_match = _DEGREE_RE.search(line)

        tail = line[school_match.end():]
        if range_match:
            tail = tail.replace(range_match.group(0), "")
        if degree_match:
            tail = tail.replace(degree_match.group(0), "")
        end = ""
        if range_match:
            raw_end = range_match.group(2).strip()
            end = "至今" if raw_end.lower() in ("至今", "现在", "今", "present") \
                else _norm_ym(raw_end)
        entry = {
            "school": school_match.group(0),
            "major": re.sub(_EDGE_STRIP, "", tail),
            "degree": degree_match.group(0) if degree_match else "",
            "start": _norm_ym(range_match.group(1)) if range_match else "",
            "end": end,
        }
        if entry not in items:
            items.append(entry)
    return [entry for entry in items if entry["school"]]


def _ensure_educations(data: dict, text: str) -> dict:
    """LLM 漏了 / 弄丢了教育明细 → 用原文正则结果强制回填（问题 1）。"""
    parsed = _clean_educations(data.get("educations"))
    if any(str(edu.get("school") or "").strip() for edu in parsed):
        return data
    found = extract_educations(text)
    if not found:
        return data
    print(f"[教育兜底] LLM 未返回 school，已从简历原文补回 {len(found)} 条教育经历")
    data = dict(data)
    data["educations"] = found
    if not str(data.get("education") or "").strip():
        data["education"] = found[0].get("degree") or ""
    return data


def _dict_to_resume(data: dict) -> Resume:
    educations = _clean_educations(data.get("educations"))
    education = str(data.get("education") or "").strip()
    if not education and educations:
        education = educations[0].get("degree", "")

    return Resume(
        name=data.get("name", "匿名"),
        skills=data.get("skills", []),
        experience=data.get("experience", []),
        projects=data.get("projects", []),
        education=education,
        city=data.get("city", ""),
        educations=educations,
    )


if __name__ == "__main__":
    sample = """
    张三
    邮箱：zhangsan@example.com  手机：13800000000
    现居：北京
    
    【教育背景】
    北京某大学 计算机科学与技术 本科 2022-2026
    
    【技能】
    - 编程语言：Python、Go、JavaScript
    - 框架：LangChain、FastAPI、Chroma
    - 工具：Git、Docker、Linux
    
    【实习经历】
    2025.06 - 2025.09  某创业公司  后端开发实习生
    负责 RESTful API 开发，使用 FastAPI + PostgreSQL。
    
    【项目经历】
    项目：JD 知识库问答系统
    技术：RAG、Chroma、Chainlit
    描述：基于 RAG 的岗位信息问答系统，支持混合检索和引用溯源。
    """

    resume = parse_text(sample)
    print(f"姓名：{resume.name}")
    print(f"技能：{resume.skills}")
    print(f"教育：{resume.education}")
    print(f"城市：{resume.city}")
    print(f"实习：{resume.experience}")
    print(f"项目：{resume.projects}")