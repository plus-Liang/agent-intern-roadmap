"""
简历解析。
输入 PDF / TXT / 纯文本，输出结构化 Resume 对象。
"""
import json
import re
from pathlib import Path
from shared.llm_client import chat
from agent.tools.resume_match import Resume


EXTRACT_PROMPT = """你是简历解析专家。从下面的简历文本中提取结构化信息。

简历文本：
{resume_text}

请输出 JSON（只输出 JSON，不要其他内容）：
{{
  "name": "姓名",
  "skills": ["技能1", "技能2"],
  "experience": [
    {{"company": "公司名", "role": "岗位", "months": 数字, "description": "工作描述"}}
  ],
  "projects": [
    {{"name": "项目名", "tech": ["技术1"], "desc": "项目描述"}}
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
"""


def parse_text(text: str) -> Resume:
    """从纯文本解析简历"""
    prompt = EXTRACT_PROMPT.format(resume_text=text)

    for attempt in range(1, 4):
        try:
            raw = chat([{"role": "user", "content": prompt}])
            data = _parse_json(raw)
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


def _dict_to_resume(data: dict) -> Resume:
    return Resume(
        name=data.get("name", "匿名"),
        skills=data.get("skills", []),
        experience=data.get("experience", []),
        projects=data.get("projects", []),
        education=data.get("education", ""),
        city=data.get("city", ""),
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