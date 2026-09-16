import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from retriever import retrieve, format_context
from reranker import rerank
from generator import generate

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STRUCTURED_PATH = BASE_DIR / "data" / "jd_structured.json"

# 教育程度排序（数字越大要求越高）
EDUCATION_ORDER = {"不限": 0, "本科": 2, "硕士": 3, "博士": 4}

_client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)
CHAT_MODEL = os.getenv("ARK_CHAT_MODEL", "deepseek-v4-flash-ga-260731")


# ---------- 工具函数 ----------

def load_structured() -> list[dict]:
    return json.loads(STRUCTURED_PATH.read_text(encoding="utf-8"))


def _llm_json(prompt: str, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=30,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            return json.loads(content)
        except Exception as e:
            last_err = e
            print(f"[第{attempt}次失败] {e}")
            if attempt < retries:
                import time
                time.sleep(attempt * 2)
    raise last_err


def classify_question(question: str) -> dict:
    """让 LLM 判断问题类型和比较字段"""
    prompt = f"""你是问题分类器。分析用户问题，输出 JSON。

问题：{question}

输出格式（只输出 JSON，不要其他内容）：
{{
  "type": "comparison" 或 "factual",
  "field": "salary_max" 或 "salary_min" 或 "days_per_week" 或 "education" 或 null,
  "direction": "max" 或 "min" 或 null
}}

判断规则：
- 包含"最高/最低/最多/最少/哪个更"等比较词 → type="comparison"
- field 从以下选：
  - 薪资相关 → "salary_max"（上限）或 "salary_min"（下限）
  - 出勤相关 → "days_per_week"
  - 学历相关 → "education"
- direction："最高/最多" → "max"，"最低/最少" → "min"
- 其他问题 type="factual"，field 和 direction 为 null
"""
    try:
        return _llm_json(prompt)
    except Exception as e:
        print(f"[分类失败，降级走 RAG] {e}")
        return {"type": "factual", "field": None, "direction": None}


# 各字段的回答模板
FIELD_TEMPLATES = {
    "salary_max": "{company} 薪资最高，{salary_min}-{salary_max}/天",
    "salary_min": "{company} 薪资最低，{salary_min}-{salary_max}/天",
    "days_per_week": "{company} 每周出勤要求最少，{days_per_week}天/周",
    "education": "{company} 学历要求最低，为「{education}」",
}


def handle_comparison(question: str, field: str, direction: str) -> dict:
    """比较类问题：本地计算 + 模板生成回答"""
    data = load_structured()

    # 学历字段特殊处理：转成数字比较
    if field == "education":
        for d in data:
            d["_edu_order"] = EDUCATION_ORDER.get(d["education"], 99)
        key = "_edu_order"
    else:
        key = field

    if direction == "max":
        best = max(data, key=lambda x: x[key])
    else:
        best = min(data, key=lambda x: x[key])

    # 用模板拼回答（不依赖 LLM，100% 准确）
    template = FIELD_TEMPLATES.get(field, "{company}")
    answer_text = template.format(**best)

    return {
        "question": question,
        "answer": answer_text,
        "hits": [],
        "route": "structured",
    }


# ---------- 主流程 ----------

def answer(question: str, top_k: int = 5) -> dict:
    """主入口：先分类，再路由"""
    # 1. LLM 判断问题类型
    intent = classify_question(question)

    # 2. comparison → 结构化查询
    if intent.get("type") == "comparison" and intent.get("field"):
        return handle_comparison(question, intent["field"], intent["direction"])

    # 3. factual → RAG
    candidates = retrieve(question, top_k=12)
    hits = rerank(question, candidates, top_k=top_k)
    context = format_context(hits)
    answer_text = generate(question, context)
    return {
        "question": question,
        "answer": answer_text,
        "hits": hits,
        "route": "rag",
    }


if __name__ == "__main__":
    questions = [
        "哪些岗位要求 Python？",
        "哪个岗位薪资最高？",
        "哪个岗位薪资最低？",
        "哪个岗位每周出勤最少？",
        "哪个岗位对学历要求最低？",
        "有没有远程实习岗位？",
        "会踢足球",
    ]

    for q in questions:
        print(f"\n{'='*70}")
        print(f"❓ {q}")
        print(f"{'='*70}")
        result = answer(q)
        print(f"[路由：{result['route']}]")
        print(result["answer"])
        if result["hits"]:
            sources = [h["metadata"]["company"] for h in result["hits"]]
            print(f"\n📎 检索来源：{', '.join(sources)}")