import json
import time
from shared.config import DATA_DIR
from shared.llm_client import chat
from rag.retriever import retrieve, format_context
from rag.reranker import rerank
from rag.generator import generate
import os

USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"


    

EDUCATION_ORDER = {"不限": 0, "本科": 2, "硕士": 3, "博士": 4}

FIELD_TEMPLATES = {
    "salary_max": "{company} 薪资最高，{salary_min}-{salary_max}/天",
    "salary_min": "{company} 薪资最低，{salary_min}-{salary_max}/天",
    "days_per_week": "{company} 每周出勤要求最少，{days_per_week}天/周",
    "education": "{company} 学历要求最低，为「{education}」",
}

STRUCTURED_PATH = DATA_DIR / "jd_structured.json"

# 结构化数据不可用时的统一提示：comparison 路由降级走 RAG
STRUCTURED_UNAVAILABLE_MSG = "结构化数据不可用，降级走 RAG"

# 结构化数据能读到、但没有任何一条记录凑齐所需字段时的提示
STRUCTURED_MISSING_FIELD_MSG = "结构化数据缺少必要字段，降级走 RAG"

# 每个比较字段算出答案所需的最小字段集：缺任一项都无法安全比较或成文。
# 不在表里的 field 退化成 ("company", field)。
REQUIRED_FIELDS = {
    "salary_max": ("company", "salary_min", "salary_max"),
    "salary_min": ("company", "salary_min", "salary_max"),
    "days_per_week": ("company", "days_per_week"),
    "education": ("company", "education"),
}


def _has_fields(record, required: tuple[str, ...]) -> bool:
    """记录是 dict，且所需字段都存在、值不为 None，才算可用。"""
    return isinstance(record, dict) and all(
        record.get(name) is not None for name in required
    )


def load_structured() -> list[dict]:
    """读取结构化数据；读不到（文件不存在/内容损坏）时返回 []，不抛异常。

    结构化数据是 comparison 路由的加速路径，属于可选数据：
    缺失时由调用方降级走 RAG，不能因为读不到而让整条链路崩溃。
    """
    try:
        data = json.loads(STRUCTURED_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[{STRUCTURED_UNAVAILABLE_MSG}] 读取 {STRUCTURED_PATH} 失败：{e}")
        return []
    if isinstance(data, dict):
        return [data]
    return data if isinstance(data, list) else []


def _llm_json(prompt: str, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            content = chat([{"role": "user", "content": prompt}])
            content = content.strip()
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
                time.sleep(attempt * 2)
    raise last_err


def classify_question(question: str) -> dict:
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
  - 薪资相关：
  - "最高" → field="salary_max"
  - "最低" → field="salary_max"（也用上限比较，因为下限都是200区分不出）
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


def _rag_answer(question: str, top_k: int, route: str) -> dict:
    # 宽召回：有 reranker 时召回 12，否则直接取 top_k
    candidates = retrieve(question, top_k=12 if USE_RERANKER else top_k)

    if USE_RERANKER:
        from rag.reranker import rerank  # 延迟导入，避免云端 ImportError
        hits = rerank(question, candidates, top_k=top_k)
    else:
        hits = candidates[:top_k]

    context = format_context(hits)
    answer_text = generate(question, context)
    return {
        "question": question,
        "answer": answer_text,
        "hits": hits,
        "route": route,
    }


def handle_comparison(question: str, field: str, direction: str) -> dict:
    data = load_structured()

    if not data:
        # 文件缺失 / 内容是空数组：都算结构化数据不可用。
        # 不猜答案，降级走 RAG，让生成层基于检索片段回答
        print(f"[{STRUCTURED_UNAVAILABLE_MSG}] 问题：{question}")
        return _rag_answer(question, top_k=5, route="rag_fallback")

    # 缺字段的记录直接跳过，避免后面 max/min 取键时 KeyError 崩溃
    required = REQUIRED_FIELDS.get(field, ("company", field))
    valid = [d for d in data if _has_fields(d, required)]

    if not valid:
        # 能读到数据，但没有一条凑齐字段：同样不猜，降级走 RAG
        print(
            f"[{STRUCTURED_MISSING_FIELD_MSG}] 问题：{question}；"
            f"需要字段 {'/'.join(required)}，"
            f"{len(data)} 条记录中没有一条字段齐全"
        )
        return _rag_answer(question, top_k=5, route="rag_fallback")

    if field == "education":
        for d in valid:
            d["_edu_order"] = EDUCATION_ORDER.get(d["education"], 99)
        key = "_edu_order"
    else:
        key = field

    if direction == "max":
        best = max(valid, key=lambda x: x[key])
        if field == "salary_max":
            answer_text = f"{best['company']} 薪资最高，{best['salary_min']}-{best['salary_max']}/天"
        elif field == "days_per_week":
            answer_text = f"{best['company']} 每周出勤要求最少，{best['days_per_week']}天/周"
        elif field == "education":
            answer_text = f"{best['company']} 学历要求最高，为「{best['education']}」"
        else:
            answer_text = f"{best['company']}"
    else:
        best = min(valid, key=lambda x: x[key])
        if field == "salary_max":
            answer_text = f"{best['company']} 薪资最低，{best['salary_min']}-{best['salary_max']}/天"
        elif field == "days_per_week":
            answer_text = f"{best['company']} 每周出勤要求最多，{best['days_per_week']}天/周"
        elif field == "education":
            answer_text = f"{best['company']} 学历要求最低，为「{best['education']}」"
        else:
            answer_text = f"{best['company']}"

    return {
        "question": question,
        "answer": answer_text,
        "hits": [],
        "route": "structured",
    }


def answer(question: str, top_k: int = 5) -> dict:
    intent = classify_question(question)

    if intent.get("type") == "comparison" and intent.get("field"):
        return handle_comparison(question, intent["field"], intent["direction"])
    
    return _rag_answer(question, top_k=top_k, route="rag")


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