from retriever import retrieve, format_context
from generator import generate


def answer(question: str, top_k: int = 5) -> dict:
    """完整 RAG 流程：检索 → 拼 Prompt → 生成"""
    hits = retrieve(question, top_k=top_k)
    context = format_context(hits)
    answer_text = generate(question, context)

    return {
        "question": question,
        "answer": answer_text,
        "hits": hits,
    }


if __name__ == "__main__":
    questions = [
        "哪些岗位要求 Python？",
        "有没有测试相关的岗位？",
        "哪个岗位对学历要求最低？",
        "有没有远程实习岗位？",
        "会踢足球",
    ]

    for q in questions:
        print(f"\n{'='*70}")
        print(f"❓ {q}")
        print(f"{'='*70}")
        result = answer(q)
        print(result["answer"])
        if result["hits"]:
            sources = [f"{h['metadata']['company']}" for h in result["hits"]]
            print(f"\n📎 检索到的来源：{', '.join(sources)}")