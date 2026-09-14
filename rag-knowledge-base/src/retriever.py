from vector_store import search


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """检索相关 chunk，不过滤，交给模型判断相关性"""
    return search(query, top_k=top_k)


def format_context(hits: list[dict]) -> str:
    """把检索结果格式化成可以塞进 Prompt 的文本"""
    if not hits:
        return "（未检索到相关内容）"

    lines = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        lines.append(
            f"[片段{i}] 来源：{m['company']} | {m['title']} | {m['city']}\n"
            f"{h['text']}"
        )
    return "\n\n---\n\n".join(lines)


if __name__ == "__main__":
    for q in ["哪些岗位要求 Python？", "有没有测试相关的岗位？", "会踢足球"]:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        hits = retrieve(q, top_k=3)
        for h in hits:
            m = h["metadata"]
            print(f"[{h['distance']:.4f}] {m['company']} | {m['title']}")