from pathlib import Path
from retriever import retrieve

BASE_DIR = Path(__file__).resolve().parent.parent

questions = [
    "哪个岗位对学历要求最低？",
    "有没有远程实习岗位？",
    "会踢足球",
]

for q in questions:
    print(f"\n{'='*70}")
    print(f"问题：{q}")
    print(f"{'='*70}")
    hits = retrieve(q, top_k=10, max_distance=10.0)  # 关掉过滤，看原始距离
    if not hits:
        print("（检索完全无结果）")
        continue
    for h in hits[:5]:
        m = h["metadata"]
        print(f"  [{h['distance']:.4f}] {m['company']} | {m['title']}")
        print(f"    {h['text'][:80]}...")