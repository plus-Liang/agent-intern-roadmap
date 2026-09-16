import json
import sys
from pathlib import Path

# 让脚本能找到 src 里的模块
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from rag_pipeline import answer  # noqa: E402


def load_cases(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_eval():
    cases = load_cases(BASE_DIR / "evaluation" / "test_cases.json")

    results = []
    for case in cases:
        print(f"\n{'='*70}")
        print(f"[{case['id']}] {case['type']} | {case['question']}")
        print(f"预期：{case['expected']}")
        print(f"{'-'*70}")

        try:
            result = answer(case["question"])
            print(f"实际：{result['answer'][:400]}")
            if result.get("hits"):
                sources = [h["metadata"]["company"] for h in result["hits"]]
                print(f"检索来源：{', '.join(sources)}")
        except Exception as e:
            print(f"[调用失败] {e}")
            results.append({
                "id": case["id"],
                "type": case["type"],
                "question": case["question"],
                "verdict": "error",
            })
            continue

        verdict = input("\n是否正确？(y/n/skip): ").strip().lower()
        results.append({
            "id": case["id"],
            "type": case["type"],
            "question": case["question"],
            "verdict": verdict,
        })

    # ... 统计部分保持不变

    # 按类型统计
    print(f"\n\n{'='*70}")
    print("评估结果统计")
    print(f"{'='*70}")

    type_stats = {}
    for r in results:
        t = r["type"]
        if t not in type_stats:
            type_stats[t] = {"correct": 0, "total": 0}
        type_stats[t]["total"] += 1
        if r["verdict"] == "y":
            type_stats[t]["correct"] += 1

    total_correct = 0
    total = 0
    for t, s in type_stats.items():
        rate = s["correct"] / s["total"] * 100
        print(f"{t:<12} {s['correct']}/{s['total']} = {rate:.0f}%")
        total_correct += s["correct"]
        total += s["total"]

    print(f"\n整体准确率：{total_correct}/{total} = {total_correct/total*100:.1f}%")

    # 保存结果
    out = BASE_DIR / "evaluation" / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到：{out}")


if __name__ == "__main__":
    run_eval()