#!/usr/bin/env python
"""
Agent（ReAct）评估脚本。

用法：
    python agent/evaluation/run_agent_eval.py                  # 跑全部 10 个任务
    python agent/evaluation/run_agent_eval.py --ids 1,6,9      # 只跑指定任务（试跑/调试）
    python agent/evaluation/run_agent_eval.py --verbose        # 打印 Agent 每轮的 thought/action
    python agent/evaluation/run_agent_eval.py --no-seed        # 不在临时库里预置投递记录
    python agent/evaluation/run_agent_eval.py --keep-db        # 跑完保留 test.db（默认删除）

流程：
    1. 把 APP_DB_PATH 指向 agent/evaluation/test.db，隔离真实库
    2. 逐个任务调用 agent.react_agent.run()，从 result["steps"] 提取工具调用序列与轮次
    3. 每个任务结束后人工输入 y/n/s 判定
    4. 打印分类汇总表，结果写入 agent/evaluation/agent_results.json
    5. 删除 test.db

注意（顺序很重要）：
    APP_DB_PATH 必须在 import agent.storage 之前设置 —— agent/storage.py 在模块导入时
    就把该环境变量固化成 DB_PATH（见 agent/storage.py 第 16-19 行），之后再改不生效。
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent.parent          # 仓库根目录
TEST_DB = EVAL_DIR / "test.db"
TASKS_PATH = EVAL_DIR / "test_tasks.json"
OUT_PATH = EVAL_DIR / "agent_results.json"

# --------------------------------------------------------------------------
# 必须在 import agent.storage 之前设置。用绝对路径，避免受当前工作目录影响。
# --------------------------------------------------------------------------
os.environ["APP_DB_PATH"] = str(TEST_DB)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent import storage                          # noqa: E402
from agent.react_agent import run as run_agent     # noqa: E402


# 预置到临时库的投递记录：让 list_tracking / 聚合类任务有数据可查。
# 空库会让第 7、8 题退化成“答没有记录就行”，失去区分度。
SEED_APPLICATIONS = [
    ("阶跃星辰", "Agent 开发实习生"),
    ("腾讯", "大模型算法实习生"),
]

# match_resume 需要一份简历：react_agent 只在传入 resume_data 时，
# 才会把 action_input 里的 "current" 替换成真实简历（见 react_agent.py 第 121-123 行）。
DEFAULT_RESUME = {
    "name": "张三",
    "skills": ["Python", "PyTorch", "RAG", "LangChain", "SQL"],
    "experience": ["2025.03-至今 某科技公司 算法实习生：参与 RAG 问答系统开发与评测"],
    "projects": ["基于 ReAct 的求职助手 Agent（工具调用 + 评估集）"],
    "education": "本科",
    "city": "北京",
}


# --------------------------------------------------------------------------
# 临时库管理
# --------------------------------------------------------------------------

def cleanup_db():
    """删除临时库文件（含 SQLite 可能产生的 -wal/-shm/-journal 边车文件）"""
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = Path(str(TEST_DB) + suffix)
        if path.exists():
            path.unlink()


def prepare_db(seed: bool = True) -> list:
    """建临时库并可选预置记录，返回实际预置的记录列表"""
    cleanup_db()
    storage.init_db()

    seeded = []
    if seed:
        for company, title in SEED_APPLICATIONS:
            app_id = storage.create_application(company, title, "mock", "")
            seeded.append({"id": app_id, "company": company, "title": title})
    return seeded


# --------------------------------------------------------------------------
# 结果提取
# --------------------------------------------------------------------------

def extract_actual_tools(steps: list) -> list:
    """从 react_agent.run() 的 steps 里提取实际调用的工具序列"""
    return [s.get("action") for s in steps if s.get("type") == "action"]


def count_turns(steps: list) -> int:
    """实际消耗的 LLM 轮次：取 steps 里最大的 turn（含产出 final_answer 的那轮）"""
    turns = [s.get("turn", 0) for s in steps]
    return max(turns) if turns else 0


def tools_matched(task: dict, actual_tools: list) -> bool:
    """工具序列是否与预期完全一致（顺序敏感）"""
    return list(task.get("expected_tools", [])) == list(actual_tools)


def _ask_verdict() -> tuple:
    """人工判定，返回 (verdict, reason)"""
    while True:
        try:
            raw = input("判定 (y=成功 / n=失败 / s=跳过): ").strip().lower()
        except EOFError:
            print()
            return "skip", "无人工判定输入（stdin 已结束）"

        if raw in ("y", "yes"):
            return "y", ""
        if raw in ("n", "no"):
            try:
                reason = input("失败原因（可留空直接回车）: ").strip()
            except EOFError:
                reason = ""
            return "n", reason or "人工判定失败"
        if raw in ("s", "skip", ""):
            return "skip", ""
        print("请输入 y / n / s")


def _auto_verdict(task: dict, actual_tools: list) -> tuple:
    """自动判定（仅用于无人值守试跑）：按工具序列是否一致判定"""
    if tools_matched(task, actual_tools):
        return "y", "工具序列一致（自动判定）"
    return "n", (
        "工具序列不一致：预期 {}，实际 {}（自动判定）".format(
            task.get("expected_tools"), actual_tools
        )
    )


# --------------------------------------------------------------------------
# 主评估逻辑
# --------------------------------------------------------------------------

def evaluate_agent(tasks: list, verbose: bool = False, auto_judge: bool = False) -> dict:
    """跑每个任务，记录实际工具序列 / 轮次 / 成功与否 / 失败原因，返回统计"""
    details = []

    for index, task in enumerate(tasks, 1):
        expected_tools = list(task.get("expected_tools", []))

        print(f"\n{'=' * 70}")
        print(f"[{task['id']}] ({index}/{len(tasks)}) {task['type']} | {task['task']}")
        print(f"预期工具：{expected_tools if expected_tools else '（不调用工具）'}")
        print(f"预期行为：{task['expected_behavior']}")
        print(f"验收标准：{task['acceptance']}")
        print("-" * 70)

        error = ""
        answer = ""
        actual_tools = []
        turns = 0

        try:
            result = run_agent(task["task"], resume_data=DEFAULT_RESUME, verbose=verbose)
            steps = result.get("steps", [])
            actual_tools = extract_actual_tools(steps)
            turns = count_turns(steps)
            answer = result.get("answer", "")
        except Exception as e:                      # 网络/解析/工具异常都算任务失败
            error = f"运行异常：{type(e).__name__}: {e}"

        matched = tools_matched(task, actual_tools)

        print("实际工具：{}  共 {} 次调用 / {} 轮".format(
            actual_tools if actual_tools else "（没有调用工具）",
            len(actual_tools),
            turns,
        ))
        print(f"工具序列匹配：{'✓' if matched else '✗'}")
        print(f"最终答案：{answer[:600] if answer else '（无答案）'}")

        if error:
            print(f"[错误] {error}")
            verdict, reason, judged_by = "n", error, "error"
        elif auto_judge:
            verdict, reason = _auto_verdict(task, actual_tools)
            judged_by = "auto"
            print(f"[自动判定] {reason}")
        else:
            verdict, reason = _ask_verdict()
            judged_by = "human"

        details.append({
            "id": task["id"],
            "type": task["type"],
            "task": task["task"],
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
            "tools_match": matched,
            "tool_calls": len(actual_tools),
            "turns": turns,
            "answer": answer[:600],
            "verdict": verdict,
            "success": verdict == "y",
            "reason": reason,
            "judged_by": judged_by,
        })

    return summarize(details)


def summarize(details: list) -> dict:
    """汇总统计并打印汇总表（格式对齐 evaluation/run_eval.py 的 RAG 评估）"""
    by_type = {}
    for d in details:
        bucket = by_type.setdefault(
            d["type"], {"total": 0, "success": 0, "turns": [], "tools_match": 0}
        )
        bucket["total"] += 1
        bucket["turns"].append(d["turns"])
        if d["success"]:
            bucket["success"] += 1
        if d["tools_match"]:
            bucket["tools_match"] += 1

    for bucket in by_type.values():
        turns = bucket.pop("turns")
        bucket["avg_turns"] = round(sum(turns) / len(turns), 1) if turns else 0.0
        bucket["success_rate"] = (
            round(bucket["success"] / bucket["total"], 3) if bucket["total"] else 0.0
        )

    total = len(details)
    success = sum(1 for d in details if d["success"])
    match = sum(1 for d in details if d["tools_match"])

    print(f"\n\n{'=' * 70}")
    print("Agent 评估结果统计")
    print(f"{'=' * 70}")
    for t, s in by_type.items():
        print("{} {}/{} = {:.0f}% | 平均 {} 轮 | 工具序列一致 {}/{}".format(
            f"{t:<14}", s["success"], s["total"], s["success_rate"] * 100,
            s["avg_turns"], s["tools_match"], s["total"],
        ))

    print("-" * 70)
    rate = (success / total * 100) if total else 0.0
    match_rate = (match / total * 100) if total else 0.0
    print(f"整体任务完成率：{success}/{total} = {rate:.1f}%")
    print(f"工具调用序列一致：{match}/{total} = {match_rate:.1f}%")

    skipped = [d["id"] for d in details if d["verdict"] == "skip"]
    if skipped:
        print(f"跳过未判定：{skipped}")

    return {
        "total": total,
        "success": success,
        "success_rate": round(success / total, 3) if total else 0.0,
        "by_type": by_type,
        "details": details,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_tasks(path=TASKS_PATH) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent（ReAct）评估脚本")
    parser.add_argument("--ids", help="只跑指定任务 id，逗号分隔，如 1,6,9")
    parser.add_argument("--verbose", action="store_true", help="打印 Agent 每轮 thought/action")
    parser.add_argument("--no-seed", action="store_true", help="不在临时库里预置投递记录")
    parser.add_argument("--keep-db", action="store_true", help="跑完保留 test.db（默认删除）")
    parser.add_argument("--auto-judge", action="store_true",
                        help="按工具序列自动判定，不询问人工（仅用于无人值守试跑）")
    parser.add_argument("--out", default=str(OUT_PATH), help="结果输出路径")
    parser.add_argument("--tasks", default=str(TASKS_PATH), help="任务文件路径")
    args = parser.parse_args()

    all_tasks = load_tasks(args.tasks)
    tasks = all_tasks
    if args.ids:
        try:
            wanted = {int(x) for x in args.ids.split(",") if x.strip()}
        except ValueError:
            print(f"--ids 格式不对：{args.ids}")
            return 1
        tasks = [t for t in all_tasks if t["id"] in wanted]
        if not tasks:
            print(f"没有匹配的任务 id：{args.ids}")
            return 1

    print(f"任务文件：{args.tasks}（本次跑 {len(tasks)}/{len(all_tasks)} 个）")
    print(f"临时数据库：{TEST_DB}")
    print(f"真实数据库：{BASE_DIR / 'agent' / 'data' / 'applications.db'}（不会被写入）")

    seeded = prepare_db(seed=not args.no_seed)
    if seeded:
        companies = ", ".join(item["company"] for item in seeded)
        print(f"已在临时库预置 {len(seeded)} 条投递记录：{companies}")

    try:
        stats = evaluate_agent(tasks, verbose=args.verbose, auto_judge=args.auto_judge)
    finally:
        if args.keep_db:
            print(f"\n保留临时库：{TEST_DB}")
        else:
            cleanup_db()
            print(f"\n已删除临时库：{TEST_DB}")

    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_ids": [d["id"] for d in stats["details"]],
            "partial_run": len(stats["details"]) != len(all_tasks),
            "judged_by": "auto" if args.auto_judge else "human",
            "seeded_applications": seeded,
            "db_path_used": str(TEST_DB),
        },
        **stats,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存到：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
