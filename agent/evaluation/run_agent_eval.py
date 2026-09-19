#!/usr/bin/env python
"""
Agent（ReAct）评估脚本。

用法：
    python agent/evaluation/run_agent_eval.py                  # 跑全部 10 个任务（人工判定）
    python agent/evaluation/run_agent_eval.py --ids 3,7,10a     # 只跑指定任务（试跑/调试）
    python agent/evaluation/run_agent_eval.py --verbose        # 打印 Agent 每轮的 thought/action
    python agent/evaluation/run_agent_eval.py --no-seed        # 不在临时库里预置投递记录
    python agent/evaluation/run_agent_eval.py --keep-db        # 跑完保留每个任务的临时库（默认逐个删除）
    python agent/evaluation/run_agent_eval.py --auto --ids 3,7,10a   # LLM 自动判定，全程无人值守
    python agent/evaluation/run_agent_eval.py --compare --ids 3,7,10a  # 人工 + LLM 同时判定，对比一致性
    python agent/evaluation/run_agent_eval.py --list-failures  # 汇总历史失败案例（按 task_id 归并）

判定方式（三选一）：
    --auto       用 agent/evaluation/judge.py（LLM-as-Judge）自动判定，不再要人工输入
    --compare    先人工判定、再 LLM 判定，两个结果都保存，末尾报告一致性
    --auto-judge 老的无 LLM 兜底：只看工具序列是否一致（留着兼容旧用法）

流程：
    1. **每个任务一个独立的临时库** agent/evaluation/test_{task_id}.db（见下方「数据库隔离」）
    2. 逐个任务调用 agent.react_agent.run()，从 result["steps"] 提取工具调用序列与轮次
    3. 每个任务结束后判定（人工 / LLM / 两者都做），随后删除该任务的临时库
    4. 打印分类汇总表，结果写入 agent/evaluation/agent_results.json（可用 --out 改路径）
    5. LLM 判定模式下，把失败的题追加进 agent/evaluation/failures/{timestamp}.json

数据库隔离（为什么一个任务一个库文件）：
    Round 5 全量评估里第 7、8 题被误判成「编造」——根因是 `delete_tracking` 之类
    的写操作会改到共享的 test.db，前一个任务把后一个任务该看到的预置数据删掉了，
    Agent 返回的是**真实数据**，判定员却以为它在编。修法是每个任务开始时
    重建一份**独立文件**的库（预置数据 + 清空），任务结束就删掉：
    - 文件级隔离比「同一个文件删了重建」更硬：即使有残留连接/边车文件，
      也不可能写进下一个任务要用的库；
    - 每个任务拿到的都是同样的预置数据（阶跃星辰 + 腾讯两条），与执行顺序无关；
    - 出问题时残留的 test_10a.db 还能直接打开排查，跑完即删也不占地方。

注意（顺序很重要）：
    APP_DB_PATH 必须在 import agent.storage 之前设置 —— agent/storage.py 在模块导入时
    就把该环境变量固化成 DB_PATH（见 agent/storage.py 第 16-19 行），之后再改环境变量不生效。
    因此切换每个任务的库时直接改写 storage.DB_PATH（storage._get_conn 每次都读这个全局）。
"""
import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent.parent          # 仓库根目录
TEST_DB = EVAL_DIR / "test.db"             # 旧版共享临时库（仅用于清理历史残留）
TASKS_PATH = EVAL_DIR / "test_tasks.json"
OUT_PATH = EVAL_DIR / "agent_results.json"
FAILURES_DIR = EVAL_DIR / "failures"       # 失败案例归档目录

# --------------------------------------------------------------------------
# 必须在 import agent.storage 之前设置。用绝对路径，避免受当前工作目录影响。
# 这里先指向旧版共享库，import 之后每个任务会再切到自己的 test_{id}.db。
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
# 临时库管理（一个任务一个库文件）
# --------------------------------------------------------------------------

def task_db_path(task_id) -> Path:
    """某个任务专属的临时库路径：agent/evaluation/test_{task_id}.db"""
    return EVAL_DIR / f"test_{task_id}.db"


def _db_sidecars(path) -> list:
    return [Path(str(path) + suffix) for suffix in ("", "-wal", "-shm", "-journal")]


def cleanup_db(path=TEST_DB, quiet: bool = False):
    """删除临时库文件（含 SQLite 可能产生的 -wal/-shm/-journal 边车文件）。

    Windows 上如果还有没关掉的连接，unlink 会抛 PermissionError：
    这里吞掉并给出警告，绝不因为删不掉临时文件而中断评估。
    """
    for candidate in _db_sidecars(path):
        for attempt in (1, 2):
            if not candidate.exists():
                break
            try:
                candidate.unlink()
                break
            except OSError as e:
                if attempt == 2:
                    if not quiet:
                        print(f"[警告] 临时库删除失败（可以手工删）：{candidate}（{e}）")
                else:
                    gc.collect()
                    time.sleep(0.2)


def cleanup_stale_dbs():
    """清理上一轮跑崩时残留的临时库（test_*.db / test.db），避免跨轮污染"""
    for candidate in list(EVAL_DIR.glob("test_*.db")) + [TEST_DB]:
        cleanup_db(candidate, quiet=True)


def activate_db(db_path) -> Path:
    """把 storage 的 DB_PATH 切到指定临时库。

    storage 在 import 时就从 APP_DB_PATH 固化了 DB_PATH，之后改环境变量不生效，
    所以这里直接改写模块全局（storage._get_conn 每次调用都读它）。
    agent.tools_registry / react_agent 都是通过 `from agent import storage` 用的同一个
    模块对象，因此这次改写对工具调用同样生效。
    """
    db_path = Path(db_path)
    os.environ["APP_DB_PATH"] = str(db_path)      # 给可能重新 import storage 的代码兜底
    storage.DB_PATH = db_path
    return db_path


def prepare_db(db_path=None, seed: bool = True) -> list:
    """为**一个任务**重建临时库并可选预置记录，返回实际预置的记录列表

    顺序：删掉旧文件（含边车）→ 切 DB_PATH → 建表 → 预置。
    """
    db_path = Path(db_path) if db_path else TEST_DB
    cleanup_db(db_path)
    activate_db(db_path)
    storage.init_db()

    seeded = []
    if seed:
        for company, title in SEED_APPLICATIONS:
            app_id = storage.create_application(company, title, "mock", "")
            seeded.append({"id": app_id, "company": company, "title": title})
    return seeded


def _assert_isolated(db_path):
    """断言当前真的在写临时库，绝不允许打到 agent/data/applications.db"""
    real_db = (BASE_DIR / "agent" / "data" / "applications.db").resolve()
    if Path(storage.DB_PATH).resolve() == real_db:
        raise RuntimeError(f"DB_PATH 指向了真实库，已中止：{real_db}")


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


def expected_sequences(task: dict) -> list:
    """任务可接受的工具序列：expect + 可选的 expected_tools_alternatives"""
    sequences = [list(task.get("expected_tools", []) or [])]
    for extra in task.get("expected_tools_alternatives") or []:
        if extra is not None:
            sequences.append(list(extra))
    return sequences


def tools_matched(task: dict, actual_tools: list) -> bool:
    """工具序列是否命中任意一条可接受序列（顺序敏感）

    第 10b 题那种「不调工具或只调 list_tracking 都算对」的题用
    task["expected_tools_alternatives"] 声明备选序列。
    """
    return list(actual_tools) in expected_sequences(task)


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
    """无 LLM 的自动判定（--auto-judge）：按工具序列是否一致判定"""
    if tools_matched(task, actual_tools):
        return "y", "工具序列一致（自动判定）"
    return "n", (
        "工具序列不一致：预期 {}（可接受 {}），实际 {}（自动判定）".format(
            task.get("expected_tools"), expected_sequences(task)[1:] or "无", actual_tools
        )
    )


def judge_verdict(task: dict, answer: str, steps: list) -> dict:
    """调 judge.py 做 LLM 判定；judge 自身出问题也返回兜底 dict，不中断评估"""
    try:
        from agent.evaluation.judge import judge_task
        return judge_task(task, {"answer": answer, "steps": steps})
    except Exception as e:                          # noqa: BLE001 - 判定器不可用不该拖垮评估
        message = f"judge 不可用：{type(e).__name__}: {e}"
        return {
            "success": False, "score": 0, "reason": message, "issues": [message],
            "hallucination": None, "hallucinations": [],
            "tool_score": 0, "answer_score": 0,
            "judged_by": "judge_error", "error": message,
        }


def _judge_fields(judge: dict) -> dict:
    """把 judge 结果摊平成写进结果 JSON 的字段"""
    if not judge:
        return {
            "judge_success": None,
            "judge_score": None,
            "judge_reason": None,
            "judge_issues": [],
            "judge_hallucination": None,
            "judge_hallucinations": [],
            "judge_tool_score": None,
            "judge_answer_score": None,
        }
    return {
        "judge_success": judge.get("success"),
        "judge_score": judge.get("score"),
        "judge_reason": judge.get("reason"),
        "judge_issues": judge.get("issues", []),
        "judge_hallucination": judge.get("hallucination"),
        "judge_hallucinations": judge.get("hallucinations", []),
        "judge_tool_score": judge.get("tool_score"),
        "judge_answer_score": judge.get("answer_score"),
    }


def _print_judge(judge: dict):
    print("[LLM 判定] {}　得分 {}/100（工具 {} + 答案 {}）".format(
        "成功" if judge.get("success") else "失败",
        judge.get("score"),
        judge.get("tool_score"),
        judge.get("answer_score"),
    ))
    print(f"          理由：{judge.get('reason')}")
    for issue in judge.get("issues", []) or []:
        print(f"          - {issue}")


# --------------------------------------------------------------------------
# 失败案例归档（agent/evaluation/failures/）
# --------------------------------------------------------------------------

def _failure_record(detail: dict, generated_at: str) -> dict:
    """把一条失败的结果摊平成失败档案（字段名对齐任务书给的清单）"""
    return {
        "task_id": detail["id"],
        "type": detail["type"],
        "task": detail["task"],
        "expected_tools": detail["expected_tools"],
        "actual_tools": detail["actual_tools"],
        "answer": detail["answer"],
        "judge_reason": detail.get("judge_reason"),
        "judge_issues": detail.get("judge_issues") or [],
        "judge_score": detail.get("judge_score"),
        "hallucination": detail.get("judge_hallucination"),
        "hallucinations": detail.get("judge_hallucinations") or [],
        "verdict": detail["verdict"],
        "reason": detail["reason"],
        "judged_by": detail["judged_by"],
        "turns": detail["turns"],
        "db_path": detail.get("db_path"),
        "generated_at": generated_at,
    }


def save_failures(details: list, judged_by: str = "llm_judge",
                  failures_dir=None, generated_at: str = None) -> Path | None:
    """把本轮失败的题写进 failures/{timestamp}.json（没有失败也会写一个空列表文件）。

    文件是一个 JSON 数组，每个元素含：
    task_id / task / actual_tools / answer / judge_reason / judge_issues 等字段。
    返回写入路径；失败列表为空时同样返回路径（留着可以直接看到「这一轮全过」）。
    """
    failures_dir = Path(failures_dir or FAILURES_DIR)
    generated_at = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    failures = [
        _failure_record(d, generated_at) for d in details if not d.get("success")
    ]

    failures_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = failures_dir / f"{stamp}.json"
    index = 1
    while out_path.exists():                      # 同一秒跑两次也不覆盖
        out_path = failures_dir / f"{stamp}_{index}.json"
        index += 1

    payload = {
        "generated_at": generated_at,
        "judged_by": judged_by,
        "total": len(details),
        "failed": len(failures),
        "failures": failures,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _read_failure_file(path: Path) -> list:
    """读一个失败档案文件，兼容两种写法：纯数组 / {"failures": [...]}"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(data, dict):
        data = data.get("failures") or []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def list_failures(failures_dir=None) -> dict:
    """汇总所有历史失败案例，**按 task_id 归并**。

    返回 {task_id: {...}}，每个 task_id 下：
      task / fail_count / first_seen / last_seen / reasons / issues /
      hallucinations / runs（每次失败的原始记录，按时间正序）

    用法：
        from agent.evaluation.run_agent_eval import list_failures
        for task_id, info in list_failures().items():
            print(task_id, info["fail_count"], info["task"])
    """
    failures_dir = Path(failures_dir or FAILURES_DIR)
    merged = {}

    if not failures_dir.is_dir():
        return merged

    for path in sorted(failures_dir.glob("*.json")):
        for record in _read_failure_file(path):
            task_id = str(record.get("task_id", "?"))
            bucket = merged.setdefault(task_id, {
                "task_id": task_id,
                "task": record.get("task", ""),
                "type": record.get("type", ""),
                "fail_count": 0,
                "first_seen": record.get("generated_at", ""),
                "last_seen": record.get("generated_at", ""),
                "reasons": [],
                "issues": [],
                "hallucinations": [],
                "runs": [],
            })
            seen_at = record.get("generated_at", "")
            bucket["fail_count"] += 1
            if seen_at and (not bucket["first_seen"] or seen_at < bucket["first_seen"]):
                bucket["first_seen"] = seen_at
            if seen_at and seen_at > (bucket["last_seen"] or ""):
                bucket["last_seen"] = seen_at
            if record.get("judge_reason") and record["judge_reason"] not in bucket["reasons"]:
                bucket["reasons"].append(record["judge_reason"])
            for issue in record.get("judge_issues") or []:
                if issue not in bucket["issues"]:
                    bucket["issues"].append(issue)
            for hit in record.get("hallucinations") or []:
                if hit not in bucket["hallucinations"]:
                    bucket["hallucinations"].append(hit)
            if not bucket["task"] and record.get("task"):
                bucket["task"] = record["task"]
            bucket["runs"].append({
                "file": path.name,
                "generated_at": seen_at,
                "actual_tools": record.get("actual_tools"),
                "answer": record.get("answer"),
                "judge_score": record.get("judge_score"),
                "judge_reason": record.get("judge_reason"),
                "judge_issues": record.get("judge_issues") or [],
                "hallucinations": record.get("hallucinations") or [],
            })

    return merged


def print_failures(failures_dir=None):
    """在命令行打印历史失败汇总（--list-failures）"""
    merged = list_failures(failures_dir)
    failures_dir = Path(failures_dir or FAILURES_DIR)
    print(f"失败档案目录：{failures_dir}")
    if not merged:
        print("还没有任何失败记录（或者目录里没有 *.json）。")
        return merged

    total = sum(info["fail_count"] for info in merged.values())
    print(f"历史失败 {total} 次，涉及 {len(merged)} 个任务：\n")
    for task_id, info in sorted(merged.items(), key=lambda kv: str(kv[0])):
        flag = "（含幻觉）" if info["hallucinations"] else ""
        print("[{}] 失败 {} 次{} | {}".format(task_id, info["fail_count"], flag, info["task"]))
        print(f"    最近一次：{info['last_seen']}")
        if info["reasons"]:
            print(f"    理由：{info['reasons'][0]}")
        for hit in info["hallucinations"][:2]:
            print(f"    幻觉：{hit}")
    return merged


# --------------------------------------------------------------------------
# 主评估逻辑
# --------------------------------------------------------------------------

def evaluate_agent(tasks: list, verbose: bool = False, auto_judge: bool = False,
                   use_judge: bool = False, compare: bool = False,
                   seed: bool = True, keep_db: bool = False) -> dict:
    """跑每个任务，记录实际工具序列 / 轮次 / 成功与否 / 失败原因，返回统计

    auto_judge: 老的「只看工具序列」自动判定（--auto-judge）
    use_judge:  用 LLM 判定（--auto）
    compare:    人工 + LLM 都判，对比一致性（--compare），此时以人工判定为准
    seed:       每个任务的独立库里是否预置投递记录
    keep_db:    跑完是否保留该任务的临时库（默认删除）

    每个任务开始时都会重建自己的 test_{id}.db，任务结束就删掉，
    因此任务之间不可能通过数据库互相污染。
    """
    details = []
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for index, task in enumerate(tasks, 1):
        expected_tools = list(task.get("expected_tools", []))

        print(f"\n{'=' * 70}")
        print(f"[{task['id']}] ({index}/{len(tasks)}) {task['type']} | {task['task']}")
        print(f"预期工具：{expected_tools if expected_tools else '（不调用工具）'}")
        if task.get("expected_tools_alternatives"):
            print(f"可接受的其他工具序列：{task['expected_tools_alternatives']}")
        print(f"预期行为：{task['expected_behavior']}")
        print(f"验收标准：{task['acceptance']}")

        # ---- 每个任务一份全新库：重建 + 预置，隔离前一个任务的写操作 ----
        db_path = task_db_path(task["id"])
        seeded = prepare_db(db_path, seed=seed)
        _assert_isolated(db_path)
        print("-" * 70)
        print("独立临时库：{}（预置 {} 条：{}）".format(
            db_path.name, len(seeded),
            "、".join(item["company"] for item in seeded) or "无",
        ))

        error = ""
        answer = ""
        actual_tools = []
        turns = 0
        steps = []

        try:
            result = run_agent(task["task"], resume_data=DEFAULT_RESUME, verbose=verbose)
            steps = result.get("steps", [])
            actual_tools = extract_actual_tools(steps)
            turns = count_turns(steps)
            answer = result.get("answer", "")
        except Exception as e:                      # 网络/解析/工具异常都算任务失败
            error = f"运行异常：{type(e).__name__}: {e}"
        finally:
            if keep_db:
                print(f"保留临时库：{db_path}")
            else:
                cleanup_db(db_path)

        matched = tools_matched(task, actual_tools)

        print("实际工具：{}  共 {} 次调用 / {} 轮".format(
            actual_tools if actual_tools else "（没有调用工具）",
            len(actual_tools),
            turns,
        ))
        print(f"工具序列匹配：{'✓' if matched else '✗'}")
        print(f"最终答案：{answer[:600] if answer else '（无答案）'}")

        judge = None
        auto_result = None
        human_verdict, human_reason = None, ""

        if error:
            print(f"[错误] {error}")
            verdict, reason, judged_by = "n", error, "error"
            if use_judge or compare:
                print("[LLM 判定] 运行异常、没有可判定的答案，跳过判定")
        elif compare:
            # 先人工后自动：避免先把 LLM 结论摆出来影响人的判断
            human_verdict, human_reason = _ask_verdict()
            judge = judge_verdict(task, answer, steps)
            _print_judge(judge)
            auto_result = "y" if judge.get("success") else "n"
            verdict, reason, judged_by = human_verdict, human_reason, "human"
        elif use_judge:
            judge = judge_verdict(task, answer, steps)
            _print_judge(judge)
            auto_result = "y" if judge.get("success") else "n"
            verdict = auto_result
            reason = judge.get("reason", "")
            judged_by = "llm_judge"
        elif auto_judge:
            verdict, reason = _auto_verdict(task, actual_tools)
            judged_by = "auto"
            print(f"[自动判定] {reason}")
        else:
            verdict, reason = _ask_verdict()
            judged_by = "human"

        # 一致性：只有人工和 LLM 都给出 y/n 时才算（跳过 / 运行异常不算）
        consistent = None
        if compare and human_verdict in ("y", "n") and auto_result in ("y", "n"):
            consistent = human_verdict == auto_result
            print(f"[一致性] 人工={human_verdict} LLM={auto_result} → "
                  f"{'一致' if consistent else '不一致'}")

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
            "human_verdict": human_verdict,
            "human_reason": human_reason,
            "auto_verdict": auto_result,
            "consistent": consistent,
            "db_path": str(db_path),
            **_judge_fields(judge),
        })

    stats = summarize(details)
    stats["generated_at"] = generated_at
    return stats


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

    # ---- LLM 自动判定汇总（--auto / --compare） ----
    judge_summary = None
    judged = [d for d in details if d.get("judge_success") is not None]
    if judged:
        judge_success = sum(1 for d in judged if d["judge_success"])
        scores = [d["judge_score"] for d in judged if isinstance(d.get("judge_score"), (int, float))]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
        hallucinated = [d["id"] for d in judged if d.get("judge_hallucination")]
        judge_summary = {
            "judged": len(judged),
            "success": judge_success,
            "success_rate": round(judge_success / len(judged), 3),
            "avg_score": avg_score,
            "hallucinations": len(hallucinated),
            "hallucinated_tasks": hallucinated,
        }
        print("LLM 判定：{}/{} = {:.0f}% | 平均 {:.1f}/100 分".format(
            judge_success, len(judged), judge_success / len(judged) * 100, avg_score,
        ))
        if hallucinated:
            print(f"幻觉否决：{len(hallucinated)} 个任务 {hallucinated}")

    # ---- 人工 vs LLM 一致性（--compare） ----
    compare_summary = None
    compared = [d for d in details if d.get("consistent") is not None]
    if compared:
        agree = sum(1 for d in compared if d["consistent"])
        compare_summary = {
            "compared": len(compared),
            "agree": agree,
            "disagree": len(compared) - agree,
            "agree_rate": round(agree / len(compared), 3),
            "both_pass": sum(1 for d in compared if d["human_verdict"] == "y" and d["auto_verdict"] == "y"),
            "both_fail": sum(1 for d in compared if d["human_verdict"] == "n" and d["auto_verdict"] == "n"),
            "human_pass_judge_fail": sum(1 for d in compared if d["human_verdict"] == "y" and d["auto_verdict"] == "n"),
            "judge_pass_human_fail": sum(1 for d in compared if d["human_verdict"] == "n" and d["auto_verdict"] == "y"),
        }
        print("人工 vs LLM 判定一致：{}/{} = {:.0f}%（一致 {}，不一致 {}）".format(
            agree, len(compared), compare_summary["agree_rate"] * 100,
            agree, len(compared) - agree,
        ))
        print("  都判成功 {} | 都判失败 {} | 人工成功/LLM 失败 {} | LLM 成功/人工失败 {}".format(
            compare_summary["both_pass"], compare_summary["both_fail"],
            compare_summary["human_pass_judge_fail"], compare_summary["judge_pass_human_fail"],
        ))

    skipped = [d["id"] for d in details if d["verdict"] == "skip"]
    if skipped:
        print(f"跳过未判定：{skipped}")

    return {
        "total": total,
        "success": success,
        "success_rate": round(success / total, 3) if total else 0.0,
        "by_type": by_type,
        "judge_summary": judge_summary,
        "compare_summary": compare_summary,
        "details": details,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_tasks(path=TASKS_PATH) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent（ReAct）评估脚本")
    parser.add_argument("--ids", help="只跑指定任务 id，逗号分隔，如 3,7,10a")
    parser.add_argument("--verbose", action="store_true", help="打印 Agent 每轮 thought/action")
    parser.add_argument("--no-seed", action="store_true", help="不在临时库里预置投递记录")
    parser.add_argument("--keep-db", action="store_true",
                        help="跑完保留每个任务的 test_{id}.db（默认逐个删除）")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--auto", action="store_true",
                      help="用 LLM（judge.py）自动判定，不再要人工输入")
    mode.add_argument("--compare", action="store_true",
                      help="人工 + LLM 都判定，对比一致性，两个结果都保存")
    parser.add_argument("--auto-judge", action="store_true",
                        help="按工具序列自动判定（无 LLM 的老兜底），仅用于无人值守试跑")
    parser.add_argument("--out", default=str(OUT_PATH), help="结果输出路径")
    parser.add_argument("--tasks", default=str(TASKS_PATH), help="任务文件路径")
    parser.add_argument("--failures-dir", default=str(FAILURES_DIR),
                        help="失败案例归档目录（默认 agent/evaluation/failures/）")
    parser.add_argument("--list-failures", action="store_true",
                        help="只汇总打印历史失败案例（按 task_id 归并），不跑任务")
    args = parser.parse_args()

    if args.list_failures:
        print_failures(args.failures_dir)
        return 0

    all_tasks = load_tasks(args.tasks)
    tasks = all_tasks
    if args.ids:
        wanted = {token.strip() for token in args.ids.split(",") if token.strip()}
        tasks = [t for t in all_tasks if str(t["id"]) in wanted]
        if not tasks:
            print(f"没有匹配的任务 id：{args.ids}（现有："
                  f"{', '.join(str(t['id']) for t in all_tasks)}）")
            return 1

    if args.auto:
        judged_by = "llm_judge"
    elif args.compare:
        judged_by = "compare"
    elif args.auto_judge:
        judged_by = "auto"
    else:
        judged_by = "human"

    print(f"任务文件：{args.tasks}（本次跑 {len(tasks)}/{len(all_tasks)} 个）")
    print(f"判定方式：{judged_by}")
    print(f"临时库：每个任务一个 agent/evaluation/test_{{task_id}}.db（跑完删除）")
    print(f"真实数据库：{BASE_DIR / 'agent' / 'data' / 'applications.db'}（不会被写入）")

    cleanup_stale_dbs()                            # 清掉上一轮残留，避免跨轮污染

    seed = not args.no_seed
    print(f"预置数据：{'每个任务都重建并预置 ' + str(len(SEED_APPLICATIONS)) + ' 条投递记录' if seed else '不预置（--no-seed）'}")

    try:
        stats = evaluate_agent(
            tasks,
            verbose=args.verbose,
            auto_judge=args.auto_judge,
            use_judge=args.auto,
            compare=args.compare,
            seed=seed,
            keep_db=args.keep_db,
        )
    finally:
        if not args.keep_db:
            cleanup_stale_dbs()

    # ---- 失败案例归档：LLM 判定模式下每轮都落一个文件 ----
    failures_path = None
    if args.auto or args.compare:
        try:
            failures_path = save_failures(
                stats["details"], judged_by=judged_by,
                failures_dir=args.failures_dir, generated_at=stats["generated_at"],
            )
            failed = sum(1 for d in stats["details"] if not d["success"])
            print(f"\n失败案例已归档：{failures_path}（本轮失败 {failed} 题）")
        except Exception as e:                      # noqa: BLE001 - 归档失败不该让评估失败
            print(f"\n[警告] 失败案例归档失败（忽略）：{type(e).__name__}: {e}")

    output = {
        "meta": {
            "generated_at": stats["generated_at"],
            "task_ids": [d["id"] for d in stats["details"]],
            "partial_run": len(stats["details"]) != len(all_tasks),
            "judged_by": judged_by,
            "judge_mode": "llm" if (args.auto or args.compare) else (
                "tool_sequence" if args.auto_judge else "none"
            ),
            "seeded_applications": [
                {"company": c, "title": t} for c, t in SEED_APPLICATIONS
            ] if seed else [],
            "db_isolation": "per_task_file（每个任务重建 agent/evaluation/test_{task_id}.db 并删除）",
            "db_paths_used": [d.get("db_path") for d in stats["details"]],
            "failures_file": str(failures_path) if failures_path else None,
        },
        **stats,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存到：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
