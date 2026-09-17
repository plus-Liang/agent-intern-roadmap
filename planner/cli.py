from shared.llm_client import chat
from shared.errors import ConfigError, APIError
from shared.logger import setup_logger
from shared.storage import save_markdown
from shared.config import PLANS_DIR
from planner.prompts import build_system_prompt, build_plan_prompt

logger = setup_logger()


def main():
    logger.info("规划助手启动")
    print("=== 学习/实习规划助手 ===")

    goal = input("你的目标：").strip()
    hours = input("每周可投入时间：").strip()
    background = input("当前基础：").strip()

    if not goal or not hours or not background:
        print("输入不能为空")
        return

    messages = [
        {"role": "system", "content": build_system_prompt("diff")},
        {"role": "user", "content": build_plan_prompt(goal, hours, background)},
    ]

    print("\n正在生成计划...\n")
    try:
        answer = chat(messages)
    except (ConfigError, APIError) as e:
        print(f"生成失败：{e}")
        logger.error(f"生成失败：{e}")
        return

    print(answer)
    messages.append({"role": "assistant", "content": answer})

    # 追问循环
    print("\n--- 可以继续追问调整计划，输入 exit 退出 ---\n")
    while True:
        follow_up = input("你：").strip()
        if follow_up.lower() == "exit":
            break
        if not follow_up:
            continue

        messages.append({"role": "user", "content": follow_up})
        print("\n正在调整...\n")
        try:
            answer = chat(messages)
        except APIError as e:
            print(f"调整失败：{e}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"助手：{answer}\n")

    # 保存
    content = "\n\n".join(
        f"## {m['role']}\n\n{m['content']}"
        for m in messages if m["role"] != "system"
    )
    path = save_markdown(content, PLANS_DIR, prefix="plan")
    print(f"\n对话记录已保存到：{path}")
    logger.info(f"保存到 {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断")