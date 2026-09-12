from llm_client import chat, chat_stream
from prompts import SYSTEM_PROMPT, build_plan_prompt
from storage import save_conversation
from errors import ConfigError, APIError
from logger import setup_logger

logger = setup_logger()

def main():
    logger.info("程序启动")
    print("=== 学习/实习规划助手 ===")

    goal = input("你的目标：").strip()
    hours = input("每周可投入时间：").strip()
    background = input("当前基础：").strip()

    if not goal or not hours or not background:
        print("输入不能为空，请重新运行。")
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_prompt = build_plan_prompt(goal, hours, background)
    messages.append({"role": "user", "content": user_prompt})

    print("\n正在生成计划...\n")
    try:
        answer = print_stream(messages)
    except APIError as e:
        logger.error(f"生成初始计划失败：{e}")
        print(f"生成失败：{e}")
        return

    messages.append({"role": "assistant", "content": answer})
    print(answer)
    logger.info("初始计划生成成功")

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
            answer = print_stream(messages)
        except APIError as e:
            logger.error(f"追问失败：{e}")
            print(f"调整失败：{e}，可继续追问或输入 exit 退出")
            messages.pop()  # 把失败的那条用户消息移除，避免污染历史
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"助手：{answer}\n")

    path = save_conversation(messages, goal)
    print(f"\n对话记录已保存到：{path}")
    logger.info(f"对话保存到 {path}")

def print_stream(messages) -> str:
    full = ""
    for piece in chat_stream(messages):
        print(piece, end="", flush=True)
        full += piece
    print()
    return full

if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        print(f"配置错误：{e}")
    except KeyboardInterrupt:
        print("\n已中断")