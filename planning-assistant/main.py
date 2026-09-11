from llm_client import chat
from prompts import SYSTEM_PROMPT, build_plan_prompt
from storage import save_plan

def main():
    print("=== 学习/实习规划助手 ===")
    goal = input("你的目标：")
    hours = input("每周可投入时间：")
    background = input("当前基础：")

    user_prompt = build_plan_prompt(goal, hours, background)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    print("\n正在生成计划...\n")
    answer = chat(messages)

    print(answer)

    path = save_plan(goal,answer)
    print(f"\n计划已保存到：{path}")

if __name__ == "__main__":
    main()