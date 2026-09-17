"""
求职助手 Agent - Chainlit UI
"""
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import chainlit as cl
from agent import storage
from agent.react_agent import run as run_agent


storage.init_db()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("resume", None)
    await cl.Message(
        content=(
            "👋 我是你的求职助手 Agent。\n\n"
            "**我能做的事**：\n"
            "- 搜索实习岗位（如：帮我找北京的 Agent 实习）\n"
            "- 查看岗位详情\n"
            "- 简历匹配打分\n"
            "- 添加到投递追踪\n"
            "- 查询追踪状态\n\n"
            "**使用建议**：\n"
            "1. 先用 `/resume` 设置你的简历（或粘贴文本）\n"
            "2. 然后直接说需求，我会自动调工具\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    content = message.content.strip()

    # 命令：设置简历
    if content.startswith("/resume"):
        resume_text = content.replace("/resume", "").strip()
        if not resume_text:
            await cl.Message(
                content="用法：`/resume 你的简历文本...`\n\n简历会保存在当前会话。"
            ).send()
            return

        from agent.resume.parser import parse_text
        try:
            resume = parse_text(resume_text)
            cl.user_session.set("resume", {
                "name": resume.name,
                "skills": resume.skills,
                "experience": resume.experience,
                "projects": resume.projects,
                "education": resume.education,
                "city": resume.city,
            })
            await cl.Message(
                content=f"✅ 简历已设置\n\n"
                        f"- 姓名：{resume.name}\n"
                        f"- 技能：{', '.join(resume.skills[:8])}\n"
                        f"- 教育：{resume.education}\n"
                        f"- 城市：{resume.city}"
            ).send()
        except Exception as e:
            await cl.Message(content=f"❌ 简历解析失败：{e}").send()
        return

    # 命令：查看追踪
    if content == "/track":
        apps = storage.list_applications()
        if not apps:
            await cl.Message(content="📋 暂无投递记录").send()
            return
        lines = ["## 📋 投递追踪\n"]
        for a in apps:
            lines.append(f"- **{a['company']} | {a['title']}**  `{a['status']}`")
        await cl.Message(content="\n".join(lines)).send()
        return

    # 正常对话：走 Agent
    resume = cl.user_session.get("resume")

    async with cl.Step(name="Agent 工作中", type="tool") as step:
        step.output = "正在分析..."
        try:
            result = run_agent(content, resume_data=resume, verbose=False)
        except Exception as e:
            await cl.Message(content=f"❌ 出错了：{e}").send()
            return

    # 展示步骤
    if result.get("steps"):
        step_log = "**Agent 执行过程：**\n\n"
        for s in result["steps"]:
            if s["type"] == "action":
                step_log += (
                    f"**第 {s['turn']} 轮**\n"
                    f"- 💭 {s['thought']}\n"
                    f"- 🔧 `{s['action']}({json.dumps(s['action_input'], ensure_ascii=False)[:100]})`\n\n"
                )
        step.output = step_log

    # 流式输出最终回答
    msg = cl.Message(content="")
    for token in result["answer"]:
        await msg.stream_token(token)
    await msg.send()