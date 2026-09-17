import sys
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import chainlit as cl
from rag.rag_pipeline import answer


def clean_text(text: str, max_len: int = 600) -> str:
    text = re.sub(r"={3,}", "", text)
    text = re.sub(r"-{3,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len] + " ..."
    return text


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="你好！我是岗位 JD 知识库助手。\n\n"
                "可以问我：\n"
                "- 哪些岗位要求 Python？\n"
                "- 哪个岗位薪资最高？\n"
                "- 有没有远程实习岗位？"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content

    try:
        async with cl.Step(name="检索与生成", type="tool") as step:
            step.output = "正在检索知识库..."
            result = answer(question)
            step.output = f"路由：{result.get('route', 'unknown')}"
    except Exception as e:
        await cl.Message(
            content=f"❌ 调用失败：{e}\n\n可能是网络波动，请重新发送。"
        ).send()
        return

    msg = cl.Message(content="")
    try:
        for token in result["answer"]:
            await msg.stream_token(token)
        await msg.send()
    except Exception as e:
        await msg.update()
        await cl.Message(content=f"⚠️ 输出中断：{e}").send()
        return

    if result.get("hits"):
        lines = ["### 📎 引用来源\n"]
        for i, h in enumerate(result["hits"], 1):
            m = h["metadata"]
            cleaned = clean_text(h["text"])
            lines.append(f"**{i}. {m['company']} | {m['title']} | {m['city']}**\n")
            for line in cleaned.split("\n"):
                lines.append(f"> {line}")
            lines.append("\n---\n")
        await cl.Message(content="\n".join(lines)).send()