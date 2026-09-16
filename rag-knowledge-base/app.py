import sys
from pathlib import Path

# 让 app.py 能找到 src 里的模块
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

import chainlit as cl
from rag_pipeline import answer

import re


def clean_text(text: str, max_len: int = 600) -> str:
    """清理引用文本，避免 Markdown 误解析"""
    # 去掉长串的分隔符（= 和 -），它们会被当成标题
    text = re.sub(r"={3,}", "", text)
    text = re.sub(r"-{3,}", "", text)
    # 合并多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    # 限制长度
    if len(text) > max_len:
        text = text[:max_len] + " ..."
    return text

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

    # 流式输出
    msg = cl.Message(content="")
    try:
        for token in result["answer"]:
            await msg.stream_token(token)
        await msg.send()
    except Exception as e:
        await msg.update()
        await cl.Message(content=f"⚠️ 输出中断：{e}").send()
        return

    # 引用来源：主对话内 Markdown 折叠
        # 引用来源
       # 引用来源
    if result.get("hits"):
        lines = ["### 📎 引用来源\n"]
        for i, h in enumerate(result["hits"], 1):
            m = h["metadata"]
            cleaned = clean_text(h["text"])
            lines.append(f"**{i}. {m['company']} | {m['title']} | {m['city']}**\n")
            # 用引用块展示，每行加 >
            for line in cleaned.split("\n"):
                lines.append(f"> {line}")
            lines.append("\n---\n")
        await cl.Message(content="\n".join(lines)).send()