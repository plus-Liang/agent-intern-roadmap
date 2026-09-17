from shared.llm_client import chat

SYSTEM_PROMPT = """你是一个岗位 JD 知识库问答助手。
你只能根据提供的检索片段回答问题，不能编造。

规则：
1. 先判断检索片段是否与问题相关。
2. 如果片段与问题完全无关（例如问足球、天气等和岗位无关的内容），
   回答"资料中未找到相关信息"，不要强行编造。
3. 如果片段部分相关，只用相关部分回答，并注明来源。
4. 每条结论都要注明来源岗位名和原文片段。
5. 对比、统计类问题，逐条列出出处。

【字段使用规则】
- 薪资、城市、出勤、岗位标签、级别类型等字段：正常使用头部元信息。
- 学历字段：如果头部"学历"与正文"任职要求"中的学历要求不一致，
  以【正文任职要求】为准。如果正文中没有明确学历要求，
  则头部"学历"字段仍可作为参考。

输出格式：
结论：
依据（岗位名 + 原文）：
不确定性：
"""


def build_prompt(question: str, context: str) -> str:
    return f"""以下是检索到的岗位 JD 片段：

{context}

---

用户问题：{question}

请根据上述片段回答。"""


def generate(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, context)},
    ]
    return chat(messages)


if __name__ == "__main__":
    test_context = """[片段1] 来源：斯伦贝谢 | AI 实习生 | 北京
【任职要求】
2、熟练使用 Python，具备扎实的实操编码与项目开发能力；
"""
    print(generate("这个岗位要求 Python 吗？", test_context))