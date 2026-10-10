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


def generate(question: str, context: str, source: str = None,
             max_tokens: int = None, reasoning_effort: str = None) -> str:
    """生成答案。

    三个新增的透传参数（第 3 周加；**都不传时走 ``chat()`` 的全局兜底**）：
        source:            token 用量归因（引用溯源传自己的来源标记）
        max_tokens:        单次输出上限；不传取 LLM_MAX_TOKENS（现为 4096）
        reasoning_effort:  思考档位；不传取 LLM_DEFAULT_REASONING_EFFORT（现为 low），
                           显式传空串则走全局默认 —— 本函数只在真值时才透传

    为什么需要它们：引用溯源要用同一份 prompt 生成**带依据的长答案**，
    当年的全局默认 1024 会被截断（截断的答案最后一句是半句，标不准引用，
    还必然被判无依据）。调用方（``rag.citation.generate_answer``）仍然显式
    给足额度并压低思考档位 —— 但即便它忘了传，全局兜底也已经够用。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, context)},
    ]
    explicit = {}
    if source is not None:
        explicit["source"] = source
    if max_tokens is not None:
        explicit["max_tokens"] = max_tokens
    if reasoning_effort:
        explicit["reasoning_effort"] = reasoning_effort
    return chat(messages, **explicit)


if __name__ == "__main__":
    test_context = """[片段1] 来源：斯伦贝谢 | AI 实习生 | 北京
【任职要求】
2、熟练使用 Python，具备扎实的实操编码与项目开发能力；
"""
    print(generate("这个岗位要求 Python 吗？", test_context))