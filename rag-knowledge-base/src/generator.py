import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
CHAT_MODEL = os.getenv("ARK_CHAT_MODEL", "ep-m-20260906214614-2ndmb")

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("ARK_API_KEY")
        if not api_key:
            raise ValueError("未找到 ARK_API_KEY，请检查 .env 文件")
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


SYSTEM_PROMPT = """你是一个岗位 JD 知识库问答助手。
你只能根据提供的检索片段回答问题，不能编造。

规则：
1. 先判断检索片段是否与问题相关。
2. 如果片段与问题完全无关（例如问足球、天气等和岗位无关的内容），
   回答"资料中未找到相关信息"，不要强行编造。
3. 如果片段部分相关，只用相关部分回答，并注明来源。
4. 每条结论都要注明来源岗位名和原文片段。
5. 对比、统计类问题，逐条列出出处。

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
    """调用 ARK 对话模型生成回答"""
    client = _get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, context)},
    ]
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        timeout=60,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # 单独测试：不检索，直接给一段 context
    test_context = """[片段1] 来源：斯伦贝谢 | AI 实习生 | 北京
【任职要求】
2、熟练使用 Python，具备扎实的实操编码与项目开发能力；
"""
    answer = generate("这个岗位要求 Python 吗？", test_context)
    print(answer)