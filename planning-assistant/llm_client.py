import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

def chat(messages: list) -> str:
    """调用模型，返回文本回答"""
    response = client.chat.completions.create(
        model="deepseek-v4-pro-ga-260813",
        messages=messages
    )
    return response.choices[0].message.content