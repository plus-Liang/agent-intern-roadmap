# 学习/实习规划助手

## 解决什么问题
输入目标、可投入时间、当前基础，生成可执行的周计划和每日任务。

## 功能
- 命令行输入目标、时间、基础
- 调用大模型生成 4-6 周计划
- 保存为本地 Markdown 文件

## 技术栈
- Python
- OpenAI SDK（兼容 DeepSeek API）
- python-dotenv

## 如何运行
1. 安装依赖：`pip install openai python-dotenv`
2. 复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`
3. 运行：`python main.py`

## 目录结构
- `main.py`：主流程
- `llm_client.py`：封装模型 API 调用
- `prompts.py`：Prompt 模板
- `storage.py`：保存计划到文件

## 后续计划
- 支持多轮对话与计划调整
- 加 FastAPI 或 Streamlit 界面
- 加异常处理与日志