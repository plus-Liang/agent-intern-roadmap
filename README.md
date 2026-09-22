---
title: Agent 求职助手
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# agent-intern-roadmap

对话式求职 Agent + REST API。

> **部署提示**：Hugging Face Spaces 只读取仓库根目录的 `README.md` 作为 Space 配置，
> 因此本文件顶部的 YAML frontmatter（`sdk: docker` / `app_port: 7860`）必须保留，
> 不要删除或下移。

## 功能
- Chainlit 对话 Agent（ReAct 循环 + 工具调用）
- REST API（岗位搜索、投递追踪、Token 统计）

## 访问
- `/` → 首页（三个入口卡片）
- `/chat` → 对话 Agent
- `/docs` → API 文档

## 数据来源
- 岗位数据：GitHub 仓库的 `rag/data/cleaned_jd.json`
- 数据库：`/tmp`（云端临时存储）

## 环境变量（Space Secrets）

在 Space 的 **Settings → Variables and secrets** 里配置，
变量名以仓库根目录的 `.env.example` 为准（例如 `OPENAI_API_KEY`）。

## 本地验证

```bash
docker build -t agent-intern-roadmap .
docker run --rm -p 7860:7860 --env-file .env agent-intern-roadmap
```

## 说明
- 镜像基于 `python:3.11-slim`，以 UID 1000 的非 root 用户运行。
- 不含 Streamlit Dashboard、Playwright、torch / Reranker。
- `app_port: 7860` 与容器内 `uvicorn main:app --host 0.0.0.0 --port 7860` 一致。
