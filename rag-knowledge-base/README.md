# 岗位 JD 知识库问答

## 解决什么问题
把 8 条 Agent 实习 JD 做成可检索的知识库，输入问题就能找到相关岗位和技能要求，并显示原文出处。

## 功能（规划中）
- 加载 JD 文档
- 切块 + Embedding
- 向量检索
- 基于检索结果回答
- 显示引用来源
- 资料里没有时拒答

## 技术栈
- Python
- sentence-transformers（本地 Embedding）
- Chroma（向量库）
- ARK API（生成）
- Streamlit（界面）

## 目录结构
（待补）