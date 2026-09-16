# 岗位 JD 知识库问答

基于 RAG 的岗位信息问答系统。把 8 条 Agent/AI 应用方向实习 JD 做成可检索的知识库，输入问题就能找到相关岗位、技能要求，并显示原文出处。

## 解决什么问题

我在找 Agent 开发实习，手头有 8 条目标岗位的 JD。但：

- 想比较哪个岗位薪资最高、哪个学历要求最低，得手动翻 8 条
- 想知道哪些岗位要求 Python，要逐条读任职要求
- 想知道有没有远程实习，得看每条 JD 的标签

所以我做了这个工具：**把 JD 结构化，支持语义问答和结构化比较**。

## 功能

- **语义问答**：哪些岗位要求 Python？哪些要求 RAG 经验？
- **结构化比较**：哪个岗位薪资最高？哪个每周出勤最少？
- **拒答**：资料里没有的问题明确说"未找到"，不编造
- **引用展示**：每条回答都附原文出处，可核实
- **流式输出**：边生成边显示
- **检索步骤可视化**：能看到 Query Routing 的决策过程

## 技术栈

| 组件 | 选型 | 理由 |
|---|---|---|
| 文档加载 | 自写 `loader.py` | JD 结构简单，按「【数字】公司：」切分 |
| 切块 | 按 JD 内部标题切 | 「岗位职责」「任职要求」是天然语义边界 |
| Embedding | 火山方舟 doubao-embedding-vision | 本地 PyTorch 被 Windows 拦截，改用 API |
| 向量库 | Chroma | 自带持久化和 metadata 过滤，原型阶段最省事 |
| 关键词检索 | jieba + rank-bm25 | 补向量对精确关键词的漏召 |
| 融合算法 | RRF（Reciprocal Rank Fusion） | 不依赖分数尺度，不需调参 |
| Reranker | BAAI/bge-reranker-v2-m3 | 中文效果好，本地跑免费 |
| 生成模型 | DeepSeek-V4-Flash（火山方舟） | OpenAI 兼容，便宜 |
| UI | Chainlit | 专为 LLM 应用设计，内置对话历史和引用展示 |
| 评估 | 自建 16 题评估集 + 5 类场景 | 无现成框架适合 JD 场景 |

## 架构

```text
用户问题
  ↓
【分类路由】LLM 判断问题类型
  ├─ comparison → 结构化查询（JSON + 模板生成）
  └─ factual    → RAG
        ↓
    混合检索（BM25 + 向量）
        ↓
    RRF 融合，召回 Top 12
        ↓
    Reranker 精排，取 Top 5
        ↓
    LLM 生成回答 + 引用
        ↓
    Chainlit 展示
```

## 目录结构

```text
rag-knowledge-base/
├── app.py                    # Chainlit UI
├── chainlit.md               # Chainlit 说明页
├── data/
│   ├── jd_sample.txt         # 原始 JD 文本（8 条）
│   └── jd_structured.json    # 结构化字段（薪资/城市/学历/出勤）
├── src/
│   ├── loader.py             # 文档加载与切分
│   ├── splitter.py           # 切块（按 JD 内部标题）
│   ├── embedder.py           # Embedding（多模态 API）
│   ├── vector_store.py       # Chroma 向量库
│   ├── retriever.py          # 混合检索（BM25+向量+RRF）
│   ├── reranker.py           # Reranker 精排
│   ├── generator.py          # 调用对话模型生成
│   └── rag_pipeline.py       # 主流程 + Query Routing
├── evaluation/
│   ├── test_cases.json       # 16 个测试问题
│   ├── run_eval.py           # 评估脚本
│   └── results.md            # 评估结果记录
├── chroma_db/                # 向量库（不提交）
└── requirements.txt
```

## 快速开始

1. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 配置 `.env`：

   ```text
   ARK_API_KEY=你的key
   ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215
   ARK_CHAT_MODEL=deepseek-v4-flash-ga-260731
   ```

3. 建向量库（首次运行）：

   ```bash
   cd src
   python vector_store.py
   ```

4. 启动 Chainlit UI：

   ```bash
   cd ..
   chainlit run app.py -w
   ```

5. 浏览器打开 `http://localhost:8000`

## 评估结果

**16 个测试问题，5 类场景，整体准确率 87.5%。**

| 类型 | 数量 | 准确率 | 说明 |
|---|---|---|---|
| 事实查找 | 3 | 100% | 单条 JD 字段查询 |
| 多块合成 | 5 | 80% | 跨多条 JD 检索 |
| 对比分析 | 3 | 67% | 跨 JD 比较 + 结构化路由 |
| 资料中没有 | 3 | 100% | 拒答能力 |
| 越界无关 | 2 | 100% | 安全边界 |

**六轮评估演进**：

| 版本 | 改动 | 准确率 |
|---|---|---|
| v1 | 基线（纯向量 Top5） | 81% |
| v2 | 头部元信息标记 | 75% ↓ |
| v3 | 精确规则（只对学历） | 68.8% ↓ |
| v4 | 混合检索 Top12 | — |
| v5 | + Reranker | — |
| v6 | + 结构化路由 | **87.5%** |

详见 `evaluation/results.md`。

## 技术选型对比

### UI 框架

| 方案 | 优点 | 缺点 | 选择 |
|---|---|---|---|
| **Chainlit** | 内置对话历史、引用展示、流式输出 | 生态较小 | ✓ |
| Streamlit | 生态大，组件多 | 每次交互重跑脚本，DOM 易冲突 | |
| Gradio | 模型 Demo 简单 | 不适合 RAG 场景 | |
| FastAPI + 前端 | 生产级 | 要写前端，开发慢 | 项目三 |

**为什么选 Chainlit**：专为 LLM 应用设计，内置引用展示和流式输出，比 Streamlit 更贴 RAG 场景。

### 向量库

| 方案 | 优点 | 缺点 |
|---|---|---|
| **Chroma** | 自带持久化 + metadata 过滤 | 不支持超大规模 |
| FAISS | 性能最强 | 只做向量索引，其他要自己写 |
| Qdrant / Milvus | 生产级 | 部署复杂 |

**为什么选 Chroma**：原型阶段少写基础设施代码。规模到百万级换 Milvus。

### Embedding

| 方案 | 优点 | 缺点 |
|---|---|---|
| 本地 BGE | 免费、快 | Windows 安全策略拦截 PyTorch |
| **API（火山方舟）** | 稳定、无依赖 | 收费、有网络延迟 |

**为什么用 API**：本地方案被系统拦截，权衡后选择 API。项目三换回本地。

### 融合算法

| 方案 | 优点 | 缺点 |
|---|---|---|
| **RRF** | 不依赖分数尺度，k=60 默认值 | 忽略分数绝对值 |
| 加权求和 | 可用分数细节 | 需要归一化 + 调权重 |

**为什么选 RRF**：BM25 和向量打分标准不同，无法直接比较。RRF 只看排名，鲁棒。

## 开发中遇到的问题

### 问题 1：本地 Embedding 被 Windows 拦截

**现象**：`OSError: [WinError 4551] 应用程序控制策略已阻止此文件`，`torch\lib\shm.dll` 无法加载。

**排查**：

1. 解除文件锁定 → 无效
2. Windows 安全排除项 → 无效
3. 关闭智能应用控制 → 有风险
4. 虚拟环境重装 → 无效

**修复**：改用火山方舟 API Embedding。

**代价**：牺牲本地速度，换来稳定性。项目三换回本地或文本专用 API。

### 问题 2：多模态接口不兼容标准 Embeddings API

**现象**：`doubao-embedding-vision` 调用标准 `/embeddings` 接口报 400。

**原因**：多模态模型只能用 `/embeddings/multimodal` 端点。

**修复**：改用 `requests` 直接调用多模态端点，输入格式改为 `[{"type": "text", "text": "..."}]`。

### 问题 3：多模态接口返回结构不一致

**现象**：`TypeError: string indices must be integers`。

**原因**：多模态接口返回 `{"data": {"embedding": [...]}}`，标准接口返回 `{"data": [{"embedding": [...]}]}`。

**修复**：写 `_extract_embeddings` 兼容两种结构。

### 问题 4：切块从 8 到 28

**现象**：8 条 JD 只切出 8 个 chunk，检索不准。

**原因**：`loader.py` 用 `re.split(r"-{10,}")` 切分，JD 内部也有 `------`，把正文误切过滤。

**修复**：改为按 `【数字】公司：` 切分，保留每条 JD 完整正文。

**效果**：正文长度从 200 字恢复到 1000 字，chunk 数升到 28。

### 问题 5：头部元信息与正文冲突

**现象**：阶跃星辰头部写"学历：不限"，正文写"本科及以上"。

**原因**：头部来自招聘网站的粗筛字段，正文是 JD 原文。

**修复**：

1. 第一次方案：标记所有头部"仅供参考"→ 模型不敢用，准确率降
2. 最终方案：只对学历字段说"以正文为准"，其他字段正常使用

**收获**：修复问题要精准定位影响范围，不能一刀切。

### 问题 6：Top K 截断

**现象**：问"哪个岗位薪资最高"，阶跃星辰（正确答案）排在 6-12 名，Top K=5 截断看不到。

**修复**：两阶段检索——混合检索宽召回 Top 12，Reranker 精排 Top 5。

### 问题 7：RAG 不适合比较类问题

**现象**：比较类问题（"薪资最高"）RAG 永远答不对。

**原因**：

1. Top K 截断，看不全数据
2. 语义检索 ≠ 结构化字段提取

**修复**：Query Routing——比较类走 JSON 查询 + 模板生成，语义类走 RAG。

**面试价值**：这是理解 RAG 边界的核心案例。

### 问题 8：Chainlit 不渲染 HTML

**现象**：`<details>` 标签被当纯文本显示。

**修复**：改用 Markdown `>` 引用块，不依赖 HTML。

### 问题 9：引用文本被 Markdown 误解析

**现象**：引用里的 `====` 被解析成 H1 标题，字号突然变大。

**修复**：在 `clean_text` 里清掉 `={3,}` 和 `-{3,}`。

## 后续计划

- 换回本地 Embedding（BGE-large-zh）或文本专用 API
- 加大 Top K 到 15，解决多块合成类漏召
- Query 改写：LLM 改写子查询，支持多条件比较
- 缓存：高频查询结果缓存
- 部署：FastAPI + Chainlit 挂载，Docker 化

## 演示

（截图 + 演示视频链接）

## 文档

- `REFLECTION.md`：项目复盘与工程决策
- `evaluation/results.md`：评估结果演进