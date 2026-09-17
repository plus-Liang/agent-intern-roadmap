# 求职助手 Agent

基于 ReAct 循环的求职全流程助手。搜索岗位、匹配简历、追踪投递，一站式完成。

## 解决什么问题

找实习时，我面临几个痛点：

- **信息分散**：BOSS、实习僧、智联各投一遍，找不到统一视图
- **匹配靠感觉**：不知道自己适合哪个岗位，投了才知道不合适
- **简历重复改**：每个岗位要手动调整简历，效率低
- **进度难追踪**：投了 50 家，哪家看过、哪家约面，全靠记忆

这个 Agent 把流程串起来：**搜索 → 匹配 → 定制简历 → 追踪进度**。

## 功能

| 功能 | 说明 |
|---|---|
| **岗位搜索** | 多平台搜索（当前 mock 实现，可扩展真实平台） |
| **岗位详情** | 获取完整 JD |
| **简历匹配** | 多维度打分 + 可解释的差距分析 |
| **简历解析** | PDF / TXT → 结构化简历 |
| **简历定制** | 根据 JD 调整简历，不改变事实 |
| **投递追踪** | 7 阶段状态机 + 事件时间线 |
| **ReAct Agent** | LLM 自主决策调用哪些工具 |
| **对话 UI** | Chainlit（对话式） |
| **Dashboard** | Streamlit（表格化管理） |

## 三层架构

```text
┌─────────────────────────────────────────┐
│  UI 层                                   │
│  ├── Chainlit（对话式，Agent 入口）       │
│  └── Streamlit Dashboard（管理界面）      │
├─────────────────────────────────────────┤
│  Agent 层                                │
│  ├── ReAct 循环（思考 → 调工具 → 观察）   │
│  └── 工具注册中心                         │
├─────────────────────────────────────────┤
│  能力层                                  │
│  ├── 岗位搜索 / 详情                      │
│  ├── 简历匹配 / 解析 / 定制               │
│  └── 投递追踪（状态机 + SQLite）          │
├─────────────────────────────────────────┤
│  基础设施层（shared/）                    │
│  └── LLM 调用 / 日志 / 异常 / 配置        │
└─────────────────────────────────────────┘
```

## 目录结构

```text
agent/
├── __init__.py
├── app.py                    # Chainlit UI（对话式）
├── react_agent.py            # ReAct 循环
├── tools_registry.py         # 工具注册中心
├── state_machine.py          # 投递状态机
├── storage.py                # SQLite 存储
├── tracker.py                # 投递追踪管理
├── data/
│   └── applications.db       # SQLite（不提交）
├── tools/
│   ├── job_search.py         # 岗位搜索
│   ├── job_detail.py         # 岗位详情
│   └── resume_match.py       # 简历匹配
└── resume/
    ├── parser.py             # 简历解析（PDF/TXT）
    └── tailor.py             # 简历定制

dashboard/
└── app.py                    # Streamlit Dashboard
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 `.env`

```text
ARK_API_KEY=你的key
ARK_CHAT_MODEL=deepseek-v4-flash-ga-260731
ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215
```

### 3. 初始化数据库

```bash
python -c "from agent import storage; storage.init_db(); print('OK')"
```

### 4. 启动 Chainlit（对话式 Agent）

```bash
chainlit run agent/app.py -w
```

浏览器打开 `http://localhost:8000`。

### 5. 启动 Dashboard（管理界面）

```bash
streamlit run dashboard/app.py
```

浏览器打开 `http://localhost:8501`。

## 核心设计

### 1. ReAct 循环

Agent 每轮输出一个 JSON 决策：

```json
// 调工具
{
  "thought": "我需要先搜索岗位",
  "action": "search_jobs",
  "action_input": {"keyword": "Agent 开发", "city": "北京"}
}

// 完成回答
{
  "thought": "信息够了",
  "final_answer": "找到 3 个岗位..."
}
```

代码解析 → 执行工具 → 把结果喂回 → 循环，直到 `final_answer`。

**关键设计**：

- **最大轮次 6**：防止死循环
- **JSON 强制输出**：比自由文本好解析
- **解析失败自动重试**：给模型一次纠正机会
- **记录中间步骤**：能看到 Agent 的"思考过程"

### 2. 投递状态机

```text
已投递 → HR已读 → 约面中 → 面试中 → 已发Offer → 已接受
  ↓        ↓        ↓         ↓         ↓
      任何阶段都可 → 被拒 / 主动放弃
```

**为什么用状态机**：

- 校验合法转换，防止数据脏
- 支持统计（简历被看率、约面率）
- 每次变更记录事件，能画时间线

### 3. 多维度简历匹配

不只给一个分数，输出 4 个维度和差距分析：

```text
总分：68/100
维度：
- 技能匹配：22/40
- 经历匹配：16/30
- 学历匹配：15/15
- 城市匹配：15/15

差距：
- 缺少深度学习/强化学习基础知识
- 实习经历与 LLM 预训练相关性较弱

亮点：
- 掌握 Python，符合 JD 基础语言要求
- 有 RAG 项目经验
```

**为什么可解释**：用户知道往哪优化，不是黑盒。

### 4. 简历定制不改变事实

Prompt 里明确禁止编造，只允许"重排、重述"。

输出包含：

- **tailored**：定制后的简历
- **changes**：改了什么
- **warnings**：哪些改不了（如"缺少多智能体经验"）

避免"AI 帮你造假"的风险。

## 技术选型对比

### UI 框架

| 方案 | 对话 | 表格 | 选择 |
|---|---|---|---|
| **Chainlit** | ★★★★★ | ★★ | 对话式 Agent |
| **Streamlit** | ★★★ | ★★★★★ | Dashboard |
| Gradio | ★★★ | ★★ | |
| FastAPI + React | ★★★★★ | ★★★★★ | 学习成本高 |

**结论**：Chainlit + Streamlit 互补，覆盖两种交互场景。

### Agent 框架

| 方案 | 优点 | 缺点 |
|---|---|---|
| **手写 ReAct** | 理解原理，可控 | 代码多 |
| LangChain Agent | 快速搭建 | 黑盒，难调试 |
| AutoGPT | 自主性强 | 不稳定，成本高 |

**为什么手写**：面试重点考"你理解 Agent 原理吗"。手写一遍，每个环节都能讲清楚。

### 存储

| 方案 | 并发 | 查询 | 选择 |
|---|---|---|---|
| **SQLite** | 支持 | 索引、条件查询 | ✓ |
| JSON 文件 | 不支持 | 全量加载 | |
| PostgreSQL | 强 | 强 | 过度设计 |

**为什么 SQLite**：投递记录几百条，SQLite 够用且支持 SQL 查询。

## 开发中遇到的问题

### 问题 1：LLM 输出的 JSON 解析失败

**现象**：`json.loads` 报 `Expecting ',' delimiter`。

**原因**：模型在长字符串里用了真实换行，JSON 要求写成 `\n`。

**修复**：

1. Prompt 里明确"字符串里的换行用 `\n` 转义"
2. 解析加兜底：标准 JSON 失败时用 `json5` 容错

**效果**：从"偶尔多走一轮"变成"稳定 2 轮完成"。

### 问题 2：`job_marks` 表未建

**现象**：Dashboard 报 `sqlite3.OperationalError: no such table: job_marks`。

**原因**：`init_db()` 里没有调用 `init_marks_table()`。

**修复**：在 `init_db()` 末尾加调用，删旧数据库重建。

### 问题 3：状态机非法转换

**现象**：直接调用 `update_status` 会把"被拒"改成"约面"。

**修复**：写入前先 `validate_transition(from_status, to_status)`，非法就抛异常。

### 问题 4：简历匹配数据格式

**现象**：`match_resume` 工具传的 `resume_json` 在 Agent 里是字符串，有时是对象。

**修复**：工具里加判断，`isinstance(resume_json, str)` 时先 `json.loads`。

## 评估与验证

### 手动测试用例

| 场景 | 预期 |
|---|---|
| "帮我找北京的 Agent 实习" | 调 `search_jobs`，返回岗位列表 |
| "看看第二个岗位详情" | 调 `get_job_detail` |
| "帮我做简历匹配" | 调 `match_resume`，返回分数 |
| "把这个岗位加到追踪" | 调 `add_tracking` |
| "我投了哪些" | 调 `list_tracking` |

### Agent 执行轮数

- 简单查询（搜索）：2 轮
- 复杂任务（搜索 + 匹配 + 追踪）：4 轮
- 最多轮次限制：6

## 后续计划

- **真实平台抓取**：接实习僧、BOSS 直聘
- **邮件自动追踪**：解析邮件，自动更新状态
- **简历生成 .docx**：从 JSON 渲染成 Word
- **FastAPI 部署**：REST API + Chainlit 挂载
- **Docker 化**：一条命令启动

## 文档

- `REFLECTION.md`：项目复盘与工程决策