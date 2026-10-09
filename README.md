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

> **部署提示**：Hugging Face Spaces 只读取仓库根目录的 `README.md` 作为 Space 配置，
> 因此本文件顶部的 YAML frontmatter（`sdk: docker` / `app_port: 7860`）必须保留，
> 不要删除或下移。

**一个能自己更新数据的求职 Agent**：三平台抓岗位 → 清洗入库 → 混合检索（BM25 + 向量）→ ReAct 对话，
再配一个投递追踪看板和一组 REST 接口。

![chat](docs/images/chat.png)
![dashboard](docs/images/dashboard.png)
![api](docs/images/api.png)

---

## 1. 项目简介

求职这件事可以拆成流水线：**找岗位 → 判断匹配度 → 投递 → 跟踪结果**。
这个项目把每一段都做成可运行的一层，用同一个 Agent 串起来：

| 层 | 干什么 | 主要入口 |
|---|---|---|
| 数据层 | 抓实习僧 / 牛客 / 国家大学生就业服务平台（ncss，原 24365），清洗去重后落 `cleaned_jd.json` + SQLite | `agent/scrapers/scheduler.py` |
| RAG 层 | 岗位 JD 切块 → 本地 embedding 入库 → BM25 + 向量 RRF 融合检索（模糊需求走语义重排） | `rag/retriever.py`、`rag/vector_store.py` |
| Agent 层 | ReAct 循环 + 工具调用：搜岗位 / 看详情 / 简历匹配打分 / 改简历 / 投递跟踪 / 模拟面试；也支持**直接粘贴岗位链接或 JD 文本** | `agent/react_agent.py`、`agent/tools_registry.py` |
| 界面层 | Chainlit 对话、FastAPI + Swagger、Streamlit 数据看板 | `main.py`、`dashboard/app.py` |

> **投递追踪的定位**：它是**手动录入的个人记录管理（本地 CRM）** —— 不做自动投递，
> 也不读招聘平台的状态。招聘平台没有公开 API，自动投递违反服务条款。
> 真正自动化的是投递**之前**的三步：**找岗位、算匹配、改简历**。

仓库自带的 `rag/data/cleaned_jd.json` 有 **约 960 条**真实岗位（含六城分布，随每晚抓取持续增长），clone 下来即可检索，
不需要先跑爬虫。**要跑起来只需要一个智谱 API Key**（对话模型；embedding 是本地模型，不花钱）。

**岗位不用只在系统里搜**：在 `/chat` 里可以直接粘贴**实习僧 / 牛客 / ncss** 的岗位链接
（系统按链接里的 job_id 去岗位库取那一条），也可以把**任意平台**（Boss、智联等抓不到的）
的 JD 文本整段贴进来 —— 一样能做匹配打分、生成投递包和模拟面试。

启动后四个入口：

| 路径 | 是什么 |
|---|---|
| `http://localhost:8000/` | 首页（对话 / 看板 / API 三张卡片） |
| `http://localhost:8000/chat` | Chainlit 对话 Agent |
| `http://localhost:8000/docs` | FastAPI 自动生成的 API 文档（Swagger UI，可直接在线调试） |
| `http://localhost:8501` | Streamlit Dashboard（岗位列表 / 投递追踪 / 匹配打分 / Token 成本 / 简历管理 / 对话 Agent）⚠️ **无认证，只本地用** |

---

## 2. 架构图

```mermaid
flowchart TD
    subgraph DATA["数据层 (agent/scrapers)"]
        S1["实习僧 shixiseng<br/>Playwright + Chromium"]
        S2["牛客 niuke<br/>requests 接口"]
        S3["ncss 国家大学生就业服务平台<br/>requests 接口 · 1 req/s"]
        CL["清洗 / 去重 / 时效过滤<br/>rag/quality/cleaner.py"]
        JS["cleaned_jd.json<br/>约 960 条 · 进 git"]
        DB["jobs.db (SQLite)<br/>city / platform / publish_date 索引"]
        S1 --> CL
        S2 --> CL
        S3 --> CL
        CL --> JS
        JS -->|"ensure_db() 首次自动建库"| DB
    end

    subgraph RAG["RAG 层 (rag/)"]
        SP["splitter 按岗位切块"]
        EM["fastembed bge-small-zh-v1.5<br/>本地推理，无 API Key"]
        VS["chroma_db/jd_chunks<br/>chunk 带 platform + job_id"]
        RT["retriever<br/>BM25 召回 + 向量召回 → RRF 融合"]
        DB --> SP --> EM --> VS --> RT
    end

    subgraph AGENT["Agent 层 (agent/)"]
        RA["react_agent：ReAct 循环 + 工具调用"]
        TL["工具注册表<br/>search_jobs / job_detail / resume_match<br/>tailor_resume / pdf_export / 投递跟踪"]
        LIM["四道闸门<br/>频率 · 单次预算 · 日额度 · max_tokens"]
        UC["user_context 用户隔离 + 对话历史落库"]
        RT -->|"语义命中 job_id 反查回岗位"| RA
        RA --> TL
        LIM -.-> RA
        UC -.-> RA
    end

    subgraph UI["界面层"]
        CH["Chainlit /chat"]
        API["FastAPI /api · /docs"]
        DASH["Streamlit Dashboard :8501"]
    end

    RA --> CH
    TL --> API
    DB --> DASH
```

数据流一句话：**抓取 → 清洗 → JSON（进 git）→ SQLite（自动建）→ 向量库（一条命令重建）→ 检索 → Agent → 界面**。

### LangGraph 工作流（Agent 入口 · 阶段 1）

Agent 的入口是 `agent/react_agent_lg.py`：它把**流程明确**的两类请求交给 LangGraph 图
（图定义在 `agent/langgraph_flow.py`），其余请求原样交回 `agent/react_agent.py` 的
ReAct 循环兜底 —— 所以投递记录、投递包、模拟面试这些**多步决策**的行为一个字没变。

```mermaid
flowchart LR
    Q["用户问题"] --> R{"入口路由<br/>react_agent_lg"}
    R -->|"找 / 搜 … 岗位"| S["搜岗位图<br/>固定五步"]
    R -->|"匹配 / 打分"| M["匹配 + 反思图"]
    R -->|"其余"| RA["react_agent<br/>ReAct 兜底"]
```

**一、搜岗位图**：`[接收] → [提取参数] → [搜岗位] → [筛选] → [返回列表]`
（5 个节点 / 5 条边，**无分支**，把原来「模型自己决定搜什么、搜几次、怎么排版」
换成固定流程）。关键词 / 城市 / 条数由「提取参数」节点抽取（LLM 优先、规则兜底，
模型编出来的城市会被丢弃）；「筛选」去重并**重新编号**，保证「列表第 N 行」
恒等于系统内部的 `index=N` —— 这正是「我想投第 N 个」不会投错岗的前提。

**二、匹配 + 反思图**：6 个节点 + 1 条回边。

```mermaid
flowchart LR
    A["接收"] --> B["定位岗位"] --> C["匹配打分"] --> D["反思"]
    D -->|"合理"| E["输出"]
    D -->|"不合理<br/>最多 2 次"| F["修正"] --> C
```

「反思」节点输入「简历 + 岗位 + 打分结果」，输出 `{合理, 理由, 建议修正}`，
判据分两层：**确定性覆盖检查**（只看 JD 的**关键项** —— `requirements` 段与带「熟悉 /
必须 / 精通 / 要求」措辞的句子里的技术项，简历里有没有真实证据）＋ **LLM 语义复核**。
典型场景：简历里没有任何 RAG 经验、岗位 JD 把 RAG 列为硬性要求，打分却给了 90 分 →
反思节点标记「分数可能虚高」，修正量至少把分数拉回**证据支持的警戒线**
（关键项覆盖 ≥ 50% 用 `LG_REFLECT_INFLATION_SCORE`，默认 85；覆盖不足时降到
`LG_REFLECT_INFLATION_SCORE_LOW`，默认 70），回到线下就不再判虚高 —— 所以是
**真收敛**（实测 1 次修正 / 2 轮反思收口），重试上限只是安全网。
判据「连续两次修正后分数变化 < 5 就接受当前分」由 `LG_REFLECT_CONVERGENCE_DELTA`
（默认 5）控制。

「定位岗位」的优先级：显式 `job_id` → 「第 N 个」序号提示 → **会话里正在看的岗位**
（生成投递包时写入 `current_job`，粘贴的 JD 走另一个槽位，取更近的那个）→
最近一次搜索的第 1 条。这样「投完第 10 个再说『匹配打分』」匹配的就是第 10 个。

「看日志 / 系统日志 / docker 日志」有确定性引导（不调 LLM、也不会被当成投递记录）：
直接回 `docker compose logs app --tail 50`。

想整体切回旧版：`.env` 里设 `AGENT_ENGINE=react`（或把 `agent/app.py` 的
`run_agent` 导入改回 `agent.react_agent`）。

### 多智能体协作（复杂任务 · 批次 1）

上面两张图解决的是**单意图**任务（只找岗位，或只打分）。当用户**一句话里说了多个需求**
（「找岗位 + 匹配 + 生成投递包」）时，单条链路装不下 —— 系统会切到
`agent/complex_task_flow.py` 的**复杂任务编排图**，由三个专职 Agent 协作：

| Agent | 职责 | 输入 → 输出 |
|---|---|---|
| **Planner**（规划者） | 把需求拆成有序、可执行的步骤 | 用户原话 → `{"steps":[{action,args,desc,when}]}` |
| **Executor**（执行者） | 执行计划里的**一步**（真调工具） | 单步 + 前序结果 → 该步结构化结果 |
| **Critic**（审查者） | 审查结果是否合格，**不合格就打回** | 步骤 + 结果 → `{合格, 理由, 打回建议}` |

```mermaid
flowchart LR
    U["用户需求"] --> P["Planner 出计划"] --> E["Executor 执行第 N 步"]
    E --> C["Critic 审查"]
    C -->|"合格"| N["进入下一步"]
    C -->|"不合格<br/>打回 ≤2 次"| V["修正参数"] --> E
    C -->|"不可挽回<br/>或已重做 2 次"| D["降级收尾"]
    N --> E
    N -->|"做完了"| O["汇总输出"]
    D --> O
```

**关键不是「有 3 个 Agent」，而是「审查者能打回重做」**：

1. **打回是可执行的**：Critic 的 `打回建议` 里带**参数覆盖项**（例：搜岗位 0 条 →
   `{"city": ""}` 去掉城市限制），`revise` 节点把它**真的合并进步骤参数**再回 Executor
   重跑。只说一句「再试一次」却不改参数的审查是白烧额度 —— 那才是"假多智能体"。
   建议还会**逐级放宽**（city → job_type → keyword），不是重复同一个动作。
2. **事实层是确定性的**：0 条结果 / 分数越界 / 三件套缺文件 / 工具报错一律判不合格，
   不依赖模型；事实层已判不合格时**不再调 LLM**（省一次调用）。模型只在
   「事实层通过、需要判断分寸」时投一票。
3. **有上限、有降级**：同一步最多重做 `MULTI_AGENT_MAX_RETRIES`（默认 2）次；
   用完仍不合格就**降级** —— 返回已完成的部分 + 如实说明哪一步没做成，
   **绝不把失败包装成成功**，也绝不无限重试（另有计划长度上限与全局执行次数上限两层防护）。

计划里的步骤还支持依赖与条件：`job_id` 可以写占位符 `"$step1.job_id"`（运行时才解析），
`when` 可以是 `score >= 70` 这类条件（分数不达标就**跳过**出包，而不是失败）。

**入口怎么判「复杂」**：命中 ≥2 种意图（搜索 / 匹配 / 要投递材料），或明说
「一条龙 / 整个流程」。只要求单一需求时**不套**这套流程 —— 多智能体更贵，
单需求上它就是纯加成本。开关：`MULTI_AGENT_ENABLED`（默认开）、
`MULTI_AGENT_CRITIC_LLM`（Critic 是否额外投一票）。

### 工具的并行调用

有两条链路原来是串行的，现在改成真并行（`asyncio.gather` + 常驻线程池）：

| 位置 | 原来 | 现在 |
|---|---|---|
| `rag/retriever.py` 双路召回 | BM25 跑完再跑向量 | 两路同时跑，RRF 融合等两边到齐 |
| `agent/tools_registry.py` 生成投递包 | 简历定制跑完再跑自荐信 | 两件产物并行生成（互不依赖） |

并行时**必须 `copy_context()`**（与 `tools_registry.call_tool` 同一处坑）：任务跑在
子线程里，子线程不继承调用方的 ContextVar，不复制的话 token 用量会记到兜底用户
`local` 名下、日额度与单次预算也会拿错桶。

顺带修掉一个 **48 秒**的真瓶颈：`_job_id_of()` 以前对**每个 chunk** 重新
`get_collection()`（Chroma 客户端构造 ~11ms × 4392 chunk ≈ 48s），现在缓存集合句柄 +
预计算扁平 `{chunk_id: job_id}` 映射，热循环退化成纯 dict 查表。
实测**语义搜岗位：48.2 秒 → 84.6 毫秒**（向量路本身只要 23ms，瓶颈全在 BM25 的子集过滤）。

---

## 3. 快速开始

> 两种方式都会同时起 **Web 服务 8000** 和 **Dashboard 8501**；要改代码 / 跑抓取就用方式 B。

### 方式 A：Docker（一条命令）

```bash
git clone https://github.com/plus-Liang/agent-intern-roadmap.git
cd agent-intern-roadmap
cp .env.example .env          # 打开 .env，把 ZHIPU_API_KEY 填上
docker compose up --build     # 访问 http://localhost:8000（/chat）与 http://localhost:8501（看板）
```

- `docker-compose.yml` 里 `env_file: .env` 是**必需**的，没有 `.env` 会直接启动失败。
- 端口映射是 `8000:7860`（容器内固定 7860，与 HF Space 的 `app_port` 一致）。
- `docker compose up` 会起**两个容器**，共用同一份数据：

  | 服务 | 地址 | 是什么 |
  |---|---|---|
  | `app` | `http://localhost:8000` | FastAPI + Chainlit（`/`、`/chat`、`/docs`） |
  | `dashboard` | `http://localhost:8501` | Streamlit 看板（岗位列表 / 投递追踪 / 匹配打分 / Token 成本 / 简历管理 / 对话 Agent） |

- 两个服务**共用**同一份数据：`./rag/data`（岗位库 / `cleaned_jd.json`）、`./agent/data`（投递记录、简历、对话历史）、`./logs`（token 用量 / 反馈）、命名卷 `chroma_db`（向量库）。所以看板里的投递记录、岗位列表与 app 里完全一致。
- `dashboard` 由单独的 `Dockerfile.dashboard` 构建（`python:3.12-slim` + `requirements-hf.txt` + `requirements-dashboard.txt` 的 streamlit），**不会**把 Streamlit 塞进 app 镜像，云端 HF Space 依旧是精简镜像。
- 基础镜像与运行环境统一在 **Python 3.12**：`Dockerfile` 用 `python:3.12-slim`，
  `runtime.txt` 写 `python-3.12`，与 `numpy==2.5.3`（没有 3.11 wheel）的要求一致。
- 代码更新后重新构建：`docker compose up --build`；日常启动不需要 `--build`：`docker compose up`。
- app 容器仍然**不含 Playwright**：容器里跑不了抓取，要抓数据请用方式 B。

> ⚠️ **Dashboard 没有任何认证**（没有登录、没有密码），只能本机自己用。
> `docker-compose.yml` 因此把它绑在回环地址（`127.0.0.1:8501:8501`），局域网和公网都访问不到。
> 想让同网段的设备访问，把那一行改成 `"8501:8501"` —— **改之前先想清楚**：看板能读出并改写你的投递记录、简历、对话历史，暴露即等于把这些给别人看。
> 要放公网，请先自己套一层带认证的反向代理。

### 方式 B：本地 pip（两步）

**第 1 步：装依赖 + 配置 Key。** 需要 **Python 3.12+**（`numpy 2.5.3` 没有 3.11 的 wheel）。

```bash
pip install -r requirements.txt
cp .env.example .env      # 只需填 ZHIPU_API_KEY，其余留空即用默认值
```

Key 在 <https://open.bigmodel.cn/> 申请（`glm-5.3-flash` 有免费额度）。embedding 走本地
`fastembed`，首次使用会自动下载 `BAAI/bge-small-zh-v1.5`（约 90MB），不需要任何 embedding Key。

**第 2 步：起服务。**

```bash
python start.py
```

`start.py` 会同时拉起两个进程：`http://localhost:8000`（首页 / `/chat` / `/docs`）和
`http://localhost:8501`（Dashboard），Ctrl+C 一起停。

也可以只起 Web 服务（不要 Dashboard）：

```bash
python main.py                 # 等价于 uvicorn main:app --host 0.0.0.0 --port 8000
```

> 必须在**项目根目录**启动：Chainlit 要读到 `.chainlit/` 与 `chainlit.md`。
> Dashboard 单独启动请用 `streamlit run dashboard/app.py`（不要用 `python -m streamlit`，会因模块遮蔽报错）。

### 可选但推荐：建向量索引

`chroma_db/` 不进 git，新 clone 下来是空的。不建也能跑（`search_jobs` 会退回 SQLite 关键词检索），
但**模糊需求那类查询（"想找偏大模型落地、能写工程代码的实习"）拿不到语义结果**。建一次即可：

```bash
python -m rag.vector_store --rebuild      # 从 jobs.db 全量重建（约几分钟，本地 embedding）
```

`jobs.db` 同样不进 git，但**不需要手动建**：`rag/data/db.py:ensure_db()` 会在它不存在时
自动从 `cleaned_jd.json` 建库（`cleaned_jd.json` 比库新时也会自动增量同步）。

### 岗位类型（实习 / 正式 / 兼职）

`jobs` 表有 `job_type` 列（`实习` / `正式` / `兼职` / 空），`search_jobs` 支持按它过滤：
用户说「找实习」就只返回实习岗，说「找岗位」则返回全部类型。判定口径集中在
`shared/job_type.py`（抓取打标、清洗、入库、搜索过滤、老库回填共用同一份实现）：

| 平台 | 类型信号 | 判定 |
|---|---|---|
| 实习僧 `shixiseng` | 搜索 URL 固定带 `type=intern` | 平台属性即实习（标题带「兼职」另算） |
| 牛客 `niuke` | 接口 `recruitType=2`（实习频道） | 平台属性即实习 |
| 教育部平台 `ncss` | 接口 `recruitType` 恒为 0、详情页无类型字段 | 只能按标题「实习」判断，其余算正式 |

> ⚠️ **不能**用「标题里有没有实习」来判牛客的数据：实测实习频道 180 条里只有 51.1%
> 的标题带「实习」字样，按标题判会把近一半真实习岗判丢。
> `ncss` 那 923 条标题不带「实习」的岗位按**正式**兜底（它是校招池），这样搜实习时
> 不会被混进来；代价是「标题不带实习、其实招实习生」的岗位会被漏判成正式。

**老库升级**：`rag/data/db.py:migrate_db()` 会给已有的 `jobs.db` 补 `job_type` 列并
回填存量数据（幂等，已挂在 `ensure_db()` 里，`search_jobs` 第一次查询就自动完成）；
库不存在时从 `cleaned_jd.json` 重建，走同一套判定，不需要手工 SQL。

### Hugging Face Space 部署

- 顶部 frontmatter 的 `sdk: docker` / `app_port: 7860` 就是 Space 配置，**不能删**。
- 变量在 Space 的 **Settings → Variables and secrets** 里配置，变量名以 `.env.example` 为准。
- 镜像以 UID 1000 的非 root 用户运行；容器文件系统是临时的，`jobs.db`、`logs/`、
  `agent/data/*.db` 重启即丢，需长期保存请挂持久卷或外部数据库。

---

## 4. 配置说明

全部变量见仓库根目录的 [`.env.example`](.env.example)，复制成 `.env` 后按需打开注释。
下表逐变量一句话，**除 `ZHIPU_API_KEY` 外都有可用的默认值**。

### 模型

| 变量 | 说明 |
|---|---|
| `ZHIPU_API_KEY` | **必填**。对话模型的 Key，缺失时 `shared/config.py:check_config()` 直接报错 |
| `ZHIPU_BASE_URL` | 智谱 API 地址，默认 `https://open.bigmodel.cn/api/paas/v4` |
| `ZHIPU_CHAT_MODEL` | 对话模型名，默认 `glm-5.3-flash` |
| `LOCAL_EMBEDDING_MODEL` | 本地 embedding 模型，默认 `BAAI/bge-small-zh-v1.5`（无需 Key） |
| `USE_RERANKER` | 是否启用重排器，默认 `false`（开启需要 torch / sentence-transformers） |

#### 换模型：只改 `.env` 三个变量

> 变量名是历史遗留（`ZHIPU_*`），**它的值可以指向任何 OpenAI 兼容服务**——不是只能连智谱。
> `shared/llm_client.py` 实际请求的是 `{ZHIPU_BASE_URL}/chat/completions`，Key 走 `Authorization: Bearer`。

```bash
# 以 DeepSeek 为例，三行全部换掉（换智谱自家模型只改最后一行）
ZHIPU_API_KEY=sk-xxxxxxxx                  # 目标服务的 Key
ZHIPU_BASE_URL=https://api.deepseek.com/v1 # OpenAI 兼容根地址，**不要**带 /chat/completions
ZHIPU_CHAT_MODEL=deepseek-chat             # 模型名，原样填服务商文档里的名字
```

| 能直接用 | 不能直接用（要改代码） |
|---|---|
| 智谱全系：`glm-4-flash` / `glm-4.5-air` / `glm-4.6v` / `glm-5.3-flash` 等 | 需要私有 SDK 的模型（如 Anthropic 原生协议） |
| 其他 OpenAI 兼容 API：DeepSeek / Moonshot / 通义 / OpenAI 等 | 本地 Ollama（要改 `shared/llm_client.py` 的调用方式，不能只改 `.env`） |

**注意事项**

- **embedding 不受换模型影响**：检索用的 embedding 是本地 `fastembed` 模型（`BAAI/bge-small-zh-v1.5`），
  在本地推理，不调 API、不花钱，换对话模型不用动它，也不用换 Key。
- **换非智谱模型的 `reasoning_effort` 已自动适配，不用管**。项目给思考模型默认传
  `reasoning_effort=low`（注入点在 `shared/llm_client.py` 的 `_build_payload()`）。有些
  OpenAI 兼容网关不认这个参数，会直接回 400（`invalid_request_error` / `unknown_parameter`）。
  客户端一旦从报错里认出「是 reasoning_effort 不被支持」，就会自动**去掉该参数重试一次**，
  并把「这个模型不吃 reasoning_effort」记进进程内缓存，之后对同一模型的请求不再注入——
  换模型不会因为这个参数失败，只会在日志里多打一行 `[llm] INFO`。
  若想彻底不发这个参数，也可以把 `.env` 里 `REACT_LLM_REASONING_EFFORT` /
  `RESUME_LLM_REASONING_EFFORT` / `MATCH_LLM_REASONING_EFFORT` /
  `COVER_LETTER_LLM_REASONING_EFFORT` 全部**留空**（空串即不注入，默认值是 `low`）。
- 输出额度（`LLM_MAX_TOKENS` / `REACT_LLM_LONG_MAX_TOKENS` 等）按 token 计，换到的服务若单次输出上限更小，
  记得把 `REACT_LLM_LONG_MAX_TOKENS` 一起调小，否则会经常 `finish_reason=length`。

**操作步骤（3 步）**

> ⚠️ **改完 `.env` 必须 `docker compose down && docker compose up -d`，不能只 `restart`。**
> `docker compose restart` 只是重启容器里的进程，**不会重新读取 `.env`**，改的模型 / Key 不会生效，
> 很容易被误判成「换模型失败」。本地 pip 方式同理：Ctrl+C 后重新 `python start.py`。

```bash
# ① 编辑 .env，改 ZHIPU_CHAT_MODEL（换厂商时 ZHIPU_API_KEY / ZHIPU_BASE_URL 一起改）
# ② 重建容器让新配置生效（只 restart 不会重读 .env！）
docker compose down && docker compose up -d   # 本地 pip 方式：Ctrl+C 后重新 python start.py
# ③ 打开 http://localhost:8000/chat，发一句「你好」
#    能正常回话 = 换模型成功；若报错，先看日志里是不是 base_url / Key 的问题
```

### 对话与多用户

| 变量 | 说明 |
|---|---|
| `CHAT_HISTORY_DB` | 对话历史库路径，默认 `agent/data/chat_history.db` |
| `CHAT_HISTORY_STORE_STEPS` | 是否把工具调用过程也落库，默认 `false`（只存问答） |
| `CHAT_AUTH_ENABLED` | 是否开启登录，默认 `false`（不开时所有人归到默认用户 `local`） |
| `CHAT_ADMIN_USER` | 管理员用户名，默认 `admin` |
| `CHAT_ADMIN_PASSWORD` | 管理员密码，**没有默认值**：开了认证却不设会直接启动失败（刻意如此） |
| `CHAINLIT_AUTH_SECRET` | Chainlit 的 JWT 签名密钥，不设会自动生成并落到 `agent/data/.chainlit_secret` |
| `MAX_HISTORY` | 每轮注入的历史消息条数上限，留空用代码默认 |

### 限流与成本控制（四道闸门）

| 变量 | 说明 |
|---|---|
| `RATE_LIMIT_ENABLED` | 四道闸门的总开关，默认 `true`；置 `false` 时**四道全关**（含 `max_tokens` 与单次预算） |
| `RATE_PER_MIN` | 闸门 3 单用户频率：每分钟补充的请求数，默认 6 |
| `RATE_BURST` | 闸门 3 令牌桶容量（允许连发几次），默认 3 |
| `LLM_MAX_TOKENS` | 闸门 1 单次输出上限，默认 1024，`0` = 不限；ReAct **日常工具调用轮**用它 |
| `RESUME_LLM_MAX_TOKENS` | 简历解析单次输出上限，默认 4096（思考模型会先吃额度，1024 不够） |
| `RESUME_LLM_REASONING_EFFORT` | 简历解析思考档位，默认 `low`，空 = 不注入 |
| `REACT_LLM_LONG_MAX_TOKENS` | ReAct **长输出轮**（完整简历 / 长列表）单次输出上限，默认 8192 |
| `REACT_LLM_REASONING_EFFORT` | ReAct 思考档位，默认 `low`，空 = 不注入 |
| `MATCH_LLM_MAX_TOKENS` | `match_resume` 工具**内部**那次 LLM 调用（不走 ReAct 分档）的单次输出上限，默认 4096 |
| `MATCH_LLM_REASONING_EFFORT` | `match_resume` 工具内部调用的思考档位，默认 `low`，空 = 不注入 |
| `COVER_LETTER_LLM_MAX_TOKENS` | 投递包内部两次 LLM 调用（**简历定制** + **自荐信**，不走 ReAct 分档）的单次输出上限，默认 4096 |
| `COVER_LETTER_LLM_REASONING_EFFORT` | 投递包内部调用的思考档位，默认 `low`，空 = 不注入 |
| `RUN_TOKEN_BUDGET` | 闸门 2 单次请求累计 token 上限，默认 30000，触顶**降级收尾不拒绝请求** |
| `DAILY_TOKENS_PER_USER` | 闸门 4 单用户日额度，默认 200000，`0` = 不限 |
| `GLOBAL_DAILY_TOKENS` | 闸门 4 全局日额度，默认 2000000，`0` = 不限 |
| `MAX_CONCURRENCY` | 全局并发上限（`asyncio.Semaphore`），默认 3 |

复杂任务编排（`agent/complex_task_flow.py`，见「多智能体协作」小节）：

| 变量 | 说明 |
|---|---|
| `MULTI_AGENT_ENABLED` | 多智能体总开关，默认 `1`；置 `0` 后复杂任务退回单流程 / ReAct（等于改造前行为） |
| `MULTI_AGENT_MAX_RETRIES` | 同一步最多被 Critic 打回重做几次，默认 2；用完仍不合格就**降级**（返回已完成部分 + 如实说明） |
| `MULTI_AGENT_MAX_PLAN_STEPS` | 计划最多几步，默认 5（Planner 也是模型，必须防它写 20 步） |
| `MULTI_AGENT_MAX_EXECUTIONS` | 全局执行次数上限，默认 12（防「步骤多 × 每步重试多」的组合爆炸） |
| `MULTI_AGENT_CRITIC_LLM` | Critic 是否额外投一票（LLM 语义复核），默认 `1`；事实层判据永远不依赖模型，且事实层已判不合格时不再调 LLM |
| `MULTI_AGENT_PACKAGE_THRESHOLD` | 生成投递包的分数门槛，默认 70（计划里 `when: score >= N` 的默认值） |

> 两个已知点：**日额度默认值偏小**（2M 只够约 10 个满额度用户），公开试用前需按真实消耗重估；
> **`max_tokens` 目前和总开关耦合**，想"只关限流、保留输出上限"需要把它拆成独立开关（待实现）。

### 路径

| 变量 | 说明 |
|---|---|
| `FASTEMBED_CACHE_PATH` | embedding 模型缓存目录，默认 `~/.cache/fastembed`；容器里建议指到持久盘 |
| `TOKEN_DB_PATH` | 用量库路径（日额度的真相源，重启不丢），默认 `logs/token_usage.db` |
| `FEEDBACK_DB_PATH` | 用户反馈库路径，默认 `logs/feedback.db` |

### 抓取（只有自己更新数据时才需要）

| 变量 | 说明 |
|---|---|
| `SCHEDULER_PLATFORMS` | 抓取平台，默认 `shixiseng,niuke,ncss` |
| `SCHEDULER_CITY` | 城市池，默认 `广州,深圳,北京,上海,杭州,成都` |
| `SCHEDULER_KEYWORDS` | 关键词池，留空则用 `config/scraping.yaml` 里的 59 个全行业词 |
| `SCHEDULER_BATCH_SIZE` | 单晚抓取组合数，默认 180（`0` = 全量，很慢，别用） |
| `SCHEDULER_CHUNK_SIZE` | 进度日志的打印间隔（每 N 个组合一行），默认 12，`0` = 不分块 |

> 城市池 / 关键词池的完整配置在 [`config/scraping.yaml`](config/scraping.yaml)，
> 环境变量优先级高于该文件。另外代码还读三个 `.env.example` 未列的变量：
> `SCHEDULER_HOUR` / `SCHEDULER_MINUTE`（Web 服务内置定时抓取的时刻，默认 03:00）、
> `SCHEDULER_RUN_ON_STARTUP`（启动时立刻抓一次，默认关）。

---

## 5. 数据说明

**clone 下来就能用，不用先爬。**

| 项 | 现状 |
|---|---|
| `rag/data/cleaned_jd.json` | **约 960 条**（随抓取增长），进 git（约 2MB），clone 即有 |
| 平台分布 | 实习僧 `shixiseng` / 国家大学生就业服务平台 `ncss` / 牛客 `niuke` 三平台，比例随抓取轮次变化 |
| 六城分布 | 上海 / 北京 / 深圳 / 杭州 / 成都 / 广州 六城为主，另有少量「全国」岗位 |
| 字段 | `platform` `job_id` `title` `company` `city` `salary` `url` `description` `publish_date` |
| `rag/data/jobs.db` | **不进 git**，首次查询时由 `ensure_db()` 自动从 JSON 建库（幂等） |
| `chroma_db/` | **不进 git**，需要 `python -m rag.vector_store --rebuild` 建一次（见第 3 节） |
| 运行期数据 | `logs/`、`agent/data/*.db`、`agent/data/user_profile.json` 均在 `.gitignore` 里 |

几点口径：

- **权威数据源是 `cleaned_jd.json`**。`jobs.db` 是它的 SQLite 镜像，只增不删；
  两者条数可能略有差异（同步时序、清洗/去重摘掉记录后库里残留旧条目）。
  新 clone 不存在这个问题，从头建库即为 JSON 里当前的条数。
- **清洗门槛按平台配置**：`rag/quality/cleaner.py:PLATFORM_MIN_DESC_LEN = {"ncss": 100}`，
  其余平台沿用全局 200 字。原因是 ncss 上有约四分之一岗位的"职位详情"是空壳（只剩标题回显），
  100 字门槛能拦掉空壳、又不误伤 100–199 字的精炼 JD。
- **抓取覆盖面是滚动的**：全池 = 3 平台 × 6 城 × 59 词 = **1062** 个组合，
  默认每晚 180 个组合，约 6 晚覆盖一轮；进度记在 `scrape_history` 表里。
- **ncss 有全局 1 req/s 限速**，且每个岗位要多请求一次详情页，所以它是最慢的平台
  （实测 60 个组合 12–63 分钟，抖动大）；实习僧约 12.5s/组合，牛客约 1.5s/组合。
- **clone 后拿到的是快照，不会自动更新。** 要最新数据：`git pull`，
  或自己跑 `python -m agent.scrapers.scheduler --once`。

---

## 6. 常见问题

### Q1. 没有 API Key 能跑吗？

页面能起来、能浏览 Dashboard，但**对话、匹配打分、改简历这些要调 LLM 的功能都用不了**——
`shared/config.py` 在缺 Key 时会抛 `ConfigError`。申请地址 <https://open.bigmodel.cn/>，
`glm-5.3-flash` 有免费额度，个人试用足够。Embedding 不用 Key（本地 fastembed）。

### Q2. 岗位数据怎么更新？

三种方式，按需要挑：

1. **手动跑一次抓取**（最直接，见 Q3）。
2. **Web 服务内置定时任务**：`main.py` 启动时会挂 APScheduler，每天 `SCHEDULER_HOUR:SCHEDULER_MINUTE`
   （默认 03:00）自动跑一次；也可以 `GET /api/scheduler/status` 看状态、
   `POST /api/scheduler/trigger` 立刻触发一次（异步，不阻塞）。
3. **GitHub Actions**：`.github/workflows/daily_scrape.yml` 每天 UTC 19:00（北京 03:00）跑一次并 commit 回仓库。
   用它需要在仓库 Secrets 里配 `ZHIPU_API_KEY`，并按需改里面的城市列表。

抓完会顺带入库：JSON → `jobs.db` 全量对齐 → 向量库**增量**写入（只重算新增/改动的 chunk，省额度）。

### Q3. 抓取怎么跑？

抓取依赖 Playwright + Chromium，**不在 `requirements.txt` 里**，单独装：

```bash
pip install -r requirements-crawl.txt
python -m playwright install chromium
python -m agent.scrapers.scheduler --once       # 真实抓取，按 batch_size 抓一批
```

常用参数与技巧：

```bash
python -m agent.scrapers.scheduler --status      # 看上次运行结果与滚动覆盖进度
python -m agent.scrapers.scheduler --selftest    # 15 项离线自检，不联网、不写盘
python -m agent.scrapers.scheduler --dry-run     # 只打印将要抓的组合，不真抓
```

- 想快速验证链路又不联网：`SCHEDULER_USE_MOCK=1 python -m agent.scrapers.scheduler --once`。
- **耗时预期**：单晚 180 个组合，实测量级 **30–60 分钟**（ncss 有 1 req/s 限速且会抖动）。
- 抓取会写 `rag/data/cleaned_jd.json`、`rag/data/scraped_jd.txt`、`jobs.db` 与向量库。
- 沙箱 / 服务器上跑 Playwright 可能因权限起不来浏览器（Windows 上是 `WinError 5`），
  需要给它足够的权限或换带桌面的环境。

### Q4. 多用户怎么开？

```bash
# .env 里
CHAT_AUTH_ENABLED=true
CHAT_ADMIN_PASSWORD=<你自己的密码>      # 不设会直接启动失败，这是刻意的
CHAINLIT_AUTH_SECRET=<随机串>           # 可选，不设会自动生成并落盘

python -m agent.chat_history adduser <用户名> <密码>      # 加号
python -m agent.chat_history setpass <用户名> <新密码>    # 改密
python -m agent.chat_history listusers                    # 列号
```

开启后：每个用户的对话历史、画像、简历、投递记录、岗位标记**互相隔离**（`contextvars` 传递当前用户），
对话按 `user_id` 固定 thread，刷新页面 / 重启进程都能续接。

两个限制（都是**已知待实现**，上生产前要知道）：

- **限流的频率桶是进程内的**，多 worker / 多副本会各自计一份，限流失真。
  `docker-compose.yml` 因此固定 `--workers 1`。要横向扩副本，需把桶换成 Redis 或落库滑动窗口。
- **`MAX_CONCURRENCY` 目前只是"压住"并发**：`run_agent` 还在事件循环里同步阻塞，
  真正并行要把 Agent 移进 `to_thread` + 线程池（连带处理 ContextVar 继承）。

### Q5. 端口冲突 / 端口怎么改？

| 端口 | 谁在用 | 怎么改 |
|---|---|---|
| 8000 | `main.py` / `uvicorn main:app`（首页、`/chat`、`/docs`） | `python main.py` 里改 `port`，或直接 `uvicorn main:app --port 8080` |
| 8501 | Streamlit Dashboard | 本地 pip：`start.py` 的 `SERVICES`；Docker：`docker-compose.yml` 里 dashboard 的 `ports`（想换宿主机端口改成 `"8502:8501"`；想公开改成 `"8501:8501"`，先看上面的无认证警告） |
| 7860 | 容器内 uvicorn / HF Space `app_port` | 改 `Dockerfile` 的 `CMD`、`docker-compose.yml` 的 `ports: "8000:7860"` 左值 |

改完记得同步首页卡片与 `docker-compose.yml` 里写死的 `localhost:8501` 链接。

### Q6. 语义检索没结果 / 首页看板卡片点不开？

- **语义检索为空**：多半是 `chroma_db/` 还没建（不进 git）。跑一次
  `python -m rag.vector_store --rebuild`。不建也能用关键词检索，只是模糊需求效果差。
- **看板卡片点不开**：首页第二张卡片指向 `http://localhost:8501`。`docker compose up` 与
  `python start.py` 都会起 Dashboard；只单独跑 `python main.py`（或只 `docker compose up app`）
  时没有 Dashboard，卡片自然点不开。

---

### Q7. Boss / 智联等平台可以用吗？

不能自动抓 —— 这些平台没有公开接口，抓取也违反服务条款。但**可以把 JD 文本复制粘贴给我**：
在 `/chat` 里直接贴整段 JD（含「岗位职责 / 任职要求」），我会把它当成一个临时岗位，
一样能做匹配打分、生成投递包和模拟面试。粘贴实习僧 / 牛客 / ncss 的岗位链接更省事，
连 JD 都不用贴（系统按链接里的 job_id 去岗位库取那一条）。

### Q8. 怎么置顶对话？

**首选**：鼠标移到侧边栏某条会话上，点右边的「…」菜单 →「📌 置顶」（已置顶的会话那一条显示「📌 取消置顶」）。

**兜底**：在会话里发 `/pin` —— 当前这个会话就会排到最上面；取消用 `/unpin`。
菜单注入依赖 Chainlit 前端 DOM（`thread-<id>` / `thread-options` / `rename-thread`），
升级 Chainlit 后可能失效，那时用命令。实现：`public/custom.js`（前端）+ `api/router.py` 的 `/threads/pin|unpin|pinned`（后端）。

- 置顶的按**置顶时间倒序**排在最上，未置顶的仍按最近更新（`updated_at`）倒序。
- 顺序由数据层决定（`agent/data_layer.py` 覆写 `get_all_user_threads`）：
  Chainlit 前端的列表顺序照单全收，所以刷新页面就能看到新顺序。
- 最多置顶 `data_layer.MAX_PINNED`（50）个；`/pin` 有回执，失败会说原因。
- 置顶状态存在 Chainlit 库 `threads.pinnedAt`（我们自己加的列，NULL = 未置顶），
  启动时 `ensure_schema()` 自动补列，不用手工迁移。

---

## 7. 测试怎么跑

全部测试都是**离线**的：不联网、不调真 LLM、不写 `rag/data/` 与真实向量库，
临时文件落在各自的一次性目录里。

```bash
# ① 调度器 / 全链路自检（15 项，不联网不写盘）
python -m agent.scrapers.scheduler --selftest

# ② 离线回归测试（4 个文件，直接当脚本跑）
python agent/tests/test_round9_fixes.py      # 合并落盘 / 关键词组 / limit / 多城市 / 孤儿清理
python agent/tests/test_ncss_offline.py      # ncss 城市映射 / 字段映射 / 限速 / 未收录城市不瞎猜
python agent/tests/test_chat_history.py      # 对话历史建库 / 续接 / 快照 / 多用户
python agent/tests/test_limits.py            # 四道闸门 / 日额度 / 单次预算 / 总开关

# ③ 端到端验证脚本
python scripts/verify_rag_e2e.py             # 模糊查询走语义、精确查询走 SQL、job_id 反查闭环
python scripts/verify_heal.py                # 实习僧浏览器故障自愈逻辑（不需要真浏览器）

# ④ 数据审计（只读）
python scripts/audit_data.py                 # 数据质量统计
```

注意事项：

- **Windows 上先设 UTF-8 输出编码**：测试会打印 `✗` 之类的符号，默认 GBK 控制台会
  `UnicodeEncodeError`，退出码变成 1（看起来像失败，其实是打印挂了）。

  ```bat
  set PYTHONIOENCODING=utf-8
  python agent/tests/test_round9_fixes.py
  ```

  PowerShell 用 `$env:PYTHONIOENCODING='utf-8'`；也可以 `set PYTHONUTF8=1`（解释器级 UTF-8 模式）。
- 测试会在 `agent/tests/` 下建 `_tmp_*` 临时目录，需要该目录**有写权限**；
  否则会看到 `unable to open database file` / `PermissionError`。
- `scripts/verify_rag_e2e.py` 与 `scripts/verify_heal.py` 用 `Path(__file__)` 自己定位仓库根，
  换机器 / 换目录都能直接跑。
- 依赖 `chroma_db/` 与 `jobs.db` 的脚本（`verify_rag_e2e.py`、`audit_data.py`）需要先按第 3 节建好数据。

---

## 8. 评测：Agent 效果量化

「改完 prompt / 换了模型，到底是变好了还是变差了？」——靠感觉答不了这个问题。
`evaluation/` 下是一套**可重复运行**的分类评测：30 道题、6 类能力，跑完给
分类准确率 + 总准确率，并逐次留档。

| 类别 | 题数 | judge_type | 在测什么 |
|---|---|---|---|
| `search` 搜岗位 | 6 | `deterministic` | 城市 / 岗位类型有没有被正确提取，过滤是不是真的生效 |
| `match` 匹配打分 | 5 | `llm`（+ 确定性区间） | 分数是否落在合理区间、反思节点是否触发、理由有没有证据 |
| `package` 投递包 | 4 | `deterministic` | 三件套是否生成、关键字段有没有丢、有没有已知错字 / 模板兜底 |
| `interview` 模拟面试 | 5 | `llm` | 连问 3 题，整组题是否贴岗位 JD、有没有串到别家公司 |
| `boundary` 边界 | 7 | `trajectory`（6）+ `deterministic`（1） | 闲聊不误调工具（断言的是**工具调用序列**）、搜不到就如实说 |
| `complex` 复杂任务 | 3 | `deterministic`（+ 轨迹断言） | 多智能体的三步协作有没有真的产出，搜索 / 匹配 / 出包的**顺序**对不对 |

### 目录与三层插件

| 文件 | 角色 |
|---|---|
| `evaluation/test_set.yaml` | **唯一题库**（题目 / 期望 / `judge_type` / 轨迹断言） |
| `evaluation/framework.py` | 三个 ABC：`BaseMetric`、`BaseProvider`、`BaseReport` + 内置插件与注册表 |
| `evaluation/run_eval.py` | 跑题（只负责执行 + 编排，判分 / 存储 / 渲染都交插件） |
| `evaluation/regression.py` | bootstrap 回归门控（两次结果的 delta + 95% 置信区间） |
| `evaluation/test_framework.py` | 框架自测（18 项，不调 LLM、不花钱） |

判分方式是**插件**，不是 if-else：`judge_type` 指向 `framework.METRICS` 里的实现，
新加一种判分 = 写一个 `BaseMetric` 子类加 `@register_metric`，主流程不用改。
题库与报告的读写走 `BaseProvider`（默认 `YamlProvider`），终端 / MD / JSON 三种呈现走
`BaseReport`。

### 轨迹断言（只看工具调用序列，不看最终文本）

`judge_type: trajectory` 的题，或任何在 `case.trajectory` 下写了断言的题，都会检查
**调用序列**（序列由 `framework.TrajectoryRecorder` 从现有工具入口采集，业务代码零改动）：

```yaml
trajectory:
  calls_tool: [search_jobs, match_resume]        # 必须调过（顺序无关）
  not_calls_tool: [generate_application_package] # 一次都不能调；写 "*" = 任何工具都不许调
  call_order: [[search_jobs, match_resume]]      # 先后顺序；任一端没调用则不算违规
```

典型用例：搜岗位时断言 `not_calls_tool: generate_application_package`（别顺手出包）；
多智能体出包前断言 `call_order: [[match_resume, generate_application_package]]`
（**先打分再出包**，顺序反了就是真 bug，最终文本再漂亮也没用）。

### bootstrap 回归门控：只在「下降超过噪声」时标红

`--compare` 不再只比两个百分数（30 题里掉 1 题就是 -3.3pp，可能只是 LLM 抖了一下），
而是对 delta 做**两级 bootstrap**（题级重采样 + 题内 run 级重采样）给出 95% 置信区间：

* CI 上界 < 0 → 🔴 回归；CI 下界 > 0 → 🟢 进步；区间跨 0 → ⚪ 噪声内，不下结论。
* 想让 CI 把 LLM 抖动也算进去，先用 `--repeat 3` 跑；`--repeat 1` 的 CI 只反映题库采样噪声。
* `regression.py --gate` 有回归时退出码为 1，可以直接当 CI 门禁用。
* 两次都跑过的题才参与 delta（题目增删单独列出，不算能力变化）。

```bash
python evaluation/run_eval.py                       # 全量（30 题，会真调 LLM，约 1-2 小时）
python evaluation/run_eval.py --category search     # 只跑某一类（改完 prompt 先跑这个）
python evaluation/run_eval.py --case match-01       # 只跑某几题
python evaluation/run_eval.py --repeat 3            # 每题跑 3 次，看稳定性
python evaluation/run_eval.py --list                # 只列题，不跑
python evaluation/run_eval.py --compare evaluation/results/<上次>.json   # 与上次对比 + CI
python evaluation/regression.py --compare a.json --current b.json        # 单独跑回归门控
python evaluation/regression.py --compare a.json --gate                  # CI 门控（有回归则退出 1）
python evaluation/test_framework.py                                      # 框架自测（不调 LLM）
```

### 多次跑：`--repeat N`（LLM 是非确定的）

同一道题、同一个模型、同一套 prompt，也可能一次过、一次不过 —— 这不是"测错了"，
而是 LLM 的固有抖动。想知道一道题到底有多稳，就得连跑几次：

```bash
python evaluation/run_eval.py --category match --repeat 3
```

`--repeat N` 在原来的准确率之外多给三个数：

| 指标 | 含义 |
|---|---|
| **单题通过率** | 这道题 N 次里过了几次（如 `4/5 = 80%`）—— 单题稳不稳 |
| **类别稳定率** | 该类别下所有题「单题通过率」的**平均值**（不是通过/总数，分子是分数、不是 0/1） |
| **不稳定题** | 单题通过率落在 **20%~80%（含端点）** 的题 —— 这几道是"再跑一次结果就变"的题，最值得盯 |

结果 JSON 保留**每一次**的完整详情（`results[].runs[]`：判词 / 检查项 / extra / 耗时），
题级 `passed` 取**多数票**（通过率 ≥ 50%），这样 `--repeat 1` 与老口径逐字一致，
`--repeat N` 也能继续用 `--compare` 对老结果。代价是耗时按 N 倍增长
（`--category match --repeat 3` 约 25-40 分钟），建议**先 `--repeat 3` 跑抖动大的类别**，
全量仍用 `--repeat 1`。

结果怎么看：

- 终端直接打印「分类准确率 + 总准确率 + 每道失败题的原因 + 轨迹断言序列」；
- 落到 `evaluation/results/<时间戳>.json`（机器可读，含每题明细与插件口径）和同名 `.md`（可读表）；
- 加了 `--compare` 会多打一段 bootstrap 回归报告（总体 / 分类 / 判分方式 / 逐题的 delta +
  95% CI + 🔴🟢⚪ 判定）——**不再用「掉了几题」当结论**；
- 匹配打分的「差距」有一层**确定性溯源过滤**（`agent/tools_registry.split_gaps_by_jd`）：
  每一项都要能在 JD 原文里找到对应关键词（英文按整词比对、中文按 ≥3 字滑窗），
  找不到的会被移出 `gaps`、单列为「通用建议」—— 防止把 JD 根本没提的技术栈
  （典型：JD 没写 TensorFlow / PyTorch，差距里却冒出它们）写成「岗位要求的差距」。

改题库：

- 只改 `evaluation/test_set.yaml` 一个文件：加题 = 复制一段改字段，每题必带
  `id / category / question / expect / judge_type`；`judge_type` 必须能在
  `framework.METRICS` 里找到插件，否则加载时直接报错（早失败好过跑一半炸）；
- 确定性题把期望写进 `expect`（城市 / 类型 / 条数 / 区间 / 必须出现或禁止出现的字），
  LLM 裁判题写 `expect.score_range` 或 `expect.topics` 加 `must_not_mention` 这类硬约束，
  轨迹题写 `trajectory.calls_tool / not_calls_tool / call_order`；
- 题里的 `job_id` 取自 `rag/data/jobs.db`。换库后某条被删，该题会以「岗位查不到」
  失败并写明原因，**不会静默算通过**。

隔离与前提：

- 评测用一个独立的 `eval_runner` 身份跑，简历库 / 投递包 / 导出目录都指向
  `evaluation/_artifacts/`，**不动你的真实简历库和投递记录**；
- 需要 `rag/data/jobs.db` 里已有数据、`.env` 里有可用的 API Key（确定性题不花钱，
  LLM 裁判题和匹配 / 面试题会真调模型，注意用量）；
- LLM 裁判用的是**独立的 judge prompt**，和被测的那套 prompt 不是一回事；
  但裁判目前跑的是**同一个模型**（项目只配了一个），这是已知局限 —— 换模型时
  裁判的标准也会跟着变，所以跨模型的分数只能看趋势、不能当绝对结论。

与 `agent/evaluation/` 的关系：那一套是「ReAct 工具调用轨迹」的深度评估
（10 个任务、每题一个独立临时库、LLM-as-Judge、失败归档），看的是**过程**；
`evaluation/` 这一套看的是**能力准确率**，快、便宜、可对比，适合每轮改动后回归。
两者互补，不是替代。

---

## 扩展指南：想改什么，改哪里

按"想改的东西"查表，**大部分需求只动一个文件、不用改代码**：

| 你想改什么 | 改哪个文件 |
|---|---|
| 加城市 / 加关键词 | [`config/scraping.yaml`](config/scraping.yaml)（**只改这一个文件，不用改代码**；环境变量 `SCHEDULER_CITY` / `SCHEDULER_KEYWORDS` 优先级更高） |
| 加平台 | 在 `agent/scrapers/` 下新建抓取器（参考 `agent/scrapers/niuke.py`），再在 `agent/scrapers/scheduler.py` 里注册 |
| 换 LLM / 换 embedding | 根目录 `.env`（变量名见 [`.env.example`](.env.example) 与第 4 节） |
| 改 Agent 提示词 | `agent/react_agent.py` 的 `STATIC_PREFIX` |
| 改对话 UI | `agent/app.py`（Chainlit）；改看板 → `dashboard/` |
| 加 REST 接口 | `api/router.py` |
| 调 RAG（召回 / 融合 / 重排） | `rag/retriever.py`（BM25 + 向量 RRF）与 `agent/tools_registry.py`（分流 / 语义重排）；改完记得 `python -m rag.vector_store --rebuild` |
| 加测试 | `agent/tests/` |

---

## 目录结构

```
agent/        对话 Agent：app.py（Chainlit）、react_agent.py、工具、抓取器、记忆与限流
rag/          RAG：切块、embedding、向量库、混合检索 + quality/（清洗、审计）
api/          FastAPI 路由（岗位 / 投递 / Token / 定时抓取）
dashboard/    Streamlit 看板（岗位列表、投递追踪、匹配打分、Token 成本、简历管理、对话 Agent）
shared/       公共层：配置、LLM 客户端、限流、token 记账、用户上下文
config/       scraping.yaml：城市池 + 关键词池 + 调度参数
scripts/      每日抓取的 Windows 任务脚本 + 验证 / 审计脚本
docs/images/  README 截图（对话 Agent / Dashboard / API 文档）
```

## License

[MIT](LICENSE)
