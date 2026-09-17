# 学习/实习规划助手

## 解决什么问题

我有一个真实需求：从零基础到投递 Agent 开发实习，需要一个可执行的学习路径。
手动规划容易漏项、节奏不对，而且计划随时可能要调整。
所以我做了这个工具，输入目标、时间、基础，生成可执行的周计划，并支持多轮追问调整。

## 功能

- 输入目标、每周可投入时间、当前基础
- 生成 3–6 周可执行的周计划和每日任务
- 支持多轮追问调整，模型记住上下文
- 输出模式可切换：默认只输出改动部分，用户要求时输出完整计划
- 流式显示，边生成边看
- 失败自动重试，配置错误友好提示
- 对话记录保存到本地，日志记录到 `logs/app.log`

## 技术栈

- Python 3.12
- 火山方舟 ARK API（OpenAI 兼容接口）
- Streamlit（Web 界面）
- python-dotenv（环境变量管理）

## 如何运行

1. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 复制 `.env.example` 为 `.env`，填入你的配置：

   ```text
   ARK_API_KEY=你的真实key
   ARK_MODEL=你的接入点ID
   ```

3. 启动 Web 界面：

   ```bash
   streamlit run app.py
   ```

4. 或者用命令行版本：

   ```bash
   python main.py
   ```

## 目录结构

| 文件 | 作用 |
|---|---|
| `app.py` | Streamlit 网页界面，处理输入、渲染对话、模式切换 |
| `main.py` | CLI 版本，保留作为对比 |
| `llm_client.py` | 封装 ARK API 调用，支持流式输出与重试机制 |
| `prompts.py` | Prompt 模板与输出模式（diff/full）控制 |
| `storage.py` | 保存对话记录为 Markdown 文件 |
| `errors.py` | 自定义异常：ConfigError、APIError |
| `logger.py` | 日志配置，写入 logs/app.log |

## 核心设计

### 1. 消息历史

模型本身无记忆，每轮调用时把完整 `messages` 列表发给 API，模型才能"记住"之前说过的话。

### 2. 输出模式控制

用 `session_state` 记录当前模式，通过关键词检测判断用户意图：

- 默认 `diff`：只输出改动部分
- 用户说"完整/全部/都输出"→ 切换为 `full`：输出完整计划
- 用户说"只输出改动"→ 切回 `diff`
- 模式一旦切换，后续保持，直到用户明确再次切换

### 3. 流式输出

用 `chat_stream` 逐字返回，用户 2–5 秒就能看到内容，而不是等 30 秒一次性出现。

Streamlit 界面里不用 `st.write_stream`，改用 `st.empty()` + `markdown` 手动流式，容器固定，避免前端 DOM 冲突。

### 4. 异常处理

- `ConfigError`：配置错误（Key 缺失），不重试，直接提示用户
- `APIError`：网络/服务端错误，重试 3 次，等待时间 2/4/6 秒递增
- 配置检查延迟到函数调用时，避免 import 阶段抛异常导致上层 try 接不住

## 开发中遇到的问题

### 问题 1：函数参数顺序错误

`save_plan(goal, content)` 和调用方 `save_plan(answer, goal)` 参数顺序不一致，导致文件名和内容错位。
**修复**：统一参数顺序，函数定义和调用一一对应。

### 问题 2：配置异常接不住

把 `raise ConfigError(...)` 放在 `llm_client.py` 模块顶层，`import` 时就抛异常，`main()` 里的 `try/except` 还没生效，接不住。
**修复**：改为延迟检查，把 Key 校验移进 `_get_client()` 函数，调用时才抛。

### 问题 3：非流式输出太慢

用户要等 30–60 秒才看到内容。
**修复**：改用流式输出，首字延迟降到 2–5 秒。

### 问题 4：Streamlit DOM 冲突

流式输出时浏览器报 `removeChild` / `insertBefore`，页面卡住。
**原因**：`st.write_stream` 动态创建/销毁 DOM 容器，和 Streamlit 的 rerun 机制冲突。
**修复**：改用 `st.empty()` 创建一个固定容器，每次 `markdown` 覆盖同一元素。

### 问题 5：追问框不显示

生成计划后，底部追问输入框没出现。
**原因**：`st.chat_input` 写在 `if/else` 分支里，点按钮那轮走的是 `if` 分支，Streamlit 不会自动重跑。
**修复**：把 `chat_input` 独立成一段，用 `if started` 控制显示。

### 问题 6：ARK API 首次调用超时

`timeout=30` 秒，首次请求容易超时。
**修复**：改为 90 秒，配合重试机制，第二次通常成功。

## 后续计划

- 加 token 计数与历史压缩，避免长对话超限
- 流式失败时保存已生成的部分内容
- 支持导出计划为 PDF
- 部署到 Streamlit Cloud，提供在线 demo
