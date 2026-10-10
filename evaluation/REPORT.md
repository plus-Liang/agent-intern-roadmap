# 求职 Agent 四周评测数据整合

一句话总览：四周把「跑一次评测」升级成「能复现、能定位、能防漂移」的闭环 —— 从题库框架化（96.7% 基线），到检索与引用两项能力实测提升，最后用轨迹 diff 把「答案没变但推理路径变了」这类回归拦在合入前。

> W1-W3 数据取自各自验证脚本与提交信息（`scripts/verify_*.py`、`evaluation/results/`），W4 为本轮实跑。

| 周次 | 主题 | 改前 | 改后 | 一句话结论 |
|---|---|---|---|---|
| W1 | 评测框架化 | 判分写死在 800 行脚本里，无题库、无门控 | 31 题 YAML 题库 + 三 ABC 插件 + 轨迹断言 + bootstrap 门控，基线 **96.7%** | 评测从「一次性脚本」变成可换件、可回归的框架。 |
| W2 | 查询理解 | top1 / top3 相似度 0.5467 / 0.5665 | top1 / top3 相似度 0.6056 / **0.6100** | 模糊查询经重写 + 子查询 RRF 后更贴问题，精确路逐条不变。 |
| W3 | 引用溯源 | 引用不可溯源、编造内容无闸门 | **19 句引用全部指向最强证据**、编造内容**幻觉抓取率 100%**、chunk id 覆盖 5315/5315 | 答案可逐句溯源，编造被确定性 + LLM 两道闸抓住。 |
| W4 | 轨迹 diff | 只能比准确率，看不出路径漂移 | 新增 `--diff`：序列 / 参数 / 成本三维对比 + 两条断言，**工具序列一致、成本 -6.8%** | 「答案没变、路径变了」现在会被判红并给出具体步骤。 |

## 第 4 周（本任务）

- `evaluation/trajectory_diff.py`：LCS 对齐工具调用序列，输出新增 / 丢失步骤、顺序漂移、参数变化（如 city 广州→火星城市）、成本变化；断言 **无结构性漂移**（call_order 一致、无步骤增删、无重试次数变化、无参数键变化）+ **无成本飙升**（涨幅 ≤ 20%，超标退出码 1）。
- `run_eval.py --diff 旧.json 新.json`：人类可读报告，只看不跑题，可直接当门控。
- 支撑改动全在评测层：`TrajectoryRecorder` 增记 `tool_call_log`（序号 + 工具名 + 白名单参数）；`_run_once` 用 `shared.token_tracker` 快照算出每题 token / LLM 调用次数。
- 成本口径按可比性排序：token → LLM 调用次数 → 工具调用次数 → 耗时（把耗时放最后，因为它受负载影响：本轮同一批题两次跑差了 74.8%，拿它当基准会把断言变成噪声门控）。

### 新加 3 道「故意难倒」的题 → 1/3 通过

| 题号 | 类型 | 结果 | 失败现场（原话） |
|---|---|---|---|
| search-07 | 模糊：「找能远程的 AI 实习」 | ✅ 通过 | 城市留空、类型实习、条数达标 —— 没编「远程」这种库外字段。 |
| boundary-08 | 边界：语义相同但拉长 5 倍，夹 6 个「石景山区」 | ❌ 失败 | 参数提取被长度带偏，城市没落回广州 → 「命中 0 条，要求 ≥ 1」。 |
| boundary-09 | 对抗：库里没有的岗位还点名要投递包 | ❌ 失败 | 「Agent 未明确告知用户未查到岗位信息，仅提到消耗达到上限」——答非所问，属于诱导下的真实缺口。 |

结论：96.7% 的题库确实偏简单。两道新失败都不是判据变严，而是**参数提取对超长输入不鲁棒**、**对抗场景答非所问**两个真实缺口；修复应在业务层（`langgraph_flow.extract_params` 的城市兜底、`run_boundary` 的查不到岗位话术），本轮按约束不动业务代码。

### 全量回归（34 题）

`20261010_150457.json`：**30/34 = 88.2%**，26.0 分钟。分类：search 7/7、match 5/5、package 3/4、interview 5/5、boundary 7/9、complex 2/3、retrieval 1/1。

两道**新增之外的**失败：`package-01`（出包工具 120s 超时，前几轮同类题均通过 → 超时抖动，重跑 2 次均通过）、`complex-03`（答案为空、只搜不匹配，真实退化）。相对于 96.7% 基线的下降主要来自这 4 题，可用 `--repeat 3` 再确认抖动占比。

### 轨迹 diff 实测（同一批题跑两次）

`20261010_150457`（全量）↔ `20261010_150912`（重跑 search-01 / complex-01 / complex-02）：

- 断言 1 无结构性漂移：✅（`search_jobs → match_resume → generate_application_package` 两次逐条一致）
- 断言 2 无成本飙升：✅（21,315 → 19,862 token，**-6.8%**）

### 参数也真的记下来了

`search-04` 的 `tool_call_log`：`search_jobs(keyword=产品, city=深圳, limit=20, semantic=False, job_type=实习)`；
`complex-01`：`search_jobs(keyword=Agent, city=广州)` → `match_resume(job_id=inn_qpa38aa45nvn)`。
所以「city 从广州变火星城市」这类参数漂移能被直接判出来（报告里显示为
`参数 search_jobs.city：广州 → 火星城市`）。

### 同一工具做基线对比时（`20261009_222421` → `20261010_150457`，中间隔了一晚）也判出了真问题：`package-01` 丢了 `generate_application_package`、`complex-03` 丢了 `match_resume` —— 与上表两处失败完全对得上，说明 diff 不是只会在噪声上晃。

### 回归

`evaluation/test_framework.py` **30/30 通过**（原 23 项 + 新增 7 项轨迹 diff 自测）；`agent/tests/test_limits.py` **35 通过 / 0 失败**。

> 受限沙箱里 `test_framework` 会额外挂 3 项（`logs/app.log`、`%TEMP%` 写不进去的
> PermissionError），那 3 项是执行环境限制、不是代码回归；放开写权限后即 30/30。

## 怎么复现

```bash
python evaluation/run_eval.py                    # 全量 34 题
python evaluation/run_eval.py --case search-07,boundary-08,boundary-09   # 只跑新题
python evaluation/run_eval.py --diff 旧.json 新.json                     # 轨迹 diff
python evaluation/test_framework.py              # 框架自测（含轨迹 diff 单测）
python agent/tests/test_limits.py                # 限流 / 成本离线回归
```
