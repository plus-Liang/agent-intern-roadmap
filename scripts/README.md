# scripts/ 每日自动抓取与推送

| 文件 | 用途 |
| --- | --- |
| `daily_task.bat` | 任务实际执行体：抓取（shixiseng + niuke）→ 清洗入库 → 有变化才 `git push` |
| `daily_task.xml` | Windows 任务计划程序的任务定义（`DailyJobScrape`），可直接导入 |
| `README.md` | 本文档：导入 / 测试 / 验证 / 删除 |

抓取命令就是 scheduler 的官方用法 `python -m agent.scrapers.scheduler --once`（见 `agent/scrapers/scheduler.py` 文件头）。
所有输出追加到 `logs\daily_task.log`，scheduler 自己的结构化日志在 `logs\scheduler.log`。

---

## 1. 导入 XML 到任务计划程序

1. 先取自己的账户名：命令行执行 `whoami`，得到形如 `desktop-abc\ljh` 的字符串。
2. 用记事本打开 `D:\agent-intern-roadmap\scripts\daily_task.xml`，
   把 `<UserId>REPLACE_WITH_YOUR_USERNAME</UserId>` 换成上一步的结果，保存（保持 UTF-8）。
   > 必须用你自己的账户：任务若以 SYSTEM 身份运行，读不到你的 git 凭据，`git push` 必失败。
3. `Win + R` → 输入 `taskschd.msc` → 打开「任务计划程序」。
4. 右侧操作栏点「导入任务…」，选择 `D:\agent-intern-roadmap\scripts\daily_task.xml`。
5. 弹窗里确认名称为 `DailyJobScrape`；若提示需要账户/密码，点「更改用户或组」→ 高级 → 立即查找 → 选当前用户，确定后按提示输入 Windows 登录密码。
6. 导入完成后，在任务列表选中 `DailyJobScrape` → 右键「属性」，逐项核对：
   - 常规：已勾选「使用最高权限运行」（HighestAvailable）
   - 触发器：每日 23:00，已启用
   - 操作：`D:\agent-intern-roadmap\scripts\daily_task.bat`，起始于 `D:\agent-intern-roadmap`
   - 条件：**未**勾选「只有在计算机使用交流电源时才启动此任务」；**未**勾选「唤醒计算机运行此任务」
   - 设置：已勾选「如果错过计划开始时间，请尽快启动任务」（补跑）

### 不想改 XML 占位符？用命令行注册等价任务

```bat
schtasks /Create /TN "DailyJobScrape" /TR "D:\agent-intern-roadmap\scripts\daily_task.bat" /SC DAILY /ST 23:00 /RL HIGHEST /F
```

注意：`schtasks` 方式默认不带「错过补跑」，注册后仍需到 GUI 的属性 → 设置里勾选「如果错过计划开始时间，请尽快启动任务」。

---

## 2. 手动测试 bat 脚本

```bat
cd /d D:\agent-intern-roadmap
scripts\daily_task.bat
```

- 这是**真实抓取**（双平台，约 20~40 分钟），且结束后会真的 `git push`。
- 控制台不输出内容，全部写进日志，另开窗口看进度：

```bat
type D:\agent-intern-roadmap\logs\daily_task.log
```

- 只想快速验证链路、不联网、不打满额度（环境变量只影响当次进程）：

```bat
set SCHEDULER_USE_MOCK=1
python -m agent.scrapers.scheduler --once
set SCHEDULER_FAST_MODE=1
python -m agent.scrapers.scheduler --once
```

- 回看上一次运行结果：`python -m agent.scrapers.scheduler --status`

---

## 3. 验证任务是否生效

- GUI：任务计划程序 → 选中 `DailyJobScrape` → 下方「历史记录」标签看每次启动/结束与返回码。
- 命令行查询：

```bat
schtasks /Query /TN "DailyJobScrape" /V /FO LIST
```

- 立刻触发一次（验证端到端，不必等到 23:00）：

```bat
schtasks /Run /TN "DailyJobScrape"
```

- 日志位置：
  - `D:\agent-intern-roadmap\logs\daily_task.log`（bat 的开始/结束时间、抓取退出码、push 结果）
  - `D:\agent-intern-roadmap\logs\scheduler.log`（抓取明细，每次运行一行 JSON 摘要）
- 推送是否成功：`git log --oneline -3`、`git status`；GitHub 上看到 `data: 自动抓取 …` 提交即成功，
  Streamlit Cloud 会跟随该 push 自动重建。

---

## 4. 禁用 / 删除任务

```bat
schtasks /Change /TN "DailyJobScrape" /DISABLE    :: 禁用（保留定义）
schtasks /Change /TN "DailyJobScrape" /ENABLE     :: 重新启用
schtasks /Delete /TN "DailyJobScrape" /F          :: 删除
```

GUI 方式：任务计划程序库 → 右键 `DailyJobScrape` → 「禁用」或「删除」。

---

## 注意事项

- **错过 23:00 会补跑**：任务不唤醒电脑，睡眠/关机期间不会启动；`StartWhenAvailable=true` 保证恢复可用后尽快执行一次。
- **需要账户已登录**：XML 用 `InteractiveToken`，任务只在你登录状态下运行——这样才用得到 git 凭据。
- **git 凭据要先存好**：至少手动成功 `git push` 一次，让 Windows 凭据管理器记住 GitHub 凭据；否则夜里推送失败（失败会写日志，不会静默）。
- **无变化不提交**：只有 `rag/data/cleaned_jd.json` 或 `rag/data/scraped_jd.txt` 有改动时才 `git commit` + `git push`，其它文件一律不暂存。
- **日志会一直追加**：长期运行可定期清理或自行截断 `logs\daily_task.log`。
