@echo off
chcp 65001 > nul
rem ===========================================================================
rem  daily_task.bat - Windows 任务计划程序每日入口
rem  作用：抓取双平台岗位（shixiseng + niuke）-> 增量入库 -> 有变化才 git push
rem  由 DailyJobScrape 任务每天 23:00 调用；也可手动双击/命令行运行做验证
rem  日志：logs\daily_task.log（追加模式，控制台无输出）
rem ===========================================================================
setlocal

set "REPO=D:\agent-intern-roadmap"
set "LOGDIR=%REPO%\logs"
set "LOG=%LOGDIR%\daily_task.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%" 2>nul

cd /d "%REPO%"
set PYTHONIOENCODING=utf-8
if errorlevel 1 (
    echo [%date% %time%] 严重错误：无法进入工作目录 %REPO% >> "%LOG%"
    endlocal
    exit /b 1
)

echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo [%date% %time%] ===== 开始：每日抓取 + 推送 ===== >> "%LOG%"
echo [%date% %time%] 工作目录: %CD% >> "%LOG%"

rem --- 1) 抓取 + 清洗 + 增量入库（CLI 见 agent/scrapers/scheduler.py 文档头）---
python -m agent.scrapers.scheduler --once >> "%LOG%" 2>&1
set "SCRAPE_RC=%ERRORLEVEL%"
echo [%date% %time%] 抓取结束，退出码=%SCRAPE_RC% >> "%LOG%"

rem --- 2) 只暂存数据产物（其它改动一律不碰）---
git add rag/data/cleaned_jd.json rag/data/scraped_jd.txt >> "%LOG%" 2>&1

rem --- 3) 有暂存改动才提交 + 推送（git diff --staged --quiet：0=无变化，1=有变化）---
git diff --staged --quiet
if errorlevel 1 (
    echo [%date% %time%] 检测到数据变化，开始提交 >> "%LOG%"
    git commit -m "data: 自动抓取 %date%" >> "%LOG%" 2>&1
    git push >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo [%date% %time%] git push 失败：请检查网络与凭据管理器中保存的 GitHub 凭据 >> "%LOG%"
    ) else (
        echo [%date% %time%] git push 成功，Streamlit Cloud 将自动更新 >> "%LOG%"
    )
) else (
    echo [%date% %time%] 数据无变化，跳过提交与推送 >> "%LOG%"
)

echo [%date% %time%] ===== 全部完成（抓取退出码=%SCRAPE_RC%）===== >> "%LOG%"
endlocal
exit /b 0
