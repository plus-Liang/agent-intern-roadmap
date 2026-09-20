"""
定时抓取调度器：抓取 → 清洗 → 增量入库，一条链路跑完。

为什么需要它：
    之前是「手动抓取 + 每次全量重建向量库」，既慢又烧 embedding 额度。
    这里把三件事串起来，并且入库走 add_chunks_incremental：
    已经入库、内容没变的 chunk 直接跳过，只对新增/改动的 chunk 调 embedding。

依赖说明（不装新依赖）：
    本机没装 schedule 库，所以定时部分**默认用标准库**：
    schedule 可用时用 schedule.every().day.at(HH:MM)；
    不可用时用标准库 sched + time.monotonic 的绝对时间推算（不受系统时间抖动影响）。
    两条路径的行为一致：挂起后每天在 SCHEDULER_TIME 跑一次。

用法：
    python -m agent.scrapers.scheduler --once            # 立刻跑一次
    python -m agent.scrapers.scheduler --daily           # 挂起，每天跑一次
    python -m agent.scrapers.scheduler --status          # 看上次运行时间和结果
    python -m agent.scrapers.scheduler --once --dry-run  # 不抓网络，只验证链路与日志
    python -m agent.scrapers.scheduler --selftest        # 离线自测（假数据 + 临时向量库）

日志：logs/scheduler.log（同时打到控制台），每次运行一行 JSON 摘要。
状态：--status 读的是一份小的 JSON 状态文件，位置见 _state_path()，
      默认放系统临时目录，不往仓库里写文件。
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# 允许 `python agent/scrapers/scheduler.py` 直接跑：把仓库根目录放进 sys.path，
# 否则直接执行脚本时拿不到 agent / rag / shared 这些顶层包。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_REPO_ROOT))

from shared.config import DATA_DIR, LOG_DIR, ROOT_DIR        # noqa: E402
from agent import storage                                    # noqa: E402
from agent import user_profile                               # noqa: E402

# ---------------------------------------------------------------------------
# 常量与默认配置
# ---------------------------------------------------------------------------
LOG_PATH = LOG_DIR / "scheduler.log"

# 默认运行时刻（24 小时制 HH:MM），可用环境变量 SCHEDULER_TIME 覆盖
DEFAULT_RUN_TIME = os.getenv("SCHEDULER_TIME", "09:00")

# 抓取规模：默认小步跑，避免第一次定时任务就把站点和额度打满
DEFAULT_MAX_PAGES = int(os.getenv("SCHEDULER_MAX_PAGES", "2"))
DEFAULT_LIMIT_PER_KEYWORD = int(os.getenv("SCHEDULER_LIMIT_PER_KEYWORD", "40"))
DEFAULT_LIMIT_TOTAL = int(os.getenv("SCHEDULER_LIMIT_TOTAL", "100"))

# 清洗参数（与 rag/quality/cleaner.py 的默认口径一致）
DEFAULT_MAX_AGE_DAYS = int(os.getenv("SCHEDULER_MAX_AGE_DAYS", "60"))
DEFAULT_MIN_DESC_LEN = int(os.getenv("SCHEDULER_MIN_DESC_LEN", "200"))

# 一个关键词都没有时的兜底关键词（与项目里实际在用的那组保持一致）
FALLBACK_KEYWORDS = ["Agent", "大模型", "LLM", "RAG", "智能体"]

CITY_SPLIT_RE = re.compile(r"[,，、;；/\s]+")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def setup_logger(log_path=None, verbose: bool = True) -> logging.Logger:
    """配置 scheduler 专用 logger：写 logs/scheduler.log + 控制台。

    重复调用不会叠加 handler（会先清掉旧的），免得一行日志打两遍。
    """
    logger = logging.getLogger("agent.scrapers.scheduler")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # noqa: BLE001
            pass

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    target = Path(log_path) if log_path else LOG_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:                       # 日志目录不可写也不该让任务挂掉
        print(f"[scheduler] 无法写入日志文件 {target}：{exc}")

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


# ---------------------------------------------------------------------------
# 状态文件（--status 读它）
# ---------------------------------------------------------------------------
def _state_path(state_file=None) -> Path:
    """状态文件位置。

    优先显式传入 / 环境变量 SCHEDULER_STATE_FILE；
    否则放系统临时目录下按仓库路径哈希区分的子目录里——
    状态属于运行期产物，不该往仓库里塞文件。
    """
    if state_file:
        return Path(state_file)
    env_path = os.getenv("SCHEDULER_STATE_FILE")
    if env_path:
        return Path(env_path)
    import hashlib
    import tempfile

    tag = hashlib.sha1(str(ROOT_DIR).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"agent-intern-roadmap-{tag}" / "scheduler_state.json"


def load_state(state_file=None) -> dict:
    """读上次运行状态；没有/坏了都返回 {}（--status 不会因此报错）"""
    path = _state_path(state_file)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: dict, state_file=None) -> Path:
    """写状态文件（先写临时文件再替换，避免半截 JSON）"""
    path = _state_path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def _record_state(result: dict, state_file=None, logger: logging.Logger = None) -> None:
    """把本次运行结果落进状态文件（失败只记日志，不影响任务本身）"""
    state = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ok": bool(result.get("ok")),
        "keywords": result.get("keywords", []),
        "city": result.get("city", ""),
        "scraped": result.get("scraped", 0),
        "cleaned": result.get("cleaned", 0),
        "chunks": result.get("chunks", 0),
        "incremental": result.get("incremental", {}),
        "out_dir": result.get("out_dir", ""),
        "converted": result.get("converted", 0),
        "error": result.get("error", ""),
    }
    try:
        save_state(state, state_file)
    except OSError as exc:
        if logger:
            logger.warning("状态文件写入失败：%s", exc)


# ---------------------------------------------------------------------------
# 配置：关键词 / 城市
# ---------------------------------------------------------------------------
def _split_list(value) -> list[str]:
    """把 "广州,深圳" / ["广州","深圳"] 这类值拆成去空去重的列表"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [str(v) for v in value]
    else:
        parts = CITY_SPLIT_RE.split(str(value))
    out, seen = [], set()
    for part in parts:
        text = part.strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def resolve_config(profile: dict = None, logger: logging.Logger = None,
                   allow_fallback: bool = True) -> dict:
    """决定这次抓什么：关键词列表 + 城市。

    优先级：环境变量（SCHEDULER_KEYWORDS / SCHEDULER_CITY）> user_profile.json > 兜底。

    关键词一个都没有时用 FALLBACK_KEYWORDS 兜底并告警——定时任务最怕"静默
    什么都不做"，宁可打日志说清用了兜底关键词。allow_fallback=False（dry-run）
    时不打这条告警也不兜底：那次根本不抓网络，提示"用了兜底关键词"只会误导。
    """
    profile = profile if isinstance(profile, dict) else user_profile.load_profile()

    keywords = _split_list(
        os.getenv("SCHEDULER_KEYWORDS") or os.getenv("SCRAPE_KEYWORDS")
    ) or _split_list(profile.get("target_keywords"))
    if not keywords:
        if allow_fallback:
            keywords = list(FALLBACK_KEYWORDS)
            if logger:
                logger.warning(
                    "没配置关键词（SCHEDULER_KEYWORDS / user_profile.target_keywords 都是空），"
                    "本次用兜底关键词：%s", "、".join(keywords),
                )

    cities = _split_list(
        os.getenv("SCHEDULER_CITY") or os.getenv("SCRAPE_CITY")
    ) or _split_list(profile.get("target_cities"))
    city = cities[0] if cities else "广州"
    if len(cities) > 1 and logger:
        logger.info(
            "配置了多个城市 %s，本次先用第一个「%s」抓取（多城市需要分别跑，后续可拆分）",
            "、".join(cities), city,
        )

    return {
        "keywords": keywords,
        "cities": cities,
        "city": city,
        "max_pages": DEFAULT_MAX_PAGES,
        "limit_per_keyword": DEFAULT_LIMIT_PER_KEYWORD,
        "limit_total": DEFAULT_LIMIT_TOTAL,
    }


# ---------------------------------------------------------------------------
# 抓取
# ---------------------------------------------------------------------------
def scrape_jobs(keywords: list[str], city=None, max_pages=DEFAULT_MAX_PAGES,
                limit_per_keyword=DEFAULT_LIMIT_PER_KEYWORD,
                limit_total=DEFAULT_LIMIT_TOTAL,
                headless: bool = True, scraper=None) -> list:
    """调 agent.scrapers.shixiseng.search_multi_keywords 抓取岗位。

    scraper: 参数用于注入假实现（离线自测），不传就用真实的
              search_multi_keywords（async 函数）。两种都能接：
              async 的直接 await，同步返回列表的包一层 coroutine。
    headless 默认 True：定时任务在后台跑，不该弹浏览器窗口。
    """
    if scraper is None:
        from agent.scrapers.shixiseng import search_multi_keywords as scraper

    async def _run():
        result = scraper(
            keywords,
            city=city,
            max_pages_per_keyword=max_pages,
            limit_total=limit_total,
            limit_per_keyword=limit_per_keyword,
            headless=headless,
            fetch_detail=True,
        )
        if inspect.isawaitable(result):
            return await result
        return result

    return asyncio.run(_run())


def _job_to_dict(job) -> dict:
    """Job dataclass → 清洗模块要的 dict（字段与 rag/quality/cleaner.py 对齐）"""
    if isinstance(job, dict):
        return job
    return {
        "platform": getattr(job, "platform", "") or "",
        "job_id": getattr(job, "job_id", "") or "",
        "title": getattr(job, "title", "") or "",
        "company": getattr(job, "company", "") or "",
        "city": getattr(job, "city", "") or "",
        "salary": getattr(job, "salary", "") or "",
        "url": getattr(job, "url", "") or "",
        "description": getattr(job, "description", "") or "",
        "publish_date": getattr(job, "publish_date", "") or "",
    }


# ---------------------------------------------------------------------------
# 清洗 → chunk
# ---------------------------------------------------------------------------
def _load_cleaner():
    """导入 rag/quality/cleaner.py。

    rag/quality 没有 __init__.py（不是包），所以这里把该目录加进 sys.path 后
    直接按模块名导入，避免为了加个 __init__.py 去动 rag/ 的结构。
    """
    import importlib
    import sys

    quality_dir = str(ROOT_DIR / "rag" / "quality")
    if quality_dir not in sys.path:
        sys.path.insert(0, quality_dir)
    return importlib.import_module("cleaner")


def clean_jobs(jobs: list[dict], city: str = "广州",
               max_age_days: int = DEFAULT_MAX_AGE_DAYS,
               min_desc_len: int = DEFAULT_MIN_DESC_LEN) -> dict:
    """调 cleaner.clean_jobs 清洗（相关性/城市/时效/正文长度四道过滤）"""
    cleaner = _load_cleaner()
    return cleaner.clean_jobs(
        [_job_to_dict(job) for job in jobs],
        city=city,
        max_age_days=max_age_days,
        min_desc_len=min_desc_len,
    )


# 单条 JD 的纯文本排版（与 rag/data_converter.py 的 _format_job 保持一致）
SEPARATOR = "-" * 88


def _format_job_text(index: int, job: dict) -> str:
    """把一条清洗后的 Job 排成 data_converter 的文本块格式"""
    company = (job.get("company") or "").strip() or "未知"
    title = (job.get("title") or "").strip() or "未知"
    city = (job.get("city") or "").strip() or "未知"
    salary = (job.get("salary") or "").strip()
    url = (job.get("url") or "").strip()
    publish_date = (job.get("publish_date") or "").strip()
    description = (job.get("description") or "").strip()
    return "\n".join((
        f"【{index}】公司：{company}",
        f"岗位：{title}",
        f"城市：{city} ｜ 薪资：{salary}",
        f"来源链接：{url}",
        f"发布时间：{publish_date}",
        SEPARATOR,
        description,
        SEPARATOR,
    ))


def jobs_to_jd_records(jobs: list[dict]) -> list[dict]:
    """清洗后的 Job dict 列表 → [{"company", "title", "city", "content"}]

    灌进 splitter 之前先补上 loader 会加的头部元信息（公司/岗位/城市/薪资/
    链接/发布时间），这样 chunk 正文里带来源信息、检索命中后能定位到岗位，
    也让 ID 的 company/title 字段有值。
    """
    records = []
    for index, job in enumerate(jobs or [], start=1):
        if not isinstance(job, dict):
            continue
        records.append({
            "company": (job.get("company") or "").strip() or "未知",
            "title": (job.get("title") or "").strip() or "未知",
            "city": (job.get("city") or "").strip() or "未知",
            "content": _format_job_text(index, job),
        })
    return records


def build_chunks(jobs: list[dict], chunk_size: int = 600) -> list[dict]:
    """清洗后的 Job dict 列表 → chunk 列表（走项目原有的 rag.splitter）"""
    from rag.splitter import split_jds

    records = jobs_to_jd_records(jobs)
    if not records:
        return []
    return split_jds(records, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# 落盘（可选）：刷新 RAG 语料
# ---------------------------------------------------------------------------
def write_cleaned(jobs: list[dict], out_dir=None) -> Path:
    """把清洗结果写成 JSON（默认 rag/data/cleaned_jd.json）"""
    target_dir = Path(out_dir) if out_dir else DATA_DIR
    target = target_dir / "cleaned_jd.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def write_rag_text(jobs: list[dict], out_dir=None) -> tuple[Path, int]:
    """把清洗结果转成 RAG 语料纯文本（默认 rag/data/scraped_jd.txt）。

    这么做的原因：agent/tools/job_search.py 和 RAG 检索读的都是
    scraped_jd.txt，定时抓完不刷新它，新岗位在 Agent 里就搜不到。
    优先复用 rag/data_converter.json_to_rag_format（与 data_converter CLI
    产出完全一致）；它不可用时用本地兜底（排版逐字符对齐）。
    """
    target_dir = Path(out_dir) if out_dir else DATA_DIR
    target = target_dir / "scraped_jd.txt"
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        # data_converter 的入口是「读 JSON 文件」，而这里数据在内存里；
        # 它有 _load_jobs/_format_job 两个私有函数，直接复用比落一个临时 JSON 干净。
        from rag.data_converter import _format_job
    except Exception:                           # noqa: BLE001 - 导入失败就用本地实现
        return target, _write_rag_text_local(jobs, target)

    blocks = [_format_job(i, job) for i, job in enumerate(jobs or [], start=1)]
    text = "\n\n".join(blocks)
    if text:
        text += "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return target, len(blocks)


def _write_rag_text_local(jobs: list[dict], target: Path) -> int:
    """data_converter 不可用时的兜底实现（排版与它完全一致）"""
    blocks = [_format_job_text(i, job) for i, job in enumerate(jobs or [], start=1)]
    text = "\n\n".join(blocks)
    if text:
        text += "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return len(blocks)


# ---------------------------------------------------------------------------
# 主任务
# ---------------------------------------------------------------------------
def run_daily_job(config: dict = None, out_dir=None, state_file=None,
                  logger: logging.Logger = None, headless: bool = True,
                  scraper=None, dry_run: bool = False) -> dict:
    """跑一次完整任务：抓取 → 清洗 → 生成 chunk → 增量入库 →（可选）落盘。

    返回摘要 dict（同时写进日志和状态文件）。异常一律在这里兜住并记进日志：
    定时任务里抛异常=任务静默死掉，比失败更糟。

    参数：
        config:      resolve_config() 的结果；不传就现取
        out_dir:     落盘目录；默认写 rag/data/（保持 Agent 能搜到新岗位）
        state_file:  状态文件路径（--status 读的就是它）
        dry_run:     True 时跳过真实抓取（只验证链路），并且不落盘
    """
    log = logger or setup_logger()
    config = config or resolve_config(logger=log, allow_fallback=not dry_run)
    keywords = config.get("keywords") or []
    city = config.get("city")
    started = time.time()

    log.info(
        "===== 定时抓取开始：关键词=%s 城市=%s dry_run=%s =====",
        "、".join(keywords), city or "不限", dry_run,
    )

    result = {
        "ok": False,
        "keywords": keywords,
        "city": city or "",
        "scraped": 0,
        "cleaned": 0,
        "chunks": 0,
        "incremental": {"added": 0, "updated": 0, "skipped": 0},
        "out_dir": str(out_dir) if out_dir else "",
        "converted": 0,
        "error": "",
    }

    try:
        # 1) 抓取
        if dry_run:
            jobs = []
            log.info("[1/4] dry-run：跳过真实抓取（不访问实习僧）")
        else:
            jobs = scrape_jobs(
                keywords,
                city=city,
                max_pages=config.get("max_pages", DEFAULT_MAX_PAGES),
                limit_per_keyword=config.get("limit_per_keyword", DEFAULT_LIMIT_PER_KEYWORD),
                limit_total=config.get("limit_total", DEFAULT_LIMIT_TOTAL),
                headless=headless,
                scraper=scraper,
            )
            log.info("[1/4] 抓取完成：%d 条", len(jobs))
        result["scraped"] = len(jobs)

        # 2) 清洗
        cleaned = clean_jobs(
            jobs,
            city=city or "广州",
            max_age_days=config.get("max_age_days", DEFAULT_MAX_AGE_DAYS),
            min_desc_len=config.get("min_desc_len", DEFAULT_MIN_DESC_LEN),
        )
        kept = cleaned.get("jobs", [])
        result["cleaned"] = len(kept)
        removed = cleaned.get("removed", {})
        log.info(
            "[2/4] 清洗完成：输入 %d 条 → 保留 %d 条（移除 相关性%d/城市%d/时效%d/正文太短%d）",
            cleaned.get("total_input", len(jobs)), len(kept),
            removed.get("relevance", 0), removed.get("city", 0),
            removed.get("age", 0), removed.get("length", 0),
        )

        # 3) 生成 chunk
        chunks = build_chunks(kept)
        result["chunks"] = len(chunks)
        log.info("[3/4] 切块完成：%d 个 chunk", len(chunks))

        # 4) 增量入库（已存在且没变的不重算向量）
        if chunks:
            from rag.vector_store import add_chunks_incremental, get_collection

            stats = add_chunks_incremental(chunks, get_collection())
        else:
            stats = {"added": 0, "updated": 0, "skipped": 0}
            log.info("[4/4] 没有可入库的 chunk（本次抓取无有效岗位）")
        result["incremental"] = stats
        log.info(
            "[4/4] 增量入库完成：新增 %d，更新 %d，跳过 %d（跳过的没花 embedding 额度）",
            stats.get("added", 0), stats.get("updated", 0), stats.get("skipped", 0),
        )

        # 5) 落盘：刷新 cleaned_jd.json 与 RAG 语料文本
        if dry_run:
            log.info("[落盘] dry-run：跳过写文件（cleaned_jd.json / scraped_jd.txt 保持原样）")
        else:
            json_path = write_cleaned(kept, out_dir)
            txt_path, count = write_rag_text(kept, out_dir)
            result["converted"] = count
            log.info("[落盘] 已写入 %s 与 %s（%d 条）", json_path, txt_path, count)

        result["ok"] = True
    except Exception as exc:                     # noqa: BLE001 - 定时任务必须活下来
        result["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("任务失败：%s", result["error"])

    result["elapsed"] = round(time.time() - started, 1)
    log.info(
        "===== 定时抓取结束：%s，耗时 %.1fs（抓取 %d / 清洗后 %d / chunk %d / 入库 +%d ~%d =%d）=====",
        "成功" if result["ok"] else f"失败（{result['error']}）", result["elapsed"],
        result["scraped"], result["cleaned"], result["chunks"],
        result["incremental"].get("added", 0), result["incremental"].get("updated", 0),
        result["incremental"].get("skipped", 0),
    )
    _record_state(result, state_file=state_file, logger=log)
    return result


# ---------------------------------------------------------------------------
# --status
# ---------------------------------------------------------------------------
def format_status(state: dict = None, state_file=None) -> str:
    """把状态文件内容渲染成给终端看的多行文本。

    state_file 传进来时报告**实际用的那个路径**——否则用户用 --state-file
    指定了别处，回显里却永远是默认路径，排查时会以为状态没写进去。
    """
    state = state if isinstance(state, dict) else load_state(state_file)
    if not state:
        return ("还没有运行记录。\n"
                "先跑一次：python -m agent.scrapers.scheduler --once")

    inc = state.get("incremental") or {}
    lines = [
        f"上次运行时间：{state.get('last_run') or '（未知）'}",
        f"结果：{'✅ 成功' if state.get('ok') else '❌ 失败'}"
        + (f"　错误：{state['error']}" if state.get("error") else ""),
        f"关键词：{'、'.join(state.get('keywords') or []) or '（未记录）'}",
        f"城市：{state.get('city') or '（不限）'}",
        f"抓取：{state.get('scraped', 0)} 条　清洗后：{state.get('cleaned', 0)} 条　"
        f"chunk：{state.get('chunks', 0)} 个",
        f"增量入库：新增 {inc.get('added', 0)}，更新 {inc.get('updated', 0)}，"
        f"跳过 {inc.get('skipped', 0)}",
    ]
    if state.get("out_dir"):
        lines.append(f"输出目录：{state['out_dir']}")
    lines.append(f"状态文件：{_state_path(state_file)}")
    lines.append(f"日志文件：{LOG_PATH}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --daily 调度
# ---------------------------------------------------------------------------
def _parse_run_time(text: str) -> tuple[int, int]:
    """解析 "HH:MM"；非法值回退到 DEFAULT_RUN_TIME（并提示）"""
    raw = str(text or "").strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    fallback = re.fullmatch(r"(\d{1,2}):(\d{2})", DEFAULT_RUN_TIME.strip())
    hour, minute = (int(fallback.group(1)), int(fallback.group(2))) if fallback else (9, 0)
    print(f"[scheduler] 运行时刻 {text!r} 不合法，改用 {hour:02d}:{minute:02d}")
    return hour, minute


def _next_run_delay(hour: int, minute: int, now: datetime = None) -> float:
    """算距离下一个 HH:MM 还有多少秒（今天已过则顺延到明天）"""
    now = now or datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def loop_daily(run_time: str = DEFAULT_RUN_TIME, logger: logging.Logger = None,
               config: dict = None, state_file=None, out_dir=None,
               max_runs: int = 0, headless: bool = True) -> int:
    """挂起，每天在 run_time 跑一次。

    schedule 库可用就用它；不可用（本机现状）用标准库 sched：
    sched 按 monotonic 时钟调度，不受系统时间被改动的影响。
    max_runs > 0 时跑够次数就退出（自测用）。
    """
    log = logger or setup_logger()
    hour, minute = _parse_run_time(run_time)
    log.info("已挂起：每天 %02d:%02d 执行一次（Ctrl+C 退出）", hour, minute)

    runs = 0
    try:
        try:
            import schedule as _schedule          # 可选：装了就用它
        except ImportError:
            _schedule = None

        if _schedule is not None:
            log.info("定时后端：schedule 库")
            _schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(
                run_daily_job, config=config, out_dir=out_dir,
                state_file=state_file, logger=log, headless=headless,
            )
            while not max_runs or runs < max_runs:
                _schedule.run_pending()
                time.sleep(20)
        else:
            import sched as _sched

            log.info("定时后端：标准库 sched（未安装 schedule 库）")
            scheduler = _sched.scheduler(time.monotonic, time.sleep)

            def _tick():
                nonlocal runs
                run_daily_job(config=config, out_dir=out_dir, state_file=state_file,
                              logger=log, headless=headless)
                runs += 1
                if not max_runs or runs < max_runs:
                    scheduler.enter(_next_run_delay(hour, minute), 1, _tick)

            scheduler.enter(_next_run_delay(hour, minute), 1, _tick)
            scheduler.run()
    except KeyboardInterrupt:
        log.info("收到 Ctrl+C，定时任务退出（已跑 %d 次）", runs)
        return 0
    return 0


# ---------------------------------------------------------------------------
# 离线自测（--selftest）：假抓取 + 临时向量库 + 临时输出目录
# ---------------------------------------------------------------------------
def _make_workdir(prefix: str) -> Path:
    """在系统临时目录下建一个**直属**的临时目录并返回它。

    不用 tempfile.mkdtemp：某些受限环境（带沙箱的执行器）只放开系统临时目录
    本身，不允许在其中再建随机子目录，mkdtemp 出来的目录不可写，
    sqlite/chroma 会直接打不开。自己在 tempdir 下拼唯一名字兼容性更好。
    """
    import tempfile
    import uuid

    path = Path(tempfile.gettempdir()) / f"{prefix}{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _selftest() -> int:
    """不需要网络、不碰真实库的自测：

    - 关键词解析（环境变量 / 画像 / 兜底）
    - run_daily_job 全链路（用假 scraper；embedding 被替换成确定性假向量）
    - 状态文件写入 + format_status 渲染
    全程只用自己的临时目录，不写 logs/、不写 rag/data/、不动 chroma_db/。
    """
    import shutil
    import tempfile

    work = _make_workdir("scheduler_selftest_")
    logger = setup_logger(log_path=work / "scheduler.log", verbose=True)
    checks = []

    def check(label, fn, detail=""):
        try:
            out = fn()
            checks.append(True)
            print(f"  [PASS] {label}" + (f"（{out or detail}）" if (out or detail) else ""))
        except Exception as exc:                 # noqa: BLE001
            checks.append(False)
            print(f"  [FAIL] {label} → {type(exc).__name__}: {exc}")

    def _fail(message: str):
        """在条件表达式里抛断言（check() 会把它记成失败项）"""
        raise AssertionError(message)

    # 自测数据的两条岗位分别用来验证「保留」与「被城市过滤」。
    # ⚠️ 正文必须 >= cleaner 的 min_desc_len（默认 200 字），否则会被
    #    「正文太短」过滤掉——第一版自测就踩了这个坑，这里故意写足长度。
    fake_jobs = [
        {   # 应当保留：Agent 关键词 + 广州 + 正文够长
            "platform": "fake", "job_id": "fake_1", "title": "Agent 开发实习生",
            "company": "自测科技", "city": "广州", "salary": "200-300/天",
            "url": "https://example.com/intern/fake_1",
            "publish_date": date.today().strftime("%Y-%m-%d"),
            "description": (
                "【岗位职责】\n"
                "1. 参与公司 Agent 平台的开发与调试，负责 RAG 检索链路的实现与优化，"
                "把业务文档接进知识库并持续改善召回效果。\n"
                "2. 与产品、算法同学一起把大模型能力落到具体业务场景，"
                "包括意图识别、工具调用与结果评估。\n"
                "3. 编写必要的单元测试与评估脚本，保证上线质量，"
                "并把踩过的坑沉淀成文档。\n"
                "【任职要求】\n"
                "1. 熟悉 Python，了解 LLM 应用开发常见范式，能独立读英文技术文档。\n"
                "2. 有 RAG、向量数据库或 Agent 框架（如 LangChain）实践经验者优先。\n"
                "3. 每周可到岗 4 天以上，实习期不少于 3 个月。"
            ),
        },
        {   # 应当被城市过滤掉（正文同样写足长度，确保是"城市"这一条把它筛掉的）
            "platform": "fake", "job_id": "fake_2", "title": "大模型算法实习生",
            "company": "北京科技", "city": "北京", "salary": "300-400/天",
            "url": "https://example.com/intern/fake_2",
            "publish_date": date.today().strftime("%Y-%m-%d"),
            "description": (
                "【岗位职责】\n"
                "1. 参与大模型训练与推理优化相关工作，跟进业界最新进展。\n"
                "2. 负责数据处理、评测集构建与实验记录整理。\n"
                "【任职要求】\n"
                "1. 熟悉 PyTorch 与分布式训练，了解 Transformer 结构。\n"
                "2. 有 LLM 微调或推理加速经验者优先，能读英文论文。"
            ),
        },
    ]

    # ---- 隔离准备：真实向量库路径先换掉，再跑任何会入库的代码 ----
    import rag.vector_store as vs

    real_db_path = vs.DB_PATH
    real_embed = vs.embed_texts
    real_chroma = Path(real_db_path)
    before = ({p: (p.stat().st_size, p.stat().st_mtime) for p in real_chroma.rglob("*")
               if p.is_file()} if real_chroma.is_dir() else {})
    work_db = work / "chroma"
    vs.DB_PATH = str(work_db)                 # 关键：get_collection() 读的是这个模块属性

    # embedding 换成确定性假向量：自测不调 API、不消耗额度
    def fake_embed_texts(texts):
        return [[float(len(t) % 7), 1.0, 0.5] for t in texts]

    vs.embed_texts = fake_embed_texts

    try:
        # ---- 1. 关键词解析 ----
        profile_path = work / "user_profile.json"
        profile_path.write_text(
            json.dumps({"target_cities": ["广州"], "target_keywords": ["Agent", "RAG"]},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        saved_profile_path = user_profile.PROFILE_PATH
        user_profile.PROFILE_PATH = profile_path
        try:
            cfg = resolve_config(logger=logger)
            check("1. 关键词/城市从画像读取",
                  lambda: (
                      f"keywords={cfg['keywords']} city={cfg['city']}"
                      if cfg["keywords"] == ["Agent", "RAG"] and cfg["city"] == "广州"
                      else _fail(f"解析结果不对：{cfg}")
                  ))

            os.environ["SCHEDULER_KEYWORDS"] = "智能体"
            cfg_env = resolve_config(logger=logger)
            check("2. 环境变量优先于画像",
                  lambda: ("SCHEDULER_KEYWORDS=智能体 生效"
                           if cfg_env["keywords"] == ["智能体"] else _fail(str(cfg_env))),
                  detail="")
        finally:
            os.environ.pop("SCHEDULER_KEYWORDS", None)
            user_profile.PROFILE_PATH = saved_profile_path

        # ---- 2. 全链路（假 scraper，真 chroma，只是换了 DB_PATH） ----
        import chromadb
        tmp_client = chromadb.PersistentClient(path=str(work_db))
        tmp_col = tmp_client.get_or_create_collection(
            name="jd_chunks", metadata={"hnsw:space": "cosine"}
        )

        cfg = {"keywords": ["Agent"], "city": "广州", "max_pages": 1,
               "limit_per_keyword": 5, "limit_total": 5}
        out_dir = work / "rag_out"
        state_file = work / "state.json"

        assert str(vs.DB_PATH) == str(work_db), "临时向量库未生效，自测中止"
        first = run_daily_job(
            config=cfg, out_dir=out_dir, state_file=state_file, logger=logger,
            scraper=lambda *a, **k: fake_jobs,
        )

        check("3. 全链路 run_daily_job 成功",
              lambda: (
                  f"抓取 {first['scraped']} → 清洗 {first['cleaned']} → "
                  f"chunk {first['chunks']} → 入库 +{first['incremental']['added']}"
                  if first["ok"] and first["scraped"] == 2 and first["cleaned"] == 1
                  and first["chunks"] > 0 and first["incremental"]["added"] == first["chunks"]
                  else _fail(str(first))
              ))

        check("4. 落盘产物存在（cleaned_jd.json / scraped_jd.txt）",
              lambda: ("两个文件都写了（%d / %d 字节）" % (
                  (out_dir / "cleaned_jd.json").stat().st_size,
                  (out_dir / "scraped_jd.txt").stat().st_size,
              ) if (out_dir / "cleaned_jd.json").is_file()
                  and (out_dir / "scraped_jd.txt").is_file()
                  else _fail("落盘文件缺失")))

        # 增量语义：同一批 chunk 再跑两遍
        chunks = build_chunks(clean_jobs(fake_jobs, city="广州")["jobs"])
        stats1 = vs.add_chunks_incremental(chunks, tmp_col)
        stats2 = vs.add_chunks_incremental(chunks, tmp_col)
        check("5. 同一批 chunk 第二次全部 skipped",
              lambda: (f"第二次 {stats2}（共 {len(chunks)} 个 chunk）"
                       if stats2 == {"added": 0, "updated": 0, "skipped": len(chunks)}
                       else _fail(f"第一次 {stats1} / 第二次 {stats2}")))

        check("6. 状态文件写入 + --status 渲染",
              lambda: (format_status(load_state(state_file), state_file).splitlines()[0]
                       if load_state(state_file).get("ok") is True
                       else _fail("状态文件内容不对")))

        check("7. 日志写进了 scheduler.log",
              lambda: (f"scheduler.log {len((work / 'scheduler.log').read_text(encoding='utf-8').splitlines())} 行"
                       if (work / "scheduler.log").is_file()
                       and "定时抓取结束" in (work / "scheduler.log").read_text(encoding="utf-8")
                       else _fail("日志文件缺失或没有本次运行记录")))
    finally:
        vs.embed_texts = real_embed
        vs.DB_PATH = real_db_path

        if before:
            def _real_db_untouched():
                after = {p: (p.stat().st_size, p.stat().st_mtime)
                         for p in real_chroma.rglob("*") if p.is_file()}
                if after != before:
                    raise AssertionError("真实向量库被改动了")
                return f"{len(before)} 个文件 size/mtime 未变"

            check("8. 真实 chroma_db/ 未被写入", _real_db_untouched)

        if all(checks) and checks:
            shutil.rmtree(work, ignore_errors=True)
            print(f"\n临时目录已清理：{work}")
        else:
            print(f"\n有失败项，保留临时目录便于排查：{work}")

    passed = checks.count(True)
    print(f"自测结果：{passed}/{len(checks)} 通过")
    return 0 if checks and passed == len(checks) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.scrapers.scheduler",
        description="定时抓取 → 清洗 → 增量入库",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="立刻跑一次（抓取→清洗→增量入库）")
    mode.add_argument("--daily", action="store_true",
                      help=f"挂起，每天跑一次（默认 {DEFAULT_RUN_TIME}，可用 --time 指定）")
    mode.add_argument("--status", action="store_true", help="查看上次运行时间和结果")
    mode.add_argument("--selftest", action="store_true",
                      help="离线自测（假数据 + 临时向量库，不抓网络、不碰真实库）")

    parser.add_argument("--time", default=DEFAULT_RUN_TIME,
                        help=f"--daily 的运行时刻 HH:MM（默认 {DEFAULT_RUN_TIME}）")
    parser.add_argument("--keywords", default="",
                        help="逗号分隔的关键词，覆盖画像/环境变量（默认读画像）")
    parser.add_argument("--city", default="", help="城市，覆盖画像/环境变量")
    parser.add_argument("--out-dir", default="",
                        help="落盘目录（默认 rag/data/，测试请指向临时目录）")
    parser.add_argument("--state-file", default="",
                        help="状态文件路径（默认系统临时目录）")
    parser.add_argument("--log-file", default="",
                        help=f"日志路径（默认 {LOG_PATH}）")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                        help=f"每个关键词最多翻几页（默认 {DEFAULT_MAX_PAGES}）")
    parser.add_argument("--limit-per-keyword", type=int, default=DEFAULT_LIMIT_PER_KEYWORD,
                        help=f"每个关键词候选上限（默认 {DEFAULT_LIMIT_PER_KEYWORD}）")
    parser.add_argument("--limit-total", type=int, default=DEFAULT_LIMIT_TOTAL,
                        help=f"合并去重后的总上限（默认 {DEFAULT_LIMIT_TOTAL}）")
    parser.add_argument("--headed", action="store_true",
                        help="显示浏览器窗口（默认 headless）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只验证链路：不抓网络、不写文件")
    parser.add_argument("--quiet", action="store_true", help="不往控制台打日志")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.selftest:
        return _selftest()

    state_file = args.state_file or None

    # --status 是纯读命令：只读状态文件，不配 logger、不打开日志文件。
    # （日志文件可能被正在运行的 --daily 进程占着，只读命令不该因为写不进去而报错。）
    if args.status:
        print(format_status(load_state(state_file), state_file))
        return 0

    logger = setup_logger(log_path=args.log_file or None, verbose=not args.quiet)

    config = resolve_config(logger=logger, allow_fallback=not args.dry_run)
    if args.keywords:
        config["keywords"] = _split_list(args.keywords)
    if args.city:
        config["city"] = args.city
        config["cities"] = [args.city]
    config["max_pages"] = args.max_pages
    config["limit_per_keyword"] = args.limit_per_keyword
    config["limit_total"] = args.limit_total

    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.daily:
        return loop_daily(
            run_time=args.time, logger=logger, config=config,
            state_file=state_file, out_dir=out_dir, headless=not args.headed,
        )

    # --once（未指定任何模式时也按 --once 处理，避免"什么都不做"的困惑）
    if not args.once:
        logger.info("未指定模式，按 --once 执行（可用 --daily / --status）")
    result = run_daily_job(
        config=config, out_dir=out_dir, state_file=state_file, logger=logger,
        headless=not args.headed, dry_run=args.dry_run,
    )
    print()
    print(format_status(load_state(state_file), state_file))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
