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

# 抓取配置（城市池 / 关键词池）外置到 config/scraping.yaml：改配置不改代码。
# 优先级：环境变量 > config/scraping.yaml > user_profile.json / DEFAULT_KEYWORDS。
# 导入失败（config 包不在 sys.path 等）时置 None，退回本文件既有的兜底逻辑。
try:                                                         # noqa: E402
    from config import loader as scraping_config
except ImportError:                                          # pragma: no cover
    scraping_config = None

# 多平台抽象层。base.py 只依赖标准库，模块级导入不会拖入 playwright
# （真正的平台实现仍在 scraper_registry() 里惰性导入，见下方说明）。
from agent.scrapers.base import PlatformScraper, RawJob       # noqa: E402

# 归档文件名与 cleaner 共用一份定义：两边各写一个字符串，改了一边忘了另一边，
# 归档就会分裂成两个文件。cleaner 不是包（rag/quality 没有 __init__.py），
# 这里用与 _load_cleaner() 相同的方式把它导进来。
ARCHIVE_FILENAME = "cleaned_jd_archive.json"
try:                                                         # pragma: no cover
    from rag.quality.cleaner import ARCHIVE_FILENAME         # noqa: E402
except ImportError:                                          # rag.quality 无包结构时的兜底
    pass

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

# 详情页并发路数（透传给抓取器的 detail_concurrency）。
# 默认 None = 用抓取器自己的常量 / SHIXISENG_DETAIL_CONCURRENCY，**不改默认调用形状**：
# 只有显式配置 SCHEDULER_DETAIL_CONCURRENCY 时才把这个参数传给 scraper，
# 免得自测里注入的假 scraper 因为多出一个关键字参数而报错。
DEFAULT_DETAIL_CONCURRENCY = None
_env_detail_concurrency = (os.getenv("SCHEDULER_DETAIL_CONCURRENCY") or "").strip()
if _env_detail_concurrency:
    try:
        DEFAULT_DETAIL_CONCURRENCY = int(_env_detail_concurrency)
    except ValueError:
        DEFAULT_DETAIL_CONCURRENCY = None

# ---------------------------------------------------------------------------
# 参数外置 + 快速模式 + mock + 诊断模式（全部只用环境变量，不改代码就能调参）
#
# 为什么要有它们：调一次并发参数就要跑一次约 40 分钟的真实抓取，验证成本太高。
#   SCHEDULER_FAST_MODE=1  只抓 1 城市 × 1 关键词 × 1 页 × 5 条 → 1~2 分钟出结果
#   SCHEDULER_USE_MOCK=1   切到 MockScraper 假数据跑通整条链路（不联网，几秒）
#   SCHEDULER_DIAGNOSE=1   跑完打印各环节**平均**耗时（列表页/详情页/page）：
#                          一次运行同时拿到总量与单价，不用为了看数字重跑
#
# 并发数本身也在抓取器那边外置（两个数各自独立，默认 3 × 3 = 9 个标签页）：
#   SHIXISENG_LIST_CONCURRENCY=3    列表页并发（1~8）
#   SHIXISENG_DETAIL_CONCURRENCY=3  详情页并发（1~8）
# ---------------------------------------------------------------------------
def _env_flag(name: str) -> bool:
    """布尔型环境变量：1/true/yes/on（大小写不敏感）为真，未设置或其余值为假。"""
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


FAST_MODE = _env_flag("SCHEDULER_FAST_MODE")
USE_MOCK = _env_flag("SCHEDULER_USE_MOCK")
DIAGNOSE = _env_flag("SCHEDULER_DIAGNOSE")

# 快速模式的规模：整条链路压到 1~2 分钟，只用来验证"改完还能不能跑通"。
FAST_MAX_PAGES = 1
FAST_LIMIT_PER_KEYWORD = 5
FAST_LIMIT_TOTAL = 5

# 清洗参数（与 rag/quality/cleaner.py 的默认口径一致）
DEFAULT_MAX_AGE_DAYS = int(os.getenv("SCHEDULER_MAX_AGE_DAYS", "60"))
DEFAULT_MIN_DESC_LEN = int(os.getenv("SCHEDULER_MIN_DESC_LEN", "200"))

# 一个关键词都没有时的兜底关键词**组**。
#
# 为什么是"组"而不是平铺的列表：抓取用的英文词（LLM / RAG / Agent）和中文词
# （大模型 / 检索增强生成 / 智能体）描述的是同一个概念，但实习僧的搜索是按字面
# 匹配的——同一个岗位用中文写"检索增强生成"、用英文写"RAG"，必须两个词都搜一遍
# 才不漏。组内是**同义词**（各自搜一次，结果合并去重），组间是**不同概念**。
#
# ⚠️ 诚实说明：这批真实数据里"检索增强生成"出现的 4 条全都同时写了 "RAG"，
#    所以中文同义词对**当前语料**不产生额外命中。它的价值在于覆盖只写中文的岗位
#    （下面的同义词增益自测就是拿这种人工样本验证的），而不是修一个现存的 bug。
DEFAULT_KEYWORDS = [
    ["Agent", "智能体"],
    ["大模型", "LLM", "大语言模型"],
    ["RAG", "检索增强生成"],
]

# 向后兼容：老代码/老文档引用的平铺列表，由 DEFAULT_KEYWORDS 推导（顺序、去重都固定）
FALLBACK_KEYWORDS = [word for group in DEFAULT_KEYWORDS for word in group]

# 默认抓取平台：多平台架构下调度器按这个列表**逐个平台、顺序执行**抓取。
# 默认启用 shixiseng（约 16 分钟）+ niuke（约 5 秒），总耗时几乎不变。
# 可用 --platforms 或环境变量 SCHEDULER_PLATFORMS="shixiseng,nowcoder" 覆盖（见 resolve_config）。
# 平台名 -> PlatformScraper 子类的注册表见 scraper_registry()。
DEFAULT_PLATFORMS = ["shixiseng", "niuke"]

# 合并落盘：超过这个天数的记录移进归档文件（口径与 cleaner.DEFAULT_STALE_DAYS 一致）
DEFAULT_STALE_DAYS = int(os.getenv("SCHEDULER_STALE_DAYS", "90"))

# 分批滚动抓取：每次只抓「最久未抓」的 N 个 (平台, 城市, 关键词) 组合，
# 多轮跑下来滚动覆盖全池。0 = 不分批（一次抓全池，与改造前完全一致）。
# 取值优先级：环境变量 SCHEDULER_BATCH_SIZE > config/scraping.yaml 的 schedule.batch_size
# > 这里的兜底值（0，保证 yaml 缺失时行为不变）。
DEFAULT_BATCH_SIZE = int(os.getenv("SCHEDULER_BATCH_SIZE", "0") or 0)


def _fmt_duration(seconds: float) -> str:
    """把秒数格式化成中文可读时长（给抓取进度日志用）。"""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}小时{m}分"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"


def _resolve_chunk_size(config: dict = None) -> int:
    """决定本次的 chunk_size：SCHEDULER_CHUNK_SIZE > config（yaml）> 0。

    <=0 表示不分块（旧行为：整池一次 gather）；非法值退回 0。
    """
    raw = os.getenv("SCHEDULER_CHUNK_SIZE")
    if raw is None or not str(raw).strip():
        raw = (config or {}).get("chunk_size", 0)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return max(value, 0)


def _make_progress_cb(logger, platform: str, total: int, chunk_size: int,
                      every: int = 20):
    """生成抓取进度回调：每 every 个组合（以及最后一个）向调度器日志写一行。

    为什么需要：实习僧列表页单组合要几十秒，整池 300+ 组合一次跑几小时，
    期间调度器日志一行都没有（2026-09-25 全量跑 1h42m 被误判为卡死）。
    这个回调让「还要多久」时刻可算。
    """
    if not logger or total <= 0:
        return None

    def _cb(done, all_total, label, jobs_count, error, elapsed):
        all_total = all_total or total
        is_last = done >= all_total
        if not is_last and every > 0 and done % every != 0:
            return
        eta = ""
        if done > 0 and not is_last:
            avg = elapsed / done
            eta = f"，预计剩余 {_fmt_duration(avg * (all_total - done))}"
        if chunk_size:
            batch = (done - 1) // chunk_size + 1
            total_batches = (all_total + chunk_size - 1) // chunk_size
            batch_label = f"[第 {batch}/{total_batches} 批] "
        else:
            batch_label = ""
        logger.info(
            "[进度][平台 %s] %s%d/%d 组合（%.1f%%，已用 %s%s）｜%s：%d 条%s",
            platform, batch_label, done, all_total,
            100.0 * done / max(all_total, 1), _fmt_duration(elapsed), eta,
            label, jobs_count, f"，失败：{error}" if error else "",
        )

    return _cb


def _resolve_batch_size(config: dict = None) -> int:
    """决定本次的 batch_size：SCHEDULER_BATCH_SIZE > config（yaml/兜底）> 0。

    非法值（空串 / 非数字 / 负数）一律退回 DEFAULT_BATCH_SIZE，绝不因为一个
    配置写错就让定时任务抓不到东西。
    """
    raw = os.getenv("SCHEDULER_BATCH_SIZE")
    if raw is None or not str(raw).strip():
        raw = (config or {}).get("batch_size", DEFAULT_BATCH_SIZE)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        if raw not in (None, ""):
            logging.getLogger(__name__).warning(
                "batch_size=%r 不是整数，改用默认值 %d", raw, DEFAULT_BATCH_SIZE)
        return max(DEFAULT_BATCH_SIZE, 0)
    return value if value > 0 else 0


def _combo_key(platform, city, keyword) -> tuple[str, str, str]:
    """统一组合键口径：None / 空值都当 ""（与 rag/data/db.py 的历史表一致）。"""
    return (str(platform or ""), str(city or ""), str(keyword or ""))

CITY_SPLIT_RE = re.compile(r"[,，、;；/\s]+")

# 关键词组分隔：分号只用来切"组"，逗号用来切"同义词"。
# 这样环境变量里写 `Agent,智能体;大模型,LLM` 能表达"两组"，
# 而写 `智能体`（只有一个词）也不会被误当成多组。
GROUP_SPLIT_RE = re.compile(r"[;；]+")

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

    formatter = logging.Formatter(("[FAST] " if FAST_MODE else "") + LOG_FORMAT,
                                  datefmt=LOG_DATEFMT)

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
        "keyword_groups": result.get("keyword_groups", []),
        "platforms": result.get("platforms", []),
        "city": result.get("city", ""),
        "cities": result.get("cities", []),
        "per_city": result.get("per_city", {}),
        "per_platform": result.get("per_platform", {}),
        "db": result.get("db", {}),
        "sync": result.get("sync", {}),
        "scraped": result.get("scraped", 0),
        "cleaned": result.get("cleaned", 0),
        "chunks": result.get("chunks", 0),
        "incremental": result.get("incremental", {}),
        "merge": result.get("merge", {}),
        "archived": result.get("archived", 0),
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


def split_keyword_groups(value) -> list[list[str]]:
    """把配置里的关键词解析成**组**（组内同义词，组间不同概念）。

    容忍这几种写法（配置来源杂，写错格式不该让定时任务挂掉）：
        ["Agent", "智能体"]                  -> [["Agent"], ["智能体"]]   每个词自成一组
        [["Agent", "智能体"], ["RAG"]]       -> 原样保留
        ["Agent,智能体", "RAG,检索增强生成"]  -> [["Agent","智能体"], ["RAG","检索增强生成"]]
        "Agent,智能体;RAG,检索增强生成"        -> [["Agent","智能体"], ["RAG","检索增强生成"]]
        "智能体"                             -> [["智能体"]]

    规则：先用分号切"组"，组内再用 CITY_SPLIT_RE（逗号/顿号/空格）切"同义词"。
    为什么要支持"列表元素内部再带逗号"：user_profile.json 是手写的，
    有人会写成 ["Agent,智能体"]，按元素逐个当组的话这两个同义词会被拆成两组，
    语义上没错（还是都搜），但日志里"同义词"的分组信息就丢了。
    """
    if value is None:
        return []

    # 1) 先把最外层拆成"组"。
    #    ⚠️ 不能用 [str(v) for v in value]：元素本身可能是 list（嵌套写法
    #    [["Agent","智能体"],["RAG"]]），str() 会把它变成 "['Agent', '智能体']"
    #    这种带引号的字符串，再按逗号切就成了 "['Agent'" / "'智能体']"。
    #    所以嵌套元素直接当"组"原样保留，只对标量元素做字符串切分。
    raw_groups = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, (list, tuple, set)):
                raw_groups.append([str(w) for w in item])
            else:
                raw_groups.append(str(item))
    else:
        raw_groups = GROUP_SPLIT_RE.split(str(value))

    # 2) 每组再拆同义词，去掉空串并保持"组内去重、组间也去重"
    groups: list[list[str]] = []
    seen_words: set[str] = set()
    for raw in raw_groups:
        raw_words = raw if isinstance(raw, (list, tuple, set)) else CITY_SPLIT_RE.split(raw or "")
        words = []
        for word in raw_words:
            text = str(word).strip()
            if text and text not in seen_words:
                seen_words.add(text)
                words.append(text)
        if words:
            groups.append(words)
    return groups


def flatten_keyword_groups(groups) -> list[str]:
    """组列表 → 平铺关键词列表（抓取接口要的就是平铺的）。"""
    out, seen = [], set()
    for group in groups or []:
        words = group if isinstance(group, (list, tuple, set)) else [group]
        for word in words:
            text = str(word).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _format_keyword_groups(groups) -> str:
    """组列表 → 日志用的可读文本：`Agent/智能体、大模型/LLM`"""
    parts = []
    for group in groups or []:
        words = [str(w) for w in (group if isinstance(group, (list, tuple, set)) else [group])]
        if words:
            parts.append("/".join(words))
    return "、".join(parts)


def _apply_fast_mode(config: dict, logger: logging.Logger = None) -> dict:
    """快速模式（SCHEDULER_FAST_MODE=1）：把本次抓取缩到最小规模。

    只留第一个城市 + 第一个关键词，翻页 1 页、每词 5 条、总量 5 条 —— 目的不是抓全，
    而是把「抓取 → 清洗 → 落库 → 切块 → 入库 → 落盘」整条链路压到 1~2 分钟，让
    "改完代码 / 改完并发参数到底还能不能跑通"这件事不用再等 40 分钟的完整抓取。

    幂等：被调用两次（resolve_config 一次，命令行参数覆盖后再一次）不会来回折腾 ——
    第二次发现规模已经是快速模式的值就不再改、也不再打日志。**不能**用"打过标记就
    跳过"的写法：命令行参数是在两次调用之间生效的，那样会让 --max-pages 2 反过来
    盖掉快速模式（实测踩过这个坑）。
    只改本次要抓的**规模**，清洗 / 入库逻辑一行都不动。
    """
    if not FAST_MODE:
        return config

    # 收敛前的规模：只用来判断"这次调用到底改没改"，决定要不要打日志
    before = (list(config.get("keywords") or []),
              [c for c in (config.get("cities") or []) if str(c).strip()],
              config.get("max_pages"), config.get("limit_per_keyword"),
              config.get("limit_total"))

    # 关键词：**整组收敛到一个词**。只截断 groups[0] 是不够的 ——
    # `SCHEDULER_KEYWORDS=Agent,RAG,LLM` 在语义上是"一个同义词组的三个词"，
    # run_daily_job 会把组重新平铺回 3 个关键词，快速模式就名存实亡了。
    words = list(config.get("keywords")
                 or flatten_keyword_groups(config.get("keyword_groups") or []))[:1]
    config["keywords"] = words
    config["keyword_groups"] = [words] if words else []

    cities = [c for c in (config.get("cities") or []) if str(c).strip()][:1]
    if cities:
        config["cities"] = cities
        config["city"] = cities[0]

    config["max_pages"] = FAST_MAX_PAGES
    config["limit_per_keyword"] = FAST_LIMIT_PER_KEYWORD
    config["limit_total"] = FAST_LIMIT_TOTAL

    after = (words, cities, config["max_pages"], config["limit_per_keyword"],
             config["limit_total"])
    if logger and before != after:
        logger.info(
            "[FAST] 快速模式：城市=%s 关键词=%s 页数=%d 每词上限=%d 总上限=%d"
            "（只验证链路能不能跑通，不是生产抓取口径）",
            "、".join(cities) or "不限",
            "、".join(words) or "（无）",
            config["max_pages"], config["limit_per_keyword"], config["limit_total"],
        )
    return config


_SOURCE_LABELS = {
    "env": "环境变量",
    "config": "config/scraping.yaml",
    "profile": "user_profile.json",
    "default": "内置默认",
}


def _load_scraping_config() -> dict:
    """读外置抓取配置（config/scraping.yaml）。

    读不到/loader 不可用就返回空壳，由 resolve_config 走本文件内置兜底；
    配置问题绝不能拖垮定时任务。
    """
    if scraping_config is None:
        return {"cities": [], "keyword_groups": [], "sources": {}}
    try:
        return scraping_config.load_scraping_config()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "读 config/scraping.yaml 失败：%s；改用内置默认值", exc)
        return {"cities": [], "keyword_groups": [], "sources": {}}


def format_config_sources(config: dict) -> str:
    """启动时打印一行生效配置，确认 config/scraping.yaml 有没有被读到。

    形如：`[配置] 城市来源=config/scraping.yaml（5 个） 关键词来源=config/scraping.yaml（7 个）`
    """
    cities = config.get("cities") or []
    keywords = (flatten_keyword_groups(config.get("keyword_groups") or [])
                or list(config.get("keywords") or []))
    cities_label = _SOURCE_LABELS.get(config.get("cities_source"), "内置默认")
    keywords_label = _SOURCE_LABELS.get(config.get("keywords_source"), "内置默认")
    return (f"[配置] 城市来源={cities_label}（{len(cities)} 个） "
            f"关键词来源={keywords_label}（{len(keywords)} 个）")


def resolve_config(profile: dict = None, logger: logging.Logger = None,
                   allow_fallback: bool = True) -> dict:
    """决定这次抓什么：关键词组 + 城市列表。

    优先级（城市与关键词同口径）：
        环境变量（SCHEDULER_KEYWORDS / SCHEDULER_CITY，向后兼容 SCRAPE_*）
        > config/scraping.yaml（见 config/loader.py）
        > user_profile.json（target_keywords / target_cities）
        > 本文件内置兜底（DEFAULT_KEYWORDS / ["广州"]）

    城市池和关键词池外置到 config/scraping.yaml 之后，扩展抓取范围只改那个文件，
    不用改代码；两个 sources 字段记录本次的实际来源，启动时由
    format_config_sources() 打印成一行，方便确认配置有没有生效。

    返回值里同时给两种形状：
        keyword_groups: [[同义词...], ...]  —— 搜索引擎按"组"记录贡献
        keywords:       [平铺...]           —— 抓取接口与状态文件沿用，保持向后兼容

    关键词一个都没有时用 DEFAULT_KEYWORDS 兜底并告警——定时任务最怕"静默
    什么都不做"，宁可打日志说清用了兜底关键词。allow_fallback=False（dry-run）
    时不打这条告警也不兜底：那次根本不抓网络，提示"用了兜底关键词"只会误导。
    """
    profile = profile if isinstance(profile, dict) else user_profile.load_profile()
    # 外置配置：文件不存在/解析失败时，loader 内部已回退到内置默认值并打 WARN。
    ext_config = _load_scraping_config()
    ext_sources = ext_config.get("sources") or {}

    # 关键词：环境变量 > config/scraping.yaml > user_profile.target_keywords > 兜底。
    env_keywords = os.getenv("SCHEDULER_KEYWORDS") or os.getenv("SCRAPE_KEYWORDS")
    if env_keywords:
        groups = split_keyword_groups(env_keywords)
        keywords_source = "env"
    elif ext_sources.get("keywords") == "config":
        # yaml 平铺写 keywords 时，loader 已按内置同义词表归好组（新词各自成组）。
        groups = [list(group) for group in (ext_config.get("keyword_groups") or [])]
        keywords_source = "config"
    else:
        # 没有外置关键词配置：保持改造前的顺序——先读画像。
        groups = split_keyword_groups(profile.get("target_keywords"))
        keywords_source = "profile" if groups else "default"
    if not groups:
        keywords_source = "default"
        if allow_fallback:
            groups = [list(group) for group in DEFAULT_KEYWORDS]
            if logger:
                logger.warning(
                    "没配置关键词（SCHEDULER_KEYWORDS / config/scraping.yaml / "
                    "user_profile.target_keywords 都是空），"
                    "本次用兜底关键词组：%s", _format_keyword_groups(groups),
                )
    keywords = flatten_keyword_groups(groups)

    # 城市：**全部**都要抓，不再只取第一个（多城市在这里就展开）。
    # 优先级：环境变量 > config/scraping.yaml > user_profile.target_cities > ["广州"]。
    cities = _split_list(os.getenv("SCHEDULER_CITY") or os.getenv("SCRAPE_CITY"))
    if cities:
        cities_source = "env"
    elif ext_sources.get("cities") == "config":
        cities = [str(city).strip() for city in (ext_config.get("cities") or [])]
        cities = [city for city in cities if city]
        cities_source = "config"
    else:
        cities = _split_list(profile.get("target_cities"))
        cities_source = "profile" if cities else "default"
    if not cities:
        cities = ["广州"]
        cities_source = "default"
    if len(cities) > 1 and logger:
        logger.info("配置了 %d 个城市：%s（将逐个城市分别抓取后合并）",
                    len(cities), "、".join(cities))

    # 平台：环境变量 SCHEDULER_PLATFORMS > user_profile.target_platforms > 兜底 shixiseng。
    # 与城市/关键词同口径，写错平台名不会在这里报错——真正的平台名校验发生在
    # 抓取时（scraper_registry() 查不到就记一条失败日志并跳过该平台）。
    platforms = _split_list(
        os.getenv("SCHEDULER_PLATFORMS")
    ) or _split_list(profile.get("target_platforms")) or list(DEFAULT_PLATFORMS)
    if USE_MOCK:
        # mock 走的是**正规平台注册表路径**（不是注入式假抓取器），所以切换只动这一行：
        # 后面的遍历 / 去重 / 清洗 / 落库…全部与真实平台完全一致。
        platforms = ["mock"]
        if logger:
            logger.info("[MOCK] SCHEDULER_USE_MOCK=1：本次平台切到 MockScraper"
                        "（假数据、不联网；落盘 / 落库 / 向量库一律跳过）")
    if len(platforms) > 1 and logger:
        logger.info("配置了 %d 个平台：%s（将逐个平台分别抓取后合并）",
                    len(platforms), "、".join(platforms))

    config = {
        "keyword_groups": groups,
        "keywords": keywords,
        # 生效来源（env / config / profile / default），供启动那行 [配置] 打印
        "keywords_source": keywords_source,
        "cities_source": cities_source,
        "cities": cities,
        "city": cities[0],
        "platforms": platforms,
        "max_pages": DEFAULT_MAX_PAGES,
        "limit_per_keyword": DEFAULT_LIMIT_PER_KEYWORD,
        "limit_total": DEFAULT_LIMIT_TOTAL,
        "detail_concurrency": DEFAULT_DETAIL_CONCURRENCY,
        "stale_days": DEFAULT_STALE_DAYS,
        # 分批滚动抓取：来自 config/scraping.yaml 的 schedule.batch_size，
        # 环境变量 SCHEDULER_BATCH_SIZE 可覆盖（见 _resolve_batch_size）。
        "batch_size": _resolve_batch_size(ext_config.get("schedule") or {}),
        # 抓取进度分块（每批多少个组合打一行进度日志）：schedule.chunk_size，
        # 环境变量 SCHEDULER_CHUNK_SIZE 可覆盖；0 = 不分块。
        "chunk_size": _resolve_chunk_size(ext_config.get("schedule") or {}),
    }
    # 快速模式在这里就收敛规模；main() 用命令行参数覆盖之后再收敛一次（幂等）。
    return _apply_fast_mode(config, logger)


# ---------------------------------------------------------------------------
# 多平台抓取：平台注册表 + 遍历
# ---------------------------------------------------------------------------
# 平台名 -> PlatformScraper 子类。
#
# ⚠️ 这里用「惰性填充」而不是模块级 `from agent.scrapers.shixiseng import ...`：
#    抓取器会连带 import playwright，而 `--status` / `--selftest` 是纯离线命令，
#    不该因为浏览器依赖出问题就被拖垮。这与本文件既有做法一致（scrape_jobs
#    里也是函数内 import）。语义上 SCRAPERS 就是 {"shixiseng": ShixisengScraper}，
#    只是把"填表"的时机推迟到第一次真正抓取之前。
SCRAPERS: dict[str, type] = {}


def scraper_registry() -> dict[str, type]:
    """返回平台注册表；首次调用时把内置平台登记进去。

    新增平台：写一个 `PlatformScraper` 子类，在这里 setdefault 一行即可，
    调度逻辑（遍历 / 去重 / 落库 / 日志）完全不用动。
    """
    if not SCRAPERS:
        from agent.scrapers.shixiseng import ShixisengScraper

        SCRAPERS.setdefault("shixiseng", ShixisengScraper)
        # 牛客网走公开 JSON 接口（只用 requests，不含 playwright），
        # 与实习僧的浏览器方案互补；同样是惰性 import，离线命令不受影响。
        from agent.scrapers.niuke import NiukeScraper

        SCRAPERS.setdefault("niuke", NiukeScraper)
        # mock_scraper 只依赖 base（不含 playwright），离线端到端自测用：
        # SCHEDULER_USE_MOCK=1 时平台被切成 "mock"，几秒钟跑完整条链路。
        from agent.scrapers.mock_scraper import MockScraper

        SCRAPERS.setdefault("mock", MockScraper)
    return SCRAPERS


class _FunctionScraper(PlatformScraper):
    """把「函数式抓取器」适配成 PlatformScraper。

    只服务**注入式假实现**（离线自测 / 测试替身）与老调用方：它们传进来的是
    `func(keywords, city=..., ...) -> list`，不是 PlatformScraper 实例。
    真实平台一律走 scraper_registry()。调用形状与 scrape_jobs 保持一致，
    这样老的假 scraper 不用改就能继续用。
    """

    def __init__(self, func, platform_name: str = "fake", options: dict = None) -> None:
        self._func = func
        self.platform_name = platform_name
        self.options = options or {}

    async def search(self, keyword: str, city: str = None, limit: int = 20):
        opts = self.options
        result = self._func(
            [keyword],
            city=city,
            max_pages_per_keyword=opts.get("max_pages", 1),
            limit_total=limit,
            limit_per_keyword=limit,
            headless=opts.get("headless", True),
            fetch_detail=True,
        )
        if inspect.isawaitable(result):
            result = await result
        return [self._as_raw(job) for job in (result or [])]

    def _as_raw(self, job) -> RawJob:
        """dict / Job / RawJob 一律转成 RawJob（假数据常常直接就是 dict）。"""
        if isinstance(job, RawJob):
            return job
        data = job if isinstance(job, dict) else _job_to_dict(job)
        return RawJob(
            platform=data.get("platform") or self.platform_name,
            job_id=data.get("job_id") or "",
            title=data.get("title") or "",
            company=data.get("company") or "",
            city=data.get("city") or "",
            salary=data.get("salary") or "",
            url=data.get("url") or "",
            description=data.get("description") or "",
            publish_date=data.get("publish_date") or "",
        )


def _make_scraper(platform_name: str, options: dict, factory=None) -> PlatformScraper:
    """实例化某平台的抓取器。

    options 是各平台**公共**的连接参数（headless / max_pages / ...）。
    平台子类不认识的参数按构造函数签名过滤掉——否则以后加一个通用参数，
    就会把只实现了部分参数的平台直接打挂。
    """
    if factory is not None:
        return factory(platform_name)

    cls = scraper_registry().get(platform_name)
    if cls is None:
        raise KeyError(
            f"未知平台 {platform_name!r}（已注册：{sorted(scraper_registry())}）"
        )
    try:
        accepted = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in options.items() if k in accepted}
    except (TypeError, ValueError):
        kwargs = {}
    return cls(**kwargs)


def _load_job_db():
    """导入岗位 SQLite 模块（rag/data/db.py）。

    优先按包路径 `rag.data.db` 导入：这样和 migrate_json_to_sqlite.py 里
    `from rag.data import db` 拿到的是**同一个模块对象**，不会出现
    "两个 db 模块、两套 DB_PATH" 的隐患；拿不到时再退回老的按模块名导入
    （`rag/data` 没有 __init__.py，靠 sys.path + importlib 兜底）。
    """
    try:
        from rag.data import db as _db
        return _db
    except ImportError:
        import importlib
        import sys

        data_dir = str(ROOT_DIR / "rag" / "data")
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        return importlib.import_module("db")


def _count_json_jobs(json_path) -> int:
    """数出 JSON 里的岗位条数（顶层 list 或 {"jobs": [...]}）；读不到返回 0。"""
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if isinstance(data, dict):
        data = data.get("jobs", [])
    return len(data) if isinstance(data, list) else 0


def _sync_json_to_sqlite(json_path, logger: logging.Logger = None) -> dict:
    """收尾步骤：把 cleaned_jd.json 的**全量**同步进 SQLite。

    为什么需要它：本轮"清洗后落库"只是增量 upsert 这次抓到的岗位。用户一旦
    清空了 jobs.db，历史岗位就再也回不来了。收尾时从 JSON 全量迁一次，
    保证 **SQLite 始终 = JSON 全量**。

    复用 rag/data/migrate_json_to_sqlite.py 的 `migrate()`（**不改迁移逻辑**）；
    该模块不可用时退回 `db.import_json()`——同样是"读 JSON → upsert_jobs"。
    同步失败只记日志，不影响本轮任务结果。
    """
    path = Path(json_path)
    if not path.is_file():
        if logger:
            logger.warning("[同步] JSON 不存在，跳过同步：%s", path)
        return {}

    stats: dict = {}
    try:
        import importlib
        import sys

        data_dir = str(ROOT_DIR / "rag" / "data")
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        stats = importlib.import_module("migrate_json_to_sqlite").migrate(path) or {}
    except Exception as exc:                      # noqa: BLE001 - 迁移脚本不可用就退回 db
        if logger:
            logger.warning("[同步] migrate_json_to_sqlite 不可用（%s: %s），改用 db.import_json",
                           type(exc).__name__, exc)
        try:
            stats = _load_job_db().import_json(path) or {}
        except Exception as exc2:                 # noqa: BLE001
            if logger:
                logger.warning("[同步] JSON → SQLite 同步失败（不影响本轮结果）：%s: %s",
                               type(exc2).__name__, exc2)
            return {}

    # 返回结构要归一：migrate() 给的是嵌套的
    # {"total", "json_count", "db_total", "stats": {"added","updated","skipped"}}，
    # 而 db.import_json() 给的是平铺的 {"added","updated","skipped"}——两种都要认，
    # 否则日志会把「更新 3」错报成「更新 0」。
    inner = stats.get("stats") if isinstance(stats.get("stats"), dict) else stats
    total = stats.get("total") or stats.get("db_total") or _count_json_jobs(path)
    outcome = {
        "total": total,
        "added": inner.get("added", 0),
        "updated": inner.get("updated", 0),
        "skipped": inner.get("skipped", 0),
    }
    if logger:
        # 约定的同步日志格式：JSON 全量条数 + 本次 upsert 明细
        logger.info("[同步] JSON → SQLite: %d 条（新增 %d，更新 %d，跳过 %d）",
                    outcome["total"], outcome["added"],
                    outcome["updated"], outcome["skipped"])
    return outcome


def _persist_jobs_to_sqlite(rows: list[dict], logger: logging.Logger = None,
                            stage: str = "清洗后") -> dict:
    """把岗位写进 SQLite（rag/data/jobs.db）。

    ⚠️ **只允许写清洗后的岗位**：本函数在 cleaner 之后调用，jobs.db 是
    Agent 真正检索的数据源，里面不能出现被相关性/城市/时效/正文长度过滤掉的
    脏数据（这正是上一版"抓完立刻落库"的数据流 bug）。

    * 走既有 `db.upsert_jobs()`，**不改 db.py**；按 job_id 幂等 upsert，
      每条记录自带 platform 字段，多平台数据就靠它区分。
    * 字段映射统一由 `_job_to_dict()` 负责（job_id / platform / title / company /
      city / salary / url / description / publish_date）。
    * 写库失败不回滚也不抛异常：后面还有 chunk 与语料落盘链路，
      不能因为本地库写不进去就让整轮任务失败。
    """
    try:
        stats = _load_job_db().upsert_jobs(rows)
    except Exception as exc:                      # noqa: BLE001
        if logger:
            logger.warning("[落库] %s岗位写入 SQLite 失败（继续后续链路）：%s: %s",
                           stage, type(exc).__name__, exc)
        return {}
    if logger:
        logger.info("[落库] %s岗位已写入 SQLite：新增 %d，更新 %d，跳过 %d",
                    stage, stats.get("added", 0), stats.get("updated", 0),
                    stats.get("skipped", 0))
    return stats


def _normalize_city_label(text) -> str:
    """城市名归一（去空白、去结尾「市」），把清洗结果对回抓取时的城市标签。"""
    value = str(text or "").strip()
    return value[:-1] if value.endswith("市") else value


def _cleaned_counts_by_platform_city(jobs: list[dict]) -> dict[tuple, int]:
    """清洗后的岗位按 (平台, 城市) 计数，用于打印「清洗后 M 条」。"""
    counts: dict[tuple, int] = {}
    for job in jobs or []:
        row = _job_to_dict(job)
        key = (row.get("platform") or "unknown", _normalize_city_label(row.get("city")))
        counts[key] = counts.get(key, 0) + 1
    return counts


def scrape_multi_platform(
    platforms, cities, keywords,
    max_pages=DEFAULT_MAX_PAGES,
    limit_per_keyword=DEFAULT_LIMIT_PER_KEYWORD,
    limit_total=DEFAULT_LIMIT_TOTAL,
    headless: bool = True,
    detail_concurrency=None,
    scraper=None,
    batch_size: int = 0,
    chunk_size: int = 0,
    logger: logging.Logger = None,
) -> dict:
    """遍历 [平台 × 城市 × 关键词] 抓取 → 合并去重。

    平台之间是**顺序**的（同一 IP 同时打多个平台的风控风险最高）。
    平台内部的「城市 × 关键词」组合交给抓取器的 `search_multi()` **并发**跑列表页
    （默认 3 路、上限 5，见 ShixisengScraper.search_multi）——列表页加载原本占一次
    抓取的 87%，是最大瓶颈。抓取器没有这个接口时（例如离线自测注入的假抓取器）
    退回原来的顺序双层遍历，行为与统计口径完全一致。

    去重键是 **(platform, job_id)**：不同平台的 job_id 可能撞车，
    只用 job_id 去重会把别的平台的岗位误当重复丢掉。

    ⚠️ 本函数**只负责抓取，不碰岗位表 jobs**：抓到的原始岗位要等 run_daily_job
    清洗之后才落库，否则 jobs.db 会混进被 cleaner 过滤掉的脏数据
    （相关性差 / 过时 / 正文太短）。分批模式下唯一会写库的是进度表
    scrape_history（记录每个组合的 last_run_at），与岗位数据无关。

    分批滚动抓取（batch_size>0 时生效，默认 0 = 不分批、抓全池）：
    先按 platforms × cities × keywords 展开全池，用 db.get_stale_combos() 取
    「最久未抓」的 batch_size 个组合，本轮只抓这些，每抓完一个就
    db.record_scrape() 回写进度；下轮接着抓没抓过的，N 轮后覆盖全池。

    返回：
        {
          "jobs":              [去重合并后的原始岗位 dict]（交给清洗链路）,
          "per_platform":      {平台: 抓到条数},
          "per_city":          {城市: 抓到条数},
          "per_keyword":       {关键词: 抓到条数},
          "per_platform_city": {(平台, 城市): 抓到条数},
          "per_platform_city_kept": {(平台, 城市): 截断后条数},
          "failed":            {"平台/城市/关键词": 错误摘要},
          "raw_total":         去重前总条数,
        }
    """
    platforms = [p for p in (platforms or []) if str(p).strip()] or list(DEFAULT_PLATFORMS)
    if logger:
        logger.info("本次抓取平台（顺序执行）：%s", " -> ".join(platforms))
    cities = [c for c in (cities or []) if c is None or str(c).strip()] or [None]
    keywords = [k for k in (keywords or []) if str(k).strip()]

    result = {
        "jobs": [], "per_platform": {}, "per_city": {}, "per_keyword": {},
        "per_platform_city": {}, "failed": {}, "raw_total": 0,
        "timings": [],
    }
    if not keywords:
        if logger:
            logger.warning("没有任何关键词，跳过抓取。")
        return result

    options = {
        "headless": headless,
        "fetch_detail": True,
        "max_pages": max_pages,
        "detail_concurrency": detail_concurrency,
    }
    # 注入式假抓取器（离线自测）包成函数式适配器；真实平台走注册表
    factory = (lambda name: _FunctionScraper(scraper, name, options)) if scraper else None

    merged: dict[tuple, dict] = {}
    # merged 的插入顺序是「平台 → 城市 → 关键词」，但 job 字典里带的是**岗位自身**的
    # 城市（全国岗位可能是"全国"）。截断要按「抓取时的平台 × 城市」配额分组，
    # 所以额外记一份 key → (平台, 抓取城市) 的归属表，与 per_platform_city 同口径。
    merged_origin: dict[tuple, tuple] = {}

    # ---- 分批滚动抓取：本次只抓「最久未抓」的 batch_size 个组合 ----
    # 两支独立逻辑，互不干扰：
    #   * 记录（_record）—— 不论是否分批都写进度表，这样 batch_size=0 的老用法
    #     也在积累 last_run_at，将来切到分批时"最久未抓"的判断是准的；
    #   * 挑选（batch_plan）—— 只有 batch_size>0 才过滤组合；batch_size<=0 时
    #     batch_plan=False，抓取组合与行为跟改造前逐字一致（一次抓全池）。
    # 注入了假抓取器（离线自测 / 测试替身）时一律不写真实进度表：与
    # "假抓取器不落库" 同一口径，避免假组合污染真实进度。
    batch_size = _resolve_batch_size({"batch_size": batch_size})
    # 抓取进度分块：显式传了就用传入值；没传（0）则回落到 config/scraping.yaml 的
    # schedule.chunk_size；再没有就由抓取器用自己的默认分块（ShixisengScraper 默认 12）。
    chunk_size = chunk_size or _resolve_chunk_size(
        (_load_scraping_config().get("schedule") or {}))
    batch_db = None
    batch_selected: set[tuple] = set()
    batch_plan = False
    if scraper is None:
        try:
            batch_db = _load_job_db()
            batch_db.init_db()
        except Exception as exc:              # noqa: BLE001 - 进度表坏掉也要照常抓
            batch_db = None
            if logger:
                logger.warning("[分批] 进度表不可用（%s: %s），本次不记录抓取进度",
                               type(exc).__name__, exc)
    if batch_size > 0 and batch_db is not None:
        pool_size = len(platforms) * len(cities) * len(keywords)
        try:
            history = batch_db.get_history_stats()
            selected = batch_db.get_stale_combos(
                platforms, [str(c).strip() if c else "" for c in cities],
                keywords, limit=batch_size,
            )
            # 平台过滤（双保险）：组合池已按 platforms 展开，这里再按本次请求的平台
            # 过滤一次——`--platforms niuke` 时候选集必须 100% 来自 niuke，不能因为
            # 池子口径变化（或将来有调用方传了全平台池）混进别的平台。
            requested = {str(p).strip() for p in platforms if str(p).strip()}
            selected = [c for c in selected if str(c[0]).strip() in requested]
            selected = selected[:batch_size]
            batch_selected = {_combo_key(p, c, k) for p, c, k in selected}
            batch_plan = True
            if logger:
                dist: dict[str, int] = {}
                for p, _c, _k in selected:
                    dist[str(p)] = dist.get(str(p), 0) + 1
                logger.info("[分批] 本批平台分布：%s",
                            "，".join(f"{p} {n} 个" for p, n in dist.items()) or "（空）")
                logger.info("[分批] 全池 %d 组合，已覆盖 %d，本次抓 %d",
                            pool_size, history.get("covered_combos", 0),
                            len(batch_selected))
                logger.info("[分批] 全池 %d 组合，本批 %d 个，剩余 %d 未抓",
                            pool_size, len(batch_selected),
                            max(pool_size - len(batch_selected), 0))
                preview = "，".join(
                    f"{p}/{c or '不限'}/{k}" for p, c, k in selected[:8]
                )
                if preview:
                    logger.info("[分批] 本批覆盖：%s%s", preview,
                                f" …（共 {len(batch_selected)} 个）"
                                if len(batch_selected) > 8 else "")
        except Exception as exc:              # noqa: BLE001 - 排序失败也要照常抓
            batch_selected, batch_plan = set(), False
            if logger:
                logger.warning("[分批] 取「最久未抓」组合失败（%s: %s），本次改为抓全池",
                               type(exc).__name__, exc)

    async def _run() -> None:
        def _record(platform: str, keyword: str, city, success: bool,
                    error: str = "") -> None:
            """把单个组合的抓取结果写回进度表；进度表不可用时是空操作。"""
            if batch_db is None:
                return
            try:
                batch_db.record_scrape(platform, city, keyword, success, error)
            except Exception as exc:          # noqa: BLE001 - 记录失败不拖垮抓取
                if logger:
                    logger.warning("[分批] 记录进度失败（%s/%s/%s）：%s",
                                   platform, city or "不限", keyword, exc)

        for platform in platforms:
            # 分批：本平台只保留选中的 (关键词, 城市) 组合；一个都没有就跳过，
            # 连抓取器实例（浏览器）都不启动——这正是分批省时间的来源。
            combos = [
                (keyword, city) for city in cities for keyword in keywords
                if not batch_plan
                or _combo_key(platform, city, keyword) in batch_selected
            ]
            if not combos:
                if logger:
                    logger.info("[分批][平台 %s] 本批没有该平台的组合，跳过", platform)
                continue
            try:
                instance = _make_scraper(platform, options, factory=factory)
            except Exception as exc:              # noqa: BLE001 - 单平台失败不拖垮整轮
                result["failed"][platform] = f"{type(exc).__name__}: {exc}"
                if logger:
                    logger.error("[平台 %s] 抓取器初始化失败：%s（继续跑其他平台）",
                                 platform, result["failed"][platform])
                for keyword, city in combos:
                    _record(platform, keyword, city, False, result["failed"][platform])
                continue

            # 关键：instance 只在**平台这一层**创建一次，下面的「所有城市 × 所有关键词」
            # 共用同一个实例、且只在最后 close 一次。平台的抓取器因此可以复用同一个
            # 浏览器（ShixisengScraper 就是这么做的），而不是每个关键词重启一次
            # （实测启动+context+关闭 ≈30 秒/次，5 城市 × 7 关键词 ≈ 17.5 分钟）。
            searches_done = 0

            def _absorb(keyword: str, city, raw_jobs) -> None:
                """把单个「城市 × 关键词」组合的结果并入统计与去重表。

                抽成函数是为了让**并发批量路径**与**顺序兜底路径**共用同一套统计口径，
                保证日志与计数和改动前逐字一致（去重仍是先到先得）。
                """
                label = city or "不限"
                raw_jobs = list(raw_jobs or [])
                result["raw_total"] += len(raw_jobs)
                for bucket, name in (
                    (result["per_city"], label),
                    (result["per_keyword"], keyword),
                    (result["per_platform"], platform),
                    (result["per_platform_city"], (platform, label)),
                ):
                    bucket[name] = bucket.get(name, 0) + len(raw_jobs)
                # 统一的抓取日志格式：谁（平台）在哪儿（城市）搜什么（关键词）抓到几条
                if logger:
                    logger.info("[平台 %s][城市 %s][关键词 %s] 抓到 %d 条",
                                platform, label, keyword, len(raw_jobs))

                for raw_job in raw_jobs:
                    job = _job_to_dict(raw_job)
                    job["platform"] = job.get("platform") or platform
                    job_id = (job.get("job_id") or "").strip()
                    key = (platform, job_id) if job_id else (
                        platform,
                        f"{job.get('company')}|{job.get('title')}|{job.get('url')}",
                    )
                    if merged.setdefault(key, job) is job:
                        merged_origin[key] = (platform, label)

                # 分批进度：这个组合本次跑成功了（不分批时是空操作）
                _record(platform, keyword, city, True)

            try:
                # 组合顺序 = 城市外层、关键词内层，与原顺序遍历完全一致，
                # 所以去重时的"先到先得"归属不变。
                # combos 已在上面的分批过滤里算好（batch_plan 时只含本批选中的组合）。
                multi = getattr(instance, "search_multi", None)
                use_multi = False
                if callable(multi):
                    # 只有抓取器明确实现了带 return_groups 的批量接口才走并发路径，
                    # 免得将来别的平台签名不同被误用（那时退回顺序遍历即可）。
                    try:
                        use_multi = "return_groups" in inspect.signature(multi).parameters
                    except (TypeError, ValueError):
                        use_multi = False

                if use_multi:
                    # 列表页并发：整批交给抓取器，由它用 Semaphore(3) 并发跑列表页
                    # （详情页有自己的并发，按列表并发等比缩小，见抓取器实现）。
                    t_batch = time.monotonic()
                    # 进度回调：把「跑到第几个组合、还要多久」实时写进调度器日志，
                    # 免得几小时的抓取在日志里一片空白（看起来像卡死）。
                    _progress_cb = _make_progress_cb(
                        logger, platform, len(combos), chunk_size)
                    try:
                        grouped = await multi(
                            combos, limit=limit_per_keyword, return_groups=True,
                            progress_cb=_progress_cb,
                            chunk_size=(chunk_size or None),
                        )
                    except Exception as exc:   # noqa: BLE001 - 整批失败不拖垮其他平台
                        summary = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                        for keyword, city in combos:
                            label = city or "不限"
                            searches_done += 1
                            result["failed"][f"{platform}/{label}/{keyword}"] = summary
                            if logger:
                                logger.error("[平台 %s][城市 %s][关键词 %s] 抓取失败：%s",
                                             platform, label, keyword, summary)
                            _record(platform, keyword, city, False, summary)
                        grouped = []

                    for entry in grouped or []:
                        if not isinstance(entry, dict):
                            continue
                        keyword = str(entry.get("keyword") or "")
                        city = entry.get("city")
                        searches_done += 1
                        error = entry.get("error")
                        if error:
                            # 单组合失败：与顺序路径一样，只记失败、继续其他组合
                            result["failed"][
                                f"{platform}/{city or '不限'}/{keyword}"
                            ] = str(error)
                            if logger:
                                logger.error("[平台 %s][城市 %s][关键词 %s] 抓取失败：%s",
                                             platform, city or "不限", keyword, error)
                            _record(platform, keyword, city, False, str(error))
                            continue
                        _absorb(keyword, city, entry.get("jobs"))

                    if logger:
                        logger.info("[计时][平台 %s] 列表页并发：%d 个组合"
                                    "（城市 %d × 关键词 %d）共耗时: %.1f 秒",
                                    platform, len(combos), len(cities), len(keywords),
                                    time.monotonic() - t_batch)
                else:
                    # 顺序兜底：抓取器没有并发批量接口（注入式假抓取器等）时，
                    # 保持原来的遍历方式与日志，行为完全不变。
                    for city in cities:
                        label = city or "不限"
                        # 城市级计时：该城市所有关键词跑完（含失败的关键词）的总耗时
                        t_city = time.monotonic()
                        for keyword in keywords:
                            if batch_plan and _combo_key(
                                    platform, city, keyword) not in batch_selected:
                                continue
                            searches_done += 1
                            try:
                                raw_jobs = await instance.search(
                                    keyword, city=city, limit=limit_per_keyword
                                )
                            except Exception as exc:  # noqa: BLE001 - 单组合失败不拖垮整轮
                                key = f"{platform}/{label}/{keyword}"
                                result["failed"][key] = f"{type(exc).__name__}: {exc}"
                                if logger:
                                    logger.error(
                                        "[平台 %s][城市 %s][关键词 %s] 抓取失败：%s",
                                        platform, label, keyword, result["failed"][key])
                                _record(platform, keyword, city, False,
                                        result["failed"][key])
                                continue
                            _absorb(keyword, city, raw_jobs)
                        if logger:
                            logger.info("[计时][城市 %s] %d 个关键词共耗时: %.1f 秒",
                                        label, len(keywords), time.monotonic() - t_city)
            finally:
                try:
                    await instance.close()
                except Exception as exc:          # noqa: BLE001
                    if logger:
                        logger.warning("[平台 %s] 关闭抓取器失败：%s", platform, exc)
                if logger:
                    # 这行是"浏览器被复用而不是每次重启"的可核对凭据：
                    # 每平台 1 个实例 / 1 次 close，搜索次数 = 城市 × 关键词。
                    logger.info(
                        "[平台 %s] 本轮共 %d 次搜索（城市 %d × 关键词 %d）："
                        "抓取器实例 1 个、close 1 次（浏览器按平台复用，不随关键词重启）",
                        platform, searches_done, len(cities), len(keywords),
                    )
                # 抓取器自己维护的分段耗时账本（只加计时日志，不改抓取）。
                # 用 getattr 兜底：没有计时能力的平台（例如注入的假抓取器）直接跳过。
                platform_timings = getattr(instance, "timings", None)
                if isinstance(platform_timings, dict) and platform_timings:
                    result["timings"].append({"platform": platform, **platform_timings})

    asyncio.run(_run())

    # 截断按 **(平台, 城市)** 分配额：每个「平台 × 城市」组合各自 limit_total 条。
    # 为什么不是全局、也不是按平台：merged 是按「平台 → 城市 → 关键词」顺序插入的，
    # 按平台截断时同平台里排在后面的城市（例如杭州、成都）会被前面的城市吃光配额，
    # 日志显示抓了 47 条、清洗前却一条不剩——这就是「杭州 47 → 清洗 0」的根因。
    # 现在 limit_total 的语义是「每个平台在每个城市的**独立**上限」，互不挤占。
    jobs: list[dict] = []
    kept_by_pc: dict[tuple, int] = {}
    dropped_by_pc: dict[tuple, int] = {}
    for key, job in merged.items():
        platform_name = (key[0] if isinstance(key, tuple) and key
                         else (job.get("platform") or ""))
        origin = merged_origin.get(key)
        city_label = origin[1] if origin else _normalize_city_label(job.get("city"))
        pc = (platform_name, city_label)
        if limit_total and kept_by_pc.get(pc, 0) >= limit_total:
            dropped_by_pc[pc] = dropped_by_pc.get(pc, 0) + 1
            continue
        kept_by_pc[pc] = kept_by_pc.get(pc, 0) + 1
        jobs.append(job)
    if dropped_by_pc and logger:
        logger.info(
            "[多平台] 已按每「平台 × 城市」limit_total=%d 截断：%s"
            "（各城市独立配额，不互相挤占）",
            limit_total,
            "、".join(f"{pf}/{city} 丢弃 {n} 条"
                      for (pf, city), n in dropped_by_pc.items()),
        )
    result["jobs"] = jobs
    # 截断后的真实条数（按「平台 × 城市」）：日志用它报「截断后 N 条」，
    # 免得再拿截断前的 per_platform_city 当清洗输入数，把丢弃的量说成进了清洗。
    result["per_platform_city_kept"] = dict(kept_by_pc)
    return result


# ---------------------------------------------------------------------------
# 分段耗时汇总（只读账本，纯日志，不影响抓取）
# ---------------------------------------------------------------------------
def _merge_timings(entries) -> dict:
    """把多个平台的耗时账本合并成一份（秒数与次数都相加）。"""
    merged: dict = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if key == "platform" or not isinstance(value, (int, float)):
                continue
            merged[key] = merged.get(key, 0) + value
    return merged


def _format_timing_overview(entries) -> list[str]:
    """渲染 [计时][总览]：各分段秒数 + 次数 + 合计 + 其他开销。

    账本覆盖一次搜索的完整墙钟（search_total = 浏览器 + 列表 + 详情 + page + 其他），
    所以"其他开销"用减法得出，保证各项加起来正好等于合计 —— 这样"29 分钟去哪了"
    就能一眼看出是落在哪个分段里。
    """
    t = _merge_timings(entries)
    boot = t.get("browser_start", 0.0)
    stop = t.get("browser_stop", 0.0)
    list_sec = sum(t.get(k, 0.0) for k in ("list_goto", "list_selector", "list_wait"))
    detail = t.get("detail", 0.0)
    page = t.get("page", 0.0)
    total = t.get("search_total", 0.0)
    other = max(0.0, total - boot - stop - list_sec - detail - page)
    return [
        "[计时][总览]",
        f"  浏览器启动: {boot:.1f} 秒（{int(t.get('browser_start_count', 0))} 次）",
        f"  列表页加载: {list_sec:.1f} 秒（{int(t.get('list_goto_count', 0))} 次："
        f"goto {t.get('list_goto', 0.0):.1f}"
        f" + 选择器 {t.get('list_selector', 0.0):.1f}"
        f" + 渲染等待 {t.get('list_wait', 0.0):.1f}）",
        f"  详情页并发: {detail:.1f} 秒（{int(t.get('detail_count', 0))} 次 / "
        f"{int(t.get('detail_jobs', 0))} 条）",
        f"  page 创建/关闭: {page:.1f} 秒（{int(t.get('page_count', 0))} 次）",
        f"  其他开销: {other:.1f} 秒（含卡片解析、逐条日志、浏览器关闭 {stop:.1f} 秒等）",
        f"  合计: {total:.1f} 秒（{int(t.get('search_total_count', 0))} 次搜索）",
    ]


def _format_diagnose_overview(entries, config: dict = None) -> list[str]:
    """诊断模式（SCHEDULER_DIAGNOSE=1）：把耗时账本换算成**每个环节的平均耗时**。

    与 [计时][总览] 的区别：总览给的是"这一段一共花了多少"，诊断给的是"每一次
    花多少"——调并发参数时真正要看的是后者（列表页 33 秒/页、详情页 6 秒/条），
    因为总量会随抓取规模变化，单价才是配置的性能指标。

    纯读账本、不发任何新请求：跑一次就同时拿到总量与单价，不用为了看数字重跑。
    """
    t = _merge_timings(entries)
    list_sec = sum(t.get(k, 0.0) for k in ("list_goto", "list_selector", "list_wait"))
    list_n = int(t.get("list_goto_count", 0) or 0)
    detail_sec = t.get("detail", 0.0)
    detail_n = int(t.get("detail_count", 0) or 0)
    detail_jobs = int(t.get("detail_jobs", 0) or 0)
    page_sec = t.get("page", 0.0)
    page_n = int(t.get("page_count", 0) or 0)
    total_sec = t.get("search_total", 0.0)
    total_n = int(t.get("search_total_count", 0) or 0)

    def _avg(sec: float, n: int) -> float:
        return sec / n if n else 0.0

    config = config or {}
    cities = [c for c in (config.get("cities") or []) if str(c).strip()]
    keywords = [k for k in (config.get("keywords") or []) if str(k).strip()]
    detail_conc = (config.get("detail_concurrency")
                   or os.getenv("SHIXISENG_DETAIL_CONCURRENCY") or "3")
    list_conc = os.getenv("SHIXISENG_LIST_CONCURRENCY") or "3"
    return [
        "[诊断][平均耗时]（每次搜索 = 一个「城市 × 关键词」组合；列表页并发跑）",
        f"  生效配置: 列表并发={list_conc} × 详情并发={detail_conc}"
        f" | 页数/词={config.get('max_pages')} 条数/词={config.get('limit_per_keyword')}"
        f" | 组合数={len(cities) * len(keywords)}"
        f"（城市 {len(cities)} × 关键词 {len(keywords)}）",
        f"  列表页: 平均 {_avg(list_sec, list_n):.1f} 秒/页（{list_n} 页，合计 "
        f"{list_sec:.1f} 秒；其中 goto {t.get('list_goto', 0.0):.1f}"
        f" + 选择器 {t.get('list_selector', 0.0):.1f}"
        f" + 渲染等待 {t.get('list_wait', 0.0):.1f}）",
        f"  详情页: 平均 {_avg(detail_sec, detail_n):.1f} 秒/条"
        f"（{detail_n} 次请求 / {detail_jobs} 条正文）",
        f"  page:   平均 {_avg(page_sec, page_n) * 1000:.0f} 毫秒/次"
        f"（{page_n} 次创建+关闭）",
        f"  单组合: 平均 {_avg(total_sec, total_n):.1f} 秒/次"
        f"（{total_n} 次搜索，合计 {total_sec:.1f} 秒）",
    ]


# ---------------------------------------------------------------------------
# 抓取
# ---------------------------------------------------------------------------
def scrape_jobs(keywords: list[str], city=None, max_pages=DEFAULT_MAX_PAGES,
                limit_per_keyword=DEFAULT_LIMIT_PER_KEYWORD,
                limit_total=DEFAULT_LIMIT_TOTAL,
                headless: bool = True, detail_concurrency=None, scraper=None) -> list:
    """调 agent.scrapers.shixiseng.search_multi_keywords 抓取岗位。

    scraper: 参数用于注入假实现（离线自测），不传就用真实的
              search_multi_keywords（async 函数）。两种都能接：
              async 的直接 await，同步返回列表的包一层 coroutine。
    headless 默认 True：定时任务在后台跑，不该弹浏览器窗口。
    detail_concurrency: 详情页并发路数；None 时**不传该参数**，让抓取器用自身默认值。

    逐关键词的命中/零命中明细由抓取器自己打印（[多关键词] 前缀的日志）——
    它内部就是逐词循环，归属是准的；这里不重复统计，避免造一份口径不同的
    第二个真相。
    """
    if scraper is None:
        from agent.scrapers.shixiseng import search_multi_keywords as scraper

    async def _run():
        extra = {} if detail_concurrency is None else {"detail_concurrency": detail_concurrency}
        result = scraper(
            keywords,
            city=city,
            max_pages_per_keyword=max_pages,
            limit_total=limit_total,
            limit_per_keyword=limit_per_keyword,
            headless=headless,
            fetch_detail=True,
            **extra,
        )
        if inspect.isawaitable(result):
            return await result
        return result

    return asyncio.run(_run())


def scrape_all_cities(cities, keywords, max_pages=DEFAULT_MAX_PAGES,
                      limit_per_keyword=DEFAULT_LIMIT_PER_KEYWORD,
                      limit_total=DEFAULT_LIMIT_TOTAL,
                      headless: bool = True, detail_concurrency=None, scraper=None,
                      logger: logging.Logger = None) -> dict:
    """逐个城市抓取并合并（Task 4：不再只抓第一个城市）。

    为什么每个城市单独调一次抓取，而不是把所有城市塞进一次调用：
    抓取器只接一个 city 参数，且"广州 20 条 + 深圳 20 条"合并后才 40 条，
    若共用一个 limit_total，先跑的城市会把预算吃光、后面的城市一条都拿不到。
    所以每个城市各给一份 limit_total 配额，最后再整体去重。

    城市是**顺序**跑的（不做城市并发）：城市并发会让同一 IP 短时间内打到多个搜索
    入口，反爬风险最高；提速交给详情页并发（detail_concurrency）。

    返回：
        {
          "jobs":       [去重合并后的岗位],
          "per_city":   {城市: 条数},
          "failed":     {城市: 错误摘要},   # 单城市失败不拖垮整轮
          "raw_total":  去重前总条数,
        }
    """
    cities = [c for c in (cities or []) if str(c).strip()]
    if not cities:
        # 没有城市 = 不限城市，交给抓取器自己处理（传 None）
        cities = [None]

    merged: dict[str, object] = {}
    per_city: dict[str, int] = {}
    failed: dict[str, str] = {}
    raw_total = 0

    for city_index, city in enumerate(cities, 1):
        label = city or "不限"
        # [城市 i/N]：城市是顺序跑的，这行日志是"跑到第几个城市"的唯一凭据
        if logger:
            logger.info("[城市 %d/%d] %s", city_index, len(cities), label)
        try:
            jobs = scrape_jobs(
                keywords, city=city, max_pages=max_pages,
                limit_per_keyword=limit_per_keyword, limit_total=limit_total,
                headless=headless, detail_concurrency=detail_concurrency,
                scraper=scraper,
            ) or []
        except Exception as exc:                 # noqa: BLE001 - 单城市失败不拖垮整轮
            failed[label] = f"{type(exc).__name__}: {exc}"
            per_city[label] = 0
            if logger:
                logger.error("[城市 %s] 抓取失败：%s（继续跑其他城市）", label, failed[label])
            continue

        raw_total += len(jobs)
        new_in_city = 0
        for job in jobs:
            job_dict = _job_to_dict(job)
            key = (job_dict.get("job_id") or "").strip() or (
                f"{job_dict.get('company')}|{job_dict.get('title')}|{job_dict.get('url')}"
            )
            if key not in merged:
                merged[key] = job_dict
                new_in_city += 1
        per_city[label] = len(jobs)

        if logger:
            # 每个城市一行：这条日志是"多城市到底有没有都跑到"的唯一凭据
            logger.info(
                "[城市 %s] 抓取 %d 条，去重后新增 %d 条（累计 %d 条）",
                label, len(jobs), new_in_city, len(merged),
            )

    jobs = list(merged.values())
    if logger and len([c for c in cities if c]) > 1:
        logger.info(
            "多城市汇总：%s → 去重合并 %d 条（去重前 %d 条）%s",
            "；".join(f"{k}={v}条" for k, v in per_city.items()),
            len(jobs), raw_total,
            f"；失败 {failed}" if failed else "",
        )

    return {"jobs": jobs, "per_city": per_city, "failed": failed, "raw_total": raw_total}


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


def _job_city_text(job) -> str:
    """取一条岗位的城市文本（dict 与对象两种形态都兼容）。"""
    if isinstance(job, dict):
        return str(job.get("city") or "").strip()
    return str(getattr(job, "city", "") or "").strip()


def _city_matches(job_city, city) -> bool:
    """宽松城市匹配（复用 cleaner.city_matches，不另立一套口径）。"""
    return bool(_load_cleaner().city_matches(job_city, city))


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
# 落盘：合并（不是覆盖）
# ---------------------------------------------------------------------------
def _cleaned_path(out_dir=None) -> Path:
    return (Path(out_dir) if out_dir else DATA_DIR) / "cleaned_jd.json"


def _archive_path(out_dir=None) -> Path:
    return (Path(out_dir) if out_dir else DATA_DIR) / ARCHIVE_FILENAME


def merge_jds(existing, new, stale_days=DEFAULT_STALE_DAYS, archive_path=None,
              today=None) -> dict:
    """合并新旧岗位（按 job_id 去重 + publish_date 降序 + 过期归档）。

    真正的逻辑在 rag/quality/cleaner.py（那里已经有 _parse_date / save_cleaned，
    日期口径只有一份）。这里保留一个同名入口是因为 scheduler 的落盘语义需要
    一个稳定的、可单独测试的函数名，也方便自测直接 from scheduler import merge_jds。

    返回 cleaner.merge_jds 的 dict：{"jobs": [...], "archived": [...], "stats": {...}}
    """
    cleaner = _load_cleaner()
    return cleaner.merge_jds(
        existing, new,
        stale_days=stale_days,
        archive_path=archive_path,
        today=today,
    )


def merge_and_write_cleaned(jobs: list[dict], out_dir=None,
                            stale_days=DEFAULT_STALE_DAYS,
                            today=None) -> dict:
    """把本次清洗结果**合并**进 cleaned_jd.json（而不是覆盖）。

    这是 Round 8「抓 5 条把 15 条冲成 5 条」的修复点：
        读现有文件 → 按 job_id 去重合并 → publish_date 降序 →
        超期记录移入归档文件 → 写回合并后的全量。

    返回：
        {"path": Path, "archive_path": Path, "stats": {...}}
        stats 形如 {"existing":15,"new":3,"merged":18,"duplicates":0,"archived":1,"output":17}
    """
    target = _cleaned_path(out_dir)
    archive = _archive_path(out_dir)

    existing = _load_cleaner().load_jobs_if_exists(target)
    merged = merge_jds(
        existing, jobs,
        stale_days=stale_days,
        archive_path=archive,
        today=today,
    )
    _write_json_list(merged["jobs"], target)

    return {
        "path": target,
        "archive_path": archive,
        "archived_count": len(merged["archived"]),
        "stats": merged["stats"],
    }


def _write_json_list(jobs: list[dict], target: Path) -> Path:
    """把岗位列表写成顶层为 list 的 JSON（排版与 cleaner.save_cleaned 一致）"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def write_cleaned(jobs: list[dict], out_dir=None) -> Path:
    """把清洗结果写成 JSON（默认 rag/data/cleaned_jd.json）。

    ⚠️ 这是**覆盖**语义，只保留给"显式要求重写整个文件"的调用方。
    daily_job 用的是 merge_and_write_cleaned（合并语义）——Round 8 的数据丢失
    就发生在直接调这个函数上。
    """
    return _write_json_list(jobs, _cleaned_path(out_dir))


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
                  scraper=None, dry_run: bool = False, persist_db=None) -> dict:
    """跑一次完整任务：抓取（多平台）→ 清洗 → 增量落库 → 生成 chunk → 增量入库
    →（可选）落盘 → 从 JSON 全量同步 SQLite。

    返回摘要 dict（同时写进日志和状态文件）。异常一律在这里兜住并记进日志：
    定时任务里抛异常=任务静默死掉，比失败更糟。

    参数：
        config:      resolve_config() 的结果；不传就现取
        out_dir:     落盘目录；默认写 rag/data/（保持 Agent 能搜到新岗位）
        state_file:  状态文件路径（--status 读的就是它）
        dry_run:     True 时跳过真实抓取（只验证链路），并且不落盘
        persist_db:  是否把**清洗后**的有效岗位写进 SQLite（rag/data/jobs.db，
                     platform 字段区分平台）。清洗前的原始数据一律不落库。
                     None（默认）= 自动：注入了假抓取器（离线自测 / 测试替身）时不落库，
                     避免自测写真实库；True/False 可强制打开/关闭。
    """
    log = logger or setup_logger()
    config = config or resolve_config(logger=log, allow_fallback=not dry_run)
    # 老调用方可能直接传一份裸 config（不经过 resolve_config / main），这里兜一次底：
    # 幂等且静默，不会因为重复调用把规模改回来。
    config = _apply_fast_mode(config, None)

    # 关键词组（保留同义词关系用于日志），keywords 是平铺后的抓取参数
    keyword_groups = config.get("keyword_groups") or []
    if not keyword_groups:
        keyword_groups = [[k] for k in (config.get("keywords") or [])]
    keywords = flatten_keyword_groups(keyword_groups) or (config.get("keywords") or [])

    cities = [c for c in (config.get("cities") or []) if str(c).strip()]
    if not cities and config.get("city"):
        cities = [config["city"]]

    # 平台：config 里没有就现取（环境变量 > 默认），保证老调用方传裸 config 也能跑
    platforms = [p for p in (config.get("platforms") or []) if str(p).strip()]
    if not platforms:
        platforms = _split_list(os.getenv("SCHEDULER_PLATFORMS")) or list(DEFAULT_PLATFORMS)
    if USE_MOCK:
        # 环境变量优先于一切：mock 模式下绝不去碰真实平台。
        platforms = ["mock"]
    # mock 产出的是**假数据**：抓取 → 清洗 → 切块照常跑（用来验证链路），
    # 但所有**写操作**都要关掉，否则 [MOCK] 岗位会污染真实库 / 真实语料 / 向量库。
    # 这与自测里"注入式假抓取器不落库"是同一个口径。
    mock_active = USE_MOCK or "mock" in platforms
    if mock_active:
        log.info("[MOCK] 假数据模式：跳过落库 / 落盘 / 向量入库（清洗与切块照常跑）")
    started = time.time()

    log.info(
        "===== 定时抓取开始：平台=%s 关键词组=[%s] 城市=%s dry_run=%s "
        "详情页并发=%s（平台之间顺序，平台内「城市×关键词」列表页并发）=====",
        "、".join(platforms),
        _format_keyword_groups(keyword_groups),
        "、".join(cities) if cities else "不限",
        dry_run,
        config.get("detail_concurrency", DEFAULT_DETAIL_CONCURRENCY)
        or f"默认({os.getenv('SHIXISENG_DETAIL_CONCURRENCY') or '3'})",
    )

    result = {
        "ok": False,
        "keywords": keywords,
        "keyword_groups": keyword_groups,
        "platforms": platforms,
        "city": "、".join(cities),
        "cities": cities,
        "per_city": {},
        "per_platform": {},
        "db": {},
        "sync": {},
        "scraped": 0,
        "cleaned": 0,
        "chunks": 0,
        "incremental": {"added": 0, "updated": 0, "skipped": 0},
        "out_dir": str(out_dir) if out_dir else "",
        "converted": 0,
        "merge": {},
        "archived": 0,
        "error": "",
    }

    try:
        # 1) 抓取：遍历 [平台 × 城市 × 关键词]，合并去重后统一进后续链路
        if dry_run:
            jobs = []
            log.info("[1/4] dry-run：跳过真实抓取（不访问任何平台）")
        else:
            merged_scrape = scrape_multi_platform(
                platforms,
                cities,
                keywords,
                max_pages=config.get("max_pages", DEFAULT_MAX_PAGES),
                limit_per_keyword=config.get("limit_per_keyword", DEFAULT_LIMIT_PER_KEYWORD),
                limit_total=config.get("limit_total", DEFAULT_LIMIT_TOTAL),
                headless=headless,
                detail_concurrency=config.get("detail_concurrency",
                                              DEFAULT_DETAIL_CONCURRENCY),
                scraper=scraper,
                batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
                chunk_size=config.get("chunk_size", 0),
                logger=log,
            )
            jobs = merged_scrape["jobs"]
            result["per_city"] = merged_scrape["per_city"]
            result["per_platform"] = merged_scrape["per_platform"]
            log.info("[1/4] 抓取完成：%d 个平台 × %d 个城市 → 去重合并 %d 条%s",
                     len(platforms), len(merged_scrape["per_city"]), len(jobs),
                     f"（{len(merged_scrape['failed'])} 个组合失败）"
                     if merged_scrape["failed"] else "")
        result["scraped"] = len(jobs)

        # 2) 清洗
        #    多城市时 max_age_days 的基准仍是"今天"，与城市无关；
        #    城市过滤仍按单城市口径（全国岗位在每个城市的抓取结果里都保留，
        #    最后靠 merged 的 job_id 去重收敛成一条），所以这里传 city_hint：
        #    cities 里有几个城市就逐个清洗再合并，避免"只按第一个城市过滤"
        #    把第二个城市的岗位整批判成城市不符而丢掉。
        if len(cities) > 1 and jobs:
            kept, total_input, removed_sum = [], 0, {r: 0 for r in ("relevance", "city", "age", "length")}
            for city in cities:
                # 只把**属于该城市**的子集交给 cleaner：以前传全量 jobs，等于每个城市
                # 都跑一遍全量清洗，城市不符的整批丢——既浪费，又让"输入 N 条"的
                # 口径对不上。全国岗位（city == "全国"）在 city_matches 下每个城市
                # 都放行，与 cleaner 内部的宽松匹配保持一致。
                subset = [j for j in jobs if _city_matches(_job_city_text(j), city)]
                one = clean_jobs(
                    subset,
                    city=city,
                    max_age_days=config.get("max_age_days", DEFAULT_MAX_AGE_DAYS),
                    min_desc_len=config.get("min_desc_len", DEFAULT_MIN_DESC_LEN),
                )
                total_input += one.get("total_input", 0)
                for key, value in (one.get("removed") or {}).items():
                    removed_sum[key] = removed_sum.get(key, 0) + value
                kept.extend(one.get("jobs", []))
                log.info("[2/4][城市 %s] 清洗：输入 %d → 保留 %d",
                         city, one.get("total_input", 0), len(one.get("jobs", [])))
            cleaned = {
                "total_input": len(jobs),
                "removed": removed_sum,
                "jobs": kept,
            }
        else:
            cleaned = clean_jobs(
                jobs,
                city=(cities[0] if cities else "广州"),
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

        # 2.5) 落库：**清洗之后**才写 SQLite
        #      jobs.db 是 Agent 真正检索的数据源，只能放清洗后的有效岗位；
        #      清洗前的原始数据一律不落库（本轮只随 kept 继续走 chunk / 语料落盘）。
        #      persist_db=None 时自动：注入了假抓取器（离线自测 / 测试替身）就不落库，
        #      免得自测写到真实的 rag/data/jobs.db。
        if dry_run:
            log.info("[落库] dry-run：跳过写 SQLite")
        else:
            should_persist = (scraper is None) if persist_db is None else bool(persist_db)
            if mock_active:
                # 假数据绝不落真实库：即使显式 persist_db=True 也强制关掉
                should_persist = False
                log.info("[落库] mock 模式：跳过写 SQLite（假数据不落真实库）")
            if should_persist:
                # 字段映射统一走 _job_to_dict()：job_id / platform / title / company /
                # city / salary / url / description / publish_date
                result["db"] = _persist_jobs_to_sqlite(
                    [_job_to_dict(job) for job in kept], logger=log
                )
            else:
                log.info("[落库] 已关闭 SQLite 落库（persist_db=False 或存在注入式假抓取器）")

            # 每个「平台 × 城市」一行：
            # 抓到 N 条 → 截断后 K 条 → 清洗后 M 条 → 落库 M 条
            # 「截断后」取的是截断阶段真实保留的条数，不再拿截断前的抓取量冒充输入。
            cleaned_by_pc = _cleaned_counts_by_platform_city(kept)
            platform_totals: dict[str, int] = {}
            cleaned_by_city: dict[str, int] = {}
            for (pf, city_key), count in cleaned_by_pc.items():
                platform_totals[pf] = platform_totals.get(pf, 0) + count
                cleaned_by_city[city_key] = cleaned_by_city.get(city_key, 0) + count
            for (platform, city), raw_n in (merged_scrape.get("per_platform_city") or {}).items():
                label = _normalize_city_label(city or "")
                after_n = (merged_scrape.get("per_platform_city_kept") or {}).get(
                    (platform, city), raw_n)
                if not label:
                    # 「不限」城市：清洗后岗位带的是真实城市，只能按平台总量报
                    cleaned_n = platform_totals.get(platform, 0)
                elif (platform, label) in cleaned_by_pc:
                    cleaned_n = cleaned_by_pc[(platform, label)]
                else:
                    # 岗位自带的 platform 与注册名不一致时（例如注入的假数据）按城市兜底，
                    # 但**必须显式告警**：这个兜底会把"平台维度查不到"伪装成
                    # "该平台清洗后 N 条"，之前正是它掩盖了 niuke 被全局截断丢光的事实。
                    cleaned_n = cleaned_by_city.get(label, 0)
                    if logger:
                        logger.warning(
                            "[WARN] 平台 %s 的清洗计数未找到，使用城市兜底"
                            "（城市 %s：%d 条；真实平台维度计数为 0 时此值可能虚高）",
                            platform, label, cleaned_n,
                        )
                log.info("[平台 %s][城市 %s] 抓到 %d 条 → 截断后 %d 条 → 清洗后 %d 条 → 落库 %d 条",
                         platform, city, raw_n, after_n, cleaned_n,
                         cleaned_n if should_persist else 0)

        # 3) 生成 chunk
        chunks = build_chunks(kept)
        result["chunks"] = len(chunks)
        log.info("[3/4] 切块完成：%d 个 chunk", len(chunks))

        # 4) 增量入库（已存在且没变的不重算向量）
        if mock_active:
            # 假 chunk 不入向量库：embedding 是真花钱/真耗时的，且会污染检索结果
            stats = {"added": 0, "updated": 0, "skipped": 0}
            log.info("[4/4] mock 模式：跳过向量入库（假数据不入向量库）")
        elif chunks:
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

        # 5) 落盘：**合并**进 cleaned_jd.json（不是覆盖），再按合并后的全量刷新语料文本
        if dry_run or mock_active:
            log.info("[落盘] %s：跳过写文件（cleaned_jd.json / scraped_jd.txt 保持原样）",
                     "dry-run" if dry_run else "mock 模式")
        else:
            merged_result = merge_and_write_cleaned(
                kept, out_dir,
                stale_days=config.get("stale_days", DEFAULT_STALE_DAYS),
            )
            final_jobs = _load_cleaner().load_jobs_if_exists(merged_result["path"])

            # 语料文本必须用**合并后的全量**：只用本次的 kept 写，
            # 等于在 txt 上又做了一次覆盖，新岗位能搜到、老岗位反而搜不到了。
            txt_path, count = write_rag_text(final_jobs, out_dir)
            result["converted"] = count
            result["merge"] = merged_result["stats"]
            result["archived"] = merged_result["archived_count"]
            log.info(
                "[落盘] 合并写入 %s：原有 %d + 本次 %d → %d 条"
                "（重复 %d，超%d天归档 %d 条 → %s）",
                merged_result["path"],
                merged_result["stats"].get("existing", 0),
                merged_result["stats"].get("new", 0),
                merged_result["stats"].get("output", 0),
                merged_result["stats"].get("duplicates", 0),
                config.get("stale_days", DEFAULT_STALE_DAYS),
                merged_result["archived_count"],
                merged_result["archive_path"],
            )
            log.info("[落盘] 已刷新 RAG 语料 %s（合并后全量 %d 条）", txt_path, count)

        # 6) 收尾：从 cleaned_jd.json **全量**同步进 SQLite
        #    上面 2.5 只是把"本次清洗出的岗位"增量写库；用户一旦清空 jobs.db，
        #    历史岗位就再也回不来了。这里以刚落盘的 JSON 全量为准再迁一次，
        #    保证 SQLite 始终 = JSON 全量（复用 migrate_json_to_sqlite，不改迁移逻辑）。
        if dry_run:
            log.info("[同步] dry-run：跳过 JSON → SQLite 同步")
        elif should_persist:
            result["sync"] = _sync_json_to_sqlite(merged_result["path"], logger=log)
        else:
            log.info("[同步] 已关闭（persist_db=False 或存在注入式假抓取器）")

        # 分段耗时总览：把本次各平台的分段计时汇总，定位"时间到底花在哪"。
        # 只统计真实抓取；dry-run 与注入式假抓取器都没有账本。
        if not dry_run:
            timing_entries = merged_scrape.get("timings") or []
            if _merge_timings(timing_entries).get("search_total_count", 0):
                for line in _format_timing_overview(timing_entries):
                    log.info("%s", line)
            else:
                log.info("[计时][总览] 本次没有可用的分段计时（抓取器未上报）")
            if DIAGNOSE:
                # 诊断模式：不重跑、不额外发请求，直接用上面这份账本算"每次花多少"，
                # 并回显当前生效的并发/规模配置，方便"改参数 → 跑一次 → 看单价"。
                for line in _format_diagnose_overview(timing_entries, config):
                    log.info("%s", line)

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
    merge = state.get("merge") or {}
    per_city = state.get("per_city") or {}
    city_text = "、".join(state.get("cities") or []) or state.get("city") or "（不限）"
    lines = [
        f"上次运行时间：{state.get('last_run') or '（未知）'}",
        f"结果：{'[OK] 成功' if state.get('ok') else '[FAIL] 失败'}"
        + (f"　错误：{state['error']}" if state.get("error") else ""),
        f"平台：{'、'.join(state.get('platforms') or []) or '（未记录）'}",
        f"关键词：{'、'.join(state.get('keywords') or []) or '（未记录）'}",
        f"城市：{city_text}",
        f"抓取：{state.get('scraped', 0)} 条　清洗后：{state.get('cleaned', 0)} 条　"
        f"chunk：{state.get('chunks', 0)} 个",
        f"增量入库：新增 {inc.get('added', 0)}，更新 {inc.get('updated', 0)}，"
        f"跳过 {inc.get('skipped', 0)}",
    ]
    if per_city:
        lines.append("分城市：" + "　".join(f"{k} {v} 条" for k, v in per_city.items()))
    per_platform = state.get("per_platform") or {}
    if per_platform:
        lines.append("分平台：" + "　".join(f"{k} {v} 条" for k, v in per_platform.items()))
    db_stats = state.get("db") or {}
    if db_stats:
        lines.append(
            f"清洗后落 SQLite：新增 {db_stats.get('added', 0)}，"
            f"更新 {db_stats.get('updated', 0)}，跳过 {db_stats.get('skipped', 0)}"
        )
    sync_stats = state.get("sync") or {}
    if sync_stats:
        lines.append(
            f"JSON 全量同步 SQLite：{sync_stats.get('total', 0)} 条"
            f"（新增 {sync_stats.get('added', 0)}，更新 {sync_stats.get('updated', 0)}）"
        )
    if merge:
        lines.append(
            f"落盘合并：原有 {merge.get('existing', 0)} + 本次 {merge.get('new', 0)} "
            f"→ {merge.get('output', 0)} 条（重复 {merge.get('duplicates', 0)}，"
            f"归档 {state.get('archived', 0)} 条）"
        )
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

    - 关键词**组**解析（环境变量 / 画像 / 兜底 / 多种写法）
    - merge_jds 合并落盘（去重、降序、90 天归档）三个场景
    - 真跑一次「15 条已有 + 3 条新抓」确认落盘是合并不是覆盖
    - 多城市抓取（遍历所有城市并合并）
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
        # 外置配置（config/scraping.yaml）优先级高于画像，本用例要验的是"画像"这条路径，
        # 所以把配置文件路径指到不存在的文件；环境变量优先级另有用例 2 覆盖。
        saved_config_path = os.environ.get("SCRAPING_CONFIG_PATH")
        os.environ["SCRAPING_CONFIG_PATH"] = str(work / "no_such_scraping.yaml")
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
            if saved_config_path is None:
                os.environ.pop("SCRAPING_CONFIG_PATH", None)
            else:
                os.environ["SCRAPING_CONFIG_PATH"] = saved_config_path
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

        # 期望值从本次实际配置与 mock 输入动态算：多平台时每个平台各抓一遍，
        # 所以 scraped = 平台数 × 每平台 mock 条数，cleaned 同理（每平台各清洗一遍）。
        mock_platforms = max(1, len(first.get("platforms") or []))
        expected_scraped = len(fake_jobs) * mock_platforms
        expected_cleaned = len(clean_jobs(fake_jobs, city="广州")["jobs"]) * mock_platforms

        check("3. 全链路 run_daily_job 成功",
              lambda: (
                  f"抓取 {first['scraped']} → 清洗 {first['cleaned']} → "
                  f"chunk {first['chunks']} → 入库 +{first['incremental']['added']}"
                  if first["ok"] and first["scraped"] == expected_scraped
                  and first["cleaned"] == expected_cleaned
                  # added 允许 <= chunks：多平台时同一个岗位会在两个平台各切一份，
                  # chunk id 相同被增量入库去重，added 因此可能略小于 chunks。
                  and first["chunks"] > 0
                  and 0 < first["incremental"]["added"] <= first["chunks"]
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

        # ---- 3. 关键词组解析（Round 9 任务 2）----
        def _groups_ok():
            cases = [
                (DEFAULT_KEYWORDS, DEFAULT_KEYWORDS),
                ("Agent,智能体;RAG,检索增强生成",
                 [["Agent", "智能体"], ["RAG", "检索增强生成"]]),
                (["Agent,智能体", "RAG"], [["Agent", "智能体"], ["RAG"]]),
                ([["Agent", "智能体"], ["RAG"]], [["Agent", "智能体"], ["RAG"]]),
                (["Agent", "智能体"], [["Agent"], ["智能体"]]),
                ("智能体", [["智能体"]]),
                (None, []),
            ]
            bad = [f"{v!r} → {split_keyword_groups(v)}" for v, want in cases
                   if split_keyword_groups(v) != want]
            if bad:
                raise AssertionError("解析错误：" + "；".join(bad))
            return f"{len(cases)} 种写法 OK，平铺 {len(flatten_keyword_groups(DEFAULT_KEYWORDS))} 个词"

        check("9. 关键词组解析（含嵌套/分号/单值写法）", _groups_ok)

        # ---- 4. merge_jds 合并落盘（Round 9 任务 1）----
        # 用独立于 fake_jobs 的合成数据，避免和上面的城市/正文长度规则纠缠
        def _mk_merge_job(job_id, days_ago):
            return {
                "platform": "merge_test", "job_id": job_id, "title": f"岗位{job_id}",
                "company": "合并测试公司", "city": "广州", "salary": "200-300/天",
                "url": f"https://example.com/{job_id}",
                "publish_date": (date.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
                "description": "【岗位职责】参与 Agent 与 RAG 开发。" * 12,
            }

        merge_existing = [_mk_merge_job(f"m_ex_{i}", i) for i in range(15)]

        def _merge_18():
            got = len(merge_jds(merge_existing, [
                _mk_merge_job("m_new_1", 0), _mk_merge_job("m_new_2", 1),
                _mk_merge_job("m_new_3", 2),
            ])["jobs"])
            if got != 18:
                raise AssertionError(f"15+3 应为 18，实际 {got}")
            return "15 + 3（不同 id）→ 18 条"

        def _merge_dup_15():
            got = merge_jds(merge_existing, list(merge_existing))
            if len(got["jobs"]) != 15 or got["stats"]["duplicates"] != 15:
                raise AssertionError(f"15+15 重复应为 15：{got['stats']}")
            return f"15 + 15（完全重复）→ 15 条，识别重复 {got['stats']['duplicates']}"

        def _merge_stale_17():
            got = merge_jds(merge_existing, [
                _mk_merge_job("m_new_1", 0), _mk_merge_job("m_new_2", 1),
                _mk_merge_job("m_stale_91", 91),
            ])
            if len(got["jobs"]) != 17 or len(got["archived"]) != 1:
                raise AssertionError(f"含 91 天前应为 17+归档1：{got['stats']}")
            if got["archived"][0]["job_id"] != "m_stale_91":
                raise AssertionError("归档的不是那条 91 天前的记录")
            return "15 + 3（含 1 条 91 天前）→ 17 条，归档 1 条"

        def _merge_sorted():
            dates = [j["publish_date"] for j in merge_jds(
                merge_existing, [_mk_merge_job("m_new_1", 0)])["jobs"]]
            if dates != sorted(dates, reverse=True):
                raise AssertionError(f"未按 publish_date 降序：{dates}")
            return f"降序 OK（{dates[0]} → {dates[-1]}）"

        check("10. merge_jds：15+3 → 18", _merge_18)
        check("11. merge_jds：15+15 完全重复 → 15", _merge_dup_15)
        check("12. merge_jds：15+3 含 91 天前 → 17 + 归档 1", _merge_stale_17)
        check("13. merge_jds：publish_date 降序", _merge_sorted)

        # ---- 5. 落盘是「合并」不是「覆盖」（Round 8 的真实事故）----
        def _merge_write_not_overwrite():
            merge_dir = work / "merge_out"
            merge_dir.mkdir(parents=True, exist_ok=True)
            # 先放 15 条（模拟库里已有数据）
            merge_and_write_cleaned(merge_existing, out_dir=merge_dir)
            before_n = len(json.loads((merge_dir / "cleaned_jd.json").read_text(encoding="utf-8")))
            # 再合并 3 条新岗位
            res = merge_and_write_cleaned(
                [_mk_merge_job("m_brand_1", 0), _mk_merge_job("m_brand_2", 1),
                 _mk_merge_job("m_brand_3", 2)],
                out_dir=merge_dir,
            )
            after = json.loads((merge_dir / "cleaned_jd.json").read_text(encoding="utf-8"))
            if before_n != 15 or len(after) != 18:
                raise AssertionError(
                    f"落盘应 15 → 18（不是覆盖成 3），实际 {before_n} → {len(after)}；stats={res['stats']}"
                )
            return f"已有 15 + 新抓 3 → 落盘 {len(after)} 条（旧数据未被冲掉）"

        check("14. 落盘合并（不再覆盖）：15 + 3 → 18 条", _merge_write_not_overwrite)

        # ---- 6. 多城市抓取（Round 9 任务 4）----
        def _multi_city():
            calls = []

            def fake_by_city(keywords, city=None, **kwargs):
                calls.append(city)
                if city == "广州":
                    return [_mk_merge_job("gz_1", 0)]
                if city == "深圳":
                    return [_mk_merge_job("sz_1", 0)]
                return []

            res = scrape_all_cities(["广州", "深圳"], ["Agent"], scraper=fake_by_city)
            if calls != ["广州", "深圳"]:
                raise AssertionError(f"应逐个城市调用，实际 {calls}")
            if res["per_city"] != {"广州": 1, "深圳": 1} or len(res["jobs"]) != 2:
                raise AssertionError(f"多城市结果不对：{res['per_city']} / {len(res['jobs'])}")
            return f"两城市都抓到（{res['per_city']}），合并 {len(res['jobs'])} 条"

        check("15. 多城市抓取：广州 + 深圳都跑", _multi_city)

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
                        help="关键词组，逗号分隔同义词、分号分隔组，如 "
                             "\"Agent,智能体;RAG,检索增强生成\"（覆盖画像/环境变量，默认读画像）")
    parser.add_argument("--city", default="",
                        help="城市，逗号分隔可传多个（如 \"广州,深圳\"），逐个抓取后合并")
    parser.add_argument("--platforms", default="",
                        help="抓取平台，逗号分隔可传多个（如 \"shixiseng\"），逐个平台抓取后合并"
                             "（默认读环境变量 SCHEDULER_PLATFORMS，兜底 shixiseng）")
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
    # 启动第一行：确认城市池/关键词池到底读的是哪一份配置（改 yaml 后先看这行）。
    print(format_config_sources(config))
    if args.keywords:
        # 用组分隔符解析：`--keywords "Agent,智能体;RAG,检索增强生成"` 得到两组，
        # `--keywords 智能体` 得到一组一个词。两种写法都保持"同义词归组"的信息。
        config["keyword_groups"] = split_keyword_groups(args.keywords)
        config["keywords"] = flatten_keyword_groups(config["keyword_groups"])
    if args.city:
        config["city"] = args.city
        config["cities"] = _split_list(args.city)
    if args.platforms:
        config["platforms"] = _split_list(args.platforms)
    config["max_pages"] = args.max_pages
    config["limit_per_keyword"] = args.limit_per_keyword
    config["limit_total"] = args.limit_total
    if USE_MOCK:
        # 命令行 --platforms 也不能盖掉 mock 开关（开关的语义是"绝不联网"）
        config["platforms"] = ["mock"]
    # 命令行参数是"最后一次覆盖"，所以快速模式要在覆盖之后再收敛一次。
    # 这里静默（logger=None）：[FAST] 那行日志由 resolve_config 打，只打一遍就够。
    config = _apply_fast_mode(config, None)

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
