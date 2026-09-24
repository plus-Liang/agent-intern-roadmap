"""抓取配置加载器：城市池 / 关键词池 / 调度参数（config/scraping.yaml）。

目标：**改配置不改代码**。scheduler 里的硬编码值只作兜底。

优先级（真正决定取值的地方在 agent/scrapers/scheduler.py:resolve_config()）：
    环境变量 > config/scraping.yaml > user_profile.json / DEFAULT_KEYWORDS

依赖：优先 PyYAML；没装就退化成 JSON（YAML 是 JSON 的超集，只写列表/标量时
两种解析器结果一致）；都没有/解析失败 → 打印 WARN + 用内置默认值。
不新增任何第三方依赖。

对外接口：
    load_scraping_config() -> dict   # {cities, keywords, keyword_groups, schedule, sources}
    get_cities() -> list[str]
    get_keywords() -> list[str]
    get_keyword_groups() -> list[list[str]]
    get_schedule() -> dict
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 内置默认值：城市沿用现状口径（默认 5 城市），关键词为**全行业词池**
# （与 config/scraping.yaml 保持一致，改一处记得同步另一处）。
# 城市顺序沿用 agent/data/user_profile.json，避免改变抓取顺序与默认 city。
DEFAULT_CITIES = ["广州", "深圳", "北京", "上海", "杭州"]

# 全行业关键词池（互联网/金融/咨询/制造/医药/快消/教育/传媒/设计/法律/HR）。
# 相关性由这张词池保证，清洗端（rag/quality/cleaner.py）不再做二次相关性过滤。
DEFAULT_KEYWORDS = [
    # 互联网/软件
    "Agent", "大模型", "RAG", "算法工程师", "后端开发", "前端开发", "全栈",
    "Python", "Java", "Go", "C++", "测试工程师", "运维", "产品经理", "数据分析",
    # 金融
    "投行", "量化", "风控", "财务分析", "审计", "基金", "券商",
    # 咨询
    "战略咨询", "管理咨询", "行业研究",
    # 制造/硬件
    "机械工程师", "电气工程师", "自动化", "嵌入式", "FPGA", "工艺工程师",
    # 医药/生物
    "临床研究", "药物研发", "注册专员", "生物信息",
    # 快消/零售
    "市场营销", "品牌", "渠道", "供应链", "电商运营",
    # 教育
    "教师", "教研", "课程设计", "教务",
    # 传媒/内容
    "编辑", "记者", "视频剪辑", "新媒体运营",
    # 设计
    "UI设计", "平面设计", "工业设计", "交互设计",
    # 法律
    "法务", "律师助理", "合规",
    # 人力资源
    "招聘", "HRBP", "培训", "薪酬",
]

# 同义词分组（与 scheduler.DEFAULT_KEYWORDS 的分组一致）：
# 组内是同义词（各搜一次、结果合并去重），组间是不同概念。
# 配置文件里平铺写 keywords 时，靠这张表把已知同义词重新归组，
# 新词不认识就"一个词一组"。全行业词池里没有同义词，
# 因此每个词各自成一组（组数 == 词数）。
DEFAULT_KEYWORD_GROUPS = [
    ["Agent", "智能体"],
    ["大模型", "LLM", "大语言模型"],
    ["RAG", "检索增强生成"],
]

# 预留：Step 2 滚动抓取用；当前不改变抓取/清洗/入库行为。
DEFAULT_SCHEDULE = {"batch_size": 0, "max_age_days": 90}

CONFIG_FILENAME = "scraping.yaml"
# 环境变量覆盖配置文件路径（自测/多环境用；指向不存在的路径即"没有外置配置"）
CONFIG_PATH_ENV = "SCRAPING_CONFIG_PATH"


def _warn(message: str) -> None:
    print(f"WARN [config] {message}", file=sys.stderr)


def default_config_path() -> Path:
    """默认配置文件：<repo_root>/config/scraping.yaml（可用环境变量覆盖）。"""
    override = (os.getenv(CONFIG_PATH_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / "config" / CONFIG_FILENAME


def _default_config() -> dict:
    return {
        "cities": list(DEFAULT_CITIES),
        "keywords": list(DEFAULT_KEYWORDS),
        "keyword_groups": [list(group) for group in DEFAULT_KEYWORD_GROUPS],
        "schedule": dict(DEFAULT_SCHEDULE),
        "sources": {"cities": "default", "keywords": "default"},
    }


def _norm_list(value) -> list[str]:
    """把配置里的列表规整成"非空字符串、去重、保序"的列表。"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if item is None or isinstance(item, (list, tuple, dict)):
            continue
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _norm_groups(value) -> list[list[str]]:
    """规整可选的 keyword_groups: [[同义词...], ...] 嵌套写法。"""
    if not isinstance(value, (list, tuple)):
        return []
    groups: list[list[str]] = []
    seen: set[str] = set()
    for raw_group in value:
        if isinstance(raw_group, str):
            raw_group = [raw_group]
        words = [w for w in _norm_list(raw_group) if w not in seen]
        if not words:
            continue
        seen.update(words)
        groups.append(words)
    return groups


def _flatten(groups) -> list[str]:
    out: list[str] = []
    for group in groups or []:
        for word in group or []:
            if word not in out:
                out.append(word)
    return out


def group_keywords(keywords) -> list[list[str]]:
    """把平铺关键词按内置同义词表归组；不认识的词各自成组（保序）。"""
    known_index = {
        word: idx
        for idx, group in enumerate(DEFAULT_KEYWORD_GROUPS)
        for word in group
    }
    groups: list[list[str]] = []
    slot: dict[int, int] = {}
    for word in _norm_list(keywords):
        idx = known_index.get(word)
        if idx is None:
            groups.append([word])
            continue
        if idx not in slot:
            slot[idx] = len(groups)
            groups.append([])
        groups[slot[idx]].append(word)
    return groups


def _parse_text(text: str):
    """优先 PyYAML；没装则退化成 JSON 解析。"""
    try:
        import yaml  # noqa: PLC0415  可选依赖，装了就优先用
    except ImportError:  # pragma: no cover - 取决于运行环境
        _warn(f"没装 PyYAML，{CONFIG_FILENAME} 按 JSON 解析（YAML 是 JSON 超集）")
        return json.loads(text)
    return yaml.safe_load(text)


def _merge_schedule(value, base: dict) -> dict:
    merged = dict(base)
    if value is None:
        return merged
    if not isinstance(value, dict):
        _warn("schedule 不是映射（期望 batch_size / max_age_days），沿用默认调度参数")
        return merged
    for key in base:
        if key not in value:
            continue
        try:
            merged[key] = int(value[key])
        except (TypeError, ValueError):
            _warn(f"schedule.{key}={value[key]!r} 不是整数，沿用默认值 {base[key]}")
    return merged


def load_scraping_config(config_path=None) -> dict:
    """读取 config/scraping.yaml；文件不存在或解析失败 → 内置默认值。

    返回 dict：
        cities          list[str]
        keywords        list[str]       平铺关键词
        keyword_groups  list[list[str]] 同义词组（scheduler 用这个抓）
        schedule        dict            预留字段
        sources         dict            {"cities": env/config/profile/default,
                                         "keywords": ...} 供启动日志显示来源
    """
    path = Path(config_path) if config_path else default_config_path()
    if not path.is_file():
        return _default_config()

    try:
        raw = _parse_text(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 任何解析/IO 问题都回退，定时任务不能因此崩
        _warn(f"解析 {path} 失败：{exc}；改用内置默认城市/关键词")
        return _default_config()

    if not isinstance(raw, dict):
        _warn(f"{path} 顶层不是映射（期望 cities / keywords / schedule）；改用内置默认值")
        return _default_config()

    cfg = _default_config()
    sources = {"cities": "default", "keywords": "default"}

    cities = _norm_list(raw.get("cities"))
    if cities:
        cfg["cities"] = cities
        sources["cities"] = "config"
    elif raw.get("cities") is not None:
        _warn("cities 为空或格式不对（期望字符串列表），沿用内置默认城市")

    groups = _norm_groups(raw.get("keyword_groups"))
    keywords = _norm_list(raw.get("keywords"))
    if groups:
        cfg["keyword_groups"] = groups
        cfg["keywords"] = _flatten(groups)
        sources["keywords"] = "config"
    elif keywords:
        cfg["keywords"] = keywords
        cfg["keyword_groups"] = group_keywords(keywords)
        sources["keywords"] = "config"
    elif raw.get("keywords") is not None:
        _warn("keywords 为空或格式不对（期望字符串列表），沿用内置默认关键词")

    cfg["schedule"] = _merge_schedule(raw.get("schedule"), cfg["schedule"])
    cfg["sources"] = sources
    return cfg


def get_cities() -> list[str]:
    return list(load_scraping_config()["cities"])


def get_keywords() -> list[str]:
    return list(load_scraping_config()["keywords"])


def get_keyword_groups() -> list[list[str]]:
    return [list(group) for group in load_scraping_config()["keyword_groups"]]


def get_schedule() -> dict:
    return dict(load_scraping_config()["schedule"])


if __name__ == "__main__":  # pragma: no cover - 手工排查用
    _cfg = load_scraping_config()
    print(f"配置文件：{default_config_path()}")
    print(f"城市（{len(_cfg['cities'])}）：{'、'.join(_cfg['cities'])}  [来源 {_cfg['sources']['cities']}]")
    print(f"关键词（{len(_cfg['keywords'])}）：{'、'.join(_cfg['keywords'])}  [来源 {_cfg['sources']['keywords']}]")
    print(f"关键词组：{_cfg['keyword_groups']}")
    print(f"schedule：{_cfg['schedule']}")
