"""
实习僧（shixiseng.com）岗位抓取器 —— Playwright async 真实抓取实现。

用法：
    import asyncio
    from agent.scrapers.shixiseng import search_shixiseng

    jobs = asyncio.run(search_shixiseng("Agent 开发", city="北京", limit=20, max_pages=3))

    翻页说明（实测结论）：
      * 每页固定 **20 条**；`?page=N` 直接请求即可生效，与点击
        `.pagination-wrap .el-pagination button.btn-next` 结果完全一致。
      * ⚠️ 必须用站点**完整参数形态**（见 `_build_search_url`）。裸拼
        `?keyword=X&city=Y&page=N` 会被站点静默重置回 page=1 且返回 0 条，
        看起来像"这一页没岗位"，极易误判。
      * 跨页按 job_id 去重；某页没有新增岗位即停止翻页。

    发布时间说明（实测结论）：
      * **列表页没有发布时间**（3 页 × 20 张卡片全扫过，无「X天前」也无具体日期；
        筛选栏的「发布时间：」只是搜索过滤条件）。
      * 详情页 `.job_date` 内首个 span 是**刷新时间**，形如 "2026-07-07 21:51:33 刷新"，
        实测 12/12 命中；本模块统一格式化为 "YYYY-MM-DD"，拿不到则填 ""。
      * 别和 `.job_deadline`（截止日期，**截止**不是发布）混用。

    设计说明（重要）：
    实习僧搜索列表页对**标题和薪资**做了字体混淆（CMap 反爬）：
    真实文字被替换成私有区码位（U+E000~U+F8FF），页面上再通过
    动态 @font-face（/interns/iconfonts/file?rand=...）把这些码位渲染成正确形状。
    因此列表页直接取到的 title / salary 是乱码，
    而**详情页 /intern/<job_id> 没有任何混淆**（页面 PUA 字符数为 0）。

    所以本模块采用两段式抓取：
      1) 列表页解析卡片 -> 拿到 url / job_id / company / city（这些字段是明文）
      2) 对每张卡片访问详情页 -> 拿到明文 title / salary，
         以及 RAG 检索真正需要的 **JD 正文**（岗位职责 / 任职要求）

    JD 正文抓取说明（实测结论）：
      * 正文容器是 `.job_detail`（位于 .job-content .content_left .con-job .job_part 内），
        实测 3/3 命中；注意类名是 `job-content`（连字符）而不是 `job_content`。
      * 正文节点带 `white-space:pre-wrap`，用 inner_text 能拿到真实换行，
        因此可以用 \n 保留段落结构。
      * **「岗位职责」「任职要求」很多时候不是独立 DOM 段落**，而是正文里的一行纯文本
        （例如 .job_til 标题是「职位描述：」，正文内部再出现「岗位职责」「任职要求」「加分项」）。
        所以本模块做两级切分：先按 DOM 段落（.con-job + .job_til），再按正文行标题切分。
      * 详情页无字体混淆（实测 PUA 字符数 = 0），正文可直接使用。

    依赖：playwright（async API）+ 项目内 agent.tools.job_search.Job（不重复定义）。
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import random
import re
import sys
from datetime import date, timedelta
from typing import Any, Optional

from playwright.async_api import async_playwright

# ---------------------------------------------------------------------------
# 复用项目里已有的 Job dataclass，绝不重新定义
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools.job_search import Job  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BASE_URL = "https://www.shixiseng.com"
SEARCH_URL = BASE_URL + "/interns"
PLATFORM = "shixiseng"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
VIEWPORT = {"width": 1440, "height": 900}
CARD_TIMEOUT_MS = 15_000          # wait_for_selector 等卡片，15 秒
DETAIL_TIMEOUT_MS = 20_000
# 详情页限流：每次请求前随机等待 0.5~1.5 秒，降低被站点限流/风控的概率
DETAIL_SLEEP_RANGE = (0.5, 1.5)

# max_pages<=0（不限页数）时的硬上限安全阀。
# 实测 keyword=实习 无城市时有 46 页，这里给到 50 足以覆盖，同时防止无限循环。
MAX_PAGES_HARD_LIMIT = 50

DEBUG_DIR = pathlib.Path(__file__).resolve().parent
DEBUG_SCREENSHOT = DEBUG_DIR / "shixiseng_debug.png"
DEBUG_HTML = DEBUG_DIR / "shixiseng_debug.html"
DEBUG_CARDS = DEBUG_DIR / "shixiseng_debug_cards.html"

# 卡片容器候选选择器（按优先级）
CARD_SELECTORS = [
    ".intern-wrap.intern-item",
    ".intern-item",
    "div[class*='intern-item']",
    ".interns .intern-wrap",
    ".job-list .job-item",
    "div[class*='intern-wrap']",
]

# 字段候选选择器（同一类名在标题和公司上复用，故用更精确的父级限定）
FIELD_SELECTORS = {
    "title": [
        ".intern-detail__job a.title",
        ".intern-detail__job .title",
        ".intern-detail__job a",
        "a.title.ellipsis.font",
        "a.title",
        ".title",
    ],
    "company": [
        ".intern-detail__company a.title",
        ".intern-detail__company .title",
        ".intern-detail__company a",
        ".company-name",
        "[class*='company'] .title",
    ],
    "city": [
        ".intern-detail__job .city",
        ".tip .city",
        ".city",
        "[class*='city']",
    ],
    "salary": [
        ".intern-detail__job .day",
        ".day",
        "[class*='salary']",
        "[class*='money']",
    ],
}

# 详情页字段候选选择器
DETAIL_SELECTORS = {
    "title": [
        ".new_job_name",
        ".job-header .job-title",
        ".job_title",
        "h1",
        "[class*='job_name']",
        "[class*='job-title']",
    ],
    "salary": [
        ".job_money",
        ".job_msg .job_money",
        "[class*='job_money']",
        "[class*='salary']",
    ],
    "city": [
        ".job_position",
        "[class*='job_position']",
        "[class*='job_city']",
    ],
    # 「刷新时间」候选选择器（实测命中 .job_date，见下方说明）
    "publish_date": [
        ".job_date",
        ".job-header .job_date",
        "[class*='job_date']",
        "[class*='refresh']",
        "[class*='publish']",
    ],
}

# JD 正文（岗位职责 / 任职要求）候选选择器：按「精确 -> 兜底」优先级排列，
# 命中第一个非空结果即采用（不要再取最长，否则会落到 .job-content 把公司简介一起吃进来）。
DETAIL_JD_SELECTORS = [
    ".job_detail",                    # 精确：JD 正文容器（实测命中）
    ".job_part",                      # 正文外层（标题 + 正文）
    ".job_description",               # 旧版正文容器
    "[class*='job_detail']",
    "[class*='job_describe']",
    ".job-content .content_left",     # 兜底：详情区左栏（会带上投递要求/工作地点）
    ".job_content .content_left",
    ".job-content",                   # 兜底：整个详情区
    ".job_content",
    ".job-box",                       # 最后兜底：详情卡片
]

# 段落标题选择器：正文里没有任何行标题时，用 DOM 段落标题（「职位描述」）当兜底标签
DETAIL_SECTION_TITLE_SELECTORS = [".job_til", ".section_title", "h3", "h4"]

# 正文为空时最多 dump 几个详情页 HTML 供人工排查（避免刷盘）
MAX_EMPTY_JD_DUMPS = 3
DEBUG_JD_DIR = DEBUG_DIR / "shixiseng_debug_jd"

# 职责侧 / 要求侧段落标题（含常见变体）。
# 这些标题在实习僧正文里**常常只是一行纯文本**，不是独立 DOM 节点，
# 所以既用于 DOM 段落识别，也用于正文按行切分。
DUTY_HEADINGS = [
    "岗位职责", "工作职责", "职位职责", "职责描述", "岗位描述",
    "职位描述", "工作内容", "主要职责", "工作职责描述", "岗位工作内容",
]
REQ_HEADINGS = [
    "任职要求", "职位要求", "岗位要求", "工作要求", "任职资格",
    "任职条件", "任职需求", "招聘要求", "能力要求", "我们希望你",
    "加分项", "加分点",
]

# 段落标题行都很短；超过该长度不认为是标题，避免把正文长句误切成段落
MAX_HEADING_LEN = 40

# Job.description 的段落标签格式：【岗位职责】/【任职要求】/【职位描述】
SECTION_LABEL_FMT = "【{label}】"

# ---------------------------------------------------------------------------
# 发布时间（刷新型）相关常量
#
# 实测结论（2026-09 实抓，勿轻易改动）：
#   * **列表页没有任何发布时间**。卡片只有 岗位名/薪资/城市/公司/福利标签/公司简介，
#     3 页 × 20 张卡片全扫过，既无「X天前」也无具体日期；筛选栏里的
#     「发布时间：」只是搜索过滤条件（对应 URL 参数 publishTime），不是岗位数据。
#   * **详情页有刷新时间**，容器 `.job_date`，内部首个 span（class 含 cutom_font）：
#         <div class="job_date"><span class="cutom_font">2026-07-07 21:51:33</span> <span>刷新</span></div>
#     实测 12/12 命中，格式恒为 "YYYY-MM-DD HH:MM:SS"，语义是**刷新时间**（站点无首次发布时间）。
#   * 注意别和 `.job_deadline`（截止日期：2026-10-08）混用——那是截止，不是发布。
#
# 站点目前给的是绝对日期，因此「X天前」换算只是兜底（防站点改版），主路径是直接截取日期。
# ---------------------------------------------------------------------------
PUBLISH_DATE_FMT = "%Y-%m-%d"
RELATIVE_PUBLISH_RE = re.compile(r"(\d+)\s*(天|小时|分钟|周|月)\s*前")
ABS_PUBLISH_RE = re.compile(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})")
TODAY_WORDS = ("刚刚", "今天", "今日")
YESTERDAY_WORDS = ("昨天", "昨日")

# 最近一次 search_shixiseng 抓到的 job_id -> "YYYY-MM-DD" 映射。
#
# 为什么要有它：agent.tools.job_search.Job **没有 publish_date 字段**，而本任务
# 明确禁止修改 job_search.py，所以日期不塞进 Job（避免破坏 dataclass 契约），
# 而是放在这里 + 诊断输出 + shixiseng_result.json 里，供清洗器/入库环节按 job_id 取用。
# 每次 search_shixiseng 开始时会清空，避免误用上一轮的陈旧数据。
LAST_PUBLISH_DATES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _clean(text: Optional[str]) -> str:
    """去掉首尾空白并合并内部空白（详情页 inner_text 里会有 \n）。"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _clean_multiline(text: Optional[str]) -> str:
    """
    正文清洗：**保留段落结构**（与 _clean 不同，_clean 会把 \\n 压成空格）。

    做三件事：
      1) 统一换行为 \\n，清掉全角空格 / 不换行空格 / 零宽字符（实习僧正文里常见）
      2) 去掉每行首尾空白，并把行内连续空格压成一个
      3) 连续空行压成一个，去掉首尾空行
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[\u200b-\u200f\u2028\u2029\ufeff]", "", text)

    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if not ln and out and not out[-1]:
            continue          # 连续空行只保留一个（保住段落间隔）
        out.append(ln)
    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def _has_obfuscated(text: str) -> bool:
    """判断文本是否含私有区码位（实习僧字体混淆的标记）。"""
    return any(0xE000 <= ord(ch) <= 0xF8FF for ch in text or "")


def extract_job_id(url: str) -> str:
    """从 URL 最后一段提取 job_id，例如 .../intern/inn_rv4mtayw9ltj?pcm=... -> inn_rv4mtayw9ltj。"""
    if not url:
        return ""
    path = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    return path.rsplit("/", 1)[-1] if path else ""


async def _first_text(node: Any, selectors: list[str]) -> str:
    """
    依次尝试多个候选选择器，返回第一个非空文本。
    单个选择器失败不影响其他候选（robustness 要求 e）。
    """
    for sel in selectors:
        try:
            loc = node.locator(sel)
            count = await loc.count()
        except Exception:  # noqa: BLE001 - 选择器非法/节点失效都跳过
            continue
        if count == 0:
            continue
        for idx in range(min(count, 3)):
            try:
                txt = _clean(await loc.nth(idx).inner_text(timeout=3_000))
            except Exception:  # noqa: BLE001
                continue
            if txt:
                return txt
    return ""


async def _all_texts(node: Any, selector: str) -> list[str]:
    """取某选择器下所有元素的文本（用于标签），失败返回空列表。"""
    out: list[str] = []
    try:
        loc = node.locator(selector)
        count = await loc.count()
        for idx in range(min(count, 20)):
            try:
                txt = _clean(await loc.nth(idx).inner_text(timeout=2_000))
            except Exception:  # noqa: BLE001
                continue
            if txt:
                out.append(txt)
    except Exception:  # noqa: BLE001
        pass
    return out


async def _first_text_int(page: Any, selectors: list[str]) -> str:
    """_first_text 的页面级版本（选择器直接作用于 page）。"""
    for sel in selectors:
        try:
            loc = page.locator(sel)
            count = await loc.count()
        except Exception:  # noqa: BLE001
            continue
        if count == 0:
            continue
        for idx in range(min(count, 3)):
            try:
                txt = _clean(await loc.nth(idx).inner_text(timeout=3_000))
            except Exception:  # noqa: BLE001
                continue
            if txt:
                return txt
    return ""


# ---------------------------------------------------------------------------
# 发布时间解析
# ---------------------------------------------------------------------------
def _format_publish_date(raw: str) -> str:
    """
    把详情页的刷新时间文本统一格式化为 "YYYY-MM-DD"；拿不到返回 ""。

    支持的输入形态（按优先级）：
        1) 绝对日期："2026-07-07 21:51:33" / "2026-07-07" / "2026年7月7日"
        2) 相对日期："3天前" / "5小时前" / "2周前" / "1个月前"（兜底，站点目前不用）
        3) 相对词：  "刚刚" / "今天" / "昨天"

    注意：文本里通常还带「刷新」二字（如 "2026-07-07 21:51:33 刷新"），
    以及可能混入「截止日期：2026-10-08」这类别的日期；本函数只负责把**第一个**
    可识别的日期转成 YYYY-MM-DD，取错字段的责任在选择器层（.job_date 已实测正确）。
    """
    s = _clean(raw)
    if not s:
        return ""

    # 1) 绝对日期
    m = ABS_PUBLISH_RE.search(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).strftime(PUBLISH_DATE_FMT)
        except ValueError:
            return ""          # 例如 2026-02-30 这种非法日期

    # 2) 相对日期（兜底）
    rel = RELATIVE_PUBLISH_RE.search(s)
    if rel:
        n = int(rel.group(1))
        unit = rel.group(2)
        days = {"天": n, "周": n * 7, "月": n * 30, "小时": 0, "分钟": 0}[unit]
        return (date.today() - timedelta(days=days)).strftime(PUBLISH_DATE_FMT)

    # 3) 相对词
    if any(w in s for w in TODAY_WORDS):
        return date.today().strftime(PUBLISH_DATE_FMT)
    if any(w in s for w in YESTERDAY_WORDS):
        return (date.today() - timedelta(days=1)).strftime(PUBLISH_DATE_FMT)

    return ""


async def _extract_publish_date(page: Any) -> tuple[str, str]:
    """
    从详情页抽「刷新时间」，返回 (YYYY-MM-DD, 命中的选择器)。

    实测命中 `.job_date`（形如 "2026-07-07 21:51:33 刷新"）。
    拿不到时返回 ("", "")，绝不抛异常——时间字段缺失不应该拖垮整条岗位的抓取。
    """
    for sel in DETAIL_SELECTORS["publish_date"]:
        # 先试容器内首个 span（最精确），再退回容器整体文本
        for target in (f"{sel} span", sel):
            try:
                loc = page.locator(target)
                count = await loc.count()
            except Exception:  # noqa: BLE001 - 选择器非法就试下一个
                continue
            if not count:
                continue
            try:
                txt = await loc.first.inner_text(timeout=3_000)
            except Exception:  # noqa: BLE001
                continue
            fmt = _format_publish_date(txt)
            if fmt:
                return fmt, target
    return "", ""


# ---------------------------------------------------------------------------
# JD 正文（岗位职责 / 任职要求）解析
# ---------------------------------------------------------------------------
def _classify_heading_line(line: str) -> Optional[tuple[str, str, str]]:
    """
    判断某一行是不是段落标题，返回 (侧别, 标题, 标题后同一行的剩余内容)：
        side  = "duty"（职责侧）/ "req"（要求侧）
        title = 命中的标题原文（如「岗位职责」「加分项」「要求」）
        rest  = 标题后同一行的剩余内容，没有则为 ""
    不匹配返回 None。

    兼容两种写法：
        标题独占一行          -> "岗位职责"
        标题 + 冒号 + 同行内容 -> "岗位职责：1. 参与 xxx"
    另外把纯「要求：xxx」也识别为要求侧（实习僧正文里很常见）。
    """
    s = (line or "").strip()
    if not s:
        return None
    # 形式一：「标题：」或「标题：同行内容」。
    # 只检查冒号前的标题部分，因此**不受整行长度限制**——实习僧正文里
    # 「要求：Go / Python / TypeScript 至少一门，能流畅读英文技术文档…」这类
    # 带同行内容的长行很常见，误用长度阈值会漏切（实测踩过）。
    for side, names in (("duty", DUTY_HEADINGS), ("req", REQ_HEADINGS)):
        for name in names:
            for sep in ("：", ":"):
                prefix = name + sep
                if s.startswith(prefix):
                    return side, name, s[len(prefix):].strip()
    # 裸「要求」只在带冒号的时候才算标题，避免把「要求具备…」这类正文误判
    m = re.match(r"^要求\s*[:：]\s*(.*)$", s)
    if m:
        return "req", "要求", m.group(1).strip()
    # 形式二：标题独占一行（无冒号）。标题都很短，超长的一定不是标题
    if len(s) <= MAX_HEADING_LEN:
        for side, names in (("duty", DUTY_HEADINGS), ("req", REQ_HEADINGS)):
            if s in names:
                return side, s, ""
    return None


def _split_jd_text(text: str, default_label: str) -> list[tuple[str, str]]:
    """
    把 JD 正文切成 [(段落标签, 段落正文)]。

    策略：先按行标题切（职责侧 / 要求侧）；如果一个标题都没遇到，
    就整段归到 default_label（来自 DOM 段落标题，例如「职位描述」）。
    相邻的同标签段落会合并，但**保留子标题**（如「加分项」）：
    否则两段「1. 2. 3.」编号会直接连在一起，丢掉分段语义。
    """
    blocks: list[tuple[str, str, list[str]]] = []   # (标签, 子标题, 正文行)
    cur_label = ""
    cur_heading = ""
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_label, cur_heading, cur_lines
        if cur_label or any(ln.strip() for ln in cur_lines):
            blocks.append((cur_label, cur_heading, cur_lines))
        cur_label, cur_heading, cur_lines = "", "", []

    for line in text.split("\n"):
        hit = _classify_heading_line(line)
        if hit:
            side, title, rest = hit
            flush()
            cur_label = "岗位职责" if side == "duty" else "任职要求"
            cur_heading = title
            if rest:
                cur_lines.append(rest)
        else:
            cur_lines.append(line)
    flush()

    # 合并相邻同标签块，丢掉空正文
    merged: list[list[str]] = []          # [标签, 正文]
    for label, heading, lines in blocks:
        body = _clean_multiline("\n".join(lines))
        if not body:
            continue
        if merged and merged[-1][0] == label:
            # 该标签的首段已由【标签】表达，子标题与其相同则不再重复；
            # 「加分项」这类附属小节标题则保留下来
            sub = heading if heading and heading != label else ""
            merged[-1][1] += "\n" + (sub + "\n" if sub else "") + body
        else:
            merged.append([label, body])

    # 一个标题都没识别出来 -> 整段挂到默认标签
    if not merged and text.strip():
        merged = [[default_label, _clean_multiline(text)]]

    # 正文出现在任何标题之前的段落，标签为空 -> 补默认标签
    return [((label or default_label), body) for label, body in merged]


async def _dom_jd_title(page: Any) -> str:
    """
    取 DOM 段落标题作为兜底标签（实测为「职位描述」）。
    用于页面把职责和要求写在同一个段落、正文里没有行标题的情况。
    """
    for sel in DETAIL_SECTION_TITLE_SELECTORS:
        try:
            loc = page.locator(sel)
            count = await loc.count()
        except Exception:  # noqa: BLE001
            continue
        for idx in range(min(count, 5)):
            try:
                txt = _clean(await loc.nth(idx).inner_text(timeout=2_000))
            except Exception:  # noqa: BLE001
                continue
            cleaned = txt.rstrip("：:")
            if cleaned and len(cleaned) <= 12:
                return cleaned
    return "职位描述"


async def _extract_jd(page: Any) -> tuple[str, str, bool]:
    """
    从详情页抽取 JD 正文。

    返回 (description, source, obfuscated)：
        description: 带段落标签、保留 \\n 段落结构的正文；抓不到返回 ""
        source:      命中的选择器（诊断用）
        obfuscated:  正文里是否含字体混淆的私有区码位
    """
    raw = ""
    source = ""
    for sel in DETAIL_JD_SELECTORS:
        try:
            loc = page.locator(sel)
            count = await loc.count()
        except Exception:  # noqa: BLE001 - 选择器非法，试下一个
            continue
        if not count:
            continue
        best = ""
        for idx in range(min(count, 3)):
            try:
                node = loc.nth(idx)
                try:
                    txt = await node.inner_text(timeout=3_000)
                except Exception:  # noqa: BLE001 - inner_text 失败则退回原始文本
                    txt = await node.text_content(timeout=3_000)
            except Exception:  # noqa: BLE001
                continue
            txt = _clean_multiline(txt)
            if len(txt) > len(best):
                best = txt
        if best:
            raw, source = best, sel
            break

    if not raw:
        return "", "", False

    default_label = await _dom_jd_title(page)
    blocks = _split_jd_text(raw, default_label)
    description = "\n\n".join(
        SECTION_LABEL_FMT.format(label=label) + "\n" + body for label, body in blocks
    )
    return description, source, _has_obfuscated(raw)


async def _dump_detail_html(page: Any, job_id: str, index: int) -> None:
    """正文抓不到时 dump 详情页 HTML，便于人工核对选择器（最多 MAX_EMPTY_JD_DUMPS 个）。"""
    try:
        DEBUG_JD_DIR.mkdir(parents=True, exist_ok=True)
        path = DEBUG_JD_DIR / f"detail_{index:02d}_{job_id or 'unknown'}.html"
        html = await page.content()
        path.write_text(html, encoding="utf-8")
        print(f"[诊断]   已保存详情页 HTML 供排查：{path}（{len(html)} 字符）")
    except Exception as exc:  # noqa: BLE001
        print(f"[诊断]   保存详情页 HTML 失败：{type(exc).__name__}: {exc}")


async def _launch_browser(playwright: Any, headless: bool) -> Any:
    """
    启动浏览器，采用「回退策略」：chromium -> chrome -> msedge，取第一个能启动的。

    为什么需要回退（本机实测结论，勿轻易简化）：
      1) Chromium 在此环境**没有安装成功**。Playwright 的注册信息还在
         （playwright.chromium.executable_path 会返回
         C:\\Users\\<user>\\AppData\\Local\\ms-playwright\\chromium-1243\\chrome-win64\\chrome.exe），
         但该目录/可执行文件实际不存在；headless 模式还需要额外缺失的
         chromium_headless_shell。因此第 1 个候选通常会抛
         "Executable doesn't exist"，属于**预期失败**，直接落到下一个候选。
      2) 系统已安装 Google Chrome / Microsoft Edge 时，可用 Playwright 的
         channel 参数直接驱动它们。Edge 与 Chromium 同源，**内核兼容 Playwright API**，
         本机实测由 channel="msedge" 启动成功并完整跑通抓取。
      3) 一旦执行 `python -m playwright install chromium` 装好自带 chromium，
         第 1 个候选就会自动生效，无需改动本函数。

    这样既能在当前环境下开箱可用，又能在装好 chromium 后无感切换。
    """
    attempts: list[dict[str, Any]] = [
        # 第 1 候选：Playwright 自带的 chromium（本机未装成功，会失败并回退）
        {"name": "chromium(Playwright 自带)", "kwargs": {}},
        # 第 2 候选：系统安装的 Google Chrome
        {"name": "chrome(系统安装)", "kwargs": {"channel": "chrome"}},
        # 第 3 候选：系统安装的 Microsoft Edge（本机实际可用，内核兼容 Playwright API）
        {"name": "msedge(系统安装)", "kwargs": {"channel": "msedge"}},
    ]
    errors: list[str] = []
    for attempt in attempts:
        try:
            browser = await playwright.chromium.launch(headless=headless, **attempt["kwargs"])
            print(f"[诊断] 浏览器已启动：{attempt['name']} (headless={headless})")
            return browser
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{attempt['name']}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
    raise RuntimeError(
        "无法启动任何浏览器，尝试记录：\n  " + "\n  ".join(errors)
        + "\n提示：可用 `python -m playwright install chromium` 安装自带 chromium。"
    )


# ---------------------------------------------------------------------------
# 列表页解析
# ---------------------------------------------------------------------------
# 每页条数（实测 20 条/页，keyword=实习 无城市时有 46 页）
PAGE_SIZE = 20

# 站点的完整搜索参数模板。
#
# ⚠️ 用完整参数而不是裸拼 `?keyword=X&city=Y&page=N`，是本任务实测踩到过的坑：
#   裸拼 URL 会被站点**重置回 page=1 并返回 0 条**（静默失败，不报错也不跳转），
#   看起来就像"这一页没有岗位"，非常容易误判。缺任何一个参数都可能触发。
SEARCH_PARAM_ORDER = [
    "page", "type", "keyword", "area", "months", "days", "degree",
    "official", "enterprise", "salary", "publishTime", "sortType",
    "city", "internExtend",
]


def _build_search_url(keyword: str, city: Optional[str], page: int) -> str:
    """
    生成第 page 页的搜索 URL（站点完整参数形态，已实测 page 参数生效）。

    page 从 1 开始；站点固定参数沿用其默认值：
        salary=-0（薪资不限）、其余过滤项为空。
    """
    from urllib.parse import quote

    params = {
        "page": str(page),
        "type": "intern",
        "keyword": keyword,
        "area": "",
        "months": "",
        "days": "",
        "degree": "",
        "official": "",
        "enterprise": "",
        "salary": "-0",
        "publishTime": "",
        "sortType": "",
        "city": city or "",
        "internExtend": "",
    }
    # 注意：这里不能过滤空值——站点需要这些参数**存在且为空**，缺参数会导致 page 失效
    query = "&".join(
        f"{k}={quote(params[k])}" for k in SEARCH_PARAM_ORDER
    )
    return f"{SEARCH_URL}?{query}"


async def _find_card_selector(page: Any) -> tuple[str, int]:
    """wait_for_selector 等卡片出现，返回命中的选择器和数量。"""
    for sel in CARD_SELECTORS:
        try:
            await page.wait_for_selector(sel, timeout=CARD_TIMEOUT_MS, state="attached")
            count = await page.locator(sel).count()
            if count:
                return sel, count
        except Exception:  # noqa: BLE001 - 该候选超时/不存在，试下一个
            continue
    return "", 0


async def _parse_card(card: Any, index: int) -> Optional[dict[str, Any]]:
    """解析单张卡片，失败返回 None（不影响其他卡片）。"""
    try:
        href = ""
        for sel in (".intern-detail__job a.title", ".intern-detail__job a", "a"):
            try:
                loc = card.locator(sel).first
                if await loc.count():
                    href = await loc.get_attribute("href") or ""
                    if href:
                        break
            except Exception:  # noqa: BLE001
                continue
        if not href:
            return None

        url = href if href.startswith("http") else BASE_URL + href
        job_id = extract_job_id(url)

        raw_title = await _first_text(card, FIELD_SELECTORS["title"])
        company = await _first_text(card, FIELD_SELECTORS["company"])
        city = await _first_text(card, FIELD_SELECTORS["city"])
        salary = await _first_text(card, FIELD_SELECTORS["salary"])
        tags = await _all_texts(card, ".intern-label") or None

        return {
            "index": index,
            "url": url,
            "job_id": job_id,
            "title_raw": raw_title,
            "company": company,
            "city": city,
            "salary_raw": salary,
            "tags": tags,
            "title_obfuscated": _has_obfuscated(raw_title),
            "salary_obfuscated": _has_obfuscated(salary),
        }
    except Exception as exc:  # noqa: BLE001 - 单卡失败不影响整体
        print(f"[诊断] 卡片 #{index} 解析失败：{type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 详情页解析（拿明文 title / salary / JD 正文）
# ---------------------------------------------------------------------------
async def _fetch_detail(page: Any, url: str) -> dict[str, Any]:
    """
    访问详情页，返回明文 title/salary/city + JD 正文 description。

    失败返回空字典（调用方按「字段缺失」处理，不中断整体）。
    """
    out: dict[str, Any] = {}
    try:
        # 限流：每次详情请求前随机等待 0.5~1.5 秒，避免高频请求被站点限流
        delay = random.uniform(*DETAIL_SLEEP_RANGE)
        await asyncio.sleep(delay)

        await page.goto(url, wait_until="domcontentloaded", timeout=DETAIL_TIMEOUT_MS)
        try:
            await page.wait_for_selector(".job-header, .new_job_name", timeout=10_000)
        except Exception:  # noqa: BLE001 - 结构变了也别直接放弃
            pass
        await page.wait_for_timeout(300)

        title = await _first_text_int(page, DETAIL_SELECTORS["title"])
        if not title or _has_obfuscated(title):
            # 兜底：详情页 <title> 形如
            # "Agent 开发工程师实习招聘-北京脑利科技实习生招聘-实习僧"
            page_title = await page.title()
            if page_title:
                guess = re.split(r"实习招聘|招聘-|_|-实习僧", page_title)[0]
                guess = _clean(guess)
                if guess and not _has_obfuscated(guess):
                    title = guess

        salary = await _first_text_int(page, DETAIL_SELECTORS["salary"])
        if not salary:
            body = await page.inner_text("body")
            match = re.search(r"\d[\d\s\-–~]*\s*/\s*天", body)
            if match:
                salary = _clean(match.group(0))

        city = await _first_text_int(page, DETAIL_SELECTORS["city"])

        # 刷新时间（发布时间）—— 列表页没有这个字段，只能从详情页取
        publish_date, publish_src = await _extract_publish_date(page)

        # JD 正文（岗位职责 / 任职要求）—— RAG 检索的主料
        description, desc_source, desc_obfuscated = await _extract_jd(page)

        if title and not _has_obfuscated(title):
            out["title"] = title
        if salary and not _has_obfuscated(salary):
            out["salary"] = salary
        if city and not _has_obfuscated(city):
            out["city"] = city
        out["publish_date"] = publish_date
        out["publish_date_source"] = publish_src
        out["description"] = description
        out["description_source"] = desc_source
        out["description_obfuscated"] = desc_obfuscated
    except Exception as exc:  # noqa: BLE001 - 详情页失败不应中断整体
        print(f"[诊断] 详情页失败 {url} -> {type(exc).__name__}: {str(exc).splitlines()[0][:140]}")
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
async def search_shixiseng(
    keyword: str,
    city: str = None,
    limit: int = 20,
    headless: bool = False,
    fetch_detail: bool = True,
    max_pages: int = 3,
) -> list[Job]:
    """
    抓取实习僧搜索页岗位（支持翻页 + 跨页去重）。

    参数：
        keyword: 搜索关键词，如 "Agent 开发"
        city: 城市，如 "北京"；None 表示不限
        limit: 返回数量上限
        headless: 是否无头模式（默认 False，可视化便于调试）
        fetch_detail: 是否访问详情页补齐明文 title/salary、JD 正文与发布时间
                      （列表页 title/salary 被字体混淆，且正文/发布时间只在详情页，
                       强烈建议 True）
        max_pages: 最多翻多少页（每页实测 20 条，默认 3 页 ≈ 60 条）。
                   <=0 表示不设上限，翻到没有新岗位为止（有硬上限安全阀）。

    返回：
        list[Job]（复用 agent.tools.job_search.Job）

    去重：
        按 job_id 去重（同一条岗位重复出现时只留最先遇到的一条）。
        另维护模块级 LAST_PUBLISH_DATES（job_id -> "YYYY-MM-DD"）：
        Job dataclass 没有 publish_date 字段且不允许改 job_search.py，
        所以日期不塞进 Job，放在那里供清洗/入库环节按 job_id 取用。
    """
    if max_pages is None or max_pages < 0:
        max_pages = 0          # 0 = 不设上限

    # 每轮清空，避免调用方误读上一轮的日期
    LAST_PUBLISH_DATES.clear()

    print("=" * 70)
    print(f"[诊断] keyword={keyword!r} city={city!r} limit={limit} "
          f"max_pages={max_pages or '不限'}  headless={headless}")

    jobs: list[Job] = []
    raw_unique: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    page_stats: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await _launch_browser(p, headless)
        context = await browser.new_context(
            viewport=VIEWPORT,           # 要求 b) viewport 1440x900
            user_agent=UA,               # 要求 a) UA 伪装
            locale="zh-CN",
            extra_http_headers={"Accept-Language": "zh-CN,zh;q=0.9"},
        )
        page = await context.new_page()
        try:
            # ------------------------------------------------------------------
            # 翻页抓列表：每页解析卡片 -> 按 job_id 去重累加
            # ------------------------------------------------------------------
            page_no = 1
            while True:
                if max_pages and page_no > max_pages:
                    print(f"[诊断] 已达 max_pages={max_pages}，停止翻页。")
                    break

                url = _build_search_url(keyword, city, page_no)
                print("-" * 70)
                print(f"[诊断] 第 {page_no} 页 URL：{url}")

                try:
                    resp = await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                except Exception as exc:  # noqa: BLE001 - 单页导航失败不放弃整轮
                    print(f"[诊断][警告] 第 {page_no} 页导航失败："
                          f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}")
                    break

                print(f"[诊断] 第 {page_no} 页 HTTP 状态："
                      f"{resp.status if resp else 'N/A'}  最终 URL：{page.url}")
                # 站点在参数不全时会静默把 page 重置回 1，这里明确告警，避免误判成"本页没岗位"
                if page_no > 1 and f"page={page_no}" not in page.url:
                    print(f"[诊断][警告] 站点未接受 page={page_no}，最终 URL 为 {page.url}"
                          "（可能被重置回第 1 页）。")

                card_selector, card_count = await _find_card_selector(page)
                print(f"[诊断] 第 {page_no} 页命中卡片选择器：{card_selector!r}  "
                      f"卡片数：{card_count}")

                # 等页面渲染稳定一点再截图
                await page.wait_for_timeout(1_500)

                # 要求 d) 截图（只截第 1 页，避免每页都刷盘）
                if page_no == 1:
                    try:
                        await page.screenshot(path=str(DEBUG_SCREENSHOT), full_page=True)
                        print(f"[诊断] 已保存截图：{DEBUG_SCREENSHOT}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[诊断] 截图失败：{type(exc).__name__}: {exc}")

                if not card_count:
                    # 要求 7) 选择器找不到 -> 保存截图 + HTML 便于人工分析
                    print(f"[诊断] 第 {page_no} 页未找到任何岗位卡片，停止翻页。")
                    if page_no == 1:
                        await _dump_debug(page, note="没有命中任何卡片选择器")
                    break

                cards = page.locator(card_selector)
                page_items: list[dict[str, Any]] = []
                page_dupes = 0
                for i in range(card_count):
                    item = await _parse_card(cards.nth(i), i)
                    if not item:
                        continue
                    jid = item["job_id"]
                    if jid and jid in seen_ids:
                        page_dupes += 1
                        continue
                    if jid:
                        seen_ids.add(jid)
                    item["page"] = page_no
                    page_items.append(item)

                raw_unique.extend(page_items)
                page_stats.append({
                    "page": page_no,
                    "cards": card_count,
                    "parsed": len(page_items) + page_dupes,
                    "new": len(page_items),
                    "dupes": page_dupes,
                })

                # 要求：诊断输出每页抓到几条、累计多少条、去重后多少条
                print(f"[诊断] 第 {page_no} 页小结：卡片 {card_count} 条，"
                      f"解析成功 {len(page_items) + page_dupes} 条，"
                      f"本页去重丢弃 {page_dupes} 条，新增 {len(page_items)} 条")
                print(f"[诊断] 累计：抓取总数 {sum(s['parsed'] for s in page_stats)} 条，"
                      f"去重后 {len(raw_unique)} 条")

                # 保存首卡 HTML，便于后续核对选择器（只存第 1 页第一张）
                if page_no == 1:
                    try:
                        first_html = await cards.first.inner_html()
                        DEBUG_CARDS.write_text(first_html, encoding="utf-8")
                        print(f"[诊断] 首卡 HTML 已保存：{DEBUG_CARDS}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[诊断] 保存首卡 HTML 失败：{type(exc).__name__}: {exc}")

                # 翻页终止条件：本页没有新增岗位（可能到末页，也可能站点把翻页重置了）
                if not page_items:
                    print(f"[诊断] 第 {page_no} 页没有新增岗位，停止翻页。")
                    break

                # 安全阀：max_pages<=0（不限）时避免无限翻
                if not max_pages and page_no >= MAX_PAGES_HARD_LIMIT:
                    print(f"[诊断][警告] 达到硬上限 {MAX_PAGES_HARD_LIMIT} 页，停止翻页。")
                    break

                page_no += 1

            print("-" * 70)
            print("[诊断] 翻页汇总（每页：卡片 / 解析 / 去重丢弃 / 新增）：")
            for st in page_stats:
                print(f"[诊断]   第 {st['page']:2d} 页：卡片 {st['cards']:3d}  "
                      f"解析 {st['parsed']:3d}  去重丢弃 {st['dupes']:3d}  新增 {st['new']:3d}")
            total_parsed = sum(s["parsed"] for s in page_stats)
            total_dupes = sum(s["dupes"] for s in page_stats)
            print(f"[诊断] 抓取总数 {total_parsed} 条，跨页去重丢弃 {total_dupes} 条，"
                  f"去重后 {len(raw_unique)} 条")

            if not raw_unique:
                await _dump_debug(page, note="卡片存在但全部解析失败")
                return jobs

            candidates = raw_unique[:limit] if limit else raw_unique
            need_detail = candidates if fetch_detail else []
            obf_count = sum(
                1 for it in candidates
                if it["title_obfuscated"] or it["salary_obfuscated"]
                or not it["title_raw"] or not it["salary_raw"]
            )
            print(f"[诊断] 需要访问详情页的岗位数：{len(need_detail)}"
                  f"（候选 {len(candidates)}/{len(raw_unique)}，limit={limit}，"
                  f"其中 title/salary 需补全 {obf_count} 条）")
            if need_detail:
                print("[诊断] 说明：实习僧列表页对 标题/薪资 做了字体混淆（私有区码位），"
                      "且 JD 正文与发布时间只存在于详情页，故逐条补抓详情页。")

            empty_jd: list[str] = []
            missing_date: list[str] = []
            if need_detail:
                for n, item in enumerate(need_detail, 1):
                    print(f"[诊断] ({n}/{len(need_detail)}) 详情页：{item['url']}")
                    detail = await _fetch_detail(page, item["url"])
                    item["title_detail"] = detail.get("title", "")
                    item["salary_detail"] = detail.get("salary", "")
                    item["city_detail"] = detail.get("city", "")
                    item["description"] = detail.get("description", "")
                    item["description_source"] = detail.get("description_source", "")
                    item["description_obfuscated"] = detail.get("description_obfuscated", False)
                    item["publish_date"] = detail.get("publish_date", "")
                    item["publish_date_source"] = detail.get("publish_date_source", "")

                    # 要求：打印每条 JD 正文长度 + 发布时间
                    print(f"[诊断]     JD 正文长度：{len(item['description'])} 字符"
                          f"（来源选择器：{item['description_source'] or '未命中'}）")
                    print(f"[诊断]     发布时间：{item['publish_date'] or '（未取到）'}"
                          f"（来源选择器：{item['publish_date_source'] or '未命中'}）")
                    if not item["publish_date"]:
                        missing_date.append(item["job_id"])
                    if not item["description"]:
                        # 要求：正文为空必须明确报出来，不许静默
                        print(f"[诊断][警告] 岗位 {item['job_id']} 的 JD 正文为空！"
                              f" url={item['url']}")
                        empty_jd.append(item["job_id"])
                        if len(empty_jd) <= MAX_EMPTY_JD_DUMPS:
                            await _dump_detail_html(page, item["job_id"], n)
                    elif item["description_obfuscated"]:
                        print(f"[诊断][警告] 岗位 {item['job_id']} 的 JD 正文含字体混淆字符，"
                              "入库前需人工确认。")

            # 组装 Job，最多 limit 条
            flagged = 0
            for item in raw_unique:
                if len(jobs) >= limit:
                    break
                title = item.get("title_detail") or item["title_raw"]
                salary = item.get("salary_detail") or item["salary_raw"]
                city_val = item.get("city_detail") or item["city"]
                if _has_obfuscated(title) or _has_obfuscated(salary):
                    flagged += 1
                    title = _clean(title)
                    salary = _clean(salary)
                if item.get("publish_date"):
                    LAST_PUBLISH_DATES[item["job_id"]] = item["publish_date"]
                jobs.append(
                    Job(
                        platform=PLATFORM,
                        job_id=item["job_id"],
                        title=title,
                        company=item["company"],
                        city=city_val,
                        salary=salary,
                        url=item["url"],
                        tags=item.get("tags"),
                        # 说明：Job 只有一个 description 字段（不改 job_search.py），
                        # 这里存「岗位职责 + 任职要求」的拼接文本，用【】标签分段。
                        description=item.get("description", ""),
                    )
                )

            print(f"[诊断] 最终产出 Job 数：{len(jobs)}")
            if flagged:
                print(f"[诊断] 其中 {flagged} 条的 title/salary 仍含字体混淆字符（详情页补全失败）")

            # 发布时间汇总（要求：拿不到就明确告警，不许静默）
            dated = [LAST_PUBLISH_DATES.get(j.job_id, "") for j in jobs]
            filled = [d for d in dated if d]
            print(f"[诊断] 发布时间汇总：{len(filled)}/{len(jobs)} 条拿到日期"
                  f"（格式 YYYY-MM-DD）")
            if filled:
                print(f"[诊断]   最早 {min(filled)}，最新 {max(filled)}")
            if missing_date:
                print(f"[诊断][警告] 以下岗位未取到发布时间：{', '.join(missing_date)}")

            # 要求：JD 正文长度汇总 + 空正文明确告警
            jd_lens = [len(j.description or "") for j in jobs]
            if jd_lens:
                empty_n = sum(1 for x in jd_lens if x == 0)
                print(f"[诊断] JD 正文长度汇总：min={min(jd_lens)} "
                      f"max={max(jd_lens)} 平均={sum(jd_lens) // len(jd_lens)} 字符；"
                      f"空正文 {empty_n}/{len(jd_lens)} 条")
                if empty_n:
                    print(f"[诊断][警告] 以下岗位 JD 正文为空：{', '.join(empty_jd) or '（见上方逐条告警）'}")
            elif not fetch_detail:
                print("[诊断] fetch_detail=False，未抓取 JD 正文与发布时间。")

            if not jobs:
                await _dump_debug(page, note="卡片存在但全部解析失败")

        finally:
            try:
                await context.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                await browser.close()
            except Exception:  # noqa: BLE001
                pass

    print("=" * 70)
    return jobs


async def _dump_debug(page: Any, note: str = "") -> None:
    """保存截图 + HTML，便于人工分析选择器。"""
    print(f"[诊断] 保存调试快照（{note}）")
    try:
        await page.screenshot(path=str(DEBUG_SCREENSHOT), full_page=True)
        print(f"[诊断]   截图 -> {DEBUG_SCREENSHOT}")
    except Exception as exc:  # noqa: BLE001
        print(f"[诊断]   截图失败：{type(exc).__name__}: {exc}")
    try:
        html = await page.content()
        DEBUG_HTML.write_text(html, encoding="utf-8")
        print(f"[诊断]   HTML -> {DEBUG_HTML} ({len(html)} 字符)")
        classes = re.findall(r'class="([^"]*intern[^"]*)"', html)
        from collections import Counter
        print("[诊断]   含 intern 的 class TOP10：")
        for cls, num in Counter(classes).most_common(10):
            print(f"[诊断]     {num:3d}  {cls}")
    except Exception as exc:  # noqa: BLE001
        print(f"[诊断]   保存 HTML 失败：{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 直接运行测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys as _sys

    KEYWORD = "Agent 开发"
    CITY = "北京"
    LIMIT = 20
    MAX_PAGES = 3
    # 冒烟测试用：--no-detail 跳过详情页（列表页的 title/salary 会是混淆字符，正文也为空）
    FETCH_DETAIL = "--no-detail" not in _sys.argv
    if "--limit" in _sys.argv:
        LIMIT = int(_sys.argv[_sys.argv.index("--limit") + 1])
    if "--max-pages" in _sys.argv:
        MAX_PAGES = int(_sys.argv[_sys.argv.index("--max-pages") + 1])

    results = asyncio.run(
        search_shixiseng(KEYWORD, city=CITY, limit=LIMIT, headless=False,
                         fetch_detail=FETCH_DETAIL, max_pages=MAX_PAGES)
    )

    print()
    print(f"抓到岗位数：{len(results)}")
    print("=" * 70)
    for i, job in enumerate(results, 1):
        desc = job.description or ""
        print(f"{i:2d}. title   : {job.title}")
        print(f"    company : {job.company}")
        print(f"    city    : {job.city}")
        print(f"    salary  : {job.salary}")
        print(f"    url     : {job.url}")
        print(f"    job_id  : {job.job_id}")
        print(f"    发布时间: {LAST_PUBLISH_DATES.get(job.job_id, '') or '（未取到）'}")
        print(f"    JD 正文 : {len(desc)} 字符")
        if not desc:
            print("    [警告] JD 正文为空！")
        else:
            preview = desc.replace("\n", " / ")
            print(f"    正文预览: {preview[:140]}{'…' if len(preview) > 140 else ''}")
        print("-" * 70)

    out_path = DEBUG_DIR / "shixiseng_result.json"
    payload = [
        {
            "platform": j.platform,
            "job_id": j.job_id,
            "title": j.title,
            "company": j.company,
            "city": j.city,
            "salary": j.salary,
            "url": j.url,
            "tags": j.tags,
            # 发布时间（详情页 .job_date 的刷新时间，已格式化为 YYYY-MM-DD；拿不到为 ""）
            # 注意：不放进 Job dataclass（不改 job_search.py），这里从映射里取
            "publish_date": LAST_PUBLISH_DATES.get(j.job_id, ""),
            # 正文：岗位职责 + 任职要求，用【】标签分段、\n 保留段落结构
            "description": j.description,
            "description_chars": len(j.description or ""),
        }
        for j in results
    ]
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    empty = [j.job_id for j in results if not (j.description or "")]
    if empty:
        print(f"[警告] 有 {len(empty)} 条岗位 JD 正文为空：{', '.join(empty)}")
    nodate = [j.job_id for j in results if not LAST_PUBLISH_DATES.get(j.job_id)]
    if nodate:
        print(f"[警告] 有 {len(nodate)} 条岗位未取到发布时间：{', '.join(nodate)}")
    print(f"结果已保存：{out_path}")
