"""
实习僧（shixiseng.com）岗位抓取器 —— Playwright async 真实抓取实现。

用法：
    import asyncio
    from agent.scrapers.shixiseng import search_shixiseng

    jobs = asyncio.run(search_shixiseng("Agent 开发", city="北京", limit=20))

设计说明（重要）：
    实习僧搜索列表页对**标题和薪资**做了字体混淆（CMap 反爬）：
    真实文字被替换成私有区码位（U+E000~U+F8FF），页面上再通过
    动态 @font-face（/interns/iconfonts/file?rand=...）把这些码位渲染成正确形状。
    因此列表页直接取到的 title / salary 是乱码，
    而**详情页 /intern/<job_id> 没有任何混淆**（页面 PUA 字符数为 0）。

    所以本模块采用两段式抓取：
      1) 列表页解析卡片 -> 拿到 url / job_id / company / city（这些字段是明文）
      2) 对每张卡片访问详情页 -> 拿到明文 title / salary（可选，fetch_detail=True）

    依赖：playwright（async API）+ 项目内 agent.tools.job_search.Job（不重复定义）。
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import random
import re
import sys
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
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _clean(text: Optional[str]) -> str:
    """去掉首尾空白并合并内部空白（详情页 inner_text 里会有 \n）。"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


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
# 详情页解析（拿明文 title / salary）
# ---------------------------------------------------------------------------
async def _fetch_detail(page: Any, url: str) -> dict[str, str]:
    """访问详情页，返回明文 title/salary/city。失败返回空字典。"""
    out: dict[str, str] = {}
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

        if title and not _has_obfuscated(title):
            out["title"] = title
        if salary and not _has_obfuscated(salary):
            out["salary"] = salary
        if city and not _has_obfuscated(city):
            out["city"] = city
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
) -> list[Job]:
    """
    抓取实习僧搜索页岗位。

    参数：
        keyword: 搜索关键词，如 "Agent 开发"
        city: 城市，如 "北京"；None 表示不限
        limit: 返回数量上限
        headless: 是否无头模式（默认 False，可视化便于调试）
        fetch_detail: 是否访问详情页补齐明文 title/salary
                      （列表页这两个字段被字体混淆，强烈建议 True）

    返回：
        list[Job]（复用 agent.tools.job_search.Job）
    """
    from urllib.parse import quote

    params = f"keyword={quote(keyword)}"
    if city:
        params += f"&city={quote(city)}"
    url = f"{SEARCH_URL}?{params}"

    print("=" * 70)
    print(f"[诊断] 访问 URL：{url}")
    print(f"[诊断] keyword={keyword!r} city={city!r} limit={limit}  headless={headless}")

    jobs: list[Job] = []
    parsed_count = 0
    card_count = 0

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
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            print(f"[诊断] HTTP 状态：{resp.status if resp else 'N/A'}  最终 URL：{page.url}")

            card_selector, card_count = await _find_card_selector(page)
            print(f"[诊断] 命中卡片选择器：{card_selector!r}  卡片数：{card_count}")

            # 等页面渲染稳定一点再截图
            await page.wait_for_timeout(1_500)

            # 要求 d) 截图（首次调试用）
            try:
                await page.screenshot(path=str(DEBUG_SCREENSHOT), full_page=True)
                print(f"[诊断] 已保存截图：{DEBUG_SCREENSHOT}")
            except Exception as exc:  # noqa: BLE001
                print(f"[诊断] 截图失败：{type(exc).__name__}: {exc}")

            if not card_count:
                # 要求 7) 选择器找不到 -> 保存截图 + HTML 便于人工分析
                print("[诊断] 未找到任何岗位卡片，保存页面快照供人工分析。")
                await _dump_debug(page, note="没有命中任何卡片选择器")
                return jobs

            cards = page.locator(card_selector)
            raw_items: list[dict[str, Any]] = []
            for i in range(card_count):
                item = await _parse_card(cards.nth(i), i)
                if item:
                    raw_items.append(item)
            parsed_count = len(raw_items)
            print(f"[诊断] 卡片解析成功数：{parsed_count}/{card_count}")

            # 保存首卡 HTML，便于后续核对选择器
            try:
                first_html = await cards.first.inner_html()
                DEBUG_CARDS.write_text(first_html, encoding="utf-8")
                print(f"[诊断] 首卡 HTML 已保存：{DEBUG_CARDS}")
            except Exception as exc:  # noqa: BLE001
                print(f"[诊断] 保存首卡 HTML 失败：{type(exc).__name__}: {exc}")

            # 只对「最终会返回的那 limit 条」补抓详情页：
            # 详情页有限流（每条 0.5~1.5s + 页面加载），不必要地多抓会拖慢并增加被风控概率
            candidates = raw_items[:limit] if limit else raw_items
            need_detail = [
                it for it in candidates
                if fetch_detail and (it["title_obfuscated"] or it["salary_obfuscated"]
                                     or not it["title_raw"] or not it["salary_raw"])
            ]
            print(f"[诊断] 需要访问详情页补全的岗位数：{len(need_detail)}"
                  f"（候选 {len(candidates)}/{parsed_count}，limit={limit}）")
            if need_detail:
                print("[诊断] 说明：实习僧列表页对 标题/薪资 做了字体混淆（私有区码位），"
                      "详情页为明文，故补抓详情页。")

            if fetch_detail:
                for n, item in enumerate(need_detail, 1):
                    print(f"[诊断] ({n}/{len(need_detail)}) 详情页：{item['url']}")
                    detail = await _fetch_detail(page, item["url"])
                    item["title_detail"] = detail.get("title", "")
                    item["salary_detail"] = detail.get("salary", "")
                    item["city_detail"] = detail.get("city", "")

            # 组装 Job，最多 limit 条
            flagged = 0
            for item in raw_items:
                if len(jobs) >= limit:
                    break
                title = item.get("title_detail") or item["title_raw"]
                salary = item.get("salary_detail") or item["salary_raw"]
                city_val = item.get("city_detail") or item["city"]
                if _has_obfuscated(title) or _has_obfuscated(salary):
                    flagged += 1
                    title = _clean(title)
                    salary = _clean(salary)
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
                        description="",
                    )
                )

            print(f"[诊断] 最终产出 Job 数：{len(jobs)}")
            if flagged:
                print(f"[诊断] 其中 {flagged} 条的 title/salary 仍含字体混淆字符（详情页补全失败）")

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
    # 冒烟测试用：--no-detail 跳过详情页（列表页的 title/salary 会是混淆字符）
    FETCH_DETAIL = "--no-detail" not in _sys.argv
    if "--limit" in _sys.argv:
        LIMIT = int(_sys.argv[_sys.argv.index("--limit") + 1])

    results = asyncio.run(
        search_shixiseng(KEYWORD, city=CITY, limit=LIMIT, headless=False,
                         fetch_detail=FETCH_DETAIL)
    )

    print()
    print(f"抓到岗位数：{len(results)}")
    print("=" * 70)
    for i, job in enumerate(results, 1):
        print(f"{i:2d}. title   : {job.title}")
        print(f"    company : {job.company}")
        print(f"    city    : {job.city}")
        print(f"    salary  : {job.salary}")
        print(f"    url     : {job.url}")
        print(f"    job_id  : {job.job_id}")
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
            "description": j.description,
        }
        for j in results
    ]
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果已保存：{out_path}")
