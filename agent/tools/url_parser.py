# -*- coding: utf-8 -*-
"""粘贴入口解析：三平台岗位链接识别 + 纯文本 JD 解析。

为什么单独一个模块
------------------
这里全是**纯函数**：不查库、不碰会话态、不打模型。放进 tools_registry 会被
一千多行业务逻辑淹没，而解析规则恰恰是最需要单测直接覆盖的部分。

两条入口
--------
1. 岗位链接：实习僧 / 牛客 / ncss 的详情页 URL → (platform, job_id)。
   拿到 id 就能直接查 jobs.db，不用抓页面（链接只当"哪条岗位"的钥匙用）。
2. JD 文本：Boss / 智联等抓不到的平台的 JD，用户直接粘贴 → 抠出公司/岗位/城市/薪资
   和职责/要求两段正文，造一条**不落库**的临时岗位（job_id = pasted_<hash>）。
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Optional

# 本文件位于 <repo_root>/agent/tools/url_parser.py，parents[2] 即仓库根目录。
# 与 job_search._db_module 同样的理由：`python agent/tools/url_parser.py`
# 直接运行时 sys.path[0] 是 tools/ 目录、仓库根不在里面，绝对导入会炸。
_REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# 第一部分：岗位链接
# --------------------------------------------------------------------------

# (平台, 正则, URL 模板)；顺序即优先级，更具体的写法放前面。
#
# 三个平台的详情页形态（取自 jobs.db 里真实落库的 url 字段）：
#   shixiseng  https://www.shixiseng.com/intern/inn_jzvdmurpamdk?pcm=pc_SearchList
#   niuke      https://www.nowcoder.com/jobs/detail/466427
#   ncss       https://www.ncss.cn/student/jobs/detail/YWF6i2YnSonM311DyjNd2q.html
#              https://www.ncss.cn/student/jobs/<id>/detail.html   （另一种写法）
#
# 域名匹配故意不给 www. 前缀：用户从 App 复制出来的常常是裸域名。
_URL_PATTERNS = (
    (
        "shixiseng",
        re.compile(r"shixiseng\.com/intern/([A-Za-z0-9_-]{4,})", re.IGNORECASE),
        "https://www.shixiseng.com/intern/{id}",
    ),
    (
        "niuke",
        re.compile(r"nowcoder\.com/jobs?/detail/([A-Za-z0-9_-]{3,})", re.IGNORECASE),
        "https://www.nowcoder.com/jobs/detail/{id}",
    ),
    (
        "ncss",
        re.compile(r"ncss\.cn/student/jobs/detail/([A-Za-z0-9_-]{4,})", re.IGNORECASE),
        "https://www.ncss.cn/student/jobs/detail/{id}.html",
    ),
    (
        "ncss",
        re.compile(
            r"ncss\.cn/student/jobs/([A-Za-z0-9_-]{4,})/detail\.html", re.IGNORECASE
        ),
        "https://www.ncss.cn/student/jobs/{id}/detail.html",
    ),
)


def find_job_urls(text) -> list[tuple[str, str, str]]:
    """扫出文本里所有三平台岗位链接。

    返回 [(platform, job_id, url)]，按出现顺序、按 (platform, job_id) 去重。
    一条都没有就返回空列表（这一步**不查库**，库里有没有是调用方的事）。
    """
    raw = str(text or "")
    if not raw:
        return []

    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for platform, pattern, template in _URL_PATTERNS:
        for match in pattern.finditer(raw):
            job_id = match.group(1)
            key = (platform, job_id)
            if key in seen:
                continue
            seen.add(key)
            found.append((platform, job_id, template.format(id=job_id)))
    return found


def parse_job_url(text) -> Optional[tuple[str, str, str]]:
    """只认第一条命中的岗位链接；没有就返回 None。"""
    urls = find_job_urls(text)
    return urls[0] if urls else None


# --------------------------------------------------------------------------
# 第二部分：JD 文本
# --------------------------------------------------------------------------

# 低于这个字数的一律不当 JD —— 用户随口一句话（"今天天气真好"）里也可能
# 恰好有"工作内容"这种词，长度是最省事的粗筛。
JD_MIN_CHARS = 100

# 判定"像 JD"的强标记：这些词几乎只出现在招聘文案里。
JD_MARKERS = (
    "岗位职责", "工作职责", "主要职责", "职位描述", "岗位描述", "工作内容",
    "任职要求", "任职资格", "岗位要求", "职位要求", "任职条件", "岗位需求",
    "加分项", "你将负责", "我们希望你",
)

_CITY_POOL = (
    "广州", "深圳", "北京", "上海", "杭州", "成都", "南京", "武汉", "西安", "苏州",
    "长沙", "重庆", "天津", "青岛", "合肥", "厦门", "郑州", "济南", "福州", "大连",
    "宁波", "无锡", "东莞", "佛山", "珠海", "香港",
)

_COMPANY_KEYS = ("公司名称", "招聘公司", "企业名称", "公司", "企业")
_TITLE_KEYS = ("岗位名称", "职位名称", "招聘岗位", "岗位", "职位")
_DEGREES = ("博士", "硕士", "本科", "大专", "专科", "中专", "不限")

# 公司名兜底：不带 key 的文案里也常有「XX科技有限公司」
_COMPANY_FALLBACK = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9（）()]{2,24}?(?:有限公司|股份有限公司|集团|研究院|实验室|银行))"
)
_SALARY = re.compile(
    r"\d+\s*[-~—到]\s*\d+\s*(?:元|千|k|K|万)?\s*(?:/|每)?\s*(?:天|月|日|小时|年)?"
)


def looks_like_jd(text) -> bool:
    """这段文本像不像一份 JD：够长 + 至少一个招聘专属标记。

    两条都要满足。只看标记会误判（"今天的工作内容是写代码"），
    只看长度更会误判（用户粘贴一大段别的文章）。
    """
    raw = str(text or "")
    if len(raw.strip()) < JD_MIN_CHARS:
        return False
    return any(marker in raw for marker in JD_MARKERS)


def _clean_line(line: str) -> str:
    """去掉 markdown 装饰与首尾空白，便于按行抠字段。"""
    return re.sub(r"^[\s#>*\-•·]+", "", str(line or "")).strip()


def _value_by_keys(lines: list[str], keys: tuple[str, ...]) -> str:
    """在行首附近找「公司：XXX」这类写法，返回冒号后面的值。"""
    for key in keys:
        for line in lines:
            head, sep, tail = line.partition("：")
            if not sep:
                head, sep, tail = line.partition(":")
            if not sep:
                continue
            if key in head and len(head) <= len(key) + 6:
                value = tail.strip().strip("】] ")
                if value:
                    return value[:60]
    return ""


def _first_title_line(lines: list[str]) -> str:
    """没有「岗位：XXX」时，取第一行短文本当岗位名（招聘帖首行基本都是标题）。"""
    for line in lines:
        if not line or len(line) > 40:
            continue
        if any(marker in line for marker in JD_MARKERS):
            continue
        return line[:60]
    return ""


def parse_pasted_jd(text) -> dict:
    """把一段粘贴的 JD 文本抠成岗位 dict（字段与 JobDetail 对齐）。

    job_id 用正文内容的 sha1 前 12 位（`pasted_<hash>`）：
    同一段文本重复粘贴拿到同一个 id，会话态里不会越积越多。
    """
    raw = str(text or "").strip()
    job_id = "pasted_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    lines = [_clean_line(line) for line in raw.splitlines()]
    lines = [line for line in lines if line]

    company = _value_by_keys(lines, _COMPANY_KEYS)
    if not company:
        fallback = _COMPANY_FALLBACK.search(raw)
        company = fallback.group(1) if fallback else ""

    title = _value_by_keys(lines, _TITLE_KEYS) or _first_title_line(lines)

    city = next((c for c in _CITY_POOL if c in raw), "")
    salary_match = _SALARY.search(raw)
    salary = re.sub(r"\s+", "", salary_match.group(0)) if salary_match else ""
    education = next((d for d in _DEGREES if d in raw), "")

    # 正文分节复用岗位详情那套标记（「岗位职责 / 任职要求 / 加分项」），
    # 保证临时岗位与库里岗位在 match_resume / 投递包眼里长得一模一样。
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from agent.tools.job_detail import _split_description

    description, requirements, bonus = _split_description(raw)

    return {
        "platform": "pasted",
        "job_id": job_id,
        "title": title,
        "company": company,
        "city": city,
        "salary": salary,
        "url": "",
        "description": description,
        "requirements": requirements,
        "bonus": bonus,
        "tags": [],
        "education": education,
    }


if __name__ == "__main__":
    samples = [
        ("实习僧", "看看这个 https://www.shixiseng.com/intern/inn_jzvdmurpamdk?pcm=pc_SearchList"),
        ("牛客", "https://www.nowcoder.com/jobs/detail/466427"),
        ("ncss A", "https://www.ncss.cn/student/jobs/detail/YWF6i2YnSonM311DyjNd2q.html"),
        ("ncss B", "https://www.ncss.cn/student/jobs/YWF6i2YnSonM311DyjNd2q/detail.html"),
        ("无关", "今天天气真好，出去走走吧"),
    ]
    for label, text in samples:
        print(f"{label}: {find_job_urls(text)}")

    jd = """AI Agent 开发实习生
公司：某科技有限公司
城市：广州
薪资：300-500/天
岗位职责：
1. 参与 Agent 框架的搭建与工具调用链路开发；
2. 负责 RAG 检索链路的数据清洗与召回优化。
任职要求：
1. 本科及以上在读，计算机相关专业；
2. 熟悉 Python，了解 LangChain / Chroma 等常用库。
加分项：有大模型应用落地经验。"""
    print(f"\n像 JD：{looks_like_jd(jd)} ／ 像 JD（短句）：{looks_like_jd('今天天气真好')}")
    import json

    print(json.dumps(parse_pasted_jd(jd), ensure_ascii=False, indent=2)[:700])
