# -*- coding: utf-8 -*-
"""
岗位数据清洗（只清洗，不改写源数据）

输入：抓取器产出的岗位 dict 列表（如 agent/scrapers/shixiseng_result.json），
字段与 agent/tools/job_search.py 的 Job 对齐：
    platform / job_id / title / company / city / salary / url / publish_date / description

依次应用三道过滤（命中即计数并丢弃，一条只计一次，按下列顺序判定）：
    a) city      城市：宽松匹配——job city 等于目标城市、含目标城市（多城市串）、
                 去掉"市"后缀后含目标城市，或 job city == "全国"
    b) age       时间：publish_date 距今天 <= max_age_days（默认 180；publish_date 为空则不参与时间过滤）
    c) length    正文长度：len(description) >= min_desc_len
                 （**按平台可覆盖**，见 PLATFORM_MIN_DESC_LEN：
                  ncss 用 100，未列出的平台用 min_desc_len 的全局值 200）

原 a) relevance 相关性过滤**已停用**：项目定位从"AI 求职助手"扩展为
"全行业求职助手"后，相关性由抓取端的关键词池（config/scraping.yaml）保证，
清洗端再按 AI 词过滤会把非 AI 岗位整条丢掉。_is_relevant 恒为 True，
removed["relevance"] 恒为 0（保留字段只为不改变统计口径）。

保留的岗位按 publish_date 降序排列（最新在前），publish_date 为空的排在最后。

依赖：仅标准库 json / sys / datetime / pathlib。
"""

import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

# 本文件位于 rag/quality/，RAG_DIR = rag/，REPO_DIR = 仓库根目录
QUALITY_DIR = Path(__file__).resolve().parent
RAG_DIR = QUALITY_DIR.parent
REPO_DIR = RAG_DIR.parent

# 默认输入输出路径
DEFAULT_INPUT = REPO_DIR / "agent" / "scrapers" / "shixiseng_result.json"
DEFAULT_OUTPUT = RAG_DIR / "data" / "cleaned_jd.json"

# 时间过滤默认值：180 天
# 口径说明：牛客等平台的大厂岗位生命周期比实习僧长（挂半年是常态），
# 60/90 天会误杀仍有效的在招岗位，因此放宽到 180 天。
# 需按平台/场景调整时可由调用方传参（如 scheduler 从 config/scraping.yaml
# 的 schedule.max_age_days 读取后传入），本模块自身不读配置文件。
DEFAULT_MAX_AGE_DAYS = 180

# 合并落盘的"过期归档"阈值：超过这个天数的记录从主文件移进归档文件。
# 为什么与 max_age_days 分开：max_age_days 是"这次抓到的岗位值不值得入库"的准入门槛，
# 而本值是"已经在库里的老记录什么时候该退场"。两者口径不同，
# 合并时若沿用入库口径，会把上一次刚通过清洗、这次没被抓到的岗位顺手删掉——
# 这正是"覆盖式落盘"之外的另一条丢数据路径。所以特意单独留一档。
DEFAULT_STALE_DAYS = 90

# 过期记录的归档文件名（与 cleaned_jd.json 同目录）
ARCHIVE_FILENAME = "cleaned_jd_archive.json"

# 正文长度默认值（全局；未在 PLATFORM_MIN_DESC_LEN 里列出的平台都用它）
DEFAULT_MIN_DESC_LEN = 200

# 按平台覆盖的正文长度门槛：{平台名: 最小正文字数}。
#
# 为什么需要它：ncss（国家大学生就业服务平台）的详情页正文长度分布
# 与实习僧/牛客**完全不同**。479 条抽样实测：
#   * 23.8%（114 条）是「标题回显」—— description 与 title 逐字相同、最长 29 字，
#     说明详情页正文没解析出来（数据缺失），不是"JD 天生短"；
#   * 76.2%（365 条）有独立正文，中位数 242 字，其中 100-199 字的 85 条是
#     **真实但精炼的短 JD**（例：审计员 125 字、审计助理 146 字、法务专员 54 字），
#     title + company + city + 这段正文已足够 RAG 做基础检索。
# 用全局 200 卡它，落盘率只有 45%（北京 × 3 词端到端更极端：29 条只留 10 条 = 34%），
# 等于把"多行业覆盖"的目标砍掉一半以上。
#
# 为什么取 100：实测「标题回显」那批最长 29 字、100-199 段全是真正文，
# 100 刚好切在确定的坏数据边界之上，混入的标题回显为 0 条，同时保住了短 JD。
# 再降到 30/50 只会多收 50-99 那 34 条，里面混着"负责中学语文教学工作。"这类
# 一句话正文，噪声比收益大。
#
# 实习僧 / 牛客**不动（保持 200）**：实习僧原始抓取样本 42 条 min 223 / 中位 444，
# 库里 502 条全部 ≥200；牛客库里 116 条同样全部 ≥200、中位 578。它们的正文
# 本就都在 200 以上，放宽对它们零收益、只有引入噪声的风险。
PLATFORM_MIN_DESC_LEN = {
    "ncss": 100,
}

# 相关性关键词（**已停用，仅作历史参考**）。
# 自"全行业求职助手"改造后 _is_relevant 恒为 True，下面的词与 _hits_keyword
# 都不再参与过滤；保留它们只为记录旧口径（需要回滚时只改 _is_relevant 即可）。
RELEVANCE_KEYWORDS_ZH = ("大模型", "智能体", "大语言模型", "检索增强生成")
RELEVANCE_KEYWORDS_EN = ("agent", "llm", "rag")

# 标题侧标记（同样已停用：不再有"标题含 AI"的额外判定）
AI_TITLE_MARKER = "ai"

# 输出字段顺序 = Job 的字段顺序（不含 tags，抓取结果里也没有 tags）
# 抓取结果里的 description_chars 是抓取器自己的统计字段，不进入清洗结果
JOB_FIELDS = (
    "platform", "job_id", "title", "company", "city",
    "salary", "url", "description", "publish_date",
)

# 城市过滤的"不限城市"占位值
NATIONWIDE = "全国"

# 过滤原因键（removed 字典里始终齐全，未命中为 0）
REMOVAL_REASONS = ("relevance", "city", "age", "length")

# --- 同城镜像折叠（Round 2 / Round 3）相关常量 -----------------------------
# Round 3 结论：**不再做任何标题增强归一化**。
# Round 2 试过「剥括号内容 + 反复剥尾部职位后缀词（工程师/实习生/开发/算法/研发）」，
# 实测在 613 条上命中 18 组，但其中 16 组的正文彼此不同 —— 折的是**不同岗位**。
# 根因：剥完后标题只剩「算法」「大模型」「java」「前端」「量化」这类**岗位类别词**，
# 同公司同城常有多个不同岗位共用同一类别词，于是被错误合并。
# 典型：百度·北京「大模型算法」把 供应链AI(582字) / 电商搜索LLM(328字) /
#       Agent系统(416字) 三个不同岗位折成一条。
# 另外「括号」里装的正是区分信息（(C++)(A193922)(4)/(5)），剥掉等于丢信息。
# 因此标题只做 _norm_text（空白折叠 + 转小写），并加「正文逐字一致」闸门。


def _job_field(job, name):
    """安全取字段并转成 str；None/缺失都当 ""。"""
    if not isinstance(job, dict):
        return ""
    value = job.get(name)
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _hits_keyword(text):
    """文本是否命中任一相关性关键词（**已停用，仅作历史参考**）。

    原逻辑：中文大小写敏感、英文忽略大小写。现在不再被 _is_relevant 调用。
    """
    if not text:
        return False
    for keyword in RELEVANCE_KEYWORDS_ZH:
        if keyword in text:
            return True
    lowered = text.lower()
    for keyword in RELEVANCE_KEYWORDS_EN:
        if keyword in lowered:
            return True
    return False


def _is_relevant(title, description):
    """相关性判定：**已停用，恒为 True（全行业口径）**。

    历史：这里原本只认 Agent / LLM / 大模型 / RAG / 智能体 / 大语言模型 /
    检索增强生成，非 AI 岗位会在**入库前**被整条丢掉。

    现状：项目定位扩展为"全行业求职助手"，相关性由抓取端的全行业关键词池
    （config/scraping.yaml）保证，清洗端不需要二次过滤，因此直接放行。
    函数与调用点都保留，只是不再产生任何移除（removed["relevance"] 恒为 0）。
    """
    return True


def _parse_date(value):
    """把 publish_date 解析成 date；解析不了（含空串、脏数据）返回 None。

    只认 "YYYY-MM-DD"，也容忍 "YYYY/MM/DD" 这类写法。
    """
    if not value:
        return None
    text = str(value).strip().replace("/", "-")
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_fresh(publish_date, today, max_age_days):
    """时间过滤：距今 <= max_age_days 天算新鲜。

    publish_date 为空或解析失败 -> True（不参与时间过滤）。
    未来日期同样按天数差判断（|today - publish_date|），避免把未来的日期
    当成"负年龄"永远放行。
    """
    parsed = _parse_date(publish_date)
    if parsed is None:
        return True
    return abs((today - parsed).days) <= max_age_days


def _normalize_job(job):
    """只保留 Job 定义的字段，保证输出结构稳定。"""
    normalized = {}
    for field in JOB_FIELDS:
        normalized[field] = _job_field(job, field).strip()
    return normalized


def _norm_city(value) -> str:
    """城市名归一：去首尾空白，并去掉"市"后缀，便于宽松比较。"""
    return str(value or "").strip().replace("市", "")


def city_matches(job_city, target_city) -> bool:
    """宽松城市匹配。

    抓取结果的 jobCity 常见形态：单城市（"广州"）、带"市"后缀（"广州市"）、
    多城市串（"广州/北京/上海"、"广州、北京"）、以及不限城市的"全国"。
    只要目标城市出现在岗位城市里就保留，避免多城市岗位被误杀。

    规则（任一命中即 True）：
      1) 目标城市为"全国"（本次不限城市）
      2) 岗位城市为"全国"（不限城市岗位）
      3) 两者完全相等
      4) 目标城市是岗位城市的子串（"广州" ⊂ "广州/北京/上海"）
      5) 去掉"市"后缀后，目标城市仍是岗位城市的子串（"广州" ⊂ "广州市"）
    """
    job_city = str(job_city or "").strip()
    target_city = str(target_city or "").strip()
    # 岗位城市缺失时不放行；目标城市为空表示不限城市，放行
    if not job_city:
        return not target_city
    if not target_city:
        return True
    if target_city == NATIONWIDE or job_city == NATIONWIDE:
        return True
    if job_city == target_city:
        return True
    if target_city in job_city:
        return True
    normalized_job = _norm_city(job_city)
    normalized_target = _norm_city(target_city)
    if normalized_target and normalized_target in normalized_job:
        return True
    return False


def clean_jobs(jobs: list[dict], city: str = "广州",
               max_age_days: int = DEFAULT_MAX_AGE_DAYS,
               min_desc_len: int = DEFAULT_MIN_DESC_LEN,
               today: date | None = None,
               platform_min_desc_len: dict | None = None) -> dict:
    """
    清洗岗位列表。

    参数：
        jobs:         原始岗位 dict 列表
        city:         目标城市（宽松匹配：相等、包含、去"市"后缀包含，或 job city == "全国" 时保留）
        max_age_days: publish_date 距今天最大天数（默认 180）
        min_desc_len: 正文最小长度（**全局默认**；被平台覆盖表压过时以表为准）
        today:        基准日期，默认取系统当天；测试时可显式传入
        platform_min_desc_len: 本次调用额外追加的平台门槛覆盖（可选），
                     形如 {"ncss": 100}；与 PLATFORM_MIN_DESC_LEN 合并，
                     同名平台以本参数为准

    返回：
        {
          "total_input": 42,
          "total_output": N,
          "removed": {"relevance": X, "city": Y, "age": Z, "length": W},
          "jobs": [清洗后的 Job 列表]   # 按 publish_date 降序，日期为空的在最后
        }
    """
    if today is None:
        today = date.today()

    # 正文长度门槛：全局值 + 按平台覆盖（表里没有的平台用全局值）。
    # 平台名统一小写去空白，避免大小写/空格差异导致覆盖失效，静默退回 200。
    thresholds = dict(PLATFORM_MIN_DESC_LEN)
    if platform_min_desc_len:
        thresholds.update({
            str(name).strip().lower(): int(value)
            for name, value in platform_min_desc_len.items()
        })

    removed = {reason: 0 for reason in REMOVAL_REASONS}
    kept = []

    for job in jobs or []:
        title = _job_field(job, "title")
        description = _job_field(job, "description")
        job_city = _job_field(job, "city").strip()
        publish_date = _job_field(job, "publish_date").strip()
        platform = _job_field(job, "platform").strip().lower()

        # a) 相关性：已停用（_is_relevant 恒为 True，全行业口径）；
        #    保留这一步只为让 removed 的键与历史统计口径保持一致。
        if not _is_relevant(title, description):
            removed["relevance"] += 1
            continue

        # b) 城市：宽松匹配（目标城市、"全国"、多城市串、带"市"后缀都算命中）
        if not city_matches(job_city, city):
            removed["city"] += 1
            continue

        # c) 时间：空日期不参与过滤
        if not _is_fresh(publish_date, today, max_age_days):
            removed["age"] += 1
            continue

        # d) 正文长度：按平台取门槛（ncss 100 / 未列出的平台用 min_desc_len，见
        #    PLATFORM_MIN_DESC_LEN 里"为什么按平台"的实测依据）
        if len(description) < thresholds.get(platform, min_desc_len):
            removed["length"] += 1
            continue

        kept.append(_normalize_job(job))

    # 按 publish_date 降序；空日期（解析失败）统一排在最后
    kept.sort(
        key=lambda item: (
            _parse_date(item.get("publish_date")) is not None,
            _parse_date(item.get("publish_date")) or date.min,
        ),
        reverse=True,
    )

    return {
        "total_input": len(jobs or []),
        "total_output": len(kept),
        "removed": removed,
        "jobs": kept,
    }


def load_jobs(path) -> list[dict]:
    """读 JSON 岗位文件；支持顶层是 list 或单个 dict。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # 容忍 {"jobs": [...]} 这类包装
        data = data.get("jobs", [data])
    return [item for item in data if isinstance(item, dict)]


def _print_summary(result):
    """打印清洗汇总和前 5 条。"""
    removed = result["removed"]
    print("=" * 60)
    print("岗位数据清洗汇总")
    print("=" * 60)
    print("输入 %d 条 → 输出 %d 条" % (result["total_input"], result["total_output"]))
    print("-" * 60)
    print("按过滤原因移除：")
    labels = {
        "relevance": "相关性（已停用：全行业口径，由抓取端关键词池保证，恒为 0）",
        "city": "城市（非目标城市且非全国）",
        "age": "时间（publish_date 超出天数上限）",
        "length": "正文长度（description 太短；门槛按平台：ncss 100 / 其余 200）",
    }
    for reason in REMOVAL_REASONS:
        print("  %-9s %2d 条  —— %s" % (reason, removed.get(reason, 0), labels[reason]))
    print("  合计移除 %d 条" % sum(removed.values()))
    print("-" * 60)
    print("保留前 5 条（按 publish_date 降序）：")
    for index, job in enumerate(result["jobs"][:5], start=1):
        print("  %d. %s | %s | %s | 正文 %d 字" % (
            index,
            job.get("title") or "(无标题)",
            job.get("company") or "(无公司)",
            job.get("publish_date") or "(无日期)",
            len(job.get("description") or ""),
        ))
    if not result["jobs"]:
        print("  （无）")


def load_jobs_if_exists(path) -> list[dict]:
    """读 JSON 岗位文件；**文件不存在 / 坏了都返回 []**，不抛异常。

    与 load_jobs 的区别：那个是 CLI 入口，输入路径是用户指定的，读不到就该报错；
    这个是「合并落盘」的前置读——首次运行时文件还不存在是**正常情况**，
    不该让定时任务因此失败。
    """
    target = Path(path)
    if not target.is_file():
        return []
    try:
        return load_jobs(target)
    except (OSError, ValueError):
        return []


def _job_identity(job: dict) -> str:
    """去重键：**平台 + job_id**；没有 job_id 就用「平台|公司|岗位|链接」兜底。

    job_id 只在**平台内**唯一：不同平台用自增数字当 id 是常态，
    只用 job_id 当键会让两个平台的岗位互相覆盖（niuke 的岗位被 shixiseng 顶掉），
    所以键里必须带 platform。

    job_id 可能是空的；全用空串当键会把这些互不相同的岗位错误地合并成一条，
    所以退到内容签名。
    """
    platform = _job_field(job, "platform").strip()
    job_id = _job_field(job, "job_id").strip()
    if job_id:
        return f"id:{platform}|{job_id}"
    return "sig:" + "|".join((
        platform,
        _job_field(job, "company").strip(),
        _job_field(job, "title").strip(),
        _job_field(job, "url").strip(),
    ))


def _job_sort_key(job: dict):
    """排序键：publish_date 降序，日期为空的排最后。

    返回 (1, date) / (0, date.min)：配合 reverse=True 让有日期的排在前面，
    且日期新的更靠前；日期为空的统一沉底（date.min 保证空日期之间不炸）。
    """
    parsed = _parse_date(job.get("publish_date"))
    return (parsed is not None, parsed or date.min)


def _is_stale(publish_date, today, stale_days) -> bool:
    """是否已过期到该归档：距今**严格大于** stale_days 天。

    publish_date 为空/解析失败 -> False（不参与归档，保守保留）。
    未来日期按绝对天数差判断，与 _is_fresh 口径一致。
    """
    parsed = _parse_date(publish_date)
    if parsed is None:
        return False
    return abs((today - parsed).days) > stale_days


def _norm_text(value) -> str:
    """文本归一（镜像比对用）：去首尾空白、内部空白折叠成一个、转小写。

    与 agent/scrapers/niuke.py 的 _norm_text 口径一致。此处**不 import 抓取器**：
    cleaner 属于 rag/quality 数据层，不该反向依赖 agent/scrapers（会把
    playwright 之类的重依赖拖进清洗链路），因此保留一份等价实现。
    """
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _norm_mirror_title(value) -> str:
    """标题的镜像归一：**只做 _norm_text**。

    Round 2 曾在此处再剥括号内容 + 压平分隔符 + 反复剥尾部职位后缀词，
    Round 3 已全部删除，原因见文件上方 MIRROR 常量处的说明：
    增强归一化会把标题削成岗位类别词，导致同城不同岗位被折成一条（16/18 组误伤）。
    """
    return _norm_text(value)


def _normalized_description(job: dict) -> str:
    """取用于「正文是否逐字一致」比较的归一化正文。

    比较口径：description 先 _norm_text（压空白 + 转小写）再比较。
    为什么用 strip 后的逐字一致而不是相似度：本轮的判定是「宁可漏折、不可误伤」，
    只有正文完全相同的才是铁证镜像；差一个字的（如字节豆包组 536/538）
    一律不折，交给人工或后续规则。
    """
    return _norm_text(_job_field(job, "description"))


def _job_mirror_key(job: dict) -> tuple:
    """同城镜像判定键：(公司, 标题, 城市)，全部走现有 _norm_text / _norm_city。

    - 公司：_norm_text（空白折叠 + 转小写）
    - 标题：_norm_text（**不剥后缀、不剥括号**）
    - 城市：_norm_city（去「市」后缀）

    **城市刻意保留在键里**：本项目的检索/看板是按城市查岗的
    （见 agent/tools/job_search.py 与 _norm_city 的存在理由），
    把跨城投放折成一条会让「搜广州看不到广州岗位」。跨城同岗位
    因此**不折叠**，只折叠同一城市内的重复投放。

    只做字面归一，不引入语义等价词典——审计结论显示跨平台
    （实习僧 vs 牛客）的 title 差异是「算法工程师 ⇄ 游戏测试工程师」
    这种级别，靠规则无法覆盖，强上必然误伤。
    """
    return (
        _norm_text(_job_field(job, "company")),
        _norm_text(_job_field(job, "title")),
        _norm_city(_job_field(job, "city")),
    )


def _mirror_sort_key(job: dict):
    """镜像组内的确定性代表行排序键：数字 id 按数值升序排前，非数字 id 排后。

    与 niuke._sort_key 口径一致。选「原始 id 最小」而不是「正文最长」当代表行，
    是为了让结果**与抓取顺序无关**：同一份数据跑两次得到同一个 job_id，
    否则每次重抓都可能换一条记录当代表行。
    """
    text = _job_field(job, "job_id")
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def fold_mirrors(records: list[dict]) -> tuple[list[dict], dict]:
    """把同 (公司, 标题, 城市) 且**正文逐字一致**的镜像折叠成一条。

    调用位置：merge_jds 里 **job_id 去重之后、排序/写盘之前**。
    为什么必须晚于 job_id 去重：job_id 是平台内唯一的，跨平台 id 重复
    （两个平台都用自增数字）会互相覆盖，所以先按「平台+job_id」把真正的
    job_id 重复吃掉，再在这一步折叠「id 不同但内容同一岗位」的镜像。

    折叠闸门（Round 3 新增，关键）：
        键命中后还要看组内 description 是否**全部归一化后逐字一致**，
        一致才折叠；只要有一条不同就整组原样保留。
        为什么必须加：Round 2 只按 (公司,标题,城市) 折，18 组里 16 组
        折的是不同岗位（百度·北京「大模型算法」折掉了供应链AI /
        电商搜索LLM / Agent系统 三个不同岗位）。正文一致才是
        「同一岗位重复投放」的铁证。

    代表行规则（顺序固定，保证确定性）：
        1) 排序键 _mirror_sort_key 最小者（原始 id 最小，数字 id 优先）；
        2) description：取组内最长的那条回填（信息量最大）；
        3) salary：先取代表行自己的；代表行为空才取组内第一个非空值。
           **不做跨单位比较**——实习僧多为「/天」、牛客有「500-800/天」，
           按数字大小取最大值会把单位不同的薪资比错。

    未选中的记录**直接不写入结果**（不是打 mirror_of 标记）。

    参数：
        records: 已按 job_id 去重后的岗位 dict 列表

    返回：
        (折叠后的记录列表, stats)
        stats = {
          "before":            折叠前条数,
          "after":             折叠后条数,
          "folded":            折叠掉的条数,
          "groups":            实际折叠的组数,
          "skipped_by_content":正文不一致而跳过的组数（键命中但不敢折）,
          "removed":           [{"key": 折叠键, "size": 组内条数,
                                "rep": {代表行的 title/company/city/job_id/platform}}],
        }
    """
    before = len(records)

    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []          # 保持首现顺序，让输出与输入顺序无关但可复现
    for job in records:
        key = _job_mirror_key(job)
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(job)

    kept: list[dict] = []
    removed: list[dict] = []
    skipped_by_content = 0
    for key in order:
        group = groups[key]
        if len(group) < 2:
            kept.append(group[0])
            continue

        # 闸门：正文必须全部归一化后逐字一致，否则整组原样保留
        # （宁可漏折、不可误伤；Round 2 的教训见 docstring）
        if len({_normalized_description(j) for j in group}) > 1:
            skipped_by_content += 1
            kept.extend(group)
            continue

        ordered = sorted(group, key=_mirror_sort_key)
        rep = ordered[0]
        changes = {}

        # description：取组内最长回填
        longest = max(ordered, key=lambda j: len(_job_field(j, "description")))
        if len(_job_field(longest, "description")) > len(_job_field(rep, "description")):
            changes["description"] = _job_field(longest, "description")

        # salary：代表行为空才回填第一个非空（不做跨单位比较）
        if not _job_field(rep, "salary"):
            for job in ordered:
                if _job_field(job, "salary"):
                    changes["salary"] = _job_field(job, "salary")
                    break

        if changes:
            rep = {**rep, **changes}

        kept.append(rep)
        removed.append({
            "key": key,
            "size": len(group),
            "rep": {
                "platform": _job_field(rep, "platform"),
                "job_id": _job_field(rep, "job_id"),
                "title": _job_field(rep, "title"),
                "company": _job_field(rep, "company"),
                "city": _job_field(rep, "city"),
            },
        })

    stats = {
        "before": before,
        "after": len(kept),
        "folded": before - len(kept),
        "groups": len(removed),
        "skipped_by_content": skipped_by_content,
        "removed": removed,
    }
    return kept, stats


def merge_jds(existing, new, stale_days: int = DEFAULT_STALE_DAYS,
              archive_path=None, today: date | None = None) -> dict:
    """把「库里已有的」和「这次新抓的」按 job_id 去重合并，并归档过期记录。

    修的是什么：daily_job 原来直接 write_text 覆盖 cleaned_jd.json，
    抓 5 条就把原来 15 条冲成 5 条。落盘必须是**合并**而不是覆盖。

    规则（按顺序）：
        1. 去重：同 job_id 只留一条；**新数据覆盖旧数据**（新抓的更完整、更新；
           岗位描述被修改时也应当以最新一次抓取为准）；
        2. 排序：publish_date 降序，日期为空的排最后；
        3. 归档：距今 > stale_days 天的记录从结果里移除并放进 archived；
           归档**不是丢弃**——archive_path 给了就追加写进归档文件。

    参数：
        existing:     现有记录（list[dict]；None/空都行）
        new:          本次新抓的记录（list[dict]）
        stale_days:   过期阈值，默认 90 天
        archive_path: 归档文件路径；None 表示只返回 archived 不落盘
        today:        基准日期，默认系统当天（测试可显式传入）

    返回：dict（**不是 list**）
        {
          "jobs":     [合并+排序+去过期后的记录],
          "archived": [本次被移出主列表的过期记录],
          "stats":    {"existing": X, "new": Y, "merged": Z,
                       "duplicates": D, "archived": A, "output": Z-A},
        }

    注意 stats["merged"] 的口径：**同城镜像折叠之前**的条数
    （= existing 与 new 按平台+job_id 去重后的总数）。折叠掉的条数另记
    "mirrors_folded"，发生的折叠组数记 "mirror_groups"，这样
    「merged = output + mirrors_folded + archived」这条账才对得上。

    为什么返回 dict 而不是 list：归档必须把过期记录**交出去**，调用方才写得了
    归档文件。只返回 list 的话，"移除"和"静默丢弃"在调用方看来没有区别。
    """
    if today is None:
        today = date.today()

    existing = [j for j in (existing or []) if isinstance(j, dict)]
    new = [j for j in (new or []) if isinstance(j, dict)]

    # 1) 去重合并：先放旧的，新的同键覆盖（后写覆盖先写）
    merged: dict[str, dict] = {}
    duplicates = 0
    for job in existing:
        key = _job_identity(job)
        if key in merged:
            duplicates += 1
        merged[key] = job
    for job in new:
        key = _job_identity(job)
        if key in merged:
            duplicates += 1
        merged[key] = job

    records = list(merged.values())

    # 1b) 同城镜像折叠：job_id 去重之后、排序写盘之前。
    #     拦的是「id 不同但同公司同标题同城市」的重复投放（job_id 去重拦不住）。
    records, mirror_stats = fold_mirrors(records)

    # 2) 排序：publish_date 降序，空日期垫底
    records.sort(key=_job_sort_key, reverse=True)

    # 3) 归档：过期的移出主列表
    kept, archived = [], []
    for job in records:
        if _is_stale(job.get("publish_date"), today, stale_days):
            archived.append(job)
        else:
            kept.append(job)

    if archived and archive_path is not None:
        # 追加归档：先读回已有归档再合并，避免第二次运行把第一次的归档冲掉
        # （同一个"覆盖丢数据"的坑，不该在归档文件上再踩一次）。
        previously = load_jobs_if_exists(archive_path)
        save_cleaned(merge_archived_records(previously, archived), archive_path)

    return {
        "jobs": kept,
        "archived": archived,
        "stats": {
            "existing": len(existing),
            "new": len(new),
            "merged": mirror_stats["before"],
            "duplicates": duplicates,
            "mirrors_folded": mirror_stats["folded"],
            "mirror_groups": mirror_stats["groups"],
            "mirrors_skipped_by_content": mirror_stats["skipped_by_content"],
            "archived": len(archived),
            "output": len(kept),
        },
    }


def merge_archived_records(existing_archive, new_archive) -> list[dict]:
    """归档文件内部的合并去重（同 job_id 以新归档为准），按日期降序。"""
    merged: dict[str, dict] = {}
    for job in list(existing_archive or []) + list(new_archive or []):
        if isinstance(job, dict):
            merged[_job_identity(job)] = job
    records = list(merged.values())
    records.sort(key=_job_sort_key, reverse=True)
    return records


def save_cleaned(jobs, path) -> Path:
    """把清洗后的岗位列表写成 JSON（顶层是 list，与输入文件保持一致）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    input_path = Path(argv[0]) if argv else DEFAULT_INPUT
    output_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUTPUT

    jobs = load_jobs(input_path)
    result = clean_jobs(
        jobs,
        city="广州",
        max_age_days=DEFAULT_MAX_AGE_DAYS,
        min_desc_len=DEFAULT_MIN_DESC_LEN,
    )
    _print_summary(result)

    saved = save_cleaned(result["jobs"], output_path)
    print("-" * 60)
    print("源文件：%s" % input_path)
    print("已保存：%s" % saved)

    return result


if __name__ == "__main__":
    main()
