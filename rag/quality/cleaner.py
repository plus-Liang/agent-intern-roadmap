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

原 a) relevance 相关性过滤**已停用**：项目定位从"AI 求职助手"扩展为
"全行业求职助手"后，相关性由抓取端的关键词池（config/scraping.yaml）保证，
清洗端再按 AI 词过滤会把非 AI 岗位整条丢掉。_is_relevant 恒为 True，
removed["relevance"] 恒为 0（保留字段只为不改变统计口径）。

保留的岗位按 publish_date 降序排列（最新在前），publish_date 为空的排在最后。

依赖：仅标准库 json / sys / datetime / pathlib。
"""

import json
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

# 正文长度默认值
DEFAULT_MIN_DESC_LEN = 200

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
               today: date | None = None) -> dict:
    """
    清洗岗位列表。

    参数：
        jobs:         原始岗位 dict 列表
        city:         目标城市（宽松匹配：相等、包含、去"市"后缀包含，或 job city == "全国" 时保留）
        max_age_days: publish_date 距今天最大天数（默认 180）
        min_desc_len: 正文最小长度
        today:        基准日期，默认取系统当天；测试时可显式传入

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

    removed = {reason: 0 for reason in REMOVAL_REASONS}
    kept = []

    for job in jobs or []:
        title = _job_field(job, "title")
        description = _job_field(job, "description")
        job_city = _job_field(job, "city").strip()
        publish_date = _job_field(job, "publish_date").strip()

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

        # d) 正文长度
        if len(description) < min_desc_len:
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
        "length": "正文长度（description 太短）",
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
            "merged": len(records),
            "duplicates": duplicates,
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
