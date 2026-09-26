"""
工具注册中心。
"""
import contextvars
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path

from agent.tools.job_search import search_jobs
from agent.tools.job_detail import get_job_detail
from agent.tools.pdf_export import available_backend, export_resume_pdf, normalize_resume
from agent.tools.resume_match import match_resume_to_jd, Resume
from agent.resume.tailor import tailor_resume
from shared.llm_client import chat
from shared.user_context import get_current_user
from agent import state_machine
from agent import storage

# 语义重排的候选上限（Round 10，乙方案的修正项之一）：
# 候选 <= 80 条才启用「子集内语义重排」。超过就只走 SQL 排序 ——
# 候选多说明关键词本身已经筛得够准（精确查询没有模糊空间），
# 再跑一次 embedding + BM25 既费时又不会改变排序质量。
SEMANTIC_MAX_CANDIDATES = 80


def _semantic_probe(keyword: str, max_terms: int = 3) -> str:
    """把「自然语言需求」收成候选池用的短关键词（semantic 模式的 SQL 预过滤用）。

    为什么需要：`search_jobs` 的关键词过滤是**逐词 AND 的字面 LIKE**
    （`db.search_jobs` 里每个词都必须出现在 title/description 中）。用户说
    「想找偏大模型落地、能写工程代码的实习」时，整句当关键词去 LIKE 会**一条都命中不了**
    —— 候选池为空，后面的语义重排根本没有机会跑（实测就是这样）。
    这里的作用只是"别把候选池掐死"：抽出信息量大的短词做 OR 召回，
    真正的排序交给 `_retrieve` 的语义路。

    做法（刻意保守，不引入新依赖）：
      1) 按标点/空白切片段，丢掉疑问词与「想找/有没有/岗位/实习」这类无区分度的词；
      2) 中文片段取 2 字滑窗（「落地」「工程」这种），英文/数字词整词保留；
      3) 全都取不到时退回"去掉停用词的原句"，最后再退回原句。
    这些词是**用空格拼起来**送进 `db.search_jobs` 的，而它按空格拆词做 AND —— 所以
    这里额外把拼好的串交给 `_search` 的宽松分支，由那条分支按 OR 召回（见 `_search`）。
    """
    import re

    stop_terms = (
        "想找", "找一", "有没有", "可以", "能够", "最好", "希望", "不要", "适合",
        "岗位", "实习", "工作", "机会", "什么", "怎么", "帮我", "推荐", "一些",
        "偏", "的", "和", "与", "或", "还是", "能", "会", "要", "在", "我",
    )
    parts = [p for p in re.split(r"[\s,，、。；;：:！!？?（）()【】\[\]\"'“”‘’/\\|]+", keyword) if p]
    # 只丢"填充字"边界，不丢实义词：让「找偏」「偏大」这种跨词边界的滑窗不出现，
    # 同时保留「大模」「模型」「落地」「工程」这类真正有区分度的 2 字词。
    fillers = "的了和与或还是在要会能想找有没有最好希望不要适合什么怎么帮我推荐一些偏每就都太很"
    terms: list[str] = []
    for part in parts:
        if part in stop_terms:
            continue
        if re.search(r"[A-Za-z0-9]", part):
            terms.append(part)
        # 中文：滑窗取 2 字词，跳过纯停用词片段
        han = re.sub(r"[^\u4e00-\u9fff]", "", part)
        if len(han) < 2:
            continue
        if han in stop_terms:
            continue
        for i in range(len(han) - 1):
            bigram = han[i:i + 2]
            if bigram in stop_terms or bigram in terms:
                continue
            if bigram[0] in fillers or bigram[1] in fillers:
                continue
            terms.append(bigram)
    if not terms:
        cleaned = " ".join(t for t in parts if t not in stop_terms)
        return cleaned or keyword
    return " ".join(terms[:max_terms])


def _retrieve(*args, **kwargs):
    """懒加载 rag.retriever.retrieve。

    放函数里而不是模块顶层：`rag.retriever` → `rag.embedder` → fastembed/jieba
    这条链在导入时就有开销；而且 tools_registry 被 dashboard / job_search / rag
    多处依赖，顶层重依赖会放大启动成本。真正要语义重排时再导入。
    """
    from rag.retriever import retrieve
    return retrieve(*args, **kwargs)


def _search(keyword, city=None, limit=20, semantic=False):
    """工具 search_jobs 的实现。

    Round 10 接入 RAG（乙方案：**SQL 先过滤、子集内语义重排**）：
      1) 先用 SQL 关键词/城市过滤出候选岗位（精确查询行为完全不变）；
      2) 只有 `semantic=True` **且** 候选数 <= `SEMANTIC_MAX_CANDIDATES` 时，
         才在**这个子集内**做一次语义重排（BM25 + 向量 RRF），把更贴近意图的
         岗位排到前面。

    Round 11 关键修正：semantic 模式下**不能**把整句自然语言当 AND 关键词去 LIKE。
    实测「想找偏大模型落地、能写工程代码的实习」整句 LIKE 命中 0 条 —— 候选池为空，
    后面的语义重排一条都跑不到。所以 semantic 分支先用 `_semantic_probe`
    抽短词做 **OR 召回**（放宽召回、不放松排序），排序仍然交给语义路。

    为什么要卡 80 条：语义重排要跑一次 embedding + BM25，候选太多时
      ① 成本高、② 收益低（"北京 Python"这种精确查询本身没有模糊空间）。
      所以大结果集直接返回 SQL 排序，保持可预期。
    """
    if semantic:
        rows = _semantic_rows(keyword, city, limit)
    else:
        rows = _rows_from_jobs(search_jobs(keyword, city, limit, platform="mock"))

    if not semantic or not rows or len(rows) > SEMANTIC_MAX_CANDIDATES:
        return rows

    allowed = [r["job_id"] for r in rows if r.get("job_id")]
    if not allowed:
        return rows
    # 语义查询用「原句」去检索（要的就是整句的语义），候选池由 probe 提供
    try:
        hits = _retrieve(keyword, top_k=len(rows), allowed_job_ids=allowed)
    except Exception as exc:                    # noqa: BLE001 —— 检索坏了就退回 SQL 结果
        print(f"[search_jobs] 语义重排不可用，退回 SQL 排序：{type(exc).__name__}: {exc}")
        return rows

    by_id = {r["job_id"]: r for r in rows}
    reranked = []
    seen_ids: set = set()
    # 注意：`retrieve` 返回的是 **chunk** 级命中，同一条 JD 会有多个 chunk 命中。
    # 这里必须按 job_id 去重（用集合），不能靠 `row in reranked` 比对 —— 每条
    # 带不同 score 的副本都是不同的 dict，比对会漏掉，导致同一岗位重复出现。
    for hit in hits:
        job_id = str((hit.get("metadata") or {}).get("job_id") or "")
        if job_id in seen_ids:
            continue
        row = by_id.get(job_id)
        if row is None:
            continue
        seen_ids.add(job_id)
        row = dict(row)
        row["score"] = round(float(hit.get("score") or 0.0), 6)
        reranked.append(row)
    # 语义路没覆盖到的候选挂在后面（不能因为重排把岗位弄丢）
    reranked.extend(r for r in rows if r.get("job_id") not in seen_ids)
    return reranked


def _rows_from_jobs(jobs) -> list[dict]:
    """Job 列表 -> 工具返回用的瘦身 dict（字段口径保持不变）。"""
    return [
        {
            "job_id": j.job_id,
            "title": j.title,
            "company": j.company,
            "city": j.city,
            "salary": j.salary,
            "tags": j.tags or [],
        }
        for j in jobs
    ]


def _semantic_rows(keyword: str, city, limit: int) -> list[dict]:
    """semantic 模式的候选召回：抽短词做 **OR** 召回，而不是整句 AND LIKE。

    分三级放宽，保证候选池不空（语义路才有东西可排）：
      1) probe 出的短词，逐个 OR（`search_jobs` 的语义分支按 OR 处理多词）；
      2) 还空 → 用最先出现的 1 个短词（最稳的一次收窄）；
      3) 还空 → 退到按时间倒序的近期岗位（**上限 `SEMANTIC_MAX_CANDIDATES`**，
         否则整库都进来，语义重排的代价就失控了）。
    候选集只影响"可选范围"，最终顺序仍由 `_retrieve` 的 RRF 决定。
    """
    from agent.tools.job_search import search_jobs as _raw_search

    probe = _semantic_probe(keyword)
    per_term = max(int(limit or 20), 20)
    seen: dict[str, dict] = {}

    def _collect(rows: list[dict]) -> None:
        for row in rows:
            if len(seen) >= SEMANTIC_MAX_CANDIDATES:
                return
            jid = str(row.get("job_id") or "")
            if jid and jid not in seen:
                seen[jid] = row

    if probe:
        _collect(_rows_from_jobs(_raw_search(probe, city, per_term, platform="mock",
                                             match_any=True)))
    terms = [t for t in probe.split() if t]
    if not seen and terms:
        _collect(_rows_from_jobs(_raw_search(terms[0], city, per_term, platform="mock")))
    if not seen:
        # 最后兜底：近期岗位（跨平台、不限关键词）。用两倍上限召回再截断，
        # 留一点余量给"截断偏好"。
        _collect(_rows_from_jobs(_raw_search(
            "", city, SEMANTIC_MAX_CANDIDATES * 2, platform="mock")))
    return list(seen.values())[:SEMANTIC_MAX_CANDIDATES]


def _detail(job_id):
    d = get_job_detail("mock", job_id)
    return {
        "job_id": d.job_id,
        "title": d.title,
        "company": d.company,
        "city": d.city,
        "salary": d.salary,
        "description": d.description,
        "requirements": d.requirements,
        "education": d.education,
    }


def _match(job_id, resume_json):
    if isinstance(resume_json, str):
        resume_data = json.loads(resume_json)
    else:
        resume_data = resume_json
    resume = Resume(
        name=resume_data.get("name", "匿名"),
        skills=resume_data.get("skills", []),
        experience=resume_data.get("experience", []),
        projects=resume_data.get("projects", []),
        education=resume_data.get("education", ""),
        city=resume_data.get("city", ""),
    )
    detail = get_job_detail("mock", job_id)
    result = match_resume_to_jd(resume, detail)
    return {
        "score": result.score,
        "dimensions": result.dimensions,
        "gaps": result.gaps,
        "highlights": result.highlights,
    }


def _add_tracking(company, title, platform="mock", url=""):
    """添加投递记录"""
    app_id = storage.create_application(company, title, platform, url)
    return {"id": app_id, "company": company, "title": title, "status": "applied"}


def _list_tracking(status=None):
    """查询投递记录"""
    apps = storage.list_applications(status)
    return [
        {
            "id": a["id"],
            "company": a["company"],
            "title": a["title"],
            "status": a["status"],
            "applied_at": a["applied_at"],
        }
        for a in apps
    ]


def _locate_application(company):
    """按公司名定位投递记录，找不到直接报错（避免误改/误删别的记录）"""
    record = storage.find_application(company)
    if not record:
        raise ValueError(f"未找到公司「{company}」的投递记录")
    return record


def _update_tracking_status(company, new_status, note=""):
    """修改投递记录状态：find_application 定位 → 校验状态转换 → 更新"""
    record = _locate_application(company)
    from_status = record["status"]
    to_status = str(new_status or "").strip().lower()

    if to_status not in state_machine.STATUS:
        raise ValueError(
            "未知状态：{}，可选：{}".format(
                new_status, "/".join(state_machine.STATUS)
            )
        )
    if to_status != from_status:
        # 非法流转（如 applied → offer）会抛 ValueError，由调用方反馈给用户
        state_machine.validate_transition(from_status, to_status)

    # 状态没变时也记一条事件，把 note 留在事件流里
    storage.update_status(record["id"], to_status, note or "")

    return {
        "id": record["id"],
        "company": record["company"],
        "title": record["title"],
        "from_status": from_status,
        "to_status": to_status,
        "status_label": state_machine.get_status_label(to_status),
        "changed": to_status != from_status,
        "note": note or "",
    }


def _delete_tracking(company):
    """删除投递记录及其状态事件：find_application 定位 → delete_application"""
    record = _locate_application(company)
    storage.delete_application(record["id"])
    return {
        "deleted": True,
        "id": record["id"],
        "company": record["company"],
        "title": record["title"],
        "last_status": record["status"],
    }


def _update_tracking_notes(company, notes):
    """修改投递记录备注：find_application 定位 → storage.update_notes（不动状态）"""
    record = _locate_application(company)
    storage.update_notes(record["id"], notes or "")
    return {"company": record["company"], "notes": notes or ""}


# ========== C1：多版本简历工具 ==========
#
# 「当前使用哪份简历」是会话级状态：工具函数拿不到 Chainlit 的 user_session，
# 所以放在本模块的进程级字典里（同一个 Agent 进程内共享）。
# 进程重启后回落到 storage.get_default_resume()（默认/最新一份），不会丢功能。
#
# 多用户：改成按 user_id 分桶的**二级字典**，而不是 ContextVar。
# 两个理由：① 状态要跨轮保留（工具在子线程执行，ContextVar 传不进去也留不下）；
# ② 同一进程里多个用户并发时要各看各的。没有登录态时桶是 DEFAULT_USER_ID。

_SESSION_STATE: dict = {}


def _session_state() -> dict:
    """当前用户的状态桶（不存在则建）。"""
    user_id = get_current_user()
    state = _SESSION_STATE.get(user_id)
    if state is None:
        state = {"current_resume_id": None}
        _SESSION_STATE[user_id] = state
    return state


def save_resume_tool(name, content):
    """保存一份简历（多版本），返回 resume_id"""
    resume_id = storage.save_resume(name, content)
    return {
        "id": resume_id,
        "name": name,
        "saved": True,
        "hint": "用 use_resume 把它设为当前使用；用 list_resumes 查看所有版本。",
    }


def list_resumes_tool():
    """列出所有简历版本（不含内容）"""
    items = storage.list_resumes()
    default = storage.get_default_resume()
    default_id = default["id"] if default else None
    current_id = _session_state().get("current_resume_id")
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "created_at": r["created_at"],
            "is_default": r["id"] == default_id,
            "is_current": r["id"] == current_id,
        }
        for r in items
    ]


def get_resume_tool(resume_id):
    """取某份简历的完整内容（含 content）"""
    data = storage.get_resume(resume_id)
    if not data:
        raise ValueError(f"未找到简历：{resume_id}（可用 list_resumes 查看现有版本）")
    return data


def use_resume(resume_id):
    """把某份简历设为当前使用（本会话内生效），切换技术岗版 / 产品岗版"""
    data = storage.get_resume(resume_id)
    if not data:
        raise ValueError(f"未找到简历：{resume_id}（可用 list_resumes 查看现有版本）")
    _session_state()["current_resume_id"] = str(resume_id).strip()
    return {
        "id": data["id"],
        "name": data.get("name", ""),
        "current": True,
        "hint": "之后需要简历的匹配/改写都会用这一份。",
    }


def get_current_resume():
    """取当前该用的简历：会话里 use_resume 设过的优先，否则用默认/最新一份。

    给 react_agent 用：调用方没显式传 resume_data 时，自动挂上当前简历。
    """
    current_id = _session_state().get("current_resume_id")
    if current_id:
        data = storage.get_resume(current_id)
        if data:
            return data
        _session_state()["current_resume_id"] = None    # 那份已被删，清理掉
    return storage.get_default_resume()


# ========== C4 / F1：简历导出 PDF + 一键投递包 ==========
#
# 产出目录可用环境变量覆盖（测试请指向临时目录，别往仓库里写）：
#   EXPORT_DIR   默认 agent/data/exports/
#   PACKAGE_DIR  默认 agent/data/packages/

_REPO_DIR = Path(__file__).resolve().parent.parent          # 仓库根目录
_DATA_DIR = _REPO_DIR / "agent" / "data"

EXPORT_DIR = Path(os.getenv("EXPORT_DIR", str(_DATA_DIR / "exports")))
PACKAGE_DIR = Path(os.getenv("PACKAGE_DIR", str(_DATA_DIR / "packages")))


def export_resume_pdf_tool(resume_id):
    """把一份简历导出成 PDF，返回文件路径。

    路径：agent/data/exports/{resume_id}_{时间戳}.pdf
    """
    data = storage.get_resume(resume_id)
    if not data:
        raise ValueError(f"未找到简历：{resume_id}（可用 list_resumes 查看现有版本）")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"{data['id']}_{stamp}.pdf"
    export_resume_pdf(data, str(path))

    return {
        "resume_id": data["id"],
        "name": data.get("name", ""),
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "pdf_backend": available_backend(),
    }


COVER_LETTER_PROMPT = """你是求职者本人，正在写一封投递用的自荐信（cover letter）。

【我的简历】
{resume}

【目标岗位】
公司：{company}
岗位：{title}
城市：{city}
任职要求：
{requirements}

【要求】
1. 第一人称，正文 200-300 个中文字符。
2. 结构：开头点明应聘的岗位 → 中间用简历里真实存在的技能/实习/项目说明为什么匹配
   （尽量呼应上面的任职要求）→ 结尾表达期待面试。
3. **绝对不能编造简历里没有的经历、技能、成绩或数字**。
4. 直接输出自荐信正文（可以有称呼和结尾问候），不要标题、不要 markdown 围栏、不要解释。
"""


def _safe_name(text) -> str:
    """把公司名清洗成能当目录名用的字符串（去掉 Windows 非法字符）"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", str(text or "").strip())
    return cleaned.strip("_")[:60] or "unknown"


def _norm_text(text) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _job_id_from_url(url) -> str:
    """从岗位链接里抠 job_id：优先 /job/xxx、/intern/xxx 这类路径，其次最后一段"""
    text = str(url or "").strip()
    if not text:
        return ""
    text = text.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    match = re.search(
        r"(?:job|jobs|intern|interns|position|positions|detail|details)/([A-Za-z0-9_-]{4,})",
        text,
    )
    if match:
        return match.group(1)
    tail = text.rsplit("/", 1)[-1]
    return tail if re.fullmatch(r"[A-Za-z0-9_-]{4,}", tail) else ""


def _resolve_job(record, job_id=None):
    """定位岗位详情：显式 job_id → 投递记录 URL 里的 id → 按公司/岗位名搜索兜底。

    返回 (JobDetail 或 None, 警告列表)；拿不到详情也不抛异常，让投递包照常生成。
    """
    tried = []
    candidates = [job_id, _job_id_from_url(record.get("url", ""))]
    for candidate in candidates:
        if not candidate or not str(candidate).strip():
            continue
        try:
            return get_job_detail("mock", str(candidate).strip()), []
        except Exception as e:                      # noqa: BLE001 - id 不对就换下一个办法
            tried.append(f"{candidate}（{type(e).__name__}）")

    keywords = []
    for value in (record.get("title", ""), record.get("company", "")):
        if value and value not in keywords:
            keywords.append(value)

    for keyword in keywords:
        try:
            jobs = search_jobs(keyword, None, 20, platform="mock")
        except Exception:                           # noqa: BLE001 - 搜索失败就试下一个关键词
            continue
        for job in jobs:
            if (_norm_text(job.company) == _norm_text(record.get("company"))
                    or _norm_text(job.title) == _norm_text(record.get("title"))):
                try:
                    return get_job_detail("mock", job.job_id), []
                except Exception:                   # noqa: BLE001 - 换下一个搜索结果
                    continue

    detail = "、".join(tried) if tried else "投递记录里没有可用 job_id"
    return None, [f"没能拿到岗位详情（尝试过：{detail}），岗位信息将按投递记录生成"]


def resolve_job(record, job_id=None):
    """公开版 _resolve_job：按投递记录定位岗位详情，给 Dashboard 之类的外部调用复用。

    返回 (JobDetail 或 None, 警告列表)；拿不到详情也不抛异常，调用方按 None 走兜底。
    """
    return _resolve_job(record, job_id)


def _resume_to_dataclass(data: dict) -> Resume:
    """结构化简历 dict → Resume（tailor.py / resume_match.py 用的数据类）"""
    return Resume(
        name=str(data.get("name") or "匿名"),
        skills=[str(s) for s in (data.get("skills") or []) if str(s).strip()],
        experience=list(data.get("experience") or []),
        projects=list(data.get("projects") or []),
        education=str(data.get("education") or ""),
        city=str(data.get("city") or ""),
    )


def _tailor_resume(data: dict, detail) -> tuple:
    """按岗位定制简历，返回 (简历 dict, 警告列表)；失败就退回原简历"""
    try:
        result = tailor_resume(_resume_to_dataclass(data), detail)
    except Exception as e:                          # noqa: BLE001 - LLM 失败不该让投递包生成失败
        return data, [f"简历定制失败（{type(e).__name__}: {e}），已改用原简历"]

    tailored = (result or {}).get("tailored")
    if not isinstance(tailored, dict) or not tailored:
        return data, ["简历定制返回空结果，已改用原简历"]

    warnings = [f"简历定制提醒：{w}" for w in (result.get("warnings") or [])]
    return tailored, warnings


def _fallback_cover_letter(company: str, title: str) -> str:
    return (
        f"尊敬的{company}招聘负责人：\n\n"
        f"您好！我希望应聘贵公司的「{title}」岗位。\n\n"
        f"我具备该岗位需要的技术基础，也有相关的实习与项目经历，"
        f"能够较快上手实际工作，并在过程中持续补充岗位所需的技能。\n"
        f"很期待有机会与您进一步沟通，也希望能为团队做出贡献。\n\n"
        f"（注：本段为模板兜底版本，LLM 生成失败，请手动补充项目细节后再投递。）\n\n"
        f"此致\n敬礼"
    )


def _generate_cover_letter(resume_data, record, detail) -> tuple:
    """用 LLM 生成自荐信，返回 (正文, 警告或空串)"""
    company = record.get("company", "")
    title = record.get("title", "")
    prompt = COVER_LETTER_PROMPT.format(
        resume=json.dumps(resume_data, ensure_ascii=False, default=str)[:2000],
        company=company,
        title=title,
        city=getattr(detail, "city", "") if detail else "",
        requirements=(getattr(detail, "requirements", "") or "（未获取到）")[:1500]
        if detail else "（未获取到岗位详情）",
    )

    try:
        text = (chat([{"role": "user", "content": prompt}],
                     source="application_package") or "").strip()
        if text.startswith("```"):                  # 模型偶尔会套一层围栏
            match = re.search(r"```(?:markdown|md|text)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        if text:
            return text, ""
        raise ValueError("模型返回空内容")
    except Exception as e:                          # noqa: BLE001 - 生成失败也要让投递包落地
        return (_fallback_cover_letter(company, title),
                f"自荐信生成失败（{type(e).__name__}: {e}），已用模板兜底")


def _job_info_text(record: dict, detail, job_id_used: str) -> str:
    """岗位信息 + 链接 + 投递记录，写进 job_info.txt"""
    lines = ["投递岗位信息", "=" * 40, f"公司：{record.get('company', '')}",
             f"岗位：{record.get('title', '')}"]

    url = record.get("url", "")
    if detail is not None:
        lines += [
            f"平台：{detail.platform}",
            f"job_id：{detail.job_id}",
            f"城市：{detail.city}",
            f"薪资：{detail.salary}",
            f"学历要求：{detail.education}",
            f"出勤/时长：{detail.days_per_week} {detail.duration}".strip(),
            f"标签：{'、'.join(detail.tags or []) or '（无）'}",
            "",
            "【岗位职责】",
            detail.description or "（无）",
            "",
            "【任职要求】",
            detail.requirements or "（无）",
        ]
        if detail.bonus:
            lines += ["", "【加分项】", detail.bonus]
        if detail.url:
            url = detail.url
    else:
        lines += [f"job_id：{job_id_used or '（未知）'}", "（未获取到岗位详情，以下链接来自投递记录）"]

    lines += [
        "",
        "【岗位链接】",
        url or "（投递记录里没有链接）",
        "",
        "【投递记录】",
        f"记录 id：{record.get('id', '')}",
        f"来源平台：{record.get('platform', '')}",
        f"当前状态：{record.get('status', '')}",
        f"投递时间：{record.get('applied_at', '')}",
        f"打包时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    return "\n".join(lines)


def generate_application_package(company, job_id=None):
    """生成一键投递包：简历 PDF + 自荐信 + 岗位信息。

    目录：agent/data/packages/{company}_{时间戳}/
      - resume.pdf       按岗位定制后的简历（定制失败则用当前简历）
      - cover_letter.md  LLM 生成的自荐信（200-300 字）
      - job_info.txt     岗位信息 + 链接 + 投递记录

    返回：{"company", "job_id", "package_dir", "files": {...}, "warnings": [...]}
    """
    record = storage.find_application(company)
    if not record:
        raise ValueError(
            f"未找到公司「{company}」的投递记录（可用 list_tracking 看现有记录）"
        )

    warnings = []
    detail, job_warnings = _resolve_job(record, job_id)
    warnings += job_warnings
    job_id_used = detail.job_id if detail is not None else (
        str(job_id).strip() if job_id else ""
    )

    resume_record = get_current_resume()
    if not resume_record:
        raise ValueError("还没有简历，无法生成投递包（先 save_resume 保存一份）")
    resume_data = normalize_resume(resume_record)

    tailored = False
    if detail is None:
        final_resume = resume_data or resume_record
        warnings.append("没有岗位详情，简历按原样导出（未做定制）")
    elif "_plain" in resume_data:
        final_resume = resume_data
        warnings.append("当前简历是纯文本，跳过按岗位定制（PDF 仍会导出原文）")
    else:
        final_resume, tailor_warnings = _tailor_resume(resume_data, detail)
        tailored = final_resume is not resume_data
        warnings += tailor_warnings

    cover_letter, cover_warning = _generate_cover_letter(resume_data, record, detail)
    if cover_warning:
        warnings.append(cover_warning)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = PACKAGE_DIR / f"{_safe_name(record.get('company'))}_{stamp}"
    package_dir.mkdir(parents=True, exist_ok=True)

    resume_pdf = package_dir / "resume.pdf"
    export_resume_pdf(final_resume, str(resume_pdf))

    cover_path = package_dir / "cover_letter.md"
    cover_path.write_text(
        f"# 自荐信 · {record.get('company', '')} {record.get('title', '')}\n\n{cover_letter}\n",
        encoding="utf-8",
    )

    info_path = package_dir / "job_info.txt"
    info_path.write_text(_job_info_text(record, detail, job_id_used), encoding="utf-8")

    return {
        "company": record.get("company", ""),
        "title": record.get("title", ""),
        "job_id": job_id_used,
        "package_dir": str(package_dir),
        "files": {
            "resume.pdf": str(resume_pdf),
            "cover_letter.md": str(cover_path),
            "job_info.txt": str(info_path),
        },
        "resume_tailored": tailored,
        "has_job_detail": detail is not None,
        "warnings": warnings,
    }


# ========== 运行时校验：风险分级 / 参数边界 / 超时 / 权限确认 ==========
#
# 《深入理解 AI Agent》第 6 章：工具必须有运行时校验（权限、风险、边界）。
# 每个工具在 TOOLS 里声明 risk_level / timeout / requires_confirmation，
# 由 call_tool 统一执行「存在性 → 参数 → 风险确认 → 超时执行 → 日志」五道关卡。
#
# risk_level 语义（按伤害半径）：
#   read          只读，不改任何状态
#   reversible    可撤销（改动能再改回来 / 产物可重建）
#   irreversible  不可撤销（删了就没，必须用户确认）

VALID_RISK_LEVELS = ("read", "reversible", "irreversible")
DEFAULT_TIMEOUT = 30                       # 秒，未显式声明 timeout 时的兜底


class ToolNeedsConfirmation(Exception):
    """工具风险过高，需要用户确认后才能执行。

    上层 UI 捕获它，用 name / args / risk_level 决定是否弹确认框；
    用户确认后以 call_tool(name, args, confirmed=True) 重放这次调用。
    """

    def __init__(self, name: str, args: dict, risk_level: str = "irreversible"):
        self.name = name
        self.args = dict(args or {})
        self.risk_level = risk_level
        super().__init__(
            f"工具「{name}」风险等级为 {risk_level}，需要用户确认后才能执行"
        )


class ToolTimeoutError(TimeoutError):
    """工具执行超时：结果已被放弃（后台线程可能仍在跑）。"""


_ARG_TYPES = (str, int, float, bool, list, dict, type(None))


def _validate_args(name: str, args, spec: dict) -> dict:
    """参数校验：必须是 dict、不能带未声明参数、值必须是 JSON 能表达的类型。"""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise TypeError(f"工具「{name}」的参数必须是 dict，收到 {type(args).__name__}")

    declared = set(spec.get("parameters") or {})
    unknown = sorted(str(k) for k in args if k not in declared)
    if unknown:
        raise TypeError(
            "工具「{}」不支持参数：{}；可用参数：{}".format(
                name, "、".join(unknown), "、".join(sorted(declared)) or "（无）"
            )
        )

    for key, value in args.items():
        if isinstance(value, bytes) or not isinstance(value, _ARG_TYPES):
            raise TypeError(
                f"工具「{name}」参数 {key} 类型不支持：{type(value).__name__}"
            )
    return args


def _log_tool(name: str, risk: str, status: str, **fields) -> None:
    """统一日志：[tool] name=delete_tracking risk=irreversible latency=123ms status=ok"""
    line = f"[tool] name={name} risk={risk}"
    line += "".join(f" {key}={value}" for key, value in fields.items())
    print(f"{line} status={status}")


def validate_registry() -> None:
    """注册表自检：每个工具的元数据必须齐全且合法（导入时执行一次）。"""
    for name, spec in TOOLS.items():
        risk = spec.get("risk_level")
        if risk not in VALID_RISK_LEVELS:
            raise ValueError(
                "工具「{}」risk_level={!r} 非法，可选：{}".format(
                    name, risk, "/".join(VALID_RISK_LEVELS)
                )
            )
        timeout = spec.get("timeout", DEFAULT_TIMEOUT)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"工具「{name}」timeout={timeout!r} 必须是正数")
        if not isinstance(spec.get("requires_confirmation"), bool):
            raise ValueError(f"工具「{name}」缺少 requires_confirmation(bool)")
        if not callable(spec.get("func")):
            raise ValueError(f"工具「{name}」缺少可调用的 func")


TOOLS = {
    "search_jobs": {
        "description": (
            "搜索实习岗位，返回岗位列表。默认走关键词精确匹配（结果可预期、最快）。"
            "**只有用户的模糊需求才用 semantic**：当用户描述的是「什么样的岗位」"
            "而不是具体关键词时（如「想找偏大模型落地、能写工程代码的实习」"
            "「有没有适合我的 AI 岗」），才把 semantic 传 true —— 此时会在候选集内"
            "做一次语义重排，更贴近意图；候选超过 80 条时自动退回关键词排序。"
            "关键词/城市这类精确查询（如「北京 Python」）**不要**传 semantic。"
        ),
        "parameters": {
            "keyword": "搜索关键词",
            "city": "城市（可选）",
            "limit": "数量，默认 20",
            "semantic": (
                "是否启用语义重排，默认 false。仅在用户的模糊需求（描述『什么样的岗位』、"
                "没有明确关键词）时传 true；关键词/城市精确查询保持 false。"
            ),
        },
        "func": _search,
        "risk_level": "read",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "get_job_detail": {
        "description": "获取岗位完整 JD 详情。",
        "parameters": {"job_id": "岗位 ID"},
        "func": _detail,
        "risk_level": "read",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "match_resume": {
        "description": "简历与岗位匹配打分。",
        "parameters": {
            "job_id": "岗位 ID",
            "resume_json": "简历 JSON 字符串或对象",
        },
        "func": _match,
        "risk_level": "read",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "add_tracking": {
        "description": "把岗位添加到投递追踪系统。",
        "parameters": {
            "company": "公司名",
            "title": "岗位名",
            "url": "岗位链接（可选）",
        },
        "func": _add_tracking,
        "risk_level": "reversible",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "list_tracking": {
        "description": (
            "查询投递追踪记录。排查/复核用：删除或改状态之后，用它确认库里实际还剩什么，"
            "不要凭记忆回答条数。"
        ),
        "parameters": {
            "status": "状态过滤（可选），如 applied/viewed/interview",
        },
        "func": _list_tracking,
        "risk_level": "read",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "update_tracking_status": {
        "description": (
            "修改一家公司的投递记录状态（按公司名定位记录）。"
            "只能按状态机合法流转，如 applied→viewed→interview→interviewing→offer→accepted，"
            "任意非终态都可转为 rejected/withdrawn。非法流转会被拒绝。"
        ),
        "parameters": {
            "company": "公司名（用于定位记录，支持模糊匹配）",
            "new_status": (
                "新状态：applied/viewed/interview/interviewing/"
                "offer/accepted/rejected/withdrawn"
            ),
            "note": "备注（可选）",
        },
        "func": _update_tracking_status,
        "risk_level": "reversible",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "delete_tracking": {
        "description": (
            "删除一家公司的投递记录（连同它的状态变更历史），不可恢复。"
            "一次只能删一家公司，因此不支持“删除全部记录”这类批量操作。"
        ),
        "parameters": {
            "company": "公司名（用于定位记录，支持模糊匹配）",
        },
        "func": _delete_tracking,
        "risk_level": "irreversible",
        "timeout": 30,
        "requires_confirmation": True,
    },
    "update_tracking_notes": {
        "description": (
            "修改一家公司的投递记录备注（按公司名定位记录）。"
            "只改备注，不动状态、不改投递时间。"
        ),
        "parameters": {
            "company": "公司名（用于定位记录，支持模糊匹配）",
            "notes": "新的备注内容（覆盖原备注，传空字符串表示清空）",
        },
        "func": _update_tracking_notes,
        "risk_level": "reversible",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "save_resume": {
        "description": (
            "保存一份简历（支持多版本，比如「技术岗版」「产品岗版」）。"
            "同名简历不会覆盖，每次保存都是新的一份。"
        ),
        "parameters": {
            "name": "简历名称，如「技术岗版」",
            "content": (
                "简历内容：JSON 字符串/对象（含 name/skills/experience/projects/"
                "education/city）或纯文本简历"
            ),
        },
        "func": save_resume_tool,
        "risk_level": "reversible",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "list_resumes": {
        "description": "列出已有的所有简历版本（只给 id/名称/创建时间，不含内容）。",
        "parameters": {},
        "func": list_resumes_tool,
        "risk_level": "read",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "get_resume": {
        "description": "取某一份简历的完整内容。",
        "parameters": {
            "resume_id": "简历 ID（list_resumes 返回的 id）",
        },
        "func": get_resume_tool,
        "risk_level": "read",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "use_resume": {
        "description": (
            "把某一份简历设为当前使用（本会话内生效）。"
            "用户说「换用产品岗版简历」时调用它，之后需要简历的操作都会用这一份。"
        ),
        "parameters": {
            "resume_id": "简历 ID（list_resumes 返回的 id）",
        },
        "func": use_resume,
        "risk_level": "reversible",
        "timeout": 30,
        "requires_confirmation": False,
    },
    "export_resume_pdf": {
        "description": (
            "把某一份简历导出成 PDF 文件并返回文件路径。"
            "用户说「导出简历 PDF」「把简历转成 PDF」时调用它。"
        ),
        "parameters": {
            "resume_id": "简历 ID（list_resumes 返回的 id）",
        },
        "func": export_resume_pdf_tool,
        "risk_level": "read",
        "timeout": 60,
        "requires_confirmation": False,
    },
    "generate_application_package": {
        "description": (
            "给一家公司生成一键投递包：按岗位定制的简历 PDF + 自荐信 + 岗位信息（含链接），"
            "打包到 agent/data/packages/ 下的一个目录里，返回目录路径。"
            "用户说「生成投递包」「把简历和自荐信打包」时调用它。"
        ),
        "parameters": {
            "company": "公司名（用于定位投递记录，支持模糊匹配）",
            "job_id": "岗位 ID（可选，不传就自动从投递记录/搜索结果里找）",
        },
        "func": generate_application_package,
        "risk_level": "reversible",
        "timeout": 120,
        "requires_confirmation": False,
    },
}


validate_registry()      # 导入即校验：元数据缺失/非法就早失败


def list_tools_description() -> str:
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"### {name}")
        lines.append(f"作用：{info['description']}")
        lines.append("参数：")
        for k, v in info["parameters"].items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def call_tool(name: str, args: dict = None, confirmed: bool = False):
    """执行工具，并施加运行时校验：存在性 → 参数 → 风险确认 → 超时 → 日志。

    Args:
        name: 工具名，必须在 TOOLS 中注册。
        args: 参数字典；None 按 {} 处理。
        confirmed: 上层 UI 已让用户确认过风险时传 True。

    Raises:
        ValueError: 工具未注册。
        TypeError: 参数不是 dict / 带未声明参数 / 参数类型不支持。
        ToolNeedsConfirmation: requires_confirmation=True 且未确认。
        ToolTimeoutError: 执行超过该工具的 timeout（秒）。
    """
    # 1. 工具必须存在
    if name not in TOOLS:
        raise ValueError(f"未知工具：{name}")

    spec = TOOLS[name]
    risk = spec.get("risk_level", "read")
    timeout = float(spec.get("timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)

    # 2. 参数校验：必须是 dict、不带未声明参数、值类型可 JSON 表达
    safe_args = _validate_args(name, args, spec)

    # 3. 风险确认：需要确认的工具未经确认一律不执行，也不留任何副作用
    if spec.get("requires_confirmation") and not confirmed:
        _log_tool(name, risk, "needs_confirmation", timeout=f"{timeout:g}s")
        raise ToolNeedsConfirmation(name, safe_args, risk)

    _log_tool(name, risk, "start", timeout=f"{timeout:g}s")
    started = time.perf_counter()

    # 4. 超时执行：工具跑在子线程里，主线程最多等 timeout 秒
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
    try:
        # 工具跑在子线程里，而子线程**不继承**调用方的 ContextVar ——
        # 必须显式把当前上下文带过去，否则工具里读到的用户永远是默认值。
        ctx = contextvars.copy_context()
        future = executor.submit(ctx.run, spec["func"], **safe_args)
        try:
            result = future.result(timeout=timeout)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if isinstance(exc, FutureTimeoutError) and not future.done():
                future.cancel()
                _log_tool(name, risk, "timeout", latency=f"{elapsed_ms:.0f}ms")
                raise ToolTimeoutError(
                    f"工具「{name}」执行超过 {timeout:g}s，本轮调用已放弃"
                ) from exc
            _log_tool(name, risk, "error",
                      latency=f"{elapsed_ms:.0f}ms", error=type(exc).__name__)
            raise

        # 5. 成功日志：耗时 + 状态
        _log_tool(name, risk, "ok",
                  latency=f"{(time.perf_counter() - started) * 1000:.0f}ms")
        # 6. 返回结果
        return result
    finally:
        executor.shutdown(wait=False)      # 超时后不阻塞主线程等子线程收尾


# ========== 自测：投递追踪的改/删/改备注（用临时库，不碰 agent/data/applications.db） ==========

def _run_selftest() -> int:
    """验证 find_application / update_tracking_status / update_tracking_notes / delete_tracking。

    隔离方式：APP_DB_PATH 指向临时文件，并显式改写 storage.DB_PATH。
    storage 在导入时就把环境变量固化成 DB_PATH（storage.py 第 16-19 行），
    所以本进程内必须直接改这个模块属性，否则调用仍会打到真实库。
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    real_db = Path(storage.DB_PATH)
    real_stat = real_db.stat() if real_db.exists() else None

    # 临时库位置：优先系统临时目录；若该目录不可写（受限环境/沙箱），
    # 退回 agent/evaluation/（与 run_agent_eval.py 的 test.db 同目录，跑完即删）。
    tmp_db = None
    tmp_dir = None
    for candidate in (Path(tempfile.gettempdir()) / f"tracking_selftest_{os.getpid()}",
                      Path(__file__).resolve().parent / "evaluation"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / "selftest_applications.db"
            with open(probe, "wb"):        # 探测可写性，sqlite 打不开时这里就会失败
                pass
        except OSError:
            continue
        tmp_db = probe
        tmp_dir = candidate
        break

    if tmp_db is None:
        print("找不到可写的临时目录，自测中止（未触碰任何数据库）")
        return 1

    os.environ["APP_DB_PATH"] = str(tmp_db)
    storage.DB_PATH = tmp_db

    checks = []

    def check(label, fn):
        try:
            detail = fn()
            checks.append(True)
            print(f"  [PASS] {label}" + (f"（{detail}）" if detail else ""))
        except Exception as e:                                  # 断言/工具异常都算失败
            checks.append(False)
            print(f"  [FAIL] {label} → {type(e).__name__}: {e}")

    def expect_error(fn, kind=ValueError):
        """确认某个调用按预期报错，返回错误信息"""
        try:
            fn()
        except kind as e:
            return str(e)
        raise AssertionError(f"预期抛 {kind.__name__}，但没有报错")

    try:
        storage.init_db()
        assert storage.DB_PATH == tmp_db, "临时库未生效，中止（防止污染真实库）"

        # 预置：阶跃星辰两条（验证“多条取最近”）、腾讯一条、OpenAI 一条（验证大小写）
        tencent_id = storage.create_application("腾讯", "大模型算法实习生", "mock", "")
        step_old = storage.create_application("阶跃星辰", "Agent 开发实习生", "mock", "")
        step_new = storage.create_application("阶跃星辰", "Agent 平台实习生", "mock", "")
        storage.create_application("OpenAI", "Research Intern", "mock", "")

        print(f"\n临时库：{tmp_db}")
        print(f"真实库：{real_db}（只读校验，不应被写入）\n")

        def check_find_hit():
            rec = storage.find_application("阶跃星辰")
            assert rec is not None, "应该找到记录，却返回 None"
            assert rec["id"] == step_new, f"多条匹配应取最近一条，实际拿到 {rec['id']}"
            assert storage.find_application("阶跃") is not None, "应支持模糊匹配"
            assert storage.find_application("openai") is not None, "应不区分大小写"
            return f"命中 {rec['company']} / {rec['title']}（id={rec['id']}，同公司共 2 条取最近）"

        def check_find_miss():
            assert storage.find_application("不存在公司") is None, "不该命中任何记录"
            assert storage.find_application("") is None, "空公司名应返回 None"
            return "返回 None"

        def check_update():
            out = call_tool("update_tracking_status", {
                "company": "阶跃星辰", "new_status": "viewed", "note": "HR 已读",
            })
            assert out["changed"] is True and out["to_status"] == "viewed", out
            saved = storage.get_application(out["id"])
            assert saved["status"] == "viewed", f"库里的状态是 {saved['status']}"

            events = storage.get_events(out["id"])
            assert any(e["note"] == "HR 已读" for e in events), f"事件里没有备注：{events}"

            # 合法性校验：非法流转 / 未知状态 / 公司不存在，都必须报错且不改库
            bad_transition = expect_error(lambda: call_tool("update_tracking_status", {
                "company": "腾讯", "new_status": "offer",
            }))
            bad_status = expect_error(lambda: call_tool("update_tracking_status", {
                "company": "腾讯", "new_status": "banana",
            }))
            bad_company = expect_error(lambda: call_tool("update_tracking_status", {
                "company": "不存在公司", "new_status": "viewed",
            }))
            assert storage.get_application(tencent_id)["status"] == "applied", "被拒的调用不该改库"

            return (f"applied → viewed 成功；非法流转/未知状态/找不到公司均被拒"
                    f"（{bad_transition}｜{bad_status}｜{bad_company}）")

        def check_notes():
            before = storage.get_application(step_new)
            out = call_tool("update_tracking_notes", {
                "company": "阶跃星辰", "notes": "HR 说下周约面",
            })
            assert out == {"company": "阶跃星辰", "notes": "HR 说下周约面"}, out
            saved = storage.get_application(step_new)
            assert saved["notes"] == "HR 说下周约面", f"库里的备注是 {saved['notes']!r}"
            assert saved["status"] == before["status"], "改备注不该动状态"

            missing = expect_error(lambda: call_tool("update_tracking_notes", {
                "company": "不存在公司", "notes": "x",
            }))
            call_tool("update_tracking_notes", {"company": "阶跃星辰", "notes": ""})
            assert (storage.get_application(step_new)["notes"] or "") == "", "空备注应清空"
            call_tool("update_tracking_notes", {"company": "阶跃星辰", "notes": "HR 说下周约面"})

            return f"备注落库并读回成功；找不到公司报错：{missing}"

        def check_runtime_guard():
            """运行时校验：元数据齐全、参数边界、超时中止、未知工具"""
            for tool_name, tool_spec in TOOLS.items():
                assert tool_spec.get("risk_level") in VALID_RISK_LEVELS, tool_name
                assert isinstance(tool_spec.get("timeout"), (int, float)), tool_name
                assert isinstance(tool_spec.get("requires_confirmation"), bool), tool_name

            bad_arg = expect_error(
                lambda: call_tool("list_tracking", {"nope": 1}), kind=TypeError
            )
            bad_type = expect_error(lambda: call_tool("search_jobs", "广州"), kind=TypeError)
            unknown = expect_error(lambda: call_tool("no_such_tool", {}), kind=ValueError)

            TOOLS["_selftest_sleep"] = {
                "description": "自测用：睡眠工具",
                "parameters": {"seconds": "睡眠秒数"},
                "func": lambda seconds: time.sleep(seconds) or "done",
                "risk_level": "read",
                "timeout": 1,
                "requires_confirmation": False,
            }
            try:
                timeout_err = expect_error(
                    lambda: call_tool("_selftest_sleep", {"seconds": 3}),
                    kind=ToolTimeoutError,
                )
            finally:
                TOOLS.pop("_selftest_sleep", None)

            return (f"元数据 {len(TOOLS)} 个工具齐全；未声明参数/类型错误/未知工具均被拒；"
                    f"超时被中止：{timeout_err}")

        def check_delete():
            # 不可逆工具必须先过确认：未确认要报 ToolNeedsConfirmation，且不许碰库
            unconfirmed = expect_error(
                lambda: call_tool("delete_tracking", {"company": "阶跃星辰"}),
                kind=ToolNeedsConfirmation,
            )
            assert storage.find_application("阶跃星辰") is not None, "未确认不该删掉任何记录"

            out = call_tool("delete_tracking", {"company": "阶跃星辰"}, confirmed=True)
            assert out["deleted"] is True and out["id"] == step_new, out
            assert storage.get_application(step_new) is None, "记录应该已被删除"
            assert storage.get_events(step_new) == [], "关联事件应一并删除"

            left = storage.find_application("阶跃星辰")
            assert left is not None and left["id"] == step_old, "应还有一条更早的同公司记录"
            call_tool("delete_tracking", {"company": "阶跃星辰"}, confirmed=True)
            assert storage.find_application("阶跃星辰") is None, "同公司记录应已删净"

            gone = expect_error(lambda: call_tool(
                "delete_tracking", {"company": "阶跃星辰"}, confirmed=True,
            ))
            assert storage.get_application(tencent_id) is not None, "不该误删其他公司的记录"

            return (f"删除 id={out['id']}（含事件）；未确认被拦：{unconfirmed}；"
                    f"重复删除报错：{gone}")

        check("0. call_tool 运行时校验（元数据/参数/超时）", check_runtime_guard)
        check("1. find_application('阶跃星辰') 找到记录", check_find_hit)
        check("2. find_application('不存在公司') 返回 None", check_find_miss)
        check("3. call_tool('update_tracking_status') 状态更新成功", check_update)
        check("4. call_tool('update_tracking_notes') 备注能改并落库", check_notes)
        check("5. call_tool('delete_tracking') 记录被删除", check_delete)

        if real_stat is not None:
            def check_real_db():
                now_stat = real_db.stat()
                assert (now_stat.st_mtime, now_stat.st_size) == (
                    real_stat.st_mtime, real_stat.st_size
                ), "真实库的 mtime/size 变了，可能被写入"
                return f"size={now_stat.st_size} 未变化"

            check("6.（额外）真实库未被写入", check_real_db)
    finally:
        if checks and all(checks):
            for suffix in ("", "-wal", "-shm", "-journal"):
                leftover = Path(str(tmp_db) + suffix)
                if leftover.exists():
                    leftover.unlink()
            if tmp_dir is not None and "tracking_selftest_" in tmp_dir.name:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\n临时库已删除：{tmp_db}")
        else:
            print(f"\n有失败项，保留临时库便于排查：{tmp_db}")

    passed = checks.count(True)
    print(f"自测结果：{passed}/{len(checks)} 通过")
    return 0 if passed == len(checks) and checks else 1


if __name__ == "__main__":
    raise SystemExit(_run_selftest())