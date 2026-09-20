"""
求职 Dashboard。
五个 Tab：
1. 岗位列表：搜索、标记想投/不合适
2. 投递追踪：查看投递记录和状态、生成投递包、 面试准备（预测问题 + 参考答案要点）
3. 匹配结果：简历-JD 匹配打分
4. Token 成本：LLM 用量按天/模型/来源统计
5. 简历管理：多版本简历的新建/上传/设默认/删除
"""
import sys
import json
import re
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import json5
import streamlit as st
import pandas as pd
from agent import storage
from agent import reminder
from agent.tools.job_search import search_jobs
from agent.tools.job_detail import get_job_detail
from agent.tools.resume_match import match_resume_to_jd, Resume
from agent.tools_registry import (
    export_resume_pdf_tool,
    generate_application_package,
    resolve_job,
)
from shared import token_tracker
from shared.llm_client import chat

try:
    # 以 `streamlit run dashboard/app.py` 方式启动时走这条（脚本所在目录会被
    # 放进 sys.path[0]，项目根目录由上面的 sys.path.insert 补上）
    from dashboard.styles import (
        e,
        icon,
        inject_styles,
        job_row,
        stat_card,
        stat_row,
        status_badge,
        timeline,
        STATUS_LABELS,
    )
except ModuleNotFoundError:  # pragma: no cover - 兼容 `python dashboard/app.py`
    from styles import (  # type: ignore[no-redef]
        e,
        icon,
        inject_styles,
        job_row,
        stat_card,
        stat_row,
        status_badge,
        timeline,
        STATUS_LABELS,
    )


# 初始化数据库
storage.init_db()

st.set_page_config(
    page_title="求职助手 Dashboard",
    page_icon="◎",
    layout="wide",
)

# 必须先 set_page_config、再注入样式
inject_styles()

st.title("求职助手 Dashboard")
st.markdown(
    '<p class="page-meta">岗位搜索 / 投递追踪 / 匹配打分 / Token 成本 / 简历管理'
    " —— 一个页面看完全流程。</p>",
    unsafe_allow_html=True,
)

# 投递包里几种文件的 MIME 类型（下载按钮用）
_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
}

# 表格行高：设计稿是 40px（Streamlit 1.5x 才支持 row_height，旧版本忽略）
_TABLE_ROW_HEIGHT = 40


def _render_dataframe(df, **kwargs):
    """统一封装的 st.dataframe：设计稿的 40px 行高 + 8px 圆角表格。

    ``row_height`` 是较新版本才有的参数，旧版本直接跳过，不会报错。
    """
    try:
        return st.dataframe(df, width="stretch", hide_index=True,
                            row_height=_TABLE_ROW_HEIGHT, **kwargs)
    except TypeError:
        return st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def _pipeline_spark(apps: list) -> list:
    """近 7 天每天的投递数（统计卡片的迷你柱状图用）。"""
    today = datetime.now().date()
    buckets = [0] * 7
    for item in apps:
        stamp = str(item.get("applied_at") or "")[:10]
        try:
            day = datetime.strptime(stamp, "%Y-%m-%d").date()
        except ValueError:
            continue
        offset = (today - day).days
        if 0 <= offset < 7:
            buckets[6 - offset] += 1
    return buckets


def _delta_dir(count: int) -> str:
    """转化率小标签的方向：>0 绿色上箭头，=0 灰色横杠。"""
    return "up" if count > 0 else "flat"


@st.cache_data(show_spinner=False, ttl=1800, max_entries=32)
def _resume_pdf_bytes(resume_id: str) -> bytes:
    """导出某份简历的 PDF 字节（缓存：同一份简历不会每次 rerun 都重新生成）"""
    info = export_resume_pdf_tool(resume_id)
    return Path(info["path"]).read_bytes()


# ============================================================
#  面试准备（D1 预测问题 + D2 参考答案要点）
# ============================================================
#
# 流程：投递记录 →（resolve_job）岗位 JD → LLM 出 5-8 个问题（技术/项目/行为）
#      → 每个问题配参考答案要点（只允许用简历里真实存在的内容）→ expander 展示 + 下载 md。
# 同一（公司+岗位+JD+简历）组合用 st.cache_data 缓存 1 小时，反复点不会重复烧 token。

INTERVIEW_PREP_PROMPT = """你是面试辅导教练。根据目标岗位 JD 和求职者简历，预测面试官最可能问的问题，并给出参考答案要点。

【目标岗位】
公司：{company}
岗位：{title}
岗位 JD：
{jd}

【求职者简历】
{resume}

【要求】
1. 出 5-8 个问题，按类别分布：技术 / 项目 / 行为，每类至少 1 个（category 字段只能从这三个词里选）。
2. 每个问题给 2-4 条「参考答案要点」，要具体到能照着准备（讲什么、用哪个例子、突出哪一点）。
3. 要点里出现的经历、技能、项目、数字**必须来自上面的简历**；简历里没有的，
   写成「需要补充：…」或「建议准备一个…的例子」，**绝对不能编造**经历、技能或成绩。
4. 只输出一个 JSON 对象，不要解释文字、不要 markdown 围栏：
{{
  "questions": [
    {{"category": "技术", "question": "问题原文", "answer_points": ["要点1", "要点2"], "evidence": ["简历里的依据，如 Python / RAG 项目"]}}
  ]
}}
"""

PREP_CATEGORY_ORDER = ["技术", "项目", "行为"]


def _safe_filename(text, limit: int = 40) -> str:
    """把公司/岗位名清洗成能当文件名用的字符串（去掉 Windows 非法字符）"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", str(text or "").strip())
    return cleaned.strip("_")[:limit] or "interview"


def _resume_for_prep() -> dict:
    """取当前默认简历，返回 {"name", "text", "note"}；没有简历时给出明确提示"""
    resume = storage.get_default_resume()
    if not resume:
        return {
            "name": "",
            "text": "（用户还没有保存任何简历）",
            "note": "当前没有简历：参考答案要点里不会出现任何经历/技能，只给「需要准备的素材」",
        }
    content = resume.get("content")
    if isinstance(content, (dict, list)):
        text = json.dumps(content, ensure_ascii=False, default=str)
    else:
        text = str(content or "")
    return {
        "name": resume.get("name", ""),
        "text": text[:2500] or "（简历内容为空）",
        "note": "",
    }


def _parse_prep_json(raw: str) -> dict:
    """解析 LLM 返回的面试准备 JSON：标准 json → 去 ``` 围栏 → json5 → 抓 {...} 片段"""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("模型返回空内容")

    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    braced = re.search(r"\{.*\}", text, re.DOTALL)
    if braced:
        candidates.append(braced.group(0))

    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError) as e:
            last_error = e
            try:
                data = json5.loads(candidate)      # 允许尾逗号、单引号、真换行
            except Exception as e2:                # noqa: BLE001 - json5 失败类型很多
                last_error = e2
                continue
        if isinstance(data, dict):
            return data
        last_error = ValueError("面试准备结果不是 JSON 对象")
    raise ValueError(f"无法解析面试准备 JSON：{last_error}")


def _normalize_prep(data: dict) -> list:
    """把模型返回的问题规整成 [{category, question, answer_points, evidence}]（最多 8 个）"""
    questions = (data or {}).get("questions")
    if not isinstance(questions, list):
        raise ValueError("模型没有返回 questions 列表")

    cleaned = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        question = " ".join(str(item.get("question") or "").split())
        if not question:
            continue
        cleaned.append({
            "category": str(item.get("category") or "其他").strip() or "其他",
            "question": question,
            "answer_points": [
                " ".join(str(p).split()) for p in (item.get("answer_points") or [])
                if str(p).strip()
            ],
            "evidence": [
                " ".join(str(e).split()) for e in (item.get("evidence") or [])
                if str(e).strip()
            ],
        })

    if not cleaned:
        raise ValueError("模型返回的问题列表是空的")
    return cleaned[:8]


def _ordered_categories(questions: list) -> list:
    """问题里出现过的类别，按 技术 → 项目 → 行为 排序，其他类别排后面"""
    seen = []
    for q in questions:
        if q["category"] not in seen:
            seen.append(q["category"])
    ordered = [c for c in PREP_CATEGORY_ORDER if c in seen]
    ordered += [c for c in seen if c not in ordered]
    return ordered


def _prep_markdown(record: dict, detail, questions: list, resume_name: str, warnings: list) -> str:
    """拼可下载的 markdown（问题 + 参考答案要点 + 依据）"""
    lines = [
        f"# 面试准备 · {record.get('company', '')} {record.get('title', '')}",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 求职者简历：{resume_name or '（没有简历）'}",
    ]
    if detail is not None:
        lines.append(
            f"- 岗位：{detail.company} | {detail.title} | {detail.city} | {detail.salary}"
        )
        lines.append(f"- job_id：{detail.job_id}")
    for warning in warnings:
        lines.append(f"- 注意：{warning}")
    lines.append("")

    for category in _ordered_categories(questions):
        lines += [f"## {category}", ""]
        same_category = [q for q in questions if q["category"] == category]
        for index, question in enumerate(same_category, 1):
            lines += [f"### Q{index}. {question['question']}", ""]
            if question["answer_points"]:
                lines.append("**参考答案要点**")
                lines += [f"- {point}" for point in question["answer_points"]]
                lines.append("")
            if question["evidence"]:
                lines += [f"依据：{'、'.join(question['evidence'])}", ""]

    lines += ["---", "由 Dashboard「投递追踪 → 面试准备」生成（LLM 预测，仅供参考）。"]
    return "\n".join(lines)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def _interview_prep(company: str, title: str, jd_text: str, resume_text: str) -> dict:
    """调 LLM 生成面试预测问题（按公司/岗位/JD/简历缓存，同一组合不重复烧 token）"""
    prompt = INTERVIEW_PREP_PROMPT.format(
        company=company or "（未知）",
        title=title or "（未知）",
        jd=jd_text or "（没有取到岗位 JD）",
        resume=resume_text or "（没有简历）",
    )
    raw = chat([{"role": "user", "content": prompt}], source="interview_prep")
    return {"questions": _normalize_prep(_parse_prep_json(raw))}


def generate_interview_prep(record: dict) -> dict:
    """为一条投递记录生成面试准备：岗位 JD + 当前简历 → 问题 + 参考答案要点 → markdown"""
    warnings = []
    detail, job_warnings = resolve_job(record)
    warnings += job_warnings

    if detail is not None:
        jd_text = "【岗位职责】\n{}\n【任职要求】\n{}".format(
            detail.description or "（无）", detail.requirements or "（无）"
        )
    else:
        jd_text = (
            "（没能取到岗位 JD：请按公司名和岗位名给出通用但具体的问题，"
            "并在问题里标注「需要按 JD 补充」）"
        )

    resume = _resume_for_prep()
    if resume["note"]:
        warnings.append(resume["note"])

    data = _interview_prep(
        str(record.get("company", "")),
        str(record.get("title", "")),
        jd_text,
        resume["text"],
    )
    questions = data.get("questions") or []

    return {
        "company": record.get("company", ""),
        "title": record.get("title", ""),
        "job_id": detail.job_id if detail is not None else "",
        "has_job_detail": detail is not None,
        "resume_name": resume["name"],
        "questions": questions,
        "markdown": _prep_markdown(record, detail, questions, resume["name"], warnings),
        "filename": "面试准备_{}_{}.md".format(
            _safe_filename(record.get("company")), _safe_filename(record.get("title"))
        ),
        "warnings": warnings,
    }


def render_interview_prep(payload: dict, app_id: str):
    """在 expander 里展示面试准备，并给一个 markdown 下载按钮"""
    questions = payload.get("questions") or []
    with st.expander(
        "面试准备 · {} | {}（{} 个问题）".format(
            payload.get("company", ""), payload.get("title", ""), len(questions)
        ),
        expanded=True,
    ):
        resume_name = payload.get("resume_name")
        st.caption(
            "问题按岗位 JD 预测，参考答案要点基于" + (
                f"简历「{resume_name}」" if resume_name else "（当前没有简历）"
            )
        )
        if not payload.get("has_job_detail"):
            st.info("没有取到岗位 JD，问题按公司/岗位名生成，仅供参考。")

        for category in _ordered_categories(questions):
            st.markdown(f"**{category}**")
            same_category = [q for q in questions if q["category"] == category]
            for index, question in enumerate(same_category, 1):
                st.markdown(f"**Q{index}. {question['question']}**")
                for point in question["answer_points"]:
                    st.markdown(f"- {point}")
                if question["evidence"]:
                    st.caption("依据：" + "、".join(question["evidence"]))
            st.write("")

        st.download_button(
            "下载 Markdown",
            data=payload["markdown"],
            file_name=payload["filename"],
            mime="text/markdown",
            key=f"prep_dl_{app_id}",
        )


# 五个 Tab
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "岗位列表", "投递追踪", "匹配打分", "Token 成本", "简历管理",
])


# ============================================================
# Tab 1：岗位列表
# ============================================================
with tab1:
    st.header("岗位列表")
    st.caption("搜索岗位，标记「想投 / 不合适」。")

    # 搜索栏
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        keyword = st.text_input("关键词", value="Agent 开发", key="search_kw")
    with col2:
        city = st.text_input("城市（留空不限）", value="广州", key="search_city")
    with col3:
        st.write("")
        st.write("")
        if st.button("搜索", key="search_btn"):
            with st.spinner("搜索中..."):
                jobs = search_jobs(
                    keyword, city or None, limit=20, platform="mock"
                )
                st.session_state["jobs"] = [
                    {
                        "job_id": j.job_id,
                        "title": j.title,
                        "company": j.company,
                        "city": j.city,
                        "salary": j.salary,
                        "url": j.url,
                        "tags": ", ".join(j.tags or []),
                    }
                    for j in jobs
                ]

    # 展示结果
    jobs = st.session_state.get("jobs", [])
    if not jobs:
        st.info("输入关键词后点击搜索。")
    else:
        st.write(f"共找到 **{len(jobs)}** 个岗位")

        # 构建表格数据，带上当前标记
        rows = []
        for j in jobs:
            mark = storage.get_mark(j["job_id"])
            rows.append({
                "标记": {"want": "想投", "skip": "不合适", "untagged": "—"}.get(mark, "—"),
                "岗位": j["title"],
                "公司": j["company"],
                "城市": j["city"],
                "薪资": j["salary"],
                "标签": j["tags"],
                "链接": j["url"],
            })

        df = pd.DataFrame(rows)
        _render_dataframe(df)

        # 标记区域
        st.subheader("标记操作")
        for j in jobs:
            c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 1, 1])
            with c1:
                st.write(f"**{j['company']}** | {j['title']}")
            with c2:
                st.write(f"{j['city']} | {j['salary']}")
            with c3:
                if st.button("标记想投", key=f"want_{j['job_id']}"):
                    storage.mark_job(j, "want")
                    st.rerun()
            with c4:
                if st.button("标记不合适", key=f"skip_{j['job_id']}"):
                    storage.mark_job(j, "skip")
                    st.rerun()
            with c5:
                if st.button("取消", key=f"untag_{j['job_id']}"):
                    storage.mark_job(j, "untagged")
                    st.rerun()

    # 已标记的岗位
    st.divider()
    st.subheader("我的「想投」列表")
    want_jobs = storage.get_marked_jobs("want")
    if not want_jobs:
        st.caption("暂无标记。")
    else:
        for j in want_jobs:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    job_row(
                        j["title"],
                        " | ".join(str(x) for x in [j["city"], j["salary"]] if x),
                        company=j["company"],
                        badge_html=status_badge("applied", "想投"),
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("加入投递", key=f"track_{j['job_id']}"):
                    storage.create_application(
                        j["company"], j["title"], "dashboard", j.get("url", "")
                    )
                    st.success(f"已加入投递追踪：{j['company']}")


# ============================================================
# Tab 2：投递追踪
# ============================================================
with tab2:
    st.header("投递追踪")

    # ============================================================
    #  跟进提醒区（投递超过 N 天、状态仍停在 applied）
    # ============================================================
    # 放在最顶部：用户打开这个 Tab 最该先看到的就是「哪几条该去催了」。
    # 判定逻辑全部走 agent.reminder，和 Agent 的 check_reminders 工具同一份口径。
    follow_up_days = reminder.DEFAULT_FOLLOW_UP_DAYS
    overdue = reminder.check_follow_ups(follow_up_days)
    if overdue:
        lines = [reminder.format_reminder(overdue, follow_up_days), "", "可直接跟进的记录："]
        for item in overdue:
            lines.append(
                "- `{company}` | {title}｜投递 {applied_at}（已 {days_elapsed} 天）".format(
                    company=item["company"] or "（未知公司）",
                    title=item["title"] or "（未知岗位）",
                    applied_at=item["applied_at"] or "（无投递时间）",
                    days_elapsed=item["days_elapsed"],
                )
            )
        st.warning("\n\n".join(lines))
        st.caption(
            f"提醒口径：状态仍是 `applied` 且投递超过 {follow_up_days} 天"
            "（在「查看时间线」里把状态改成 viewed / interview 后，这条提醒会自动消失）。"
        )
    else:
        st.success(f"没有超过 {follow_up_days} 天仍未跟进的投递。")

    apps = storage.list_applications()
    if not apps:
        st.info("暂无投递记录。去「岗位列表」标记想投的岗位，加入追踪。")
    else:
        # 投递漏斗：总投递 → 简历被看 → 进面 → Offer
        total = len(apps)
        viewed = sum(
            1 for a in apps
            if a["status"] in ["viewed", "interview", "interviewing", "offer", "accepted"]
        )
        interview = sum(
            1 for a in apps
            if a["status"] in ["interview", "interviewing", "offer", "accepted"]
        )
        offer = sum(1 for a in apps if a["status"] in ["offer", "accepted"])

        # 转化率：比率的分母都是总投递数
        def _pct(n: int) -> str:
            return f"{n / total:.1%}" if total else "0.0%"

        # 设计稿首页的 4 张统计卡片（.stat-card）
        stat_row([
            {
                "label": "总投递",
                "value": total,
                "unit": "份",
                "icon_name": "send",
                "variant": "accent",
                "spark": _pipeline_spark(apps),
                "hint": "全部投递记录",
            },
            {
                "label": "简历被看",
                "value": viewed,
                "unit": f"转化 {_pct(viewed)}",
                "icon_name": "search",
                "delta": _pct(viewed),
                "delta_dir": _delta_dir(viewed),
                "hint": f"{viewed} / {total}",
            },
            {
                "label": "进面",
                "value": interview,
                "unit": f"约面率 {_pct(interview)}",
                "icon_name": "calendar",
                "variant": "accent",
                "delta": _pct(interview),
                "delta_dir": _delta_dir(interview),
                "hint": f"{interview} / {total}",
            },
            {
                "label": "Offer",
                "value": offer,
                "unit": f"Offer 率 {_pct(offer)}",
                "icon_name": "check",
                "variant": "ok" if offer else "default",
                "delta": _pct(offer),
                "delta_dir": _delta_dir(offer),
                "hint": f"{offer} / {total}",
            },
        ])

        st.divider()

        # 表格
        df = pd.DataFrame([
            {
                "ID": a["id"],
                "公司": a["company"],
                "岗位": a["title"],
                "状态": STATUS_LABELS.get(a["status"], a["status"]),
                "投递时间": a["applied_at"],
            }
            for a in apps
        ])
        _render_dataframe(df)

        # 查看时间线
        st.subheader("查看时间线")
        app_ids = [a["id"] for a in apps]
        selected = st.selectbox(
            "选择一条投递记录",
            options=app_ids,
            format_func=lambda x: next(
                f"{a['company']} | {a['title']}" for a in apps if a["id"] == x
            ),
        )
        if selected:
            app = storage.get_application(selected)
            from agent.state_machine import get_status_label

            st.markdown(
                '<div class="job-panel"><header><h2>{company} | {title}</h2>'
                '<span class="sub">{status}</span>'
                '<div class="r">{badge}</div></header></div>'.format(
                    company=e(app["company"]),
                    title=e(app["title"]),
                    status=e(STATUS_LABELS.get(app["status"], app["status"])),
                    badge=status_badge(app["status"]),
                ),
                unsafe_allow_html=True,
            )

            events = storage.get_events(selected)
            timeline_events = []
            for event in reversed(events):          # 最新的排最上面
                from_label = (
                    get_status_label(event["from_status"]) if event["from_status"] else "创建"
                )
                to_label = get_status_label(event["to_status"])
                state = {
                    "offer": "done", "accepted": "done", "interview": "done",
                    "interviewing": "done", "rejected": "error", "declined": "error",
                }.get(event["to_status"], "done")
                timeline_events.append({
                    "title": f"{from_label} → {to_label}",
                    "time": event["created_at"],
                    "desc": event["note"] or "",
                    "badge_html": status_badge(event["to_status"]),
                    "state": state,
                })
            st.markdown(timeline(timeline_events), unsafe_allow_html=True)

        # 每条投递记录两个动作：一键投递包 + 面试准备（都要调 LLM，点按钮才跑）
        st.divider()
        st.subheader("投递包 / 面试准备")
        st.caption(
            "投递包 = 按岗位定制的简历 PDF + 自荐信 + 岗位信息；"
            "面试准备 = 按岗位 JD 预测 5-8 个面试问题（技术/项目/行为）+ 结合简历的参考答案要点。"
            "两个动作都会调 LLM，大概十几秒到半分钟。"
        )

        for a in apps:
            act_info, act_pkg, act_prep = st.columns([4, 1, 1])
            with act_info:
                st.write(f"**{a['company']}** | {a['title']} | 状态：{a['status']}")
            with act_pkg:
                if st.button("生成投递包", key=f"pkg_{a['id']}"):
                    with st.spinner(f"正在为「{a['company']}」生成投递包..."):
                        try:
                            result = generate_application_package(a["company"])
                        except Exception as e:
                            result = {"error": f"{type(e).__name__}: {e}"}
                    st.session_state["last_package"] = {"app_id": a["id"], "result": result}
            with act_prep:
                if st.button("面试准备", key=f"prep_{a['id']}"):
                    with st.spinner(f"正在为「{a['company']}」预测面试问题..."):
                        try:
                            result = generate_interview_prep(a)
                        except Exception as e:
                            result = {"error": f"{type(e).__name__}: {e}"}
                    st.session_state["interview_prep"] = {"app_id": a["id"], "result": result}

        # 结果区：放在按钮循环之后，给足宽度放下载按钮
        last_package = st.session_state.get("last_package")
        if last_package:
            payload = last_package["result"]
            if payload.get("error"):
                st.error(f"生成投递包失败：{payload['error']}")
            else:
                st.success(f"投递包已生成：`{payload['package_dir']}`")
                for warning in payload.get("warnings", []):
                    st.warning(warning)
                dl_cols = st.columns(len(payload["files"]))
                for col, (name, path) in zip(dl_cols, payload["files"].items()):
                    with col:
                        file_path = Path(path)
                        st.download_button(
                            name,
                            data=file_path.read_bytes() if file_path.is_file() else b"",
                            file_name=name,
                            mime=_MIME_TYPES.get(file_path.suffix, "application/octet-stream"),
                            key=f"dl_{name}_{last_package['app_id']}",
                        )

        #  面试准备结果
        prep_state = st.session_state.get("interview_prep")
        if prep_state:
            prep_payload = prep_state["result"]
            if prep_payload.get("error"):
                st.error(f"面试准备生成失败：{prep_payload['error']}")
            else:
                st.success(
                    "面试准备已生成：**{}** | {}".format(
                        prep_payload.get("company", ""), prep_payload.get("title", "")
                    )
                )
                for warning in prep_payload.get("warnings", []):
                    st.warning(warning)
                render_interview_prep(prep_payload, prep_state["app_id"])


# ============================================================
# Tab 3：匹配打分
# ============================================================
with tab3:
    st.header("简历-JD 匹配打分")

    st.caption("输入简历信息和岗位 ID，计算匹配分数和缺口分析。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("简历")
        name = st.text_input("姓名", value="张三")
        skills = st.text_area("技能（逗号分隔）", value="Python, RAG, LangChain, Chroma, Git")
        education = st.selectbox("学历", ["本科", "硕士", "博士"], index=0)
        city = st.text_input("城市", value="广州")
        exp_text = st.text_area(
            "实习经历（每行一条：公司|岗位|月数）",
            value="某创业公司|后端实习生|3",
        )
        proj_text = st.text_area(
            "项目经历（每行一条：项目名|技术|描述）",
            value="JD 知识库问答系统|RAG,Chroma,Chainlit|基于RAG的岗位问答",
        )

    with col2:
        st.subheader("目标岗位")
        job_id = st.text_input("岗位 ID", value="inn_pztab1vmuoqq")
        st.caption(
            "可用 ID 示例：inn_pztab1vmuoqq（信投智联科技 · 大模型算法）、"
            "inn_bvmuxglatdbv（科大讯飞 · 产品运营）、inn_78xqcaa6aktp（妙客莱音 · AI Agent 开发）"
        )

        if st.button("计算匹配度"):
            try:
                detail = get_job_detail("mock", job_id)

                experience = []
                for line in exp_text.strip().split("\n"):
                    parts = line.split("|")
                    if len(parts) >= 3:
                        experience.append({
                            "company": parts[0],
                            "role": parts[1],
                            "months": int(parts[2]) if parts[2].isdigit() else 0,
                        })

                projects = []
                for line in proj_text.strip().split("\n"):
                    parts = line.split("|")
                    if len(parts) >= 3:
                        projects.append({
                            "name": parts[0],
                            "tech": [t.strip() for t in parts[1].split(",")],
                            "desc": parts[2],
                        })

                resume = Resume(
                    name=name,
                    skills=[s.strip() for s in skills.split(",")],
                    experience=experience,
                    projects=projects,
                    education=education,
                    city=city,
                )

                with st.spinner("匹配中..."):
                    result = match_resume_to_jd(resume, detail)

                dims = result.dimensions
                st.session_state["last_match"] = {
                    "score": result.score,
                    "company": detail.company,
                    "title": detail.title,
                    "gaps": len(result.gaps or []),
                    "highlights": len(result.highlights or []),
                }

                # 匹配结果：总分 + 缺口/亮点数量（设计稿统计卡片）
                stat_row([
                    {
                        "label": "匹配度",
                        "value": result.score,
                        "unit": "/ 100 分",
                        "icon_name": "target",
                        "variant": "accent" if result.score >= 80 else "warn",
                        "hint": f"{detail.company} | {detail.title}",
                    },
                    {
                        "label": "补齐后可达",
                        "value": min(100, result.score + 6 * len(result.gaps or [])),
                        "unit": "/ 100 分",
                        "icon_name": "chart",
                        "delta": f"{len(result.gaps or [])} 项缺口",
                        "delta_dir": "dn" if result.gaps else "flat",
                        "hint": f"共 {len(result.gaps or [])} 项待补",
                    },
                    {
                        "label": "已命中亮点",
                        "value": len(result.highlights or []),
                        "unit": "项",
                        "icon_name": "check",
                        "variant": "ok" if result.highlights else "default",
                        "hint": "简历里可直接讲的加分项",
                    },
                ])

                # 维度得分
                st.subheader("维度得分")
                for k, v in dims.items():
                    st.write(f"- {k}: **{v}**")

                # 差距
                if result.gaps:
                    st.subheader("差距")
                    for g in result.gaps:
                        st.write(f"- {g}")

                # 亮点
                if result.highlights:
                    st.subheader("亮点")
                    for h in result.highlights:
                        st.write(f"- {h}")

                # 岗位信息
                with st.expander("查看岗位 JD"):
                    st.write(f"**{detail.company} | {detail.title}**")
                    st.write(f"城市：{detail.city}  |  薪资：{detail.salary}")
                    st.write("**任职要求：**")
                    st.write(detail.requirements)

            except Exception as e:
                st.error(f"匹配失败：{e}")


# ============================================================
# Tab 4：Token 成本
# ============================================================
with tab4:
    st.header("Token 成本")
    st.caption(
        "所有 LLM 调用的用量都会记进 `logs/token_usage.db`（按调用来源区分），"
        "这里直接看总量和分布。"
    )

    range_days = st.segmented_control(
        "统计区间",
        options=[7, 14, 30],
        default=7,
        format_func=lambda d: f"近 {d} 天",
        key="token_range",
    ) or 7

    usage = token_tracker.query_usage(days=range_days)

    # 区间环比：把窗口对半切成「后半段 vs 前半段」，算真实变化率
    daily = usage.get("daily") or []
    half = max(1, len(daily) // 2)
    recent_tokens = sum(d.get("tokens", 0) for d in daily[half:])
    earlier_tokens = sum(d.get("tokens", 0) for d in daily[:half])
    if earlier_tokens:
        change_pct = (recent_tokens - earlier_tokens) / earlier_tokens * 100
        delta_dir = "up" if change_pct > 0 else ("dn" if change_pct < 0 else "flat")
        delta_text = "{:+.1f}%".format(change_pct)
    else:
        delta_dir, delta_text = "flat", "无对比数据"

    stat_row([
        {
            "label": "总 token",
            "value": f"{usage['total_tokens']:,}",
            "unit": f"近 {range_days} 天",
            "icon_name": "coins",
            "variant": "accent",
            "delta": delta_text,
            "delta_dir": delta_dir,
            "hint": "后半段 vs 前半段",
            "spark": [d.get("tokens", 0) for d in daily],
        },
        {
            "label": "调用次数",
            "value": f"{usage['total_calls']:,}",
            "unit": "次",
            "icon_name": "send",
            "spark": [d.get("calls", 0) for d in daily],
            "hint": "区间内的 LLM 调用",
        },
        {
            "label": "平均每次",
            "value": f"{usage['total_tokens'] / usage['total_calls']:,.0f}"
                     if usage["total_calls"] else "—",
            "unit": "token / 次",
            "icon_name": "chart",
            "hint": "单次调用平均消耗",
        },
        {
            "label": "用量来源",
            "value": len(usage["by_source"]),
            "unit": "个",
            "icon_name": "package",
            "hint": "按调用来源区分",
        },
    ])

    if not usage["total_calls"]:
        st.info("这个区间还没有用量记录。跑一次对话或匹配后再来看。")
    else:
        st.subheader("按天")
        st.bar_chart(
            pd.DataFrame(usage["daily"]), x="date", y="tokens", height=260
        )
        st.caption("流式调用拿不到 usage 时会记 0，并把来源标成 `xxx:stream_no_usage`。")

        left, right = st.columns(2)
        with left:
            st.subheader("按来源")
            st.dataframe(
                pd.DataFrame([
                    {"来源": src, "token": v["tokens"], "调用次数": v["calls"]}
                    for src, v in usage["by_source"].items()
                ]),
                width="stretch",
                hide_index=True,
            )
        with right:
            st.subheader("按模型")
            st.dataframe(
                pd.DataFrame([
                    {"模型": model, "token": v["tokens"], "调用次数": v["calls"]}
                    for model, v in usage["by_model"].items()
                ]),
                width="stretch",
                hide_index=True,
            )


# ============================================================
# Tab 5：简历管理（多版本）
# ============================================================
with tab5:
    st.header("简历管理")
    st.caption(
        "同一个岗位方向存一份简历（技术岗版 / 产品岗版…）。"
        "Agent 需要简历时会用「默认」那一份，切换方向只要改默认即可。"
    )

    notice = st.session_state.pop("resume_notice", None)
    if notice:
        st.success(notice)

    resumes = storage.list_resumes()
    default_resume = storage.get_default_resume()
    default_id = default_resume["id"] if default_resume else None

    if resumes:
        newest = resumes[0]
        stat_row([
            {
                "label": "简历版本",
                "value": len(resumes),
                "unit": "份",
                "icon_name": "file",
                "variant": "accent",
                "hint": "按岗位方向各存一份",
            },
            {
                "label": "当前默认",
                "value": (default_resume or {}).get("name") or "未设置",
                "unit": "",
                "icon_name": "check",
                "variant": "ok" if default_resume else "warn",
                "hint": "Agent 取简历时用这一份",
            },
            {
                "label": "最近更新",
                "value": (newest.get("created_at") or "")[:10] or "—",
                "unit": "",
                "icon_name": "clock",
                "hint": str(newest.get("name") or ""),
            },
        ])
        st.dataframe(
            pd.DataFrame([
                {
                    "默认": "是" if r["id"] == default_id else "",
                    "名称": r["name"],
                    "ID": r["id"],
                    "创建时间": r["created_at"],
                }
                for r in resumes
            ]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("还没有简历，用下面的表单新建一份。")

    if resumes:
        st.subheader("下载 PDF")
        st.caption(
            "每份简历都能导出成 PDF（下载后可直接作为投递附件）。"
            "PDF 在点击下载时才生成，不会拖慢页面。"
        )
        for r in resumes:
            pdf_col1, pdf_col2 = st.columns([4, 1])
            with pdf_col1:
                st.markdown(
                    job_row(
                        r["name"],
                        f"{r['id']} | {r['created_at']}",
                        initials="简",
                        badge_html=(
                            status_badge("offer", "默认")
                            if r["id"] == default_id else ""
                        ),
                        accent=r["id"] == default_id,
                    ),
                    unsafe_allow_html=True,
                )
            with pdf_col2:
                st.download_button(
                    "下载 PDF",
                    data=lambda rid=r["id"]: _resume_pdf_bytes(rid),
                    file_name=f"{r['name']}_{r['id']}.pdf",
                    mime="application/pdf",
                    key=f"pdf_{r['id']}",
                )

    st.subheader("新建 / 上传简历")
    with st.form("resume_form", clear_on_submit=True):
        resume_name = st.text_input("简历名称", placeholder="技术岗版", key="resume_name")
        uploaded = st.file_uploader(
            "上传文件（.json / .txt / .md，可选；上传了就优先用文件内容）",
            type=["json", "txt", "md"],
            key="resume_upload",
        )
        resume_content = st.text_area(
            "简历内容",
            height=200,
            key="resume_content",
            placeholder=(
                '{"name": "张三", "skills": ["Python", "RAG"], '
                '"experience": [], "projects": [], "education": "本科", "city": "广州"}\n'
                "或直接粘贴纯文本简历（Agent 会在匹配时自行解析）"
            ),
        )
        submitted = st.form_submit_button("保存简历", key="resume_submit")

    if submitted:
        text = (
            uploaded.getvalue().decode("utf-8", errors="replace")
            if uploaded is not None else resume_content
        )
        if not str(text).strip():
            st.error("简历内容不能为空。")
        else:
            new_id = storage.save_resume(resume_name, text)
            saved = storage.get_resume(new_id) or {}
            st.session_state["resume_notice"] = (
                f"已保存「{saved.get('name', resume_name)}」，ID = {new_id}"
            )
            st.rerun()

    if resumes:
        st.divider()
        st.subheader("设为默认 / 删除")

        labels = {
            r["id"]: f"{r['name']}（{r['id']}）"
                     + ("　· 当前默认" if r["id"] == default_id else "")
            for r in resumes
        }
        selected = st.selectbox(
            "选择一份简历",
            options=[r["id"] for r in resumes],
            format_func=lambda rid: labels[rid],
            key="resume_pick",
        )

        col_set, col_confirm, col_del = st.columns([1, 1, 1])
        with col_set:
            if st.button("设为默认", key="resume_set_default"):
                if storage.set_default_resume(selected):
                    st.session_state["resume_notice"] = f"已把 {selected} 设为默认简历。"
                else:
                    st.session_state["resume_notice"] = f"设置失败：找不到简历 {selected}。"
                st.rerun()
        with col_confirm:
            confirm_delete = st.checkbox("确认删除", key="resume_confirm_delete")
        with col_del:
            if st.button("删除", key="resume_delete", disabled=not confirm_delete):
                ok = storage.delete_resume(selected)
                st.session_state["resume_notice"] = (
                    f"已删除简历 {selected}。" if ok else f"删除失败：找不到简历 {selected}。"
                )
                st.rerun()

        detail = storage.get_resume(selected)
        with st.expander("查看这份简历的内容"):
            content = (detail or {}).get("content")
            if isinstance(content, (dict, list)):
                st.json(content)
            else:
                st.code(str(content or "") or "（空）")
