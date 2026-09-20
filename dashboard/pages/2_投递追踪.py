"""投递追踪页（原 ``app.py`` Tab 2）。

跟进提醒、投递漏斗统计、时间线、投递包生成、面试准备（预测问题 + 参考答案要点）。

面试准备相关的模块级函数（prompt / JSON 解析 / 归一化 / markdown 拼装）原本就在
``app.py`` 顶层，这里原样迁过来；只有跨页面共用的 :func:`resume_for_prep` 移到了
``dashboard.shared``。
"""

import sys
from pathlib import Path

# 项目根目录引导（多页面模式下每个页面都是独立脚本，必须自己补 sys.path）
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import re
from datetime import datetime

import json5
import pandas as pd
import streamlit as st

from agent import reminder, storage
from agent.tools_registry import generate_application_package, resolve_job
from dashboard.shared import (
    MIME_TYPES,
    STATUS_LABELS,
    delta_dir,
    e,
    inject_styles,
    pipeline_spark,
    render_dataframe,
    resume_for_prep,
    stat_row,
    status_badge,
    timeline,
)
from shared.llm_client import chat

st.set_page_config(
    page_title="投递追踪 · 求职助手 Dashboard",
    page_icon="◎",
    layout="wide",
)

# 必须先 set_page_config、再注入样式
inject_styles()


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

    resume = resume_for_prep()
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


# ============================================================
#  页面主体
# ============================================================
st.header("投递追踪")

# ============================================================
#  跟进提醒区（投递超过 N 天、状态仍停在 applied）
# ============================================================
# 放在最顶部：用户打开这个页面最该先看到的就是「哪几条该去催了」。
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
            "spark": pipeline_spark(apps),
            "hint": "全部投递记录",
        },
        {
            "label": "简历被看",
            "value": viewed,
            "unit": f"转化 {_pct(viewed)}",
            "icon_name": "search",
            "delta": _pct(viewed),
            "delta_dir": delta_dir(viewed),
            "hint": f"{viewed} / {total}",
        },
        {
            "label": "进面",
            "value": interview,
            "unit": f"约面率 {_pct(interview)}",
            "icon_name": "calendar",
            "variant": "accent",
            "delta": _pct(interview),
            "delta_dir": delta_dir(interview),
            "hint": f"{interview} / {total}",
        },
        {
            "label": "Offer",
            "value": offer,
            "unit": f"Offer 率 {_pct(offer)}",
            "icon_name": "check",
            "variant": "ok" if offer else "default",
            "delta": _pct(offer),
            "delta_dir": delta_dir(offer),
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
    render_dataframe(df)

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
                        mime=MIME_TYPES.get(file_path.suffix, "application/octet-stream"),
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
