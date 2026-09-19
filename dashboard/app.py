"""
求职 Dashboard。
五个 Tab：
1. 岗位列表：搜索、标记想投/不合适
2. 投递追踪：查看投递记录和状态
3. 匹配结果：简历-JD 匹配打分
4. Token 成本：LLM 用量按天/模型/来源统计
5. 简历管理：多版本简历的新建/上传/设默认/删除
"""
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
from agent import storage
from agent.tools.job_search import search_jobs
from agent.tools.job_detail import get_job_detail
from agent.tools.resume_match import match_resume_to_jd, Resume
from shared import token_tracker


# 初始化数据库
storage.init_db()

st.set_page_config(
    page_title="求职助手 Dashboard",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 求职助手 Dashboard")

# 五个 Tab
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 岗位列表", "📊 投递追踪", "🎯 匹配打分", "💰 Token 成本", "📄 简历管理",
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
        if st.button("🔍 搜索", key="search_btn"):
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
                "标记": {"want": "⭐ 想投", "skip": "❌ 不合适", "untagged": "—"}.get(mark, "—"),
                "岗位": j["title"],
                "公司": j["company"],
                "城市": j["city"],
                "薪资": j["salary"],
                "标签": j["tags"],
                "链接": j["url"],
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)

        # 标记区域
        st.subheader("标记操作")
        for j in jobs:
            c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 1, 1])
            with c1:
                st.write(f"**{j['company']}** | {j['title']}")
            with c2:
                st.write(f"{j['city']} | {j['salary']}")
            with c3:
                if st.button("⭐ 想投", key=f"want_{j['job_id']}"):
                    storage.mark_job(j, "want")
                    st.rerun()
            with c4:
                if st.button("❌ 不合适", key=f"skip_{j['job_id']}"):
                    storage.mark_job(j, "skip")
                    st.rerun()
            with c5:
                if st.button("取消", key=f"untag_{j['job_id']}"):
                    storage.mark_job(j, "untagged")
                    st.rerun()

    # 已标记的岗位
    st.divider()
    st.subheader("⭐ 我的「想投」列表")
    want_jobs = storage.get_marked_jobs("want")
    if not want_jobs:
        st.caption("暂无标记。")
    else:
        for j in want_jobs:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.write(f"**{j['company']}** | {j['title']} | {j['city']} | {j['salary']}")
            with c2:
                if st.button("📥 加入投递", key=f"track_{j['job_id']}"):
                    storage.create_application(
                        j["company"], j["title"], "dashboard", j.get("url", "")
                    )
                    st.success(f"已加入投递追踪：{j['company']}")


# ============================================================
# Tab 2：投递追踪
# ============================================================
with tab2:
    st.header("投递追踪")

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

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总投递", total)
        col2.metric("简历被看", viewed)
        col3.metric("进面", interview)
        col4.metric("Offer", offer)

        # 转化率：三个比率的分母都是总投递数
        def _pct(n: int) -> str:
            return f"{n / total:.1%}" if total else "0.0%"

        st.caption("转化率（分母均为「总投递」）")
        rate1, rate2, rate3, _ = st.columns(4)
        rate1.metric("简历被看率", _pct(viewed), help=f"{viewed} / {total}")
        rate2.metric("约面率", _pct(interview), help=f"{interview} / {total}")
        rate3.metric("Offer 率", _pct(offer), help=f"{offer} / {total}")

        st.divider()

        # 表格
        df = pd.DataFrame([
            {
                "ID": a["id"],
                "公司": a["company"],
                "岗位": a["title"],
                "状态": a["status"],
                "投递时间": a["applied_at"],
            }
            for a in apps
        ])
        st.dataframe(df, width="stretch", hide_index=True)

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
            st.write(f"**{app['company']} | {app['title']}**")
            events = storage.get_events(selected)
            for e in events:
                from agent.state_machine import get_status_label
                from_label = get_status_label(e["from_status"]) if e["from_status"] else "创建"
                to_label = get_status_label(e["to_status"])
                note = f" — {e['note']}" if e["note"] else ""
                st.write(f"- `{e['created_at']}`  {from_label} → **{to_label}**{note}")


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

        if st.button("🎯 计算匹配度"):
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

                st.success(f"匹配度：**{result.score}/100**")

                # 维度得分
                st.subheader("维度得分")
                dims = result.dimensions
                for k, v in dims.items():
                    st.write(f"- {k}: **{v}**")

                # 差距
                if result.gaps:
                    st.subheader("⚠️ 差距")
                    for g in result.gaps:
                        st.write(f"- {g}")

                # 亮点
                if result.highlights:
                    st.subheader("✅ 亮点")
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

    col1, col2, col3 = st.columns(3)
    col1.metric("总 token", f"{usage['total_tokens']:,}")
    col2.metric("调用次数", f"{usage['total_calls']:,}")
    col3.metric(
        "平均每次",
        f"{usage['total_tokens'] / usage['total_calls']:,.0f}"
        if usage["total_calls"] else "—",
    )

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
        st.dataframe(
            pd.DataFrame([
                {
                    "默认": "✅" if r["id"] == default_id else "",
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
            if st.button("✅ 设为默认", key="resume_set_default"):
                if storage.set_default_resume(selected):
                    st.session_state["resume_notice"] = f"已把 {selected} 设为默认简历。"
                else:
                    st.session_state["resume_notice"] = f"设置失败：找不到简历 {selected}。"
                st.rerun()
        with col_confirm:
            confirm_delete = st.checkbox("确认删除", key="resume_confirm_delete")
        with col_del:
            if st.button("🗑️ 删除", key="resume_delete", disabled=not confirm_delete):
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
