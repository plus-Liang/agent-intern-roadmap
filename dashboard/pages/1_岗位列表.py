"""岗位列表页（原 ``app.py`` Tab 1）。

搜索岗位，标记「想投 / 不合适」，并把想投的岗位一键加入投递追踪。
"""

import sys
from pathlib import Path

# 项目根目录引导（多页面模式下每个页面都是独立脚本，必须自己补 sys.path）
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from agent import storage
from agent.tools.job_search import search_jobs
from dashboard.shared import (
    inject_styles,
    job_row,
    render_dataframe,
    status_badge,
)

st.set_page_config(
    page_title="岗位列表 · 求职助手 Dashboard",
    page_icon="◎",
    layout="wide",
)

# 必须先 set_page_config、再注入样式
inject_styles()

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
    render_dataframe(df)

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
