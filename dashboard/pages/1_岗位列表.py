"""岗位列表页（原 ``app.py`` Tab 1）。

搜索岗位，标记「想投 / 不合适」，并把想投的岗位一键加入投递追踪。
"""

import json
import sys
from datetime import datetime
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

# ============================================================
# 数据新鲜度
# ============================================================
# 搜索结果走的是 mock 平台，看不到「本地到底抓了多少数据」。用户搜一个没抓过的
# 城市（例如上海）时会得到空列表却不知道为什么，所以在搜索框下面直接读
# rag/data/cleaned_jd.json，把「这个城市本地有没有数据、数据多旧」讲清楚。

#: 抓取产出的清洗后岗位数据
CLEANED_JD = ROOT_DIR / "rag" / "data" / "cleaned_jd.json"

#: 超过这么多天就提示数据可能过时
STALE_DAYS = 7


@st.cache_data(ttl=300, show_spinner=False)
def local_jd_stats() -> dict:
    """本地已抓取数据的城市覆盖情况。

    :return: ``{"counts": {城市: 岗位数}, "mtime": float | None, "total": int}``
             文件缺失或损坏时返回空统计——这只是辅助提示，不该拦住页面。
    """
    try:
        mtime = CLEANED_JD.stat().st_mtime
        raw = json.loads(CLEANED_JD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"counts": {}, "mtime": None, "total": 0}

    records = raw if isinstance(raw, list) else (raw.get("jobs") or [])
    counts: dict[str, int] = {}
    for item in records:
        key = str((item or {}).get("city") or "").strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return {"counts": counts, "mtime": mtime, "total": len(records)}


def render_freshness(city: str) -> None:
    """提示输入城市在本地的数据覆盖与新鲜度。城市为空则什么都不显示。"""
    city = (city or "").strip()
    if not city:
        return

    stats = local_jd_stats()
    count = stats["counts"].get(city, 0)

    if count == 0:
        st.warning(
            f"该城市暂无数据（{city}）。"
            "运行 `python -m agent.scrapers.scheduler --once` 可抓取"
        )
        return

    updated = datetime.fromtimestamp(stats["mtime"]) if stats["mtime"] else None
    stamp = updated.strftime("%Y-%m-%d %H:%M") if updated else "未知"
    st.caption(f"本地数据：共 {count} 个岗位（数据更新于 {stamp}）")

    if updated and (datetime.now() - updated).days >= STALE_DAYS:
        st.warning(
            f"本地数据已 {(datetime.now() - updated).days} 天未更新"
            f"（超过 {STALE_DAYS} 天），结果可能过时。"
            "运行 `python -m agent.scrapers.scheduler --once` 可重新抓取"
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

# 数据新鲜度：搜索前先说明该城市本地有没有数据、数据多旧
render_freshness(city)

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
