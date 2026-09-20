"""求职 Dashboard 首页。

多页面结构的入口脚本（``streamlit run dashboard/app.py``）。Streamlit 会
自动把 ``dashboard/pages/`` 下的脚本挂到左侧导航，本页面作为默认落地页，
只放三块内容：

1. 页面标题 + 一句话描述
2. 4 张统计卡片（岗位总数 / 投递总数 / 待跟进 / 累计 token）
3. 快捷入口（``st.page_link`` 链到 5 个子页面）
4. 最近动态（``applications.db`` 里最后 5 条投递）

各子页面的业务逻辑原样保留在 ``dashboard/pages/`` 里，这里不重复实现。
"""

import sys
from pathlib import Path

# 项目根目录引导：dashboard/ 的上一级（agent/、shared/ 都在它下面）
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from dashboard.shared import (
    STATUS_LABELS,
    count_applications,
    count_marked_jobs,
    follow_up_overdue,
    inject_styles,
    job_row,
    recent_applications,
    reminder,
    stat_row,
    status_badge,
    total_tokens,
)

st.set_page_config(
    page_title="首页 · 求职助手 Dashboard",
    page_icon="◎",
    layout="wide",
)

# 必须先 set_page_config、再注入样式
inject_styles()

# 左侧导航里的 5 个子页面（路径相对入口脚本所在目录）
_PAGES = [
    ("pages/1_岗位列表.py", "岗位列表"),
    ("pages/2_投递追踪.py", "投递追踪"),
    ("pages/3_匹配打分.py", "匹配打分"),
    ("pages/4_Token成本.py", "Token 成本"),
    ("pages/5_简历管理.py", "简历管理"),
]

st.title("求职助手 Dashboard")
st.markdown(
    '<p class="page-meta">岗位搜索 / 投递追踪 / 匹配打分 / Token 成本 / 简历管理'
    " —— 一个页面看完全流程。</p>",
    unsafe_allow_html=True,
)

# ============================================================
#  统计概览
# ============================================================
follow_up_days = reminder.DEFAULT_FOLLOW_UP_DAYS
job_total = count_marked_jobs()
app_total = count_applications()
overdue = follow_up_overdue(follow_up_days)
overdue_total = len(overdue)
token_total = total_tokens()

stat_row([
    {
        "label": "岗位总数",
        "value": job_total,
        "unit": "个",
        "icon_name": "search",
        "variant": "accent",
        "hint": "已标记进岗位池的岗位",
    },
    {
        "label": "投递总数",
        "value": app_total,
        "unit": "份",
        "icon_name": "send",
        "hint": "applications.db 里的全部投递记录",
    },
    {
        "label": "待跟进",
        "value": overdue_total,
        "unit": "条",
        "icon_name": "bell",
        "variant": "warn" if overdue_total else "ok",
        "hint": f"投递超过 {follow_up_days} 天、状态仍是 applied",
    },
    {
        "label": "累计 token",
        "value": f"{token_total:,}",
        "unit": "token",
        "icon_name": "coins",
        "hint": "logs/token_usage.db 的全部用量",
    },
])

if overdue_total:
    st.warning(reminder.format_reminder(overdue, follow_up_days))

# ============================================================
#  快捷入口
# ============================================================
st.subheader("快捷入口")
st.caption("每个页面都有自己的左侧导航条目，这里是最快的跳转方式。")
link_cols = st.columns(len(_PAGES))
for col, (page_path, label) in zip(link_cols, _PAGES):
    with col:
        st.page_link(page_path, label=label)

# ============================================================
#  最近动态
# ============================================================
st.divider()
st.subheader("最近动态")
st.caption("`applications.db` 里最近 5 条投递记录。")

recent = recent_applications(5)
if not recent:
    st.info("暂无投递记录。去「岗位列表」标记想投的岗位，加入投递追踪。")
else:
    for item in recent:
        company = item.get("company") or "（未知公司）"
        meta = " | ".join(
            str(value) for value in
            [company, item.get("applied_at"), STATUS_LABELS.get(item.get("status"), item.get("status"))]
            if value
        )
        st.markdown(
            job_row(
                item.get("title") or "（未知岗位）",
                meta,
                company=company,
                badge_html=status_badge(item.get("status")),
            ),
            unsafe_allow_html=True,
        )
