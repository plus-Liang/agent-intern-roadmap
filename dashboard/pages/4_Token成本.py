"""Token 成本页（原 ``app.py`` Tab 4）。

按天 / 来源 / 模型统计 LLM 用量，并给出区间环比。
"""

import sys
from pathlib import Path

# 项目根目录引导（多页面模式下每个页面都是独立脚本，必须自己补 sys.path）
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from dashboard.shared import inject_styles, stat_row, token_tracker

st.set_page_config(
    page_title="Token 成本 · 求职助手 Dashboard",
    page_icon="◎",
    layout="wide",
)

# 必须先 set_page_config、再注入样式
inject_styles()

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
