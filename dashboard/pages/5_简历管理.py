"""简历管理页（原 ``app.py`` Tab 5）。

多版本简历的新建 / 上传 / 设默认 / 删除 / 导出 PDF。
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
from dashboard.shared import (
    inject_styles,
    job_row,
    resume_pdf_bytes,
    stat_row,
    status_badge,
)

st.set_page_config(
    page_title="简历管理 · 求职助手 Dashboard",
    page_icon="◎",
    layout="wide",
)

# 必须先 set_page_config、再注入样式
inject_styles()

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
                data=lambda rid=r["id"]: resume_pdf_bytes(rid),
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
