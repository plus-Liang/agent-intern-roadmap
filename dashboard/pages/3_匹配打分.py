"""简历-JD 匹配打分页（原 ``app.py`` Tab 3）。

输入简历信息和岗位 ID，计算匹配分数、维度得分和缺口分析。
"""

import json
import sys
from pathlib import Path

# 项目根目录引导（多页面模式下每个页面都是独立脚本，必须自己补 sys.path）
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from agent.tools.job_detail import get_job_detail
from agent.tools.resume_match import Resume, match_resume_to_jd
from dashboard.shared import inject_styles, stat_row

st.set_page_config(
    page_title="匹配打分 · 求职助手 Dashboard",
    page_icon="◎",
    layout="wide",
)

# 必须先 set_page_config、再注入样式
inject_styles()

st.header("简历-JD 匹配打分")

st.caption("填写简历信息并选择目标岗位，计算匹配分数和缺口分析。")

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

    # 岗位下拉：读 rag/data/cleaned_jd.json，展示"公司 | 岗位 | 城市"，实际取值是 job_id
    _jd_path = ROOT_DIR / "rag" / "data" / "cleaned_jd.json"
    _job_options: list[tuple[str, str]] = []
    try:
        with _jd_path.open(encoding="utf-8") as f:
            _jd_list = json.load(f)
        if isinstance(_jd_list, list):
            for _item in _jd_list:
                if isinstance(_item, dict) and _item.get("job_id"):
                    _job_options.append((
                        f"{_item.get('company') or '未知公司'} | "
                        f"{_item.get('title') or '未知岗位'} | "
                        f"{_item.get('city') or '未知城市'}",
                        str(_item["job_id"]),
                    ))
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"岗位数据读取失败：{exc}")

    if _job_options:
        # 同一岗位可能被重复收录（公司/岗位/城市完全一致），下拉里只保留第一次出现的 job_id
        _label_to_id: dict[str, str] = {}
        for _label, _jid in _job_options:
            _label_to_id.setdefault(_label, _jid)
        _selected = st.selectbox("岗位", list(_label_to_id), index=0)
        job_id = _label_to_id[_selected]
    else:
        job_id = None
        st.warning("未找到岗位数据，请检查 rag/data/cleaned_jd.json")

    if st.button("计算匹配度", disabled=job_id is None):
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
