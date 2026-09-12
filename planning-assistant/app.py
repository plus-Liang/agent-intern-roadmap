import streamlit as st
from prompts import build_plan_prompt, build_system_prompt
from llm_client import chat_stream
from errors import ConfigError, APIError

st.set_page_config(page_title="学习/实习规划助手", page_icon="📅")
st.title("📅 学习/实习规划助手")


# ---------- 工具函数 ----------
def detect_mode(user_input: str, current_mode: str) -> str:
    """根据用户输入判断输出模式：diff 只输出改动，full 输出完整计划"""
    text = user_input.lower()

    if any(k in text for k in [
        "完整", "全部", "都输出", "整个计划", "全文",
        "重新给我一份", "完整版", "从头到尾"
    ]):
        return "full"

    if any(k in text for k in [
        "只输出改动", "只给改动", "不用重复", "只发改动",
        "只列改动", "仅改动"
    ]):
        return "diff"

    return current_mode


# ---------- 初始化 ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": build_system_prompt("diff")}
    ]
if "started" not in st.session_state:
    st.session_state.started = False
if "output_mode" not in st.session_state:
    st.session_state.output_mode = "diff"
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("你的信息")

    if not st.session_state.started:
        goal = st.text_input("目标")
        hours = st.text_input("每周可投入时间")
        background = st.text_area("当前基础")

        if st.button("生成计划"):
            if not goal or not hours or not background:
                st.warning("请填写完整信息")
            else:
                st.session_state.started = True
                st.session_state.pending_prompt = build_plan_prompt(
                    goal, hours, background
                )
                st.rerun()
    else:
        st.success("计划已生成，可在下方继续追问")
        st.caption(f"当前输出模式：**{st.session_state.output_mode}**")

        if st.button("重置对话"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ---------- 渲染历史消息 ----------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- 处理待生成的消息 ----------
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    # 更新输出模式
    st.session_state.output_mode = detect_mode(
        prompt, st.session_state.output_mode
    )
    st.session_state.messages[0] = {
        "role": "system",
        "content": build_system_prompt(st.session_state.output_mode)
    }

    # 追加并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 手动流式输出（不用 st.write_stream，避免 DOM 冲突）
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        try:
            for piece in chat_stream(st.session_state.messages):
                full += piece
                placeholder.markdown(full)
            st.session_state.messages.append(
                {"role": "assistant", "content": full}
            )
        except ConfigError as e:
            placeholder.error(f"配置错误：{e}")
        except APIError as e:
            placeholder.error(f"生成失败：{e}")


# ---------- 底部追问输入框 ----------
if st.session_state.started:
    user_input = st.chat_input("继续追问调整计划...")
    if user_input:
        st.session_state.pending_prompt = user_input
        st.rerun()