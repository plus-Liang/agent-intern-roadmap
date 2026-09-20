"""求职 Dashboard 的视觉样式层。

设计稿：``job-dashboard-design/``（4 个可直接离线打开的 HTML 文件）。
本模块把那份设计稿的设计令牌（design token）和组件样式搬进 Streamlit：

* :func:`inject_styles` —— 注入全局 CSS（变量、Streamlit 默认样式覆盖、组件类）
* :func:`stat_card` / :func:`stat_row` —— 统计卡片（设计稿 ``.stat``）
* :func:`job_row` —— 岗位行（设计稿 ``.frow``）
* :func:`status_badge` —— 状态标签（设计稿 ``.tag``）
* :func:`timeline` —— 时间线（设计稿 ``.tl``）
* :func:`icon` —— 内联 SVG 图标（设计稿用 SVG，不用 emoji）

设计稿原始令牌（数字来自 ``:root``，未做改动）::

    --g0 #ffffff   --g25 #fbfbfa   --g50 #f7f6f4   --g100 #f0efec  --g150 #e6e5e1
    --g200 #dedcd7 --g300 #c4c1ba  --g400 #96938c  --g500 #78756e
    --g600 #5c5952 --g700 #444139  --g800 #2b2925  --g900 #1a1815

    --a50 #ecf6f8  --a100 #d3e8ec   --a300 #7db9c3  --a600 #0b626c
    --a700 #084e57 --a900 #04353c

    ok  fg #1f6b3f / bg #e8f3ec / br #c2ddcb
    wr  fg #8a5a0b / bg #faf0dd / br #e8d5ac
    er  fg #93352c / bg #fbeae8 / br #eec9c4
    in  fg #3f4c72 / bg #eceefa / br #cdd4ea

用法::

    from dashboard.styles import inject_styles, stat_row

    inject_styles()          # 必须在 set_page_config() 之后、渲染内容之前调用
    stat_row([...])          # 首页/各 Tab 的统计卡片

维护提示（改 CSS 前先看这三条）
--------------------------------
1. **本文件注入的是全局自定义 CSS**。``inject_styles()`` 把 ``_CSS`` 整段塞进
   页面，作用域是整站，所以任何一条选择器写宽了都会伤到 Streamlit 自己的组件。

2. **禁止覆盖 Streamlit 内置控制按钮**，尤其是左侧导航栏的展开/收起按钮。
   它们的 testid 是（Streamlit 1.63 实测，不是 ``stSidebarCollapsedControl``）：

   * ``[data-testid="stExpandSidebarButton"]`` —— 侧栏**收起**时的展开箭头，
     住在 ``[data-testid="stToolbar"]`` 里（就是原来写 ``visibility:hidden``
     的那个容器）。
   * ``[data-testid="stSidebarCollapseButton"]`` —— 侧栏**展开**时的收起按钮，
     住在 ``[data-testid="stSidebarHeader"]`` 里；它内部那个按钮本身还会带上
     ``[data-testid="stBaseButton-headerNoPadding"]``。

   坑点：``visibility:hidden`` 会被子元素继承。给 ``stToolbar`` 整块写
   ``visibility:hidden``（哪怕同时写了 ``display:flex``），里面的展开箭头也会
   一起消失，页面上只剩一片空白——这正是之前「左上角没有箭头」的原因。
   这两个 testid 只允许**加强**（``visibility:visible`` / ``display:flex``），
   永远不允许隐藏；通用按钮规则也必须把它们排除在外。

3. **改完必须重启服务**。Streamlit 不会热重载被 import 的模块，
   ``dashboard/styles.py`` 改动后要重启 ``streamlit run`` 才生效（浏览器刷新
   或 rerun 都不够）。

另：Streamlit 的内部 class 名会随版本变化，直接用内部 class 名写 CSS 很容易
被一次升级打断。本模块只对 ``data-testid`` 属性选择器做少量结构性覆盖
（隐藏菜单/页脚、压缩留白），视觉部分尽量靠 ``.streamlit/config.toml``
的主题配置 + 自己的 class 承载。
"""

from __future__ import annotations

import html as _html

import streamlit as st

__all__ = [
    "inject_styles",
    "stat_card",
    "stat_row",
    "job_row",
    "status_badge",
    "timeline",
    "icon",
    "STATUS_LABELS",
    "STATUS_VARIANTS",
    "e",
]

# ============================================================
# 数值与状态
# ============================================================

#: 投递状态 → 设计稿 ``.tag`` 变体。
#: 设计稿只有 ok / wr / er / in / ac / plain 六种，这里做语义映射。
STATUS_VARIANTS = {
    "applied": ("in", "已投递"),
    "viewed": ("plain", "已查看"),
    "interview": ("ac", "面试中"),
    "interviewing": ("ac", "面试中"),
    "offer": ("ok", "已 Offer"),
    "accepted": ("ok", "已接受"),
    "rejected": ("er", "未通过"),
    "declined": ("er", "已放弃"),
}

#: 状态英文值 → 中文（表格/卡片里统一走中文）
STATUS_LABELS = {key: label for key, (_, label) in STATUS_VARIANTS.items()}


def status_badge(status: str, label: str | None = None) -> str:
    """状态标签（设计稿 ``.tag`` + ``.d`` 小圆点）。

    ``status`` 是 ``agent.state_machine`` 里的英文状态值；未知值退化为
    灰底 ``plain``，不会抛异常——数据库里出现新状态时页面照样能渲染。
    """
    variant, default_label = STATUS_VARIANTS.get(str(status or ""), ("plain", str(status or "—")))
    return '<span class="tag {v}"><span class="d"></span>{t}</span>'.format(
        v=variant, t=e(label or default_label)
    )


def e(value) -> str:
    """HTML 转义（公司名/岗位名/备注里可能有 ``<`` ``>`` ``&``）。"""
    return _html.escape("" if value is None else str(value), quote=True)


# ============================================================
# 内联 SVG 图标
# ============================================================
# 设计稿的图标全部是 24x24 网格、stroke=currentColor 的内联 SVG，
# 这里保留同一套几何形状，替换掉原来散落在 app.py 里的 emoji。

_ICON_PATHS = {
    # 统计卡片
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "send": '<path d="M22 4 12 14.01l-3-3L2 18"/><path d="M16 4h6v6"/>',
    "calendar": ('<rect x="3" y="4" width="18" height="17" rx="2"/>'
                 '<path d="M3 9h18"/><path d="M8 2v4"/><path d="M16 2v4"/>'),
    "chart": '<path d="M4 19V5"/><path d="M4 19h16"/><path d="m7 15 4-5 3 3 5-7"/>',
    "check": '<path d="m5 12 5 5L19 7"/>',
    "file": ('<path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z"/>'
             '<path d="M14 2v5h5"/><path d="M9 13h6"/><path d="M9 17h4"/>'),
    "coins": '<circle cx="12" cy="12" r="9"/><path d="M12 7v10"/><path d="M9.5 10h5"/><path d="M9.5 14h5"/>',
    "flask": ('<path d="M9 3h6"/><path d="M10 3v5.5L5 18a2 2 0 0 0 1.7 3h10.6A2 2 0 0 0 19 18l-5-9.5V3"/>'
              '<path d="M7 14h10"/>'),
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "inbox": ('<path d="M3 12h5l2 3h4l2-3h5"/>'
              '<path d="M5 5h14l2 7v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-5z"/>'),
    "trash": ('<path d="M4 7h16"/><path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>'
              '<path d="M6 7l1 13a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-13"/>'),
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "alert": '<path d="M12 3 2 20h20z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
    "bell": ('<path d="M18 8a6 6 0 1 0-12 0c0 6-2 7-2 7h16s-2-1-2-7"/>'
             '<path d="M13.7 20a2 2 0 0 1-3.4 0"/>'),
    "download": '<path d="M12 3v12"/><path d="m7 11 5 5 5-5"/><path d="M5 20h14"/>',
    "package": ('<path d="M12 3 3 7.5v9L12 21l9-4.5v-9z"/><path d="m3 7.5 9 4.5 9-4.5"/>'
                '<path d="M12 21v-9"/>'),
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5"/>',
    "chevr": '<path d="m9 6 6 6-6 6"/>',
    "arrow_up": '<path d="m5 15 7-7 7 7"/>',
    "arrow_dn": '<path d="m5 9 7 7 7-7"/>',
    "minus": '<path d="M5 12h14"/>',
}


def icon(name: str, size: int = 14, width: float = 1.7, class_name: str = "") -> str:
    """内联 SVG 图标字符串（``currentColor`` 描边，随父元素颜色）。

    :param class_name: 附加的 CSS class（例如岗位行箭头用的 ``go``）
    :return: 未知图标名返回空串，方便调用方无脑写 ``icon(x)`` 而不必先判断。
    """
    path = _ICON_PATHS.get(str(name or ""))
    if not path:
        return ""
    cls = ' class="{}"'.format(e(class_name)) if class_name else ""
    return (
        '<svg{cls} width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="{w}" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true">{p}</svg>'
    ).format(cls=cls, s=size, w=width, p=path)


# ============================================================
# 全局 CSS
# ============================================================

_CSS = """
/* ============================================================
   teal-ink 设计令牌 —— 与设计稿 job-dashboard-design :root 逐项对齐
   ============================================================ */
:root,
[data-testid="stApp"],
.stApp {
  /* 中性灰 11 级（设计稿实际有 14 个值，含 --g25/--g150，这里是全域） */
  --g0:#ffffff;   --g25:#fbfbfa;  --g50:#f7f6f4;  --g100:#f0efec; --g150:#e6e5e1;
  --g200:#dedcd7; --g300:#c4c1ba; --g400:#96938c; --g500:#78756e;
  --g600:#5c5952; --g700:#444139; --g800:#2b2925; --g900:#1a1815;

  /* 强调色（深青） */
  --a50:#ecf6f8; --a100:#d3e8ec; --a300:#7db9c3; --a600:#0b626c;
  --a700:#084e57; --a900:#04353c;

  /* 语义色 */
  --ok-fg:#1f6b3f; --ok-bg:#e8f3ec; --ok-br:#c2ddcb;
  --wr-fg:#8a5a0b; --wr-bg:#faf0dd; --wr-br:#e8d5ac;
  --er-fg:#93352c; --er-bg:#fbeae8; --er-br:#eec9c4;
  --in-fg:#3f4c72; --in-bg:#eceefa; --in-br:#cdd4ea;

  /* 尺寸节奏 */
  --r-sm:4px; --r-md:6px; --r-lg:9px; --r-xl:12px;
  --sh-xs:0 1px 2px rgba(26,24,21,.05);
  --sh-sm:0 1px 3px rgba(26,24,21,.07),0 1px 2px rgba(26,24,21,.04);
  --sh-md:0 8px 24px -6px rgba(26,24,21,.14),0 2px 6px rgba(26,24,21,.05);

  /* 字体栈（系统栈，不加载任何外部字体） */
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI Variable Text","Segoe UI",
         "PingFang SC","Hiragino Sans GB","Microsoft YaHei UI","Microsoft YaHei",
         "Noto Sans SC",system-ui,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Mono","JetBrains Mono",Consolas,
         "PingFang SC","Microsoft YaHei UI",monospace;

  /* 密度：正文 13px / 表格 12.5px / 行高 40px */
  --fs-body:13px; --fs-table:12.5px; --fs-small:11.5px; --fs-tiny:11px;
  --row-h:40px; --bg-page:#fbfbfa; --brand:#0b626c; --brand-soft:#d3e8ec;
  --brand-ring:rgba(11,98,108,.20); --border:#e6e5e1;
  --shadow:0 1px 2px rgba(26,24,21,.05); --shadow-md:0 8px 24px -6px rgba(26,24,21,.14);
}

/* Streamlit 后端主题变量同步（原生组件跟随同一套色） */
:root {
  --primary-color:#0b626c;
  --background-color:#fbfbfa;
  --secondary-background-color:#f5f4f1;
  --text-color:#1a1a18;
}

/* ============================================================
   1. 覆盖 Streamlit 默认样式
   ============================================================ */

/* 1.1 去掉默认的紫色：任何残留的 Streamlit 品牌紫都换成深青 */
:root { --st-primary-color:#0b626c; }
.stApp a, .stApp a:visited { color:var(--a700); }
.stApp a:hover { color:var(--a600); }

/* 1.2 去掉默认装饰：顶部工具栏、彩色渐变头、页脚、悬浮标记

   ★★ 这里曾经写着 [data-testid="stToolbar"] { height:0; visibility:hidden }，
   导致左上角的「展开侧栏」箭头（[data-testid="stExpandSidebarButton"] 就住在
   stToolbar 内部）被一起隐藏——visibility 是会被子元素继承的，页面上只剩空白。
   现在只做「不改背景 + 不设 hidden」，绝不再对 stToolbar 用 visibility/display。 */
#MainMenu { visibility:hidden; }
footer { visibility:hidden; height:0; }
[data-testid="stToolbar"] { right:0; background:transparent; }
[data-testid="stDecoration"] { display:none; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stStatusWidget"] { display:none; }
[data-testid="stAppDeployButton"] { display:none; }

/* 1.2b ★ Streamlit 侧栏控制按钮白名单 —— 这两个按钮是导航的唯一入口，不能隐藏。

   * stExpandSidebarButton  侧栏收起时的展开箭头（在 stToolbar 里）
   * stSidebarCollapseButton 侧栏展开时的收起按钮（在 stSidebarHeader 里）
   * stSidebarCollapsedControl 旧版本 testid，顺手一起兜住，防升级后回退
   这里只做「加强」（恢复可见 + 撑成 flex），不设置任何颜色/边框，保持原生观感。 */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
  visibility:visible !important;
  display:flex !important;
  opacity:1 !important;
  align-items:center;
  justify-content:center;
}
[data-testid="stToolbar"] [data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] svg {
  width:16px !important;
  height:16px !important;
}

/* 1.3 统一字体为系统栈（覆盖 Streamlit 默认字体） */
html, body, .stApp, .stApp * ,
[data-testid="stAppViewContainer"], [data-testid="stSidebar"],
input, textarea, select, button, .stMarkdown, .stDataFrame {
  font-family:var(--sans) !important;
}
code, pre, kbd, .stCode, [data-testid="stCodeBlock"] * {
  font-family:var(--mono) !important;
}
body, .stApp { font-size:var(--fs-body); line-height:1.5; color:var(--g800); }

/* 1.4 去掉 emoji 装饰：Streamlit 提示条自带的表情图标不再显示 */
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] > p:first-child > span:first-child {
  display:none;
}

/* 1.5 版面密度：主区块收紧到设计稿的 20px 内边距 */
[data-testid="stAppViewBlockContainer"],
.block-container {
  padding-top:1.6rem; padding-bottom:2.5rem; max-width:1280px;
}
[data-testid="stSidebar"] { border-right:1px solid var(--border); }
h1, h2, h3, h4, h5 { letter-spacing:-.012em; color:var(--g800); font-weight:600; }
/* 17px 是设计稿 .pagehead h1 的实测值 */
h1 { font-size:17px; }
h2 { font-size:15px; }
h3 { font-size:13.5px; }
hr, [data-testid="stDivider"] hr { border-color:var(--border); }

/* 1.6 按钮：无阴影、hover 背景微变（设计稿 .btn / .btn.pri / .btn.ghost）

   范围说明：只匹配 stBaseButton-*，并排除三类 Streamlit 自带按钮：
   * stBaseButton-header        —— 顶部工具栏按钮，改小会难点
   * stBaseButton-headerNoPadding —— 侧栏「收起」按钮（在 stSidebarHeader 里），
                                    属于导航控件，必须保留原生观感（见 1.2b）
   * stBaseButton-elementToolbar —— 表格/图表的图标工具条（22px 宽），
                                   一套 32px 最小高度会把图标挤变形 */
.stApp [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]):not([data-testid="stBaseButton-elementToolbar"]),
.stApp [data-testid="stDownloadButton"] button {
  border-radius:var(--r-md) !important;
  box-shadow:none !important;
  border:1px solid var(--g200) !important;
  background:var(--g0) !important;
  color:var(--g600) !important;
  font-size:12.5px !important;
  font-weight:500 !important;
  min-height:32px !important;
  transition:background .12s ease, border-color .12s ease, color .12s ease;
}
.stApp [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]):not([data-testid="stBaseButton-elementToolbar"]):hover {
  background:var(--g50) !important;
  border-color:var(--g300) !important;
  color:var(--g900) !important;
}
.stApp [data-testid^="stBaseButton-"]:active,
.stApp [data-testid="stDownloadButton"] button:active { transform:none !important; }

/* 主按钮：深青实心（设计稿 .btn.pri 是近黑，这里用品牌深青做主操作色） */
.stApp [data-testid="stBaseButton-primary"],
.stApp [data-testid="stBaseButton-primaryFormSubmit"] {
  background:var(--a600) !important;
  border-color:var(--a600) !important;
  color:var(--g0) !important;
}
.stApp [data-testid="stBaseButton-primary"]:hover,
.stApp [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background:var(--a700) !important;
  border-color:var(--a700) !important;
  color:var(--g0) !important;
}
.stApp [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]):not([data-testid="stBaseButton-elementToolbar"]):focus-visible {
  outline:none !important;
  box-shadow:0 0 0 3px var(--a50) !important;
  border-color:var(--a300) !important;
}
.stApp [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]):not([data-testid="stBaseButton-elementToolbar"]):disabled,
.stApp [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]):not([data-testid="stBaseButton-elementToolbar"]):disabled:hover {
  opacity:.5 !important; background:var(--g50) !important;
  border-color:var(--g150) !important; color:var(--g400) !important;
}
/* 按钮里的图标跟着字色走，尺寸收到 13px（侧栏收起/展开按钮除外，见 1.2b） */
.stApp [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-headerNoPadding"]):not([data-testid="stBaseButton-elementToolbar"]) svg {
  width:13px; height:13px;
}

/* 1.7 输入类控件：8px 圆角 + 极淡边框 + 深青聚焦环

   Streamlit 1.63 已从 BaseWeb 切到 React Aria，控件没有 data-baseweb，
   但有更稳的 data-testid 外壳：

     文本框        [data-testid=stTextInputRootElement]  > [data-testid=stTextInputField]
     多行文本域    [data-testid=stTextAreaRootElement]   > textarea
     下拉选择      [data-testid=stSelectbox] > div[data-rac][role=group] > input[role=combobox]

   注意：不要用宽泛的 div[data-rac][data-orientation]——tabs 容器和
   tablist 也带这两个属性，会把整个 Tab 区套上一个白底边框。 */
/* 文本框：外壳是 stTextInputRootElement，内层输入框本身无边框，
   所以只给外壳上色，千万不要再给 input 加边框（会出现双层框）。 */
.stApp [data-testid="stTextInputRootElement"],
.stApp [data-testid="stNumberInputContainer"],
.stApp [data-testid="stTextAreaRootElement"],
.stApp [data-testid="stDateInputRootElement"] {
  background-color:var(--g0) !important;
  border:1px solid var(--g200) !important;
  border-radius:var(--r-md) !important;
  box-shadow:none !important;
}
.stApp [data-testid="stTextInputRootElement"]:hover,
.stApp [data-testid="stTextAreaRootElement"]:hover,
.stApp [data-testid="stNumberInputContainer"]:hover {
  border-color:var(--g300) !important;
}
.stApp [data-testid="stTextInputRootElement"]:focus-within,
.stApp [data-testid="stTextAreaRootElement"]:focus-within,
.stApp [data-testid="stNumberInputContainer"]:focus-within {
  border-color:var(--a300) !important;
  box-shadow:0 0 0 3px var(--a50) !important;
}
/* 输入元素本身：去掉自带的边框/背景，交给外壳 */
.stApp [data-testid="stTextInputField"],
.stApp [data-testid="stNumberInputField"],
.stApp .stTextArea textarea,
.stApp [data-testid="stTextAreaRootElement"] textarea {
  background:transparent !important;
  border:none !important;
  border-radius:0 !important;
  box-shadow:none !important;
  font-size:13px !important;
  color:var(--g900) !important;
}
.stApp .stTextArea textarea,
.stApp [data-testid="stTextAreaRootElement"] textarea {
  line-height:1.55 !important;
  padding:8px 10px !important;
}

/* 下拉选择（React Aria ComboBox）：触发器是 role=group 的 div */
.stApp div[data-rac][role="group"],
.stApp [data-testid="stSelectbox"] div[role="group"] {
  background-color:var(--g0) !important;
  border:1px solid var(--g200) !important;
  border-radius:var(--r-md) !important;
  box-shadow:none !important;
}
.stApp div[data-rac][role="group"]:hover { border-color:var(--g300) !important; }
.stApp div[data-rac][role="group"]:focus-within {
  border-color:var(--a300) !important;
  box-shadow:0 0 0 3px var(--a50) !important;
}
.stApp [role="combobox"] {
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
  font-size:13px !important;
  color:var(--g900) !important;
}
/* 下拉弹层 */
.stApp [role="listbox"] { font-size:12.5px !important; }
.stApp [role="option"] { font-size:12.5px !important; }
.stApp [role="option"][aria-selected="true"], .stApp [role="option"]:hover {
  background-color:var(--a50) !important; color:var(--a900) !important;
}
/* 文件上传 / 表单容器：对齐设计稿的极淡边框 + 8px 圆角 */
.stApp [data-testid="stFileUploaderDropzone"] {
  background-color:var(--g0) !important;
  border:1px dashed var(--g200) !important;
  border-radius:var(--r-md) !important;
}
.stApp [data-testid="stFileUploaderDropzone"]:hover { border-color:var(--a300) !important; }
.stApp [data-testid="stForm"] {
  border:1px solid var(--border) !important;
  border-radius:var(--r-lg) !important;
  background:var(--g0);
  box-shadow:var(--sh-xs);
}

/* 1.8 提示条：设计稿 .note 的柔和语义底色（覆盖 Streamlit 默认色调） */
[data-testid="stAlert"] {
  border-radius:var(--r-md) !important;
  border:1px solid var(--g200);
  box-shadow:none !important;
  padding:9px 11px;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { font-size:12px; line-height:1.6; }
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentSuccess"],
[data-testid="stAlertContentWarning"], [data-testid="stAlertContentError"] {
  border-radius:var(--r-md) !important;
  border-width:1px !important;
  border-style:solid !important;
  box-shadow:none !important;
  padding:8px 10px !important;
}
[data-testid="stAlertContentInfo"] {
  background-color:var(--in-bg) !important; border-color:var(--in-br) !important; color:var(--in-fg) !important;
}
[data-testid="stAlertContentSuccess"] {
  background-color:var(--ok-bg) !important; border-color:var(--ok-br) !important; color:var(--ok-fg) !important;
}
[data-testid="stAlertContentWarning"] {
  background-color:var(--wr-bg) !important; border-color:var(--wr-br) !important; color:var(--wr-fg) !important;
}
[data-testid="stAlertContentError"] {
  background-color:var(--er-bg) !important; border-color:var(--er-br) !important; color:var(--er-fg) !important;
}
[data-testid="stAlertContentInfo"] * , [data-testid="stAlertContentSuccess"] *,
[data-testid="stAlertContentWarning"] *, [data-testid="stAlertContentError"] * {
  color:inherit !important; background-color:transparent !important;
}

/* 1.9 表格 / 代码块 / 展开器 / 标签页 */
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
  border:1px solid var(--border) !important;
  border-radius:var(--r-lg) !important;
  overflow:hidden;
  box-shadow:var(--sh-xs);
}
.stApp [data-testid="stCodeBlock"] pre,
.stApp pre {
  background:var(--g50) !important;
  border:1px solid var(--border);
  border-radius:var(--r-md);
  font-size:12.5px;
}
.stApp code { font-size:12.5px; background:var(--g50); border-radius:var(--r-sm); padding:.5px 4px; }
.stApp .stMarkdown code { color:var(--a700); }
[data-testid="stExpander"] details,
[data-testid="stExpander"] {
  border:1px solid var(--border) !important;
  border-radius:var(--r-lg) !important;
  background:var(--g0);
  box-shadow:var(--sh-xs);
}
[data-testid="stExpander"] summary { font-size:12.5px; font-weight:500; }

/* 1.9b 标签页：1.63 用 React Aria（data-testid="stTab" / role="tab"），
   旧的 [data-baseweb="tab"] 选择器在这个版本上完全匹配不到，已换成新的。
   选中态的下划线由 react-aria-SelectionIndicator 渲染，这里只改颜色；
   万一它的定位依赖内联样式，最多是下划线颜色没跟上，不会破坏布局。 */
.stTabs [role="tablist"] { gap:2px; border-bottom:1px solid var(--border); }
.stTabs [data-testid="stTab"] {
  height:34px !important; padding:0 11px !important;
  font-size:12.5px !important; font-weight:500; color:var(--g500) !important;
}
.stTabs [data-testid="stTab"]:hover { color:var(--g800) !important; }
.stTabs [data-testid="stTab"] p { font-size:12.5px !important; margin:0 !important; }
.stTabs [data-testid="stTab"][data-selected="true"],
.stTabs [data-testid="stTab"][aria-selected="true"] {
  color:var(--g900) !important; font-weight:600 !important;
}
.stTabs [data-testid="stTab"][data-selected="true"] p,
.stTabs [data-testid="stTab"][aria-selected="true"] p { font-weight:600 !important; }
.stTabs .react-aria-SelectionIndicator { background-color:var(--a600) !important; }
/* 未选中的面板不占空间，选中面板顶部留白对齐设计稿 */
.stTabs [data-testid="stTabPanel"] { padding-top:13px !important; }

/* 1.10 进度条 / 滑块 / 复选 / 分段控件统一到深青 */
.stApp [data-testid="stProgress"] div[role="progressbar"] > div { background:var(--a600) !important; }
.stApp [data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; }

/* ============================================================
   2. 可复用组件类
   ============================================================ */

/* 2.1 统计卡片（设计稿 .stat / .stats） */
.stats { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:10px; margin:2px 0 12px; }
.stats[data-cols="3"] { grid-template-columns:repeat(3, minmax(0,1fr)); }
.stats[data-cols="2"] { grid-template-columns:repeat(2, minmax(0,1fr)); }
.stat-card {
  background:var(--g0); border:1px solid var(--border); border-radius:8px;
  box-shadow:var(--sh-xs); padding:13px 14px 12px; position:relative; overflow:hidden;
  min-width:0; flex:1 1 0;
}
.stat-card::after {
  content:""; position:absolute; left:0; top:0; bottom:0; width:2px; background:var(--g200);
}
.stat-card.accent::after { background:var(--a600); }
.stat-card.ok::after     { background:var(--ok-fg); }
.stat-card.warn::after   { background:var(--wr-fg); }
.stat-card.error::after  { background:var(--er-fg); }
.stat-card .lab {
  display:flex; align-items:center; gap:6px; font-size:var(--fs-small);
  color:var(--g500); font-weight:500;
}
.stat-card .lab svg { color:var(--g400); }
.stat-card .lab .info { margin-left:auto; color:var(--g300); display:inline-flex; }
.stat-card .val { display:flex; align-items:baseline; gap:6px; margin-top:8px; }
.stat-card .val b {
  font-size:26px; font-weight:600; letter-spacing:-.03em; line-height:1;
  font-variant-numeric:tabular-nums; color:var(--g900);
}
.stat-card .val span { font-size:var(--fs-small); color:var(--g400); }
.stat-card .dlt {
  display:flex; align-items:center; gap:5px; margin-top:9px;
  font-size:var(--fs-small); color:var(--g500);
}
.stat-card .dlt .chg {
  display:inline-flex; align-items:center; gap:2px; font-weight:600; font-size:var(--fs-tiny);
  padding:1px 4px; border-radius:3px;
}
.stat-card .dlt .up   { background:var(--ok-bg); color:var(--ok-fg); }
.stat-card .dlt .dn   { background:var(--er-bg); color:var(--er-fg); }
.stat-card .dlt .flat { background:var(--g100); color:var(--g500); }
/* 迷你柱状图（设计稿 .spark） */
.stat-card .spark { display:flex; align-items:flex-end; gap:2px; height:16px; margin-left:auto; }
.stat-card .spark i { width:3px; border-radius:1px; background:var(--g200); display:block; }
.stat-card .spark i.hi { background:var(--a300); }

/* 2.2 岗位行（设计稿 .frow / .job） */
.job-row {
  display:flex; align-items:center; gap:10px; padding:9px 14px;
  border-bottom:1px solid var(--g100); min-width:0;
}
.job-row:last-child { border-bottom:none; }
.job-row:hover { background:var(--g25); }
.job-row .co {
  width:26px; height:26px; border-radius:var(--r-sm); flex:none; display:grid; place-items:center;
  font-size:10.5px; font-weight:700; letter-spacing:-.02em;
  background:var(--g100); color:var(--g600); border:1px solid var(--g150);
}
.job-row .id { min-width:0; flex:1; }
.job-row .id b {
  display:block; font-size:var(--fs-table); font-weight:500; color:var(--g900);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.job-row .id span {
  display:block; font-size:var(--fs-tiny); color:var(--g400); margin-top:1px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.job-row .sc { width:74px; flex:none; text-align:right; }
.job-row .sc b { font-size:var(--fs-table); font-weight:600; font-variant-numeric:tabular-nums; }
.job-row .go { color:var(--g300); flex:none; }
.job-row:hover .go { color:var(--g600); }
.job-panel {
  background:var(--g0); border:1px solid var(--border); border-radius:var(--r-lg);
  box-shadow:var(--sh-xs); overflow:hidden;
}
.job-panel > header {
  display:flex; align-items:center; gap:8px; padding:11px 14px; border-bottom:1px solid var(--g100);
}
.job-panel > header h2 { font-size:var(--fs-table); font-weight:600; margin:0; }
.job-panel > header .sub { font-size:var(--fs-small); color:var(--g400); }
.job-panel > header .r { margin-left:auto; display:flex; align-items:center; gap:6px; }

/* 2.3 状态标签（设计稿 .tag / .badge） */
.status-badge {
  display:inline-flex; align-items:center; gap:4px; height:19px; padding:0 6px;
  border-radius:var(--r-sm); font-size:var(--fs-tiny); font-weight:500; line-height:1;
  border:1px solid var(--g200); background:var(--g0); color:var(--g600);
  white-space:nowrap; vertical-align:middle;
}
.status-badge .d { width:5px; height:5px; border-radius:50%; background:currentColor; opacity:.75; }
.status-badge.ok    { background:var(--ok-bg); border-color:var(--ok-br); color:var(--ok-fg); }
.status-badge.warn  { background:var(--wr-bg); border-color:var(--wr-br); color:var(--wr-fg); }
.status-badge.error { background:var(--er-bg); border-color:var(--er-br); color:var(--er-fg); }
.status-badge.info  { background:var(--in-bg); border-color:var(--in-br); color:var(--in-fg); }
.status-badge.accent{ background:var(--a50);  border-color:var(--a100); color:var(--a700); }
.status-badge.plain { border-color:transparent; background:var(--g100); color:var(--g600); }
.count-badge {
  display:inline-grid; place-items:center; min-width:17px; height:17px; padding:0 4px;
  border-radius:9px; background:var(--g100); border:1px solid var(--g200);
  font-size:10.5px; font-weight:600; color:var(--g600); font-variant-numeric:tabular-nums;
}

/* 2.4 时间线（设计稿 .tl） */
.timeline { position:relative; padding:2px 0 0; }
.timeline .ev { display:flex; gap:11px; padding-bottom:15px; position:relative; }
.timeline .ev:last-child { padding-bottom:2px; }
.timeline .ev::before {
  content:""; position:absolute; left:6.5px; top:16px; bottom:0; width:1px; background:var(--g150);
}
.timeline .ev:last-child::before { display:none; }
.timeline .dot {
  width:14px; height:14px; border-radius:50%; flex:none; margin-top:1.5px; z-index:1;
  background:var(--g0); border:1.5px solid var(--g300); display:grid; place-items:center;
}
.timeline .dot i { width:5px; height:5px; border-radius:50%; background:var(--g300); display:block; }
.timeline .ev.done .dot { border-color:var(--a600); background:var(--a50); }
.timeline .ev.done .dot i { background:var(--a600); }
.timeline .ev.warn .dot { border-color:var(--wr-fg); background:var(--wr-bg); }
.timeline .ev.warn .dot i { background:var(--wr-fg); }
.timeline .ev.error .dot { border-color:var(--er-fg); background:var(--er-bg); }
.timeline .ev.error .dot i { background:var(--er-fg); }
.timeline .ev .bd { min-width:0; flex:1; }
.timeline .ev .l1 { display:flex; align-items:center; gap:7px; flex-wrap:wrap; }
.timeline .ev .l1 b { font-size:var(--fs-table); font-weight:600; color:var(--g900); }
.timeline .ev .l1 time {
  margin-left:auto; font-size:var(--fs-tiny); color:var(--g400);
  font-variant-numeric:tabular-nums; white-space:nowrap;
}
.timeline .ev p { font-size:12px; color:var(--g500); margin:3px 0 0; line-height:1.55; }

/* 2.5 辅助类 */
.mono { font-family:var(--mono); font-variant-numeric:tabular-nums; }
.num  { font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1; }
.note {
  display:flex; gap:9px; padding:9px 11px; border-radius:var(--r-md);
  background:var(--a50); border:1px solid var(--a100); color:var(--a900);
}
.note svg { color:var(--a600); margin-top:1px; }
.note p { font-size:var(--fs-small); line-height:1.6; margin:0; }
.page-meta { font-size:12.5px; color:var(--g500); margin:-4px 0 12px; }

/* 2.6 窄屏回退：卡片两列，表格信息不再强行塞一行 */
@media (max-width:1100px) {
  .stats, .stats[data-cols="3"] { grid-template-columns:repeat(2, minmax(0,1fr)); }
}
@media (max-width:640px) {
  .stats, .stats[data-cols="3"], .stats[data-cols="2"] {
    grid-template-columns:minmax(0,1fr);
  }
}

/* 2.7 消除 Streamlit markdown 容器的默认外边距，让卡片网格贴合 */
[data-testid="stMarkdownContainer"] > .stats { margin-top:0; }
"""


def inject_styles() -> None:
    """注入全局 CSS（每次 rerun 都调用；CSS 很便宜，不必缓存）。

    必须在 ``st.set_page_config()`` 之后调用——``set_page_config`` 要求是
    第一个 Streamlit 命令。

    用 ``st.html`` 而不是 ``st.markdown(unsafe_allow_html=True)``：
    当内容只包含 ``<style>`` 时，Streamlit 会把它送进 event container
    而不是主容器，既不占版面，也不会被 markdown 解析器二次处理。
    """
    st.html("<style>{}</style>".format(_CSS))


# ============================================================
# HTML 组件
# ============================================================

_VARIANTS = {"default", "accent", "ok", "warn", "error"}


def _spark_html(spark: list | tuple | None) -> str:
    """迷你柱状图（设计稿 .spark）。数值会归一化到 16px 高度内。"""
    if not spark:
        return ""
    try:
        values = [float(v) for v in spark if v is not None]
    except (TypeError, ValueError):
        return ""
    if not values:
        return ""
    top = max(values) or 1.0
    bars = []
    for index, value in enumerate(values):
        height = max(3, int(round(value / top * 16)))
        cls = ' class="hi"' if index == len(values) - 1 else ""
        bars.append('<i{} style="height:{}px"></i>'.format(cls, height))
    return '<span class="spark" aria-hidden="true">{}</span>'.format("".join(bars))


def stat_card(
    label: str,
    value,
    unit: str = "",
    *,
    icon_name: str = "",
    delta: str = "",
    delta_dir: str = "flat",
    hint: str = "",
    variant: str = "default",
    spark: list | tuple | None = None,
) -> str:
    """单张统计卡片（设计稿 ``.stat``）的 HTML。

    :param label: 卡头文字
    :param value: 主数值（会原样转义，可传字符串如 ``"76"``）
    :param unit: 主数值后面的小字，如 ``"/ 100 分"``
    :param icon_name: :func:`icon` 里的图标名，空则不显示
    :param delta: 变化幅度小标签，如 ``"18.7%"``；为空则不显示
    :param delta_dir: ``up`` / ``dn`` / ``flat``
    :param hint: delta 后面的说明文字，如 ``"较上周"``
    :param variant: ``default`` / ``accent`` / ``ok`` / ``warn`` / ``error``
    :param spark: 迷你柱状图数据
    """
    variant = variant if variant in _VARIANTS else "default"
    card_class = "stat-card" + ("" if variant == "default" else " " + variant)

    label_html = icon(icon_name, 13) if icon_name else ""
    label_html += e(label)
    if hint and not delta:
        # 没有 delta 时，hint 退到卡头的 info 图标上（title 悬浮提示）
        label_html += '<span class="info" title="{}">{}</span>'.format(
            e(hint), icon("info", 12)
        )

    delta_html = ""
    if delta:
        arrow = {"up": "arrow_up", "dn": "arrow_dn"}.get(delta_dir, "minus")
        delta_html = (
            '<div class="dlt">'
            '<span class="chg {d}">{arrow}{value}</span>{hint}{spark}'
            "</div>"
        ).format(
            d=delta_dir if delta_dir in {"up", "dn", "flat"} else "flat",
            arrow=icon(arrow, 10, 2.4),
            value=e(delta),
            hint=e(hint),
            spark=_spark_html(spark),
        )
    elif spark:
        delta_html = '<div class="dlt">{}</div>'.format(_spark_html(spark))

    return (
        '<article class="{cls}">'
        '<div class="lab">{lab}</div>'
        '<div class="val"><b>{val}</b>{unit}</div>'
        "{delta}"
        "</article>"
    ).format(
        cls=card_class,
        lab=label_html,
        val=e(value),
        unit='<span>{}</span>'.format(e(unit)) if unit else "",
        delta=delta_html,
    )


def stat_row(cards: list, columns: int | None = None) -> None:
    """渲染一行统计卡片（设计稿 ``.stats`` 网格）。

    :param cards: :func:`stat_card` 返回的 HTML 字符串列表，或 dict 列表
                  （dict 会作为关键字参数转给 :func:`stat_card`）
    :param columns: 每行卡片数；默认取卡片数量，最多 4
    """
    items = []
    for card in cards or []:
        items.append(stat_card(**card) if isinstance(card, dict) else str(card))
    if not items:
        return
    cols = columns or min(len(items), 4)
    st.markdown(
        '<section class="stats" data-cols="{}">{}</section>'.format(
            int(cols), "".join(items)
        ),
        unsafe_allow_html=True,
    )


def job_row(
    title: str,
    meta: str = "",
    *,
    company: str = "",
    initials: str = "",
    score=None,
    score_max: float = 100,
    badge_html: str = "",
    accent: bool = False,
) -> str:
    """岗位行（设计稿 ``.frow``）的 HTML。

    :param title: 岗位名
    :param meta: 副标题（城市 · 薪资 · 状态…）
    :param company: 公司名（用于左侧方块首字，未给 ``initials`` 时取前 2 字）
    :param initials: 左侧方块里的文字
    :param score: 匹配分；给了就渲染右侧分数 + 进度条
    :param accent: 匹配分进度条是否用深青（否则用中性灰）
    """
    mark = e(initials or (company or "?")[:2])
    score_html = ""
    if score is not None:
        try:
            ratio = max(0.0, min(1.0, float(score) / float(score_max or 100)))
        except (TypeError, ValueError):
            ratio = 0.0
        bar_cls = "" if accent else ' class="mut"'
        score_html = (
            '<div class="sc"><b>{v}</b>'
            '<div class="bar"><i{cls} style="width:{pct:.0f}%"></i></div></div>'
        ).format(v=e(score), cls=bar_cls, pct=ratio * 100)

    return (
        '<div class="job-row">'
        '<div class="co">{mark}</div>'
        '<div class="id"><b>{title}</b><span>{meta}</span></div>'
        "{score}{badge}{go}"
        "</div>"
    ).format(
        mark=mark,
        title=e(title),
        meta=e(meta),
        score=score_html,
        badge=(' <span class="sc">{}</span>'.format(badge_html) if badge_html else ""),
        go=icon("chevr", 14, 1.8, class_name="go"),
    )


def timeline(events: list) -> str:
    """时间线（设计稿 ``.tl``）的 HTML。

    :param events: dict 列表，支持字段
        ``title`` / ``time`` / ``desc`` / ``badge_html`` / ``state``
        （``done`` / ``warn`` / ``error`` / 空）
    """
    rows = []
    for event in events or []:
        state = str(event.get("state") or "").strip()
        cls = "ev" + (" " + state if state in {"done", "warn", "error"} else "")
        badge = event.get("badge_html") or ""
        time_html = (
            "<time>{}</time>".format(e(event.get("time")))
            if event.get("time") else ""
        )
        desc = (
            "<p>{}</p>".format(e(event.get("desc")))
            if event.get("desc") else ""
        )
        rows.append(
            (
                '<div class="{cls}">'
                '<div class="dot"><i></i></div>'
                '<div class="bd">'
                '<div class="l1"><b>{title}</b>{badge}{time}</div>'
                "{desc}"
                "</div></div>"
            ).format(
                cls=cls,
                title=e(event.get("title")),
                badge=badge,
                time=time_html,
                desc=desc,
            )
        )
    if not rows:
        return ""
    return '<div class="timeline">{}</div>'.format("".join(rows))
