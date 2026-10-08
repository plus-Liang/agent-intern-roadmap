# -*- coding: utf-8 -*-
"""岗位类型（实习 / 正式 / 兼职）判定 —— 全项目**唯一**的口径来源。

为什么需要这个模块
------------------
用户说「帮我找广州的 agent 的**实习**岗位」，返回列表里却混着「算法工程师」
这类正式岗 —— 根因是 jobs 表**根本没有岗位类型字段**，三个平台抓取时也没打标，
search_jobs 自然只能按 city + keyword 过滤。

三个平台的类型信号强度完全不同（2026-10 实测，别凭感觉改）：

    * **实习僧 shixiseng**：搜索 URL 固定带 `type=intern`，**平台属性就是实习**。
      所以它的数据天然全是实习岗，不需要（也无法）逐条判类型。
      例外：站点也放「兼职」，以及少量标题不带「实习」的正式岗（可转正）。
    * **牛客 niuke**：接口用 `recruitType=2`（实习频道）取数。
      实测 180 条里只有 **51.1%** 的标题带「实习」字样 —— 同一批次里
      「大模型算法」「算法工程师」同样来自实习频道（`recruitType=1` 那一批
      标题带「实习」的是 **0%**，两批完全可分）。
      ⇒ **标题关键词不能当实习判据**，否则牛客会漏掉近一半实习岗。
    * **国家大学生就业服务平台 ncss**：列表接口的 `recruitType` 字段
      **恒为 "0"**（4 城 × 5 词共 400 条抽样全是 0），`jobType="03"` 过滤又过狠
      （北京「算法工程师」26 → 0 条），详情页也没有「岗位类型 / 工作性质」这类
      结构化字段。
      ⇒ 这个平台**只能按标题关键词打标**，且必须容忍误判。

判定优先级（先事实、后关键词、最后平台兜底）
--------------------------------------------
    1. 标题含「兼职」        → 兼职
    2. 标题含「实习 / Intern」→ 实习（跨平台最强的单条信号）
    3. 平台是实习专属平台     → 实习（实习僧 / 牛客）
    4. 标题含「社招 / 全职」  → 正式
    5. 平台是校招 / 综合就业池 → 正式（ncss；**兜底式判定，见下**）
    6. 其余                   → ""（未知，**不硬猜**；调用方按"不限"处理）

ncss 为什么按"正式"兜底（这是有意的取舍，别当成 bug）
-------------------------------------------------------
ncss 的列表接口 `recruitType` 恒为 0、详情页也没有结构化类型字段，**唯一**
可用的信号就是标题里有没有「实习」。实测 988 条里只有 65 条标题带「实习」，
其余 923 条不带 —— 如果把它们留成"未知"，用户搜「广州的算法工程师实习」
时这 923 条既不在实习结果里、也不在正式结果里，等于凭空消失（用户会以为
"这条岗位没了"）。按正式兜底后两类查询都能给出**确定**答案，代价是
「标题不带实习、其实是实习」的岗位会被算进正式（漏判，不会误放进实习结果）。
比起把正式岗混进实习列表（用户本轮报的就是这个），宁可漏判。

为什么「可转正」仍算实习、为什么第 2 步在第 3 步之前
----------------------------------------------------
实习僧的「Java开发（可转正）」「AI Agent开发（可转正）」是实习岗（可转正只是
转正通道），平台参数已经把它限定成实习了，不能再因标题把它翻成正式岗。反过来，
实习僧上标题明写「社会招聘」的条目才是真正式岗，所以第 4 步要放在平台兜底之后。
"""

from __future__ import annotations

from typing import Any

# 规范取值："" 表示未知 / 不做判定（调用方一律当成"不限"）
TYPE_INTERN = "实习"
TYPE_FULLTIME = "正式"
TYPE_PARTTIME = "兼职"
KNOWN_TYPES = (TYPE_INTERN, TYPE_FULLTIME, TYPE_PARTTIME)

# 平台属性：这些平台的抓取参数**已经**限定为实习岗，逐条缺证据时按实习处理。
# 值必须与各抓取器的 platform_name / jobs.platform 完全一致（小写）。
PLATFORM_INTERN_DEFAULT = frozenset({"shixiseng", "niuke"})

# 反向的平台属性：这些平台是**校园招聘 / 综合就业**池，实习只是其中一小部分，
# 逐条缺证据时按正式处理（否则「广州的算法工程师」这类正式岗会以"未知"身份
# 留在库里，用户搜「实习」时既不敢显示、搜「正式」时又漏掉）。
PLATFORM_FULLTIME_DEFAULT = frozenset({"ncss"})

# 岗位名里的类型关键词（大小写不敏感；intern 覆盖 Internship / Intern 等形态）
_INTERN_WORDS = ("实习", "intern", "见习", "实训")
_PARTTIME_WORDS = ("兼职", "part-time", "part time")
_FULLTIME_WORDS = ("社招", "全职", "社会招聘", "正式")

# 用户查询里表示岗位类型的词 —— 搜索侧要先把它从关键词里摘掉，
# 否则「广州的 agent 实习岗位」会退化成关键词 LIKE '%实习%'，
# 把牛客里标题不带「实习」的实习岗（实测占 49%）全部漏掉。
QUERY_TYPE_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (TYPE_INTERN, ("实习生", "实习", "intern", "见习", "实训")),
    (TYPE_FULLTIME, ("正式", "社招", "社会招聘", "全职", "校招", "应届")),
    (TYPE_PARTTIME, ("兼职", "part-time", "part time")),
)


def normalize_job_type(value: Any) -> str:
    """把任意来源的类型值规整成规范取值；认不出来返回 ""。

    兼容抓取器 / 外部数据可能给的写法：intern / 实习 / internship / fulltime /
    正式 / 全职 / 社招 …… 大小写、空格、下划线都无所谓。
    """
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if any(word in text for word in _INTERN_WORDS):
        return TYPE_INTERN
    if any(word in text for word in _PARTTIME_WORDS):
        return TYPE_PARTTIME
    if any(word in text for word in _FULLTIME_WORDS) or text in (
        "fulltime", "full_time", "full-time", "regular",
    ):
        return TYPE_FULLTIME
    return ""


def classify_job_type(
    platform: Any = "",
    title: Any = "",
    description: Any = "",
    explicit: Any = "",
) -> str:
    """判定一条岗位的类型，返回 实习 / 正式 / 兼职 / ""（未知）。

    参数：
        platform:    平台标识（jobs.platform，如 "ncss"）
        title:       岗位名
        description: JD 正文。**默认不参与判定** —— ncss 详情页正文里几乎每条
                     都会出现「实习」字样（含正式岗的"接受实习生"话术），用它
                     判类型会造成大面积假阳性。保留该形参只为将来需要时显式启用。
        explicit:    上游已经拿到的类型值（如抓取器的 job_type），能认出来就
                     直接采信，认不出来再走规则。

    本函数**纯函数、无 IO**：抓取端打标、清洗端归一、存量回填、搜索端过滤
    四处共用同一份实现，避免出现两套口径。
    """
    known = normalize_job_type(explicit)
    if known:
        return known

    name = str(title or "").lower()
    if name:
        if any(word in name for word in _PARTTIME_WORDS):
            return TYPE_PARTTIME
        if any(word in name for word in _INTERN_WORDS):
            return TYPE_INTERN

    if str(platform or "").strip().lower() in PLATFORM_INTERN_DEFAULT:
        # 平台抓取参数已限定实习（实习僧 type=intern / 牛客 recruitType=2）。
        # 注意顺序：上面两条标题规则先跑，「Java开发（可转正）」不会被误判成正式。
        return TYPE_INTERN

    if name and any(word in name for word in _FULLTIME_WORDS):
        return TYPE_FULLTIME

    if str(platform or "").strip().lower() in PLATFORM_FULLTIME_DEFAULT:
        return TYPE_FULLTIME      # ncss 等校招池：缺证据时按正式（见模块 docstring）
    return ""


def detect_query_type(text: Any) -> str:
    """从用户问题里识别「要哪一类岗位」，返回 实习 / 正式 / 兼职 / ""。

    只认明确的类型词（「实习」「正式」「兼职」「校招」……），识别不出返回 ""，
    表示**不过滤**（「帮我找广州的 agent 岗位」→ 实习和正式都要）。
    """
    lowered = str(text or "").lower()
    if not lowered:
        return ""
    for job_type, words in QUERY_TYPE_WORDS:
        if any(word in lowered for word in words):
            return job_type
    return ""


def strip_query_type_words(text: Any) -> str:
    """把问题里的类型词摘掉，留下真正该检索的关键词。

    「广州的 agent 实习岗位」→「广州的 agent 岗位」。
    只摘类型词，不动其它任何内容（调用方后续的分词/城市剥离行为不变）。
    """
    out = str(text or "")
    for _job_type, words in QUERY_TYPE_WORDS:
        for word in words:
            while True:
                lowered = out.lower()
                idx = lowered.find(word)
                if idx < 0:
                    break
                out = out[:idx] + " " + out[idx + len(word):]
    return out


def matches(row_type: Any, wanted: Any, platform: Any = "", title: Any = "") -> bool:
    """过滤判定：`wanted` 为空 → 一律放行；否则要求类型一致。

    行数据里没有类型时（旧库 / 存量 JSON）用 `classify_job_type` 现场推，
    这样即使 `job_type` 还没回填，过滤也立刻生效（**推断口径与打标口径同一份**）。
    """
    want = normalize_job_type(wanted)
    if not want:
        return True
    actual = normalize_job_type(row_type) or classify_job_type(platform, title)
    return actual == want
