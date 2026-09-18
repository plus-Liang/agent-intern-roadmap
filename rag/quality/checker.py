# -*- coding: utf-8 -*-
"""
JD 数据质量校验模块（只检测，不修复）

校验对象：
  - rag/data/scraped_jd.txt     JD 原始文本（真实入库数据），每条以「【数字】公司：」开头，
                                正文里有【岗位职责】/【任职要求】/【加分项】等段落标题
  - rag/data/jd_structured.json 结构化字段（可选，需显式传入），含 company/salary_min/salary_max/
                                city/education。真实数据没有这个文件，默认不校验结构化字段，
                                只跑文本校验；仓库里残留的那份是旧 mock 数据，公司名与
                                scraped_jd.txt 对不上，自动加载只会把 15 条全判失败。

检测项：
  a) 学历冲突：头部「学历：X」与正文「任职要求」段落里的学历要求不一致
  b) 薪资冲突：头部「薪资：X-Y/天」与 structured 里的 salary_min/salary_max 不一致
  c) 城市冲突：头部「城市：X」与 structured 里的 city 不一致
  d) 字段缺失：structured 里 city/education/salary_min/salary_max 有空的
  e) 正文缺失：jd_text 里找不到「岗位职责」或「任职要求」段落

问题分级（见 CHECK_SEVERITY）：
  errors   严重问题 —— 字段缺失、正文缺失，会让 passed=False
  warnings 提示问题 —— 头部与正文学历表述不一致，只提示，不影响 passed
  passed 只由 errors 决定：没有 errors 就是通过，warnings 不参与判定。
  没有 structured 文件时 b/c/d 三项无法执行，记在 report["skipped"] 里，不计入
  errors，也不影响 passed。

依赖：仅标准库 re / json / sys / pathlib。
"""

import json
import re
import sys
from pathlib import Path

# 本文件所在目录（rag/quality），以及 rag 目录
QUALITY_DIR = Path(__file__).resolve().parent
RAG_DIR = QUALITY_DIR.parent

# 默认数据源：真实入库的 JD 文本；结构化字段可选
DEFAULT_JD_FILE = RAG_DIR / "data" / "scraped_jd.txt"
DEFAULT_STRUCTURED_FILE = RAG_DIR / "data" / "jd_structured.json"

# 「【1】公司：阶跃星辰」——块起始行；要求行内是 【数字】公司：，以排除末尾的说明段落
BLOCK_RE = re.compile(r"^【\d+】\s*公司[:：]\s*(?P<company>\S+?)\s*$", re.MULTILINE)

# 头部字段：用 [^\S\n]* 而不是 \s*，避免跨行匹配
_WS = r"[^\S\n]*"
HEADER_CITY_RE = re.compile(r"城市" + _WS + r"[:：]" + _WS + r"(?P<city>[^｜|/\s]+)")
HEADER_EDU_RE = re.compile(r"学历" + _WS + r"[:：]" + _WS + r"(?P<edu>[^｜|/\s]+)")
HEADER_SALARY_RE = re.compile(r"薪资" + _WS + r"[:：]" + _WS + r"(?P<salary>[^｜|/\n]+)")

# 段落标题别名。同一段落可以有多个别名（如「任职要求」/「职位要求」）。
SECTION_ALIASES = {
    "duty": ("岗位职责", "工作职责", "职位描述", "工作内容", "岗位描述"),
    "requirement": ("任职要求", "职位要求", "岗位要求", "任职资格", "岗位需求"),
}
# 出现这些标题说明当前段落结束
SECTION_BREAK_KEYWORDS = (
    "加分项", "岗位福利", "实习时长", "岗位标签", "工作地点", "页面刷新",
    "来源链接", "投递截止", "备注",
)

# 学历描述 -> 归一化名称，按优先级排列
_EDU_PATTERNS = (
    ("博士", r"博士"),
    ("硕士", r"硕士|研究生"),
    ("本科", r"本科|学士"),
    ("专科", r"专科|大专"),
    # 头部「学历：不限」提取到的是裸词「不限」，正文里则可能写成「学历不限 / 不限学历 / 专业不限」
    ("不限", r"不限"),
)
_MISSING = object()

# ---------------------------------------------------------------------------
# 问题分级表：error 计入 passed=False，warning 只提示、不影响通过。
# 每个检测项的级别集中在这里声明，调整分级只需要改这一处。
# ---------------------------------------------------------------------------
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

CHECK_SEVERITY = {
    # 头部「不限」vs 正文「本科/硕士」、头部「本科」vs 正文「硕士」等
    # 表述宽严不一致的情况：属于提示，不判失败
    "education_conflict": SEVERITY_WARNING,
    # 头部与结构化字段（salary/city）不一致：本次未要求降级，保持原有语义
    "salary_conflict": SEVERITY_ERROR,
    "city_conflict": SEVERITY_ERROR,
    # 字段缺失 / 正文缺失 / 结构化记录缺失：严重问题
    "missing_field": SEVERITY_ERROR,
    "missing_section": SEVERITY_ERROR,
    "missing_record": SEVERITY_ERROR,
    # 没有 jd_structured.json 时结构化校验无法执行：不是问题，只登记跳过
    "check_skipped": SEVERITY_WARNING,
}


def _severity_of(kind):
    """查某个检测项的分级；未登记的检测项按严重问题处理。"""
    return CHECK_SEVERITY.get(kind, SEVERITY_ERROR)


def _record(errors, warnings, kind, message):
    """按 kind 的分级把 message 放进 errors 或 warnings；message 为空则忽略。"""
    if not message:
        return
    if _severity_of(kind) == SEVERITY_WARNING:
        warnings.append(message)
    else:
        errors.append(message)


def _norm_edu(value):
    """把一段文本里的学历要求归一化，取**最先出现**的那一档。

    取最先出现而不是最高档：「硕士/博士在读」这类写法表达的是「最低硕士」，
    若取最高档会得到「博士」，从而把「头部 硕士 + 正文 硕士/博士在读」误判成冲突。
    头部「学历：X」提取到的本身就是单值（不限/本科/硕士），同样适用。
    """
    if not value:
        return ""
    best = None  # (出现位置, 名称)
    for name, pattern in _EDU_PATTERNS:
        match = re.search(pattern, value)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), name)
    return best[1] if best else ""


def _norm_company(name):
    """公司名归一化，用于跨文件匹配。

    文本里的「索尼（中国）」「万兴科技Wondershare」与 JSON 里的「索尼」「万兴科技」指同一家，
    因此去掉括号注释、空格、常见公司后缀，并只保留中文与字母数字后比较。
    """
    if not name:
        return ""
    text = re.sub(r"[（(【\[].*?[)）】\]]", "", str(name))
    text = re.sub(r"[\s·・,，。.、_\-]+", "", text)
    for suffix in ("有限责任公司", "股份有限公司", "有限公司", "公司"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    return text


def _extract_header(text, pattern):
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def parse_header_salary(raw):
    """解析头部薪资文本，返回 (min, max)；无法解析返回 None。

    「500-1000/天」-> (500, 1000)；「200/天」-> (200, 200)
    """
    if not raw:
        return None
    nums = re.findall(r"\d+", raw)
    if not nums:
        return None
    values = [int(n) for n in nums[:2]]
    if len(values) == 1:
        return values[0], values[0]
    return values[0], values[1]


def _extract_required_education(text):
    """从「任职要求」段落里提取学历要求，返回归一化字符串；未提及返回 None。"""
    section = extract_section(text, "requirement")
    if section is None:
        return None
    normalized = _norm_edu(section)
    return normalized or None


def _strip_marks(line):
    return re.sub(r"[\s:：、.。,，\-—【】\[\]（）()]+", "", line)


def _is_section_header(line, keywords):
    """判断一行是不是段落标题，如「任职要求」「【岗位职责】」「职位要求：」。"""
    stripped = _strip_marks(line)
    for keyword in keywords:
        if stripped == keyword:
            return True
        if stripped in (keyword + "需求定义", keyword + "描述", keyword + "详情"):
            return True
    return False


def _is_section_break(line):
    """判断一行是不是「另一个段落」的标题（用于结束当前段落）。"""
    stripped = _strip_marks(line)
    if not stripped:
        return False
    if set(stripped) <= {"-", "="}:
        return True
    for keyword in SECTION_BREAK_KEYWORDS:
        if stripped.startswith(keyword):
            return True
    for aliases in SECTION_ALIASES.values():
        if _is_section_header(line, aliases):
            return True
    return False


def extract_section(jd_text, section):
    """提取 jd_text 里指定段落的正文；找不到返回 None。"""
    if not jd_text or section not in SECTION_ALIASES:
        return None
    aliases = SECTION_ALIASES[section]
    lines = jd_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if _is_section_header(line, aliases):
            start = index + 1
            break
    if start is None:
        return None
    collected = []
    for line in lines[start:]:
        if _is_section_break(line):
            break
        collected.append(line)
    return "\n".join(collected).strip()


def check_missing_fields(structured):
    """检测 structured 里必填字段缺失（city/education/salary_min/salary_max）。

    返回 messages 列表，由调用方按 CHECK_SEVERITY["missing_field"] 分级。
    """
    issues = []
    for field in ("city", "education", "salary_min", "salary_max"):
        value = structured.get(field, _MISSING) if isinstance(structured, dict) else _MISSING
        if value is _MISSING or value is None or (isinstance(value, str) and not value.strip()):
            issues.append("字段缺失：%s" % field)
    return issues


def check_education_conflict(header_edu, body_edu):
    """头部学历与正文学历冲突检测（任一侧缺失时不报冲突）。

    返回提示级 message（见 CHECK_SEVERITY["education_conflict"]）：头部「不限」vs
    正文「本科/硕士」、头部「本科」vs 正文「硕士」都只算 warning，不判失败。
    """
    if not header_edu or not body_edu:
        return None
    if header_edu == body_edu:
        return None
    return "学历冲突：头部'%s'，正文'%s'" % (header_edu, body_edu)


def check_salary_conflict(header_salary, structured):
    """头部薪资与 structured 薪资冲突检测。"""
    parsed = parse_header_salary(header_salary)
    if parsed is None:
        return None
    header_min, header_max = parsed
    struct_min = structured.get("salary_min") if isinstance(structured, dict) else None
    struct_max = structured.get("salary_max") if isinstance(structured, dict) else None
    if struct_min is None or struct_max is None:
        return None
    if (header_min, header_max) != (struct_min, struct_max):
        return "薪资冲突：头部'%s-%s'，结构化'%s-%s'" % (
            header_min, header_max, struct_min, struct_max,
        )
    return None


def check_city_conflict(header_city, structured):
    """头部城市与 structured 城市冲突检测。"""
    struct_city = structured.get("city") if isinstance(structured, dict) else None
    if not header_city or not struct_city:
        return None
    if str(header_city).strip() != str(struct_city).strip():
        return "城市冲突：头部'%s'，结构化'%s'" % (header_city, struct_city)
    return None


def skipped_checks():
    """没有 structured 数据时无法执行的检测项名称。"""
    return ["字段缺失", "薪资冲突", "城市冲突"]


# 「跳过校验」提示的统一前缀：控制台汇总时按前缀折叠，report.json 里仍逐条保留
SKIP_PREFIX = "跳过校验（无 jd_structured.json）："


def skipped_message(name):
    """构造一条跳过校验的提示文本。"""
    return SKIP_PREFIX + name


def check_jd_quality(jd_text: str, structured: dict | None) -> dict:
    """校验单条 JD。

    参数：
      jd_text     单条 JD 的原始文本块（含「【n】公司：」块头那一行）
      structured  结构化字段 dict；传 None 表示没有结构化数据，
                  薪资/城市/字段缺失三项不做判定

    返回 {passed: bool, errors: list[str], warnings: list[str]}
      - errors   严重问题（字段缺失、正文缺失、薪资/城市不一致）
      - warnings 提示问题（学历表述不一致）
      - passed   等价于 not errors，warnings 不影响通过
    """
    errors = []
    warnings = []

    header_edu = _norm_edu(_extract_header(jd_text, HEADER_EDU_RE))
    body_edu = _extract_required_education(jd_text)

    # a) 学历冲突 -> warning（头部与正文表述不一致，只提示）
    _record(errors, warnings, "education_conflict",
            check_education_conflict(header_edu, body_edu))

    if structured is None:
        # 真实数据没有 jd_structured.json：b) 薪资冲突 c) 城市冲突 d) 字段缺失
        # 三项无从校验，跳过。正文缺失（e）不依赖 structured，照常校验。
        for message in skipped_checks():
            _record(errors, warnings, "check_skipped", skipped_message(message))
    else:
        # b) 薪资冲突
        header_salary = _extract_header(jd_text, HEADER_SALARY_RE)
        _record(errors, warnings, "salary_conflict",
                check_salary_conflict(header_salary, structured))

        # c) 城市冲突
        header_city = _extract_header(jd_text, HEADER_CITY_RE)
        _record(errors, warnings, "city_conflict",
                check_city_conflict(header_city, structured))

        # d) 字段缺失
        for message in check_missing_fields(structured):
            _record(errors, warnings, "missing_field", message)

    # e) 正文缺失
    for section, label in (("duty", "岗位职责"), ("requirement", "任职要求")):
        if extract_section(jd_text, section) is None:
            _record(errors, warnings, "missing_section",
                    "正文缺失：找不到「%s」段落" % label)

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def load_jd_texts(jd_file):
    """读取 JD 文本文件（默认 scraped_jd.txt），按「【n】公司：」切块。

    返回 [(company, block_text), ...]；block_text 含块头那一行。
    """
    text = Path(jd_file).read_text(encoding="utf-8")
    matches = list(BLOCK_RE.finditer(text))
    if not matches:
        raise ValueError("未在 %s 中匹配到任何 JD 块（期望行形如「【1】公司：阶跃星辰」）" % jd_file)
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match.group("company").strip(), text[match.start():end]))
    return blocks


def load_structured(structured_file):
    """读取 jd_structured.json。

    返回 (records, index)：
      records  [record, ...]                 —— 保序，供兜底匹配遍历
      index    {归一化公司名: record}         —— 精确匹配用（同名归一化冲突保留先出现的）
    """
    data = json.loads(Path(structured_file).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    records = []
    index = {}
    for record in data:
        if not isinstance(record, dict):
            continue
        company = str(record.get("company", "")).strip()
        if not company:
            continue
        records.append(record)
        index.setdefault(_norm_company(company), record)
    return records, index


def _match_structured(company, records, index):
    """把文本里的公司名匹配到结构化记录，返回 (record, 匹配方式)。

    三级匹配：
      1. 归一化后完全相等
      2. 归一化后互为前缀（如文本「万兴科技Wondershare」vs JSON「万兴科技」）
    两侧都要求长度 >= 2，避免单字造成误匹配。
    """
    key = _norm_company(company)
    if key in index:
        return index[key], "normalized"
    for record in records:
        other = _norm_company(record.get("company", ""))
        if (len(key) >= 2 and len(other) >= 2
                and (key.startswith(other) or other.startswith(key))):
            return record, "prefix"
    return None, None


def check_all(jd_file: str, structured_file: str | None = None) -> dict:
    """
    读 JD 文本（和可选的结构化字段），逐条校验。

    参数：
      jd_file          JD 文本路径，默认 rag/data/scraped_jd.txt
      structured_file  结构化 JSON 路径；传 None 或文件不存在时，
                       薪资/城市/字段缺失三项整体跳过

    返回：
    {
      "total": 15,
      "passed": 13,            # 没有 errors 的条数
      "failed": 2,             # 有 errors 的条数
      "error_count": 2,        # 所有条目 errors 条数之和
      "warning_count": 45,     # 所有条目 warnings 条数之和
      "structured_used": false,# 是否用上了结构化字段
      "skipped": ["字段缺失", "薪资冲突", "城市冲突"],  # 没用结构化字段时被跳过的检测项
      "details": [
        {
          "company": "信投智联科技",
          "passed": true,
          "errors": [],
          "warnings": ["跳过校验（无 jd_structured.json）：薪资冲突"]
        },
        ...
      ]
    }
    """
    blocks = load_jd_texts(jd_file)
    structured_used = bool(structured_file) and Path(structured_file).is_file()
    if structured_used:
        records, index = load_structured(structured_file)
    else:
        records, index = [], {}

    details = []
    for company, block_text in blocks:
        if not structured_used:
            # 没有结构化数据：直接做文本校验
            result = check_jd_quality(block_text, None)
            details.append({
                "company": company,
                "passed": result["passed"],
                "errors": result["errors"],
                "warnings": result["warnings"],
            })
            continue

        structured, matched_by = _match_structured(company, records, index)
        if structured is None:
            details.append({
                "company": company,
                "passed": False,
                "errors": ["字段缺失：jd_structured.json 中找不到该公司记录"],
                "warnings": [],
            })
            continue
        result = check_jd_quality(block_text, structured)
        item = {
            "company": company,
            "passed": result["passed"],
            "errors": result["errors"],
            "warnings": result["warnings"],
        }
        if company != structured.get("company"):
            item["note"] = "公司名经归一化匹配到结构化记录：%s" % structured.get("company")
        details.append(item)

    passed = sum(1 for item in details if item["passed"])
    error_count = sum(len(item["errors"]) for item in details)
    warning_count = sum(len(item["warnings"]) for item in details)
    return {
        "total": len(details),
        "passed": passed,
        "failed": len(details) - passed,
        "error_count": error_count,
        "warning_count": warning_count,
        "structured_used": structured_used,
        "skipped": skipped_checks() if not structured_used else [],
        "details": details,
    }


def _print_items(items):
    """按 company 分组打印某一级别的明细；items 为 [(company, [message, ...]), ...]。"""
    printed = False
    for company, messages in items:
        if not messages:
            continue
        printed = True
        print("\n【%s】" % company)
        for message in messages:
            print("  - %s" % message)
    if not printed:
        print("  （无）")


def _print_report(report, jd_file, structured_file, structured_used, report_file):
    """打印校验汇总与明细。

    「跳过校验」提示在控制台折叠成一行（15 条 JD × 3 项会刷屏），
    report.json 里仍然逐条保留。
    """
    details = report["details"]
    skip_count = sum(
        1 for item in details for message in item["warnings"]
        if message.startswith(SKIP_PREFIX)
    )
    real_warnings = [
        (item["company"], [m for m in item["warnings"] if not m.startswith(SKIP_PREFIX)])
        for item in details
    ]

    print("=" * 60)
    print("JD 数据质量校验汇总")
    print("=" * 60)
    print("JD 文本：%s" % jd_file)
    print("结构化：%s" % (structured_file if structured_used else "（无，结构化校验已跳过）"))
    print("-" * 60)
    print("总数：%d" % report["total"])
    print("通过：%d" % report["passed"])
    print("失败：%d" % report["failed"])
    print("errors（严重，计入失败）：%d" % report["error_count"])
    print("warnings（提示，不影响通过）：%d" % report["warning_count"])

    if not structured_used:
        print("-" * 60)
        print("跳过校验（%d 条）——缺结构化文件，无法比对：%s"
              % (skip_count, "、".join(report["skipped"])))
        print("  提示：把结构化 JSON 作为第二个参数传入即可启用这三项校验")

    print("-" * 60)
    print("errors 明细（严重问题）：")
    _print_items([(item["company"], item["errors"]) for item in details])

    print("-" * 60)
    print("warnings 明细（提示问题）：")
    _print_items(real_warnings)

    print("-" * 60)
    print("报告已保存：%s" % report_file)
    return report


def main(argv=None):
    """命令行入口。

    用法：
      python -m rag.quality.checker                       # 只校验 scraped_jd.txt 的文本项
      python -m rag.quality.checker <jd.txt> <struct.json> # 额外校验薪资/城市/字段缺失
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    jd_file = Path(argv[0]) if argv else DEFAULT_JD_FILE
    # 结构化字段必须显式传入：真实数据没有 jd_structured.json，
    # 仓库里残留的那份是旧 mock 数据，自动加载会把 15 条真实 JD 全判失败
    structured_file = Path(argv[1]) if len(argv) > 1 else None
    report_file = QUALITY_DIR / "report.json"

    structured_used = bool(structured_file) and structured_file.is_file()
    if structured_file and not structured_used:
        print("提示：结构化文件不存在，已跳过结构化校验：%s" % structured_file)

    report = check_all(
        str(jd_file),
        str(structured_file) if structured_used else None,
    )

    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return _print_report(report, jd_file, structured_file, structured_used, report_file)


if __name__ == "__main__":
    main()
