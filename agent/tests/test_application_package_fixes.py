# -*- coding: utf-8 -*-
"""投递包三问题修复验证（问题 1 学校/时间丢失 · 问题 2 过度包装 · 问题 3 错别字）。

用法：
    python agent/tests/test_application_package_fixes.py           # 离线确定性检查
    python agent/tests/test_application_package_fixes.py --e2e     # 追加真实 LLM 端到端

离线部分不碰网络；--e2e 会真实解析原始简历 PDF + 让 LLM 定制简历并重新生成
「墨泊可士」投递包，末尾打印新 resume.pdf 的目录与关键行。
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from agent.resume import parser as resume_parser              # noqa: E402
from agent.resume import tailor as resume_tailor              # noqa: E402
from agent.tools import pdf_export                            # noqa: E402
import agent.tools_registry as tools_registry                 # noqa: E402
import agent.storage as storage                               # noqa: E402
from shared.user_context import user_scope                    # noqa: E402

PASSED = 0
FAILED = 0

# 含「上海大学」的原始简历（本机 .files 里那份真身）
ORIGINAL_RESUME = (REPO / ".files" / "8a28231b-27e9-4c2c-aae1-1a723603dd8d"
                   / "341f0010-af1c-4814-aae7-24e91ebdde0b.pdf")
COMPANY = "墨泊可士"
JOB_ID = "inn_ct9k5tle1quk"

PRAISE_WORDS = ("体现", "展现", "展示", "彰显", "自我评价")

# 自夸是「自夸动词 + 能力/素养…」的搭配，不是单个名词 —— 裸词「能力」会把
# 「岗位要求的 Python 编程能力」这类客观表述误判成自夸（实测第 1 次生成就误报过）。
_PRAISE_RE = re.compile(
    r"(?:体现|展现|展示|彰显|证明|说明|反映|表现出)(?:了|出)?[^。；\n]{0,30}?"
    r"(?:能力|素养|精神|意识|习惯|潜力|态度|热情|水平|功底|思维)"
    r"|(?:具备|拥有|注重|善于|擅长)[^。；\n]{0,25}?"
    r"(?:能力|素养|精神|意识|习惯|态度|潜力|热情|水平|功底)"
)


def self_praise_hits(text: str) -> list:
    """命中真实自夸搭配的片段（含裸词自夸词），给断言用。"""
    hits = [w for w in PRAISE_WORDS if w in text]
    hits += [m.group(0) for m in _PRAISE_RE.finditer(text)]
    return hits


def check(label: str, ok: bool, detail: str = ""):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  ✅ {label}")
    else:
        FAILED += 1
        print(f"  ❌ {label} {detail}")


def pdf_text(path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# ==========================================================================
# A. 离线确定性检查
# ==========================================================================

def test_problem2_strip_self_praise():
    print("\n[A1] 问题 2：确定性剥离自我评价")
    cases = [
        ("构建千级文档向量库，检索成本 < 5ms，体现了较强的效果调优能力。",
         ["构建千级文档向量库", "检索成本"], ["体现", "效果调优能力"]),
        ("负责数据清洗与接口联调，展现了出色的学习方法与架构抽象能力，交付 3 个模块。",
         ["负责数据清洗", "交付 3 个模块"], ["展现", "架构抽象能力"]),
        ("完成了商品检索链路优化，具备独立完成工作并持续改进方案的能力。",
         ["完成了商品检索链路优化"], ["具备独立完成", "持续改进方案的能力"]),
        ("出于对 AI Agent 技术领域的浓厚兴趣，主动学习 LangChain。",
         [], ["浓厚兴趣"]),
        ("基于 Milvus 建库 + 混合检索，召回率从 65% 提升至 86%，P95 延迟 120ms。",
         ["召回率从 65% 提升至 86%", "P95 延迟 120ms"], []),
    ]
    for raw, keep, drop in cases:
        got = resume_tailor.strip_self_praise(raw)
        check(f"保留事实：{raw[:22]}…", all(k in got for k in keep), f"→ {got}")
        check(f"删掉自夸：{raw[:22]}…", all(d not in got for d in drop), f"→ {got}")

    result = {"tailored": {"experience": [{"company": "A", "role": "R", "description":
              "搭建评测集，bad case 下降 40%，体现了较强的工程能力。"}],
              "projects": [{"name": "P", "desc": "重写检索，展现出色的架构抽象能力。"}]}}
    cleaned = resume_tailor._sanitize_result(result)
    exp_desc = cleaned["tailored"]["experience"][0]["description"]
    proj_desc = cleaned["tailored"]["projects"][0]["desc"]
    check("description 去自夸", "体现" not in exp_desc and "能力" not in exp_desc, exp_desc)
    check("desc 去自夸", "展现" not in proj_desc and "能力" not in proj_desc, proj_desc)


def test_problem3_tidy():
    print("\n[A2] 问题 3：数字/单位空格清理")
    check("< 5ms → <5ms", pdf_export._tidy("检索成本 < 5ms") == "检索成本 <5ms")
    check("5 ms → 5ms", pdf_export._tidy("延迟 5 ms") == "延迟 5ms")
    check("5m s → 5ms", pdf_export._tidy("延迟 5m s") == "延迟 5ms")
    check("千级 保留", pdf_export._tidy("构建千级文档向量库") == "构建千级文档向量库")
    check("干级 → 千级", pdf_export._tidy("构建干级文档向量库") == "构建千级文档向量库")
    check("干万 → 千万", pdf_export._tidy("日均干万级请求") == "日均千万级请求")
    check("中文正常空格不动", pdf_export._tidy("2025 年 10 月") == "2025 年 10 月")


def test_problem1_blocks():
    print("\n[A3] 问题 1：教育经历/项目时间上版面")
    data = {
        "name": "梁家浩", "city": "广州", "education": "硕士研究生",
        "educations": [{"school": "上海大学", "major": "人工智能",
                        "degree": "硕士研究生", "start": "2025.09", "end": "至今"}],
        "skills": ["Python", "LangChain"],
        "experience": [{"company": "某公司", "role": "算法实习生",
                        "start": "2025.06", "end": "2025.09", "description": "清洗 10 万条数据"}],
        "projects": [{"name": "RAG 多智能体笔记本客服系统", "tech": ["Milvus"],
                      "start": "2025.10", "end": "2026.01",
                      "desc": "构建千级文档向量库，用 < 5ms 的检索成本支撑问答"}],
    }
    text = "\n".join(b.get("text", "") for b in pdf_export.resume_blocks(data))
    check("教育经历小节", "教育经历" in text)
    check("学校名", "上海大学" in text, text)
    check("专业", "人工智能" in text)
    check("教育时间", "2025.09-至今" in text, text)
    check("实习时间", "2025.06-2025.09" in text, text)
    check("项目时间", "2025.10-2026.01" in text, text)
    check("联系方式不再重复学历档位", "教育：硕士研究生" not in text)
    check("千级原样", "千级" in text and "干" not in text)
    check("5ms 连写", "<5ms" in text, text)


def test_problem1_backfill():
    print("\n[A4] 问题 1：定制结果丢字段时从原简历补回")
    orig = {
        "name": "梁家浩", "city": "广州", "education": "硕士研究生",
        "educations": [{"school": "上海大学", "major": "人工智能",
                        "degree": "硕士研究生", "start": "2025.09", "end": "至今"}],
        "skills": ["Python"],
        "experience": [{"company": "某公司", "role": "算法实习生",
                        "start": "2025.06", "end": "2025.09", "description": "d"}],
        "projects": [{"name": "P", "start": "2025.10", "end": "2026.01", "desc": "d"}],
    }
    # LLM 把实习整段吞了、教育只剩档位、项目时间丢了
    new = {"name": "梁家浩", "education": "硕士研究生", "city": "广州",
           "skills": ["LangChain", "Python"],
           "educations": [{"school": "", "major": "", "degree": "", "start": "", "end": ""}],
           "projects": [{"name": "P", "desc": "重写后的描述"}]}
    merged = tools_registry._backfill_from_original(orig, new)
    check("实习整体补回", merged.get("experience") == orig["experience"])
    check("教育明细补回学校", merged["educations"][0]["school"] == "上海大学")
    check("项目 start 补回", merged["projects"][0].get("start") == "2025.10")
    check("项目 end 补回", merged["projects"][0].get("end") == "2026.01")
    check("定制后的描述保留", merged["projects"][0].get("desc") == "重写后的描述")
    check("技能不被原简历覆盖", merged["skills"] == ["LangChain", "Python"])


SAMPLE_RESUME_TEXT = """梁家浩
邮箱：a@b.com  手机：13800000000
现居：广州
【教育背景】
上海大学 | 人工智能（硕士研究生）| 2025.09-至今
【技能】
- 编程语言：Python、LangChain

【实习经历】
2025.06 - 2025.09  某公司  算法实习生
负责接口开发。
【项目经历】
项目：RAG 客服系统
描述：构建干级文档向量库，检索成本 5m s。
"""


def test_problem1_parser_fallback():
    print("\n[A6] 问题 1：parser 确定性教育兜底（LLM 漏字段也能救回）")
    edus = resume_parser.extract_educations(SAMPLE_RESUME_TEXT)
    check("原文正则抓到 1 条教育", len(edus) == 1, str(edus))
    if edus:
        check("school=上海大学", edus[0]["school"] == "上海大学", str(edus[0]))
        check("major=人工智能", edus[0]["major"] == "人工智能", str(edus[0]))
        check("degree 含硕士", "硕士" in edus[0]["degree"], str(edus[0]))
        check("start=2025.09", edus[0]["start"] == "2025.09", str(edus[0]))
        check("end=至今", edus[0]["end"] == "至今", str(edus[0]))
    # LLM 只回了 education=硕士（实测 f9bbd037 就是这个形态）→ 强制回填
    fixed = resume_parser._ensure_educations({"education": "硕士"}, SAMPLE_RESUME_TEXT)
    check("educations 被强制回填", bool(fixed.get("educations")), str(fixed.get("educations")))
    # LLM 本来就给对了 → 不许动它
    keep = resume_parser._ensure_educations(
        {"educations": [{"school": "清华大学", "major": "x", "degree": "本科",
                         "start": "2020.09", "end": "2024.06"}]}, SAMPLE_RESUME_TEXT)
    check("已有 school 时不覆盖", keep["educations"][0]["school"] == "清华大学")


def test_problem1_tailor_restore():
    print("\n[A7] 问题 1：定制结果丢字段时由 tailor 确定性回填")
    src = resume_parser._dict_to_resume({
        "name": "梁家浩", "city": "广州", "education": "硕士研究生",
        "skills": ["Python", "Milvus"],
        "educations": [{"school": "上海大学", "major": "人工智能",
                        "degree": "硕士研究生", "start": "2025.09", "end": "至今"}],
    })
    out = resume_tailor._restore_source_fields(
        {"tailored": {"name": "", "education": "", "skills": [], "city": "",
                      "educations": [], "projects": []}}, src)["tailored"]
    check("姓名补回", out["name"] == "梁家浩", str(out))
    check("城市补回", out["city"] == "广州", str(out))
    check("技能补回", out["skills"] == ["Python", "Milvus"], str(out))
    check("学历档位补回", out["education"] == "硕士研究生", str(out))
    check("教育明细补回", out["educations"][0]["school"] == "上海大学", str(out))
    out2 = resume_tailor._restore_source_fields(
        {"tailored": {"educations": [{"school": "北京大学", "major": "m", "degree": "硕士",
                                      "start": "2021.09", "end": "2024.06"}]}}, src)
    check("定制结果自带 school 时尊重它",
          out2["tailored"]["educations"][0]["school"] == "北京大学")


def test_problem1_library_recover():
    print("\n[A8] 问题 1：原简历自身丢了 educations 时，从同名人旧版本回填")
    fake_name = "重名回填自测-不参与投递"
    with user_scope("admin"):
        good_id = storage.save_resume(fake_name, {
            "name": fake_name, "education": "硕士研究生", "city": "广州",
            "educations": [{"school": "上海大学", "major": "人工智能",
                            "degree": "硕士研究生", "start": "2025.09", "end": "至今"}],
        })
        # 坏的那份：解析时把 educations 丢了（f9bbd037 的真实形态）
        merged = tools_registry._backfill_from_original(
            {"id": "00000000", "name": fake_name, "education": "硕士"}, {"name": fake_name})
    edus = merged.get("educations") or []
    check("从同名版本捞回教育明细",
          any(str(e.get("school")) == "上海大学" for e in edus), f"{edus}（源 {good_id}）")
    # 库里没有同名人版本时不硬编造
    merged2 = tools_registry._backfill_from_original(
        {"name": "查无此人-唯一名字-9f3a", "education": "硕士"}, {"name": "查无此人-唯一名字-9f3a"})
    check("无源时不编造", not merged2.get("educations"), str(merged2.get("educations")))


def test_problem2_tidy_all_paths():
    print("\n[A9] 问题 2：_tidy 在渲染前覆盖所有路径（技能行/联系方式/_plain）")
    data = {"name": "梁家浩", "city": "广州", "phone": "138 0000 0000",
            "education": "硕士", "skills": ["检索 5m s", "干级向量库"],
            "experience": [], "projects": []}
    out = Path(tempfile.gettempdir()) / "dsh_resume_fix_paths.pdf"
    pdf_export.export_resume_pdf(data, str(out))
    text = pdf_text(out)
    check("技能行 5m s → 5ms", "5ms" in text and "5m s" not in text, text)
    check("技能行 干级 → 千级", "千级" in text and "干级" not in text, text)
    check("联系方式数字空格清理", "13800000000" in text, text)

    plain = "梁家浩\n构建干级文档向量库，检索成本 5m s"
    out2 = Path(tempfile.gettempdir()) / "dsh_resume_fix_plain.pdf"
    pdf_export.export_resume_pdf(plain, str(out2))
    text2 = pdf_text(out2)
    check("_plain 路径 干级 → 千级", "千级" in text2 and "干级" not in text2, text2)
    check("_plain 路径 5m s → 5ms", "5ms" in text2 and "5m s" not in text2, text2)


def test_problem3_4_dedup_and_terms():
    print("\n[A10] 问题 3：技能去重 ｜ 问题 4：技术名词纠错")
    data = {"name": "梁家浩", "skills": ["LangChain", "LangGraph", "Agent 开发",
                                         "langgraph", "LangGraph", "Milvus"]}
    text = "、".join(b.get("text", "") for b in pdf_export.resume_blocks(data))
    check("LangGraph 只出现一次", text.count("LangGraph") == 1, text)
    check("其它技能一个不少",
          all(s in text for s in ("LangChain", "Agent 开发", "Milvus")), text)

    check("Llamalndex → LlamaIndex",
          pdf_export.fix_tech_terms("用 Llamalndex 做检索") == "用 LlamaIndex 做检索")
    check("Langchain → LangChain",
          pdf_export.fix_tech_terms("基于 Langchain") == "基于 LangChain")
    check("Fastapi → FastAPI", pdf_export.fix_tech_terms("Fastapi 接口") == "FastAPI 接口")
    check("正常文本不动", pdf_export.fix_tech_terms("检索耗时 5ms") == "检索耗时 5ms")

    real_chat = tools_registry.chat
    try:
        tools_registry.chat = lambda *a, **k: "我熟悉 Llamalndex 和 Langchain，也用过 Milvus。"
        body, _warn = tools_registry._generate_cover_letter(
            {}, {"company": "A", "title": "B"}, None)
    finally:
        tools_registry.chat = real_chat
    check("自荐信路径纠错 LlamaIndex", "LlamaIndex" in body and "Llamalndex" not in body, body)
    check("自荐信路径纠错 LangChain", "LangChain" in body and "Langchain" not in body, body)


def why_this_job_ok(cover: str) -> bool:
    """自荐信里有没有「为什么是这个岗位」的实质句子。

    只按关键词列表判会漏（实测模型写「以 AI Agent 为核心方向，与我的技术路线
    一致，也是我想长期深耕的方向，因此特别希望加入」——一个白名单词都没命中），
    改成「理由词 + 加入意愿词」同现即算达标。
    """
    if any(w in cover for w in ("感兴趣", "兴趣", "吸引我", "想加入", "最想",
                                "愿意", "为什么选")):
        return True
    reasons = ("一致", "契合", "匹配", "对口", "深耕", "正对应", "正是我",
               "符合", "方向", "看重", "看重的是", "因为", "正中")
    intents = ("希望加入", "想加入", "愿意加入", "期待加入", "特别希望", "很想加入",
               "想成为", "希望成", "希望有机会", "渴望加入")
    for sentence in re.split(r"[。！？\n]", cover):
        if any(r in sentence for r in reasons) and any(i in sentence for i in intents):
            return True
    return False


def test_font_glyph():
    print("\n[A5] 问题 3：字体确有「千」字形 + PDF 往返")
    font_path = pdf_export.find_cjk_font()
    check("找到 CJK 字体", bool(font_path), str(font_path))
    font = pdf_export._TrueTypeFont(str(font_path))
    gid_qian = font.gid("千")
    gid_gan = font.gid("干")
    check("「千」有字形", gid_qian not in (0, None), f"gid={gid_qian}")
    check("「千」「干」不是同一个字形", gid_qian != gid_gan, f"{gid_qian} vs {gid_gan}")

    data = {
        "name": "梁家浩", "city": "广州",
        "educations": [{"school": "上海大学", "major": "人工智能",
                        "degree": "硕士研究生", "start": "2025.09", "end": "至今"}],
        "skills": ["Python"],
        "projects": [{"name": "RAG 客服系统", "start": "2025.10", "end": "2026.01",
                      "desc": "构建千级文档向量库，检索成本 < 5ms"}],
    }
    out = Path(tempfile.gettempdir()) / "dsh_resume_fix_offline.pdf"
    pdf_export.export_resume_pdf(data, str(out))
    text = pdf_text(out)
    check("PDF 含「千级」", "千级" in text, text[:200])
    check("PDF 无「干」", "干" not in text)
    check("PDF 含「5ms」", "5ms" in text)
    check("PDF 无「5m s」", "5m s" not in text)
    check("PDF 含学校", "上海大学" in text)
    check("PDF 含教育时间", "2025.09" in text and "至今" in text)
    check("PDF 无缺字形占位 ?", "?" not in text.replace("？", ""))
    print(f"  ℹ️  离线样张：{out}（{out.stat().st_size} B）")


# ==========================================================================
# B. 端到端（真实 LLM）
# ==========================================================================

def test_e2e():
    print("\n[B] 端到端：解析原始简历 → 重新生成墨泊可士投递包")
    if not ORIGINAL_RESUME.exists():
        check(f"原始简历存在：{ORIGINAL_RESUME}", False)
        return

    with user_scope("admin"):
        resume = resume_parser.parse_file(str(ORIGINAL_RESUME))
        data = asdict(resume)
        print(f"  ℹ️  parser 输出 educations={json.dumps(data.get('educations'), ensure_ascii=False)}")
        print("  ℹ️  projects 时间=" + str(
            [(p.get("name"), p.get("start"), p.get("end"))
             for p in (data.get("projects") or [])]))

        edus = data.get("educations") or []
        check("parser 抓到学校名", any(e.get("school") == "上海大学" for e in edus), str(edus))
        check("parser 抓到专业", any("人工智能" in str(e.get("major")) for e in edus), str(edus))
        check("parser 抓到教育起止",
              any(str(e.get("start")).startswith("2025.09") for e in edus), str(edus))
        projs = data.get("projects") or []
        check("parser 抓到项目时间",
              any(str(p.get("start")) and str(p.get("end")) for p in projs),
              str([(p.get("name"), p.get("start"), p.get("end")) for p in projs]))

        rid = storage.save_resume("梁家浩-投递包修复验证", data)
        state = tools_registry._session_state()
        if isinstance(state, dict):
            state["current_resume_id"] = rid
        if not tools_registry.get_current_resume():
            storage.set_default_resume(rid)

        print(f"  ℹ️  当前简历 id={rid}（{storage.get_resume(rid).get('name')}）")
        for run in range(1, 4):
            print(f"\n  ---- 第 {run}/3 次生成 ----")
            result = tools_registry.generate_application_package(COMPANY, JOB_ID)
            print(f"  ℹ️  包目录：{result['package_dir']}")
            for w in result.get("warnings") or []:
                print(f"  ⚠️  {w}")

            text = pdf_text(result["files"]["resume.pdf"])
            if run == 1:
                print("  ---- resume.pdf 正文节选 ----")
                for line in [ln for ln in text.splitlines() if ln.strip()][:24]:
                    print(f"    {line}")

            check(f"[第{run}次] 教育经历整段在（上海大学）", "上海大学" in text, text[:300])
            check(f"[第{run}次] 专业在（人工智能）", "人工智能" in text, text[:300])
            check(f"[第{run}次] 教育时间在（2025.09-至今）",
                  "2025.09" in text and "至今" in text, text[:300])
            check(f"[第{run}次] 无「干级」", "干级" not in text)
            check(f"[第{run}次] 无「5m s」", "5m s" not in text)
            check(f"[第{run}次] 无「5 ms」", "5 ms" not in text)
            check(f"[第{run}次] LangGraph 只出现一次",
                  text.count("LangGraph") == 1, f"出现了 {text.count('LangGraph')} 次")
            found = self_praise_hits(text)
            check(f"[第{run}次] 无主观自我评价", not found, f"命中：{found}")

            cover = Path(result["files"]["cover_letter.md"]).read_text(encoding="utf-8")
            found_cover = self_praise_hits(cover)
            check(f"[第{run}次] 自荐信无主观自我评价", not found_cover, f"命中：{found_cover}")
            check(f"[第{run}次] 自荐信无错拼 Llamalndex", "Llamalndex" not in cover)
            if run == 1:
                cliche = [w for w in ("平台大", "发展前景好", "前景广阔", "氛围好", "重视人才",
                                      "行业领先", "大厂") if w in cover]
                check("自荐信无空泛套话", not cliche, f"命中：{cliche}")
                check("自荐信有「为什么这个岗位」", why_this_job_ok(cover),
                      f"正文末尾：{cover[-120:]}")
                print("  ---- cover_letter.md 末尾 ----")
                for line in [ln for ln in cover.splitlines() if ln.strip()][-4:]:
                    print(f"    {line}")
                print("  ---- cover_letter.md 全文 ----")
                print(cover)
            print(f"  ℹ️  产物体积：resume.pdf {Path(result['files']['resume.pdf']).stat().st_size} B"
                  f" / cover_letter.md {len(cover)} 字")


def main() -> int:
    print("=" * 68)
    print("投递包三问题修复验证")
    print("=" * 68)
    test_problem2_strip_self_praise()
    test_problem3_tidy()
    test_problem1_blocks()
    test_problem1_backfill()
    test_problem1_parser_fallback()
    test_problem1_tailor_restore()
    test_problem1_library_recover()
    test_problem2_tidy_all_paths()
    test_problem3_4_dedup_and_terms()
    test_font_glyph()
    if "--e2e" in sys.argv:
        test_e2e()
    print("\n" + "=" * 68)
    print(f"结果：{PASSED} 通过 / {FAILED} 失败")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
