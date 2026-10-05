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

PRAISE_WORDS = ("体现", "展现", "展示", "彰显", "能力", "素养", "精神",
                "浓厚兴趣", "自我评价")


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

        result = tools_registry.generate_application_package(COMPANY, JOB_ID)
        print(f"  ℹ️  包目录：{result['package_dir']}")
        for w in result.get("warnings") or []:
            print(f"  ⚠️  {w}")

        text = pdf_text(result["files"]["resume.pdf"])
        print("  ---- resume.pdf 正文节选 ----")
        for line in [ln for ln in text.splitlines() if ln.strip()][:24]:
            print(f"    {line}")

        check("PDF 含「上海大学」", "上海大学" in text)
        check("PDF 含「人工智能」", "人工智能" in text)
        check("PDF 含教育时间 2025.09", "2025.09" in text)
        check("PDF 含「至今」", "至今" in text)
        check("PDF 含项目时间 2025.10", "2025.10" in text)
        check("PDF 含项目结束 2026.01", "2026.01" in text)
        check("PDF 含「千级」", "千级" in text)
        check("PDF 无「干」", "干" not in text)
        check("PDF 无「5m s」", "5m s" not in text)
        check("PDF 无「5 ms」", "5 ms" not in text)
        found = [w for w in PRAISE_WORDS if w in text]
        check("PDF 无主观自我评价", not found, f"命中：{found}")

        cover = Path(result["files"]["cover_letter.md"]).read_text(encoding="utf-8")
        found_cover = [w for w in PRAISE_WORDS if w in cover]
        check("自荐信无主观自我评价", not found_cover, f"命中：{found_cover}")
        # 「为什么这家公司」：必须有一句基于 JD 具体信息、且不是放之四海皆准的套话
        cliche = [w for w in ("平台大", "发展前景好", "前景广阔", "氛围好", "重视人才",
                              "行业领先", "大厂") if w in cover]
        check("自荐信无空泛套话", not cliche, f"命中：{cliche}")
        check("自荐信有「为什么这个岗位」",
              any(w in cover for w in ("感兴趣", "兴趣", "吸引我", "想加入",
                                       "最想", "愿意", "为什么选")),
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
    test_font_glyph()
    if "--e2e" in sys.argv:
        test_e2e()
    print("\n" + "=" * 68)
    print(f"结果：{PASSED} 通过 / {FAILED} 失败")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
