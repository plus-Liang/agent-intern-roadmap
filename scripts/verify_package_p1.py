# -*- coding: utf-8 -*-
"""投递包 P1 修复的离线复核（不打网络、不真调 LLM）。

复现的真实故障（同一段粘贴的实习僧 JD，包目录 unknown_20261012_133216）：
  Bug 1  cover_letter.md 空 / 残句，流程报成功；
  Bug 2  resume.pdf 正文 447 字 vs 原简历约 1100 字（5 条 bullet → 3 条）。

用法：python scripts/verify_package_p1.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent.tools import pdf_export                              # noqa: E402
import agent.tools_registry as reg                              # noqa: E402
from agent.tools.job_detail import JobDetail                    # noqa: E402
from shared.user_context import user_scope                      # noqa: E402

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(("  ✅ " if ok else "  ❌ ") + label + (f"  {detail}" if detail and not ok else ""))


RESUME = json.loads((REPO / "agent/data/resumes/admin/07e83f22.json")
                    .read_text(encoding="utf-8"))["content"]
#: 原简历里 5 个大类的标题（用户说「丢失」的就是这几个关键词）
BIG_LABELS = ["数据层", "RAG 检索", "Agent 编排", "成本与稳定性", "质量工程"]

DETAIL = JobDetail(
    platform="pasted", job_id="pasted_verify0001", title="AI应用开发（可转正）",
    company="", city="广州", salary="200-400/天", education="本科", url="",
    description="1.负责AI技术在实际应用场景中的研发与落地；",
    requirements="1.本科及以上学历，计算机、AI 相关专业；",
    bonus="", tags=[], days_per_week="5天／周", duration="实习3个月",
)

LONG_COVER = (
    "尊敬的招聘经理：\n\n"
    "您好！我在实习僧上看到贵公司「AI应用开发（可转正）」岗位，岗位要求熟悉 "
    "Python 与 AI Agent 技术，与我这两年的方向一致，因此特别希望加入。\n\n"
    "我在求职 Agent 项目里用 Playwright 抓取三个平台数据，3×9城×59关键词=1593 "
    "组合滚动抓取，清洗去重后入库2600+条；检索用 BM25 + 向量混合检索 + RRF 融合，"
    "MRR 0.809；编排用 LangGraph 工作流与反思节点，评测发现并修复 6 个真实 bug。\n\n"
    "期待有机会面试，向您进一步说明这些实践。\n\n此致\n敬礼\n梁家浩"
)


def pdf_text(path) -> str:
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def main() -> int:
    tmp = Path(REPO / "agent/data/packages/_verify_p1")
    tmp.mkdir(parents=True, exist_ok=True)

    print("\n[Bug 2] 要点保真：原简历 5 个大类必须都在 PDF 里")
    src_len = reg._resume_text_len(RESUME)
    out_pdf = tmp / "resume_full.pdf"
    pdf_export.export_resume_pdf(RESUME, str(out_pdf))
    text = pdf_text(out_pdf)
    got_len = reg._pdf_text_len(out_pdf)
    bullets = pdf_export._split_points(RESUME["projects"][0]["desc"])
    check("每条大类标题都在 PDF 里",
          all(lbl in text for lbl in BIG_LABELS),
          str([lbl for lbl in BIG_LABELS if lbl not in text]))
    check("bullet 不再被砍到 3 条", len(bullets) >= 5, f"{len(bullets)} 条")
    check("PDF 正文长度 ≥ 原简历 80%", got_len >= src_len * 0.8,
          f"{got_len} vs {src_len}（{got_len / src_len:.0%}）")
    check("_resume_findings 对完整简历不报警",
          not reg._resume_findings(RESUME, RESUME, out_pdf))

    # 模拟「LLM 把项目描述压成 1 句」→ 校验必须报警
    shrunk = json.loads(json.dumps(RESUME, ensure_ascii=False))
    shrunk["projects"][0]["desc"] = "用 LangGraph 做了个求职助手。"
    short_pdf = tmp / "resume_shrunk.pdf"
    pdf_export.export_resume_pdf(shrunk, str(short_pdf))
    check("内容缩水时报警（Bug 2 的看门狗）",
          bool(reg._resume_findings(RESUME, shrunk, short_pdf)))

    print("\n[Bug 1] 自荐信：空 / 残句不许静默通过")
    check("残句被判不合格", bool(reg._cover_body_problems("我熟悉 LangChain。", "梁家浩")))
    check("完整信被判合格", not reg._cover_body_problems(LONG_COVER, "梁家浩"))
    check("缺称呼被抓出",
          any("称呼" in p for p in reg._cover_body_problems(
              "我做过 RAG 项目，量很大。" * 20, "梁家浩")))
    short_cover = tmp / "cover_short.md"
    short_cover.write_text("# 自荐信\n\n我熟悉 LangChain。\n", encoding="utf-8")
    check("过短文件落盘后被抓出",
          any("过短" in w for w in reg._cover_letter_findings(short_cover, RESUME, "梁家浩")))
    good_cover = tmp / "cover_ok.md"
    good_cover.write_text(LONG_COVER, encoding="utf-8")
    check("合格文件读回后无告警",
          not reg._cover_letter_findings(good_cover, RESUME, "梁家浩"),
          str(reg._cover_letter_findings(good_cover, RESUME, "梁家浩")))
    check("自荐信能对上简历经历", bool(reg._cover_match(LONG_COVER, RESUME)))

    print("\n[P2] 公司名兜底 + 薪资不再瞎填")
    check("公司名为空 → 未知公司", reg._company_label({}, DETAIL) == "未知公司")
    check("岗位详情有公司名时优先用它",
          reg._company_label({"company": "旧记录"}, JobDetail(
              platform="mock", job_id="x", title="t", company="墨泊可士", city="广州",
              salary="", url="")) == "墨泊可士")
    salary = re.search(r"薪资：(.+)", reg._job_info_text(
        {"company": "", "title": "AI应用开发（可转正）"}, DETAIL, "pasted_x"))
    check("job_info 薪资 = 200-400/天", "200-400/天" in salary.group(1), salary.group(1))
    check("job_info 公司名给出可操作提示",
          "没有写明公司名" in reg._job_info_text({"company": "", "title": "t"}, DETAIL, ""))
    check("公司名空时目录名不再叫 unknown",
          reg._safe_name(reg._company_label({}, DETAIL)) == "未知公司")

    print("\n[端到端] 粘贴 JD（无公司名）走一遍 generate_application_package")
    detail = JobDetail(**{**DETAIL.__dict__, "company": ""})
    with user_scope("admin"):
        originals = (reg.storage.find_application, reg.get_current_resume,
                     reg._record_from_job_library, reg.chat, reg.PACKAGE_DIR,
                     reg._tailor_resume)
        reg.storage.find_application = lambda company: None
        reg.get_current_resume = lambda: {"id": "verify", "content": RESUME}
        reg._record_from_job_library = lambda c, jid=None, t="": (
            {"id": "", "company": "", "title": "AI应用开发（可转正）", "url": "",
             "platform": "pasted", "status": "", "applied_at": ""}, detail)
        reg.chat = lambda *a, **k: LONG_COVER               # 假装 LLM 正常
        reg.PACKAGE_DIR = tmp / "packages"
        reg._tailor_resume = lambda data, d: (
            json.loads(json.dumps(data, ensure_ascii=False)), [])   # 假装定制没丢内容
        try:
            out = reg.generate_application_package("", detail.job_id)
        finally:
            (reg.storage.find_application, reg.get_current_resume,
             reg._record_from_job_library, reg.chat, reg.PACKAGE_DIR,
             reg._tailor_resume) = originals

    cover = Path(out["files"]["cover_letter.md"]).read_text(encoding="utf-8")
    pkg_text = pdf_text(out["files"]["resume.pdf"])
    print(f"  ℹ️  包目录：{Path(out['package_dir']).name}")
    print(f"  ℹ️  cover_letter.md {len(cover.encode('utf-8'))} B / resume.pdf 正文 "
          f"{reg._pdf_text_len(out['files']['resume.pdf'])} 字（原简历 {src_len} 字）")
    print(f"  ℹ️  warnings：{out.get('warnings')}")
    check("包目录不再叫 unknown_*",
          not Path(out["package_dir"]).name.startswith("unknown"),
          Path(out["package_dir"]).name)
    check("cover_letter.md > 500 字节", len(cover.encode("utf-8")) > 500,
          f"{len(cover.encode('utf-8'))} B")
    check("cover_letter.md 有称呼", "尊敬的" in cover)
    check("cover_letter.md 有经历匹配", bool(reg._cover_match(cover, RESUME)))
    check("cover_letter.md 有署名", "梁家浩" in cover and "敬礼" in cover)
    check("resume.pdf 5 个大类都在", all(lbl in pkg_text for lbl in BIG_LABELS),
          str([lbl for lbl in BIG_LABELS if lbl not in pkg_text]))
    check("resume.pdf 正文 ≥ 原简历 80%",
          reg._pdf_text_len(out["files"]["resume.pdf"]) >= src_len * 0.8)
    check("job_info 薪资正确", "薪资：200-400/天" in Path(out["files"]["job_info.txt"])
          .read_text(encoding="utf-8"))

    print("\n" + "=" * 60)
    print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
    for item in FAIL:
        print(f"  ❌ {item}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
