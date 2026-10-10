# -*- coding: utf-8 -*-
"""Chainlit 简历附件上传（PDF / Word 提取）离线回归测试。

跑法：
    python agent/tests/test_resume_upload.py

**不联网、不调 LLM、不起 Chainlit、不碰真实数据**：
所有样本都在仓库内 `agent/tests/_tmp_resume_upload/` 下现场生成，跑完删掉。

PDF 样本是**手工拼的最小 PDF 字节**（标准 14 号字体 + 文本流），
不依赖 reportlab 之类写库，`pypdf` 能真解析出文字 —— 这样测的是
「PDF 提取」这条路本身，而不是某个写库回读自己的自证。
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_TMP_DIR = Path(os.getenv("RESUME_UPLOAD_TEST_DIR",
                          str(Path(__file__).resolve().parent / "_tmp_resume_upload")))

from agent.resume import extractor as EX               # noqa: E402

PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f" → {detail}" if detail else ""))
    except Exception as exc:                                   # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {label} → {type(exc).__name__}: {exc}")


def section(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def _fail(msg):
    raise AssertionError(msg)


def _expect(cond, msg):
    if not cond:
        _fail(msg)


# ---------------------------------------------------------------------------
# 样本生成
# ---------------------------------------------------------------------------

def _pdf_escape(line: str) -> bytes:
    """PDF 字面字符串：必须用圆括号包起来，反斜杠 / 括号要转义。

    ⚠️ 踩过的坑：写成 `A Tj`（裸词）pypdf 提取结果是空字符串、不报错，
    样本会"看起来生成成功但永远提取不出文字"，必须带 `(...)`。
    """
    body = line.encode("latin-1")
    body = body.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    return b"(" + body + b")"


def _pdf_bytes(*lines: str) -> bytes:
    """手工拼一份能提取出文字的最小 PDF（Helvetica 标准字体 / WinAnsi 编码）。

    每行用**显式 Td 换行**（不用 `T*`）：只依赖最基础的操作符，
    没有写库参与，pypdf 能真解析出文字。
    """
    content = b"BT /F1 12 Tf 40 760 Td "
    first = True
    for line in lines:
        if not first:
            content += b"0 -16 Td "
        content += _pdf_escape(line) + b" Tj "
        first = False
    content += b"ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
         b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n").encode()
    out += b"%%EOF\n"
    return bytes(out)


def _docx_bytes() -> bytes:
    import io

    import docx

    document = docx.Document()
    document.add_paragraph("张三")
    document.add_paragraph("技能：Python / 大模型 / RAG")
    document.add_paragraph("城市：广州")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "教育"
    table.rows[0].cells[1].text = "中山大学 计算机 2027 届"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _write(name: str, data: bytes) -> Path:
    path = _TMP_DIR / name
    path.write_bytes(data)
    return path


def _fresh_dir() -> None:
    """每次跑用干净目录（上次崩溃残留的样本不该影响本次判断）。"""
    if _TMP_DIR.exists():
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
    if _TMP_DIR.exists():
        _fail(f"临时目录清不掉：{_TMP_DIR}")
    _TMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
section("1. PDF 提取（真 PDF 字节 → pypdf 解析）")

_fresh_dir()
_PDF = _write("resume.pdf", _pdf_bytes("Zhang San", "Skills: Python / RAG", "City: Guangzhou"))


def t_pdf_extract():
    text, err = EX.extract_text(_PDF)
    _expect(err is None, f"不该报错，实得：{err}")
    _expect("Zhang San" in text, f"缺第一行：{text!r}")
    _expect("RAG" in text, f"缺技能行：{text!r}")
    _expect("Guangzhou" in text, f"缺城市行：{text!r}")
    return f"{len(text)} 字"


def t_pdf_by_name_not_ext():
    """落盘文件没有扩展名时，靠 %PDF 文件头也要认出来。"""
    path = _write("noext_tmp", _PDF.read_bytes())
    text, err = EX.extract_text(path)
    _expect(err is None, f"无扩展名 PDF 应能识别，实得：{err}")
    _expect("Zhang San" in text, "无扩展名 PDF 内容不对")
    return "靠 %PDF 头识别"


def t_pdf_empty_is_降级():
    """空白 PDF（提取不到文字）→ 降级提示，不是静默成功。"""
    path = _write("blank.pdf", _pdf_bytes())
    text, err = EX.extract_text(path)
    _expect(text == "", "空 PDF 不该返回正文")
    _expect(err and "没有提取到" in err, f"降级话术不对：{err}")
    return "空 PDF 被拦住"


def t_pdf_corrupt():
    """损坏 / 假 PDF → 异常被吞成 error，绝不抛给调用方。"""
    path = _write("broken.pdf", b"%PDF-1.4\nthis is not a real pdf body\n")
    text, err = EX.extract_text(path)
    _expect(text == "", "损坏 PDF 不该返回正文")
    _expect(bool(err), "损坏 PDF 必须给原因")
    _expect("暂不支持" not in err, f"不该误判成格式不支持：{err}")
    return "异常已吞"


check("PDF 文本提取（含多行）", t_pdf_extract)
check("无扩展名 PDF 靠文件头识别", t_pdf_by_name_not_ext)
check("空白 PDF 降级（不静默成功）", t_pdf_empty_is_降级)
check("损坏 PDF 不抛异常、给原因", t_pdf_corrupt)

# ---------------------------------------------------------------------------
section("2. Word（.docx）提取")

_DOCX = _write("resume.docx", _docx_bytes())


def t_docx_extract():
    text, err = EX.extract_text(_DOCX)
    _expect(err is None, f"不该报错，实得：{err}")
    _expect("张三" in text, f"缺姓名：{text!r}")
    _expect("大模型" in text, f"缺技能：{text!r}")
    return f"{len(text)} 字"


def t_docx_table():
    """简历常把教育/技能排成表格，表格文字也必须抓到。"""
    text, _ = EX.extract_text(_DOCX)
    _expect("中山大学" in text, f"表格内容没抓到：{text!r}")
    return "表格已抓"


def t_docx_empty():
    import io

    import docx

    buffer = io.BytesIO()
    docx.Document().save(buffer)
    path = _write("empty.docx", buffer.getvalue())
    text, err = EX.extract_text(path)
    _expect(text == "", "空 docx 不该返回正文")
    _expect(bool(err), "空 docx 必须给原因")
    return "空 docx 被拦住"


check("docx 正文提取", t_docx_extract)
check("docx 表格提取", t_docx_table)
check("空 docx 降级", t_docx_empty)

# ---------------------------------------------------------------------------
section("3. 不支持格式的降级路径")


def t_txt_unsupported():
    path = _write("resume.txt", b"plain text resume")
    text, err = EX.extract_text(path)
    _expect(text == "", "txt 不该被解析")
    _expect(err and ".txt" in err, f"应点名格式：{err}")
    _expect("粘贴" in err, f"应给出粘贴文本的出路：{err}")
    return ".txt → 提示粘贴"


def t_doc_unsupported():
    """老版 .doc 是二进制格式，python-docx 打不开 → 明确归到不支持。"""
    path = _write("resume.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1 fake old word")
    text, err = EX.extract_text(path)
    _expect(text == "", ".doc 不该被解析")
    _expect(err and "docx" in err, f"应提示另存为 docx：{err}")
    return ".doc → 提示另存"


def t_missing_file():
    text, err = EX.extract_text(_TMP_DIR / "nope.pdf")
    _expect(text == "", "不存在的文件不该返回正文")
    _expect(err and "不存在" in err, f"应说明文件不存在：{err}")
    return "缺文件不炸"


def t_never_raises():
    """随机二进制 + 目录路径 + None 之类脏输入，全都不许抛。"""
    junk = _write("junk.bin", os.urandom(64))
    for weird in (junk, _TMP_DIR, Path("") / "x" / "y.pdf"):
        text, err = EX.extract_text(weird)
        _expect(text == "" and bool(err), f"{weird} 应降级：{text!r} / {err!r}")
    return "3 种脏输入全降级"


check("txt 不支持 → 提示粘贴", t_txt_unsupported)
check("老版 .doc 不支持 → 提示另存", t_doc_unsupported)
check("文件不存在 → 不抛异常", t_missing_file)
check("脏输入永不抛异常", t_never_raises)

# ---------------------------------------------------------------------------
section("4. 图片识别（模型不支持读图，必须走单独话术）")


def t_is_image():
    for name in ("a.png", "b.JPG", "c.jpeg", "d.webp"):
        path = _write(name, b"\x89PNG\r\n\x1a\nfake")
        _expect(EX.is_image(path), f"{name} 应识别为图片")
    for name in ("a.pdf", "b.docx"):
        path = _write(name, b"x")
        _expect(not EX.is_image(path), f"{name} 不该识别为图片")
    return "png/jpg/webp 命中"


def t_image_by_name_only():
    """元素只有 name、path 还没落地时也要能判出图片。"""
    _expect(EX.is_image(_TMP_DIR / "ghost.png", "简历截图.png"), "靠 name 判图失败")
    _expect(not EX.is_image(_TMP_DIR / "ghost.pdf", "简历.pdf"), "pdf 被误判成图")
    return "name 兜底可用"


check("图片扩展名识别", t_is_image)
check("仅凭 name 也能判图片", t_image_by_name_only)

# ---------------------------------------------------------------------------
section("5. app.py 接线（不改 app 逻辑，只钉住约定）")


def t_supported_sets():
    _expect(EX.PDF_EXTS == {".pdf"}, f"PDF 集合变了：{EX.PDF_EXTS}")
    _expect(EX.WORD_EXTS == {".docx"}, f"Word 集合变了：{EX.WORD_EXTS}")
    _expect(".png" in EX.IMAGE_EXTS and ".jpg" in EX.IMAGE_EXTS, "图片集合缺 png/jpg")
    return "pdf/docx/图片 三集合正确"


def t_app_uses_extractor():
    """app.py 必须真接上 extractor（不能被回退成只认文本）。"""
    source = (REPO / "agent" / "app.py").read_text(encoding="utf-8")
    _expect("from agent.resume import extractor" in source, "app.py 没 import extractor")
    _expect("extractor.extract_text(" in source, "app.py 没调用 extract_text")
    _expect("_resume_attachments(message)" in source, "app.py 没读 message.elements")
    _expect("当前模型不支持读图" in source, "app.py 缺图片话术")
    return "接线齐全"


def t_parse_text_still_works():
    """提取出的文字必须能喂进现有 /resume 解析链路（否则上传了也白搭）。"""
    from agent.resume.parser import parse_text

    text, err = EX.extract_text(_DOCX)
    _expect(err is None, f"docx 提取失败：{err}")
    resume = parse_text(text)
    _expect("张三" in (resume.name or "") or "张三" in text, f"解析没吃到姓名：{resume.name}")
    return f"parse_text ok（name={resume.name}）"


def t_parse_budget_wired():
    """本轮修复钉住：简历解析必须带大额度 + low 思考档。

    思考模型下 max_tokens 同时卡思考与正文，1024 会 content 为空、
    JSON 解析报 `Expecting value: line 1 column 1 (char 0)`。
    """
    saved = {k: os.environ.get(k)
             for k in ("RATE_LIMIT_ENABLED", "RESUME_LLM_REASONING_EFFORT",
                       "LLM_MAX_TOKENS", "LLM_DEFAULT_REASONING_EFFORT")}
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    os.environ.pop("RESUME_LLM_REASONING_EFFORT", None)
    os.environ.pop("LLM_MAX_TOKENS", None)
    os.environ.pop("LLM_DEFAULT_REASONING_EFFORT", None)
    try:
        from shared import limits
        from shared import llm_client as LC

        _expect(limits.resume_max_tokens() >= 4096,
                f"解析额度太小：{limits.resume_max_tokens()}")
        _expect(limits.resume_reasoning_effort() == "low",
                f"思考档默认不是 low：{limits.resume_reasoning_effort()!r}")
        payload = LC._build_payload([{"role": "user", "content": "x"}],
                                    "m", False, 4096, "low")
        _expect(payload.get("max_tokens") == 4096, f"payload 没带额度：{payload}")
        _expect(payload.get("reasoning_effort") == "low", f"payload 没带档位：{payload}")

        # 全局兜底契约（本轮治本的那条）：**什么都不传**也必须拿到够用的额度
        # 与低思考档 —— 调用方漏传不再是 bug。
        plain = LC._build_payload([{"role": "user", "content": "x"}], "m", False, None)
        _expect(plain.get("max_tokens", 0) >= 4096,
                f"裸调用没拿到全局兜底额度：{plain}")
        _expect(plain.get("reasoning_effort") == "low",
                f"裸调用没拿到全局默认思考档：{plain}")
        # 显式传空串 = 主动 opt-out，不能被全局默认覆盖
        opted_out = LC._build_payload([{"role": "user", "content": "x"}],
                                      "m", False, None, "")
        _expect("reasoning_effort" not in opted_out,
                f"显式空串没生效（调用方失去 opt-out）：{opted_out}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    source = (REPO / "agent" / "resume" / "parser.py").read_text(encoding="utf-8")
    _expect("reasoning_effort=effort" in source and "max_tokens=budget" in source,
            "parser.py 没把额度/档位传给 chat()")
    _expect("consume_truncated()" in source, "parser.py 没识别「输出触顶」这种失败")
    return "4096 + low 已接线"


check("PDF/Word/图片 集合定义正确", t_supported_sets)
check("app.py 已接上 extractor", t_app_uses_extractor)
check("提取文字能进 /resume 解析链路", t_parse_text_still_works)
check("简历解析已带大额度+low 思考档", t_parse_budget_wired)

# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
print("=" * 74)
if FAIL:
    for label, detail in FAIL:
        print(f"  [FAIL] {label} → {detail}")
    sys.exit(1)

shutil.rmtree(_TMP_DIR, ignore_errors=True)
print("临时样本目录已清理。")
