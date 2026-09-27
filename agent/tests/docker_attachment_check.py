# -*- coding: utf-8 -*-
"""容器内验证：Chainlit 上传落盘的文件，能否被 app.py 的附件逻辑读到并解析。

对应 bug：Docker 里真 PDF 上传后 Agent 说"没收到消息内容"。
跑法（宿主机）：
    docker cp agent/tests/docker_attachment_check.py <容器>:/tmp/
    docker exec -w /app <容器> python /tmp/docker_attachment_check.py
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, "/app")

from chainlit.config import FILES_DIRECTORY  # noqa: E402

from agent.resume import extractor  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" → {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


class FakeElement:
    """只保留 app.py 会读的字段（Chainlit 的 Element 有几十个字段）。"""

    def __init__(self, path, name, mime, type_):
        self.path = path
        self.name = name
        self.mime = mime
        self.type = type_


class FakeMessage:
    def __init__(self, elements):
        self.elements = elements
        self.content = "请保存这份简历"


print("=" * 74)
print("Docker 附件读取自检")
print("=" * 74)
print(f"cwd = {os.getcwd()}")
print(f"FILES_DIRECTORY = {FILES_DIRECTORY} exists={FILES_DIRECTORY.exists()}")

spooled = sorted(glob.glob(str(FILES_DIRECTORY / "*" / "*")))
print(f"spool 里的文件 = {spooled}")

check("上传目录存在", FILES_DIRECTORY.exists(), str(FILES_DIRECTORY))
if not spooled:
    print("\n⚠️ spool 里没有文件：请先在 /chat 页面上传一份简历，再跑本自检。")
    sys.exit(0)

# ---- 1. 每个落盘文件都要能被 extractor 读出文字（PDF 必过；docx 需要 python-docx）
for path in spooled:
    name = os.path.basename(path)
    magic = open(path, "rb").read(4)
    text, err = extractor.extract_text(path, name)
    print(f"---- {name} magic={magic!r} size={os.path.getsize(path)}")
    if path.endswith(".pdf") or magic.startswith(b"%PDF"):
        check(f"PDF 能提取文字（{name}）", bool(text) and not err,
              f"{len(text or '')} 字 err={err}")
    elif path.endswith(".docx") or magic.startswith(b"PK\x03\x04"):
        check(f"docx 能提取文字（{name}）", bool(text) and not err,
              f"{len(text or '')} 字 err={err}")

# ---- 2. app.py 的 _resume_attachments 必须认得出这些附件
from agent.app import _resume_attachments  # noqa: E402

real = spooled[0]
message = FakeMessage([FakeElement(real, os.path.basename(real),
                                   "application/pdf", "file")])
found = _resume_attachments(message)
check("app._resume_attachments 认得出真实附件", len(found) == 1,
      f"len={len(found)} path={getattr(found[0], 'path', None) if found else None}")

# ---- 3. 相对路径 / 路径不存在时也不能被判成"没附件"
relative = os.path.relpath(real, "/app") if real.startswith("/app/") else real
message = FakeMessage([FakeElement(relative, os.path.basename(real),
                                   "application/pdf", "file")])
found = _resume_attachments(message)
check("相对路径附件能被改写成可用绝对路径", len(found) == 1,
      f"relative={relative} → {getattr(found[0], 'path', None) if found else None}")

ghost = FakeMessage([FakeElement(str(FILES_DIRECTORY / "nonexistent" / "x.pdf"),
                                 "x.pdf", "application/pdf", "file")])
check("不存在的附件被跳过（不炸）", _resume_attachments(ghost) == [])

# ---- 4. Word 解析依赖 python-docx 是否装进镜像
try:
    import docx  # noqa: F401
    check("python-docx 已装（Word 上传可用）", True)
except Exception as e:  # noqa: BLE001
    check("python-docx 已装（Word 上传可用）", False, f"{type(e).__name__}: {e}")

print()
print(f"结果：{'全部通过' if not FAILURES else '失败 ' + ', '.join(FAILURES)}")
sys.exit(1 if FAILURES else 0)
