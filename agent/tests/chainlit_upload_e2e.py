# -*- coding: utf-8 -*-
"""在容器里跑通「上传附件 → on_message 读到 elements → 存进简历」全链路（不依赖浏览器）。

说的就是 Chainlit 前端的那几步：
  1. POST {BASE}/chat/project/file?session_id=...  上传文件（返回 file id）
  2. socket.io 的 client_message（带 fileReferences）触发 on_message
  3. 断言助手回复里出现了 PDF/Word 里的关键字

跑法（宿主机，容器已把 8000 映射出来）：
    python agent/tests/chainlit_upload_e2e.py
跑法（容器内）：
    docker exec -e CHAT_BASE=http://127.0.0.1:7860 -w /app <容器> python agent/tests/chainlit_upload_e2e.py

环境变量：
    CHAT_BASE  默认 http://127.0.0.1:8000
    PROBE_PDF  默认 E:\\dsh\\cache\\tmp\\resume_probe.pdf（没有会临时生成）
    PROBE_DOCX 默认 E:\\dsh\\cache\\tmp\\resume_probe.docx（没有会临时生成）
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
import zipfile

import httpx
import socketio

BASE = os.getenv("CHAT_BASE", "http://127.0.0.1:8000").rstrip("/")
CHAT = f"{BASE}/chat"
KEYWORDS = ("简历已设置", "Zhang San", "Python")

PDF_LINES = ["Zhang San", "Skills: Python / RAG / FastAPI / Docker", "City: Guangzhou"]


def make_pdf(path: str) -> str:
    """手写最小 PDF：文字必须用 (..) Tj 写，否则 pypdf 提取为空。"""
    def esc(line: str) -> bytes:
        body = line.encode("latin-1")
        body = body.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
        return b"(" + body + b")"

    content = b"BT /F1 12 Tf 40 760 Td "
    first = True
    for line in PDF_LINES:
        if not first:
            content += b"0 -16 Td "
        content += esc(line) + b" Tj "
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
    with open(path, "wb") as fh:
        fh.write(bytes(out))
    return path


def make_docx(path: str) -> str:
    """手写最小 docx（python-docx 能读）：[Content_Types] + _rels + word/document.xml。"""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Zhang San</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Skills: Python RAG Docker</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>City: Guangzhou</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)
    return path


def ensure(path: str, kind: str) -> str:
    if os.path.exists(path):
        return path
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if kind == "pdf":
        return make_pdf(path)
    return make_docx(path)


class Probe:
    def __init__(self) -> None:
        self.received: list = []
        self.done = asyncio.Event()
        self.connected = asyncio.Event()

    async def run_case(self, sio, client: httpx.AsyncClient, session_id: str,
                       path: str, mime: str) -> bool:
        self.done.clear()
        self.received.clear()
        with open(path, "rb") as fh:
            resp = await client.post(
                f"{CHAT}/project/file",
                params={"session_id": session_id},
                files={"file": (os.path.basename(path), fh, mime)},
            )
        print(f"  上传 {os.path.basename(path)} → HTTP {resp.status_code} {resp.text[:160]}")
        if resp.status_code != 200:
            return False
        payload = {
            "message": {
                "id": str(uuid.uuid4()),
                "type": "user_message",
                "output": "请保存这份简历",
                "createdAt": "2026-09-27T00:00:00+00:00",
            },
            "fileReferences": [{"id": resp.json()["id"]}],
        }
        await sio.emit("client_message", payload)
        try:
            await asyncio.wait_for(self.done.wait(), timeout=180)
        except asyncio.TimeoutError:
            print("  等待助手回复超时（180s）")
        joined = "\n".join(
            (d.get("output") or "") for d in self.received
            if isinstance(d, dict) and d.get("type") == "assistant_message"
        )
        print(f"  助手回复：{joined[:400]!r}")
        return all(k in joined for k in KEYWORDS)


async def main() -> int:
    for stream in (sys.stdout, sys.stderr):                # Windows 控制台默认 GBK，回复里有 ✅/emoji
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                   # noqa: BLE001
            pass
    only = os.getenv("PROBE_ONLY", "").strip().lower()
    pdf = ensure(os.getenv("PROBE_PDF", r"E:\dsh\cache\tmp\resume_probe.pdf"), "pdf")
    docx = ensure(os.getenv("PROBE_DOCX", r"E:\dsh\cache\tmp\resume_probe.docx"), "docx")
    for path in (pdf, docx):
        with open(path, "rb") as fh:
            print(f"样本 {path} magic={fh.read(4)!r} size={os.path.getsize(path)}")

    probe = Probe()
    sio = socketio.AsyncClient(reconnection=False)

    @sio.event
    async def connect():
        probe.connected.set()

    @sio.event
    async def disconnect(reason=None):
        print(f"  [socket] 断开：{reason}")

    @sio.on("new_message")
    async def on_new_message(data):
        probe.received.append(data)
        if isinstance(data, dict) and data.get("type") == "assistant_message" \
                and (data.get("output") or "").strip():
            probe.done.set()

    @sio.on("task_end")
    async def on_task_end(_data=None):
        probe.done.set()

    session_id = str(uuid.uuid4())
    auth = {"sessionId": session_id, "threadId": None, "clientType": "webapp",
            "userEnv": "{}", "chatProfile": None}
    # 必须用 websocket：polling 下服务端紧接着发的 task_end 会让 python-socketio
    # 报 "Unexpected packet from server, aborting" 并断开命名空间。
    await sio.connect(CHAT, socketio_path="/chat/ws/socket.io", auth=auth,
                      transports=["websocket"], wait_timeout=20)
    print(f"已连接 {CHAT} sid={sio.sid} session_id={session_id}")
    await sio.emit("connection_successful")
    await asyncio.sleep(3)

    results: dict[str, bool] = {}
    cases = [
        (pdf, "application/pdf"),
        (docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ]
    if only in ("pdf", "docx"):
        cases = [c for c in cases if c[0].lower().endswith(only)]
    async with httpx.AsyncClient(timeout=90.0) as client:
        for path, mime in cases:
            label = os.path.basename(path)
            print(f"---- 用例 {label}")
            try:
                results[label] = await probe.run_case(sio, client, session_id, path, mime)
            except Exception as exc:                            # noqa: BLE001
                print(f"  用例抛异常：{type(exc).__name__}: {exc}")
                results[label] = False

    await sio.disconnect()
    print("\n==== 结果 ====")
    for label, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
