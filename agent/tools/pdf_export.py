"""
简历导出 PDF（C4）。

对外只暴露一个函数：

    export_resume_pdf(resume: dict, output_path: str) -> str

后端选择顺序（**不安装任何新依赖**）：
1. **reportlab** —— 装了就用它（中文走内置 CID 字体 STSong-Light，不需要字体文件）；
2. **fpdf2**     —— 没 reportlab 就用它（需要系统中文字体，自动探测）；
3. **纯标准库兜底** —— 两个都没装时，用本文件自带的极简 PDF 写入器：
   - 自动探测系统中文字体（如 Windows 的 msyh.ttc / simhei.ttf），
     只把用到的字形子集化后嵌入（Type0 / CIDFontType2 / Identity-H + ToUnicode，
     pypdf 能正确抽回中文，成品 PDF 约 20-60KB）；
   - 连中文字体都找不到时，退化用 base14（Helvetica + WinAnsi），
     非 ASCII 字符降级成 `?`，至少保证 PDF 打得开、结构合法。

> 说明：本机既没有 reportlab 也没有 fpdf2，所以走的是第 3 条兜底路径。
> 装了 reportlab / fpdf2 后会自动切回前两条路径（它们的排版更好）。

环境变量：
- `PDF_CJK_FONT`：指定中文字体文件（.ttf / .ttc）路径，优先级最高。
"""
from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

# --------------------------------------------------------------------------
# 版面常量（A4，单位 pt）
# --------------------------------------------------------------------------
PAGE_W, PAGE_H = 595.28, 841.89
MARGIN = 52.0

# 每种内容块的字号 / 颜色 / 缩进
_SIZES = {"h1": 24.0, "contact": 10.5, "h2": 13.5, "body": 10.5, "bullet": 10.5, "note": 9.0}
_COLORS = {
    "h1": (0.10, 0.20, 0.45),
    "contact": (0.25, 0.25, 0.25),
    "h2": (0.10, 0.20, 0.45),
    "body": (0.10, 0.10, 0.10),
    "bullet": (0.20, 0.20, 0.20),
    "note": (0.45, 0.45, 0.45),
}
_INDENTS = {"bullet": 12.0}


# --------------------------------------------------------------------------
# 后端探测
# --------------------------------------------------------------------------

def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:                               # noqa: BLE001 - 缺依赖/导入期报错都算不可用
        return False


def available_backend() -> str:
    """返回当前可用的 PDF 后端名：reportlab / fpdf2 / stdlib"""
    if _module_available("reportlab"):
        return "reportlab"
    if _module_available("fpdf"):
        return "fpdf2"
    return "stdlib"


# --------------------------------------------------------------------------
# 中文字体探测
# --------------------------------------------------------------------------

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simkai.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]


def find_cjk_font(prefer_ttf: bool = False) -> str | None:
    """找一个可用的中文字体文件；找不到返回 None。

    prefer_ttf=True 时优先纯 .ttf（fpdf2 对 .ttc 字体集合支持不稳），
    否则按 _FONT_CANDIDATES 的顺序取第一个存在的。
    """
    env_font = os.getenv("PDF_CJK_FONT", "").strip()
    if env_font and Path(env_font).is_file():
        return env_font

    candidates = list(_FONT_CANDIDATES)
    if prefer_ttf:
        candidates.sort(key=lambda p: (not p.lower().endswith(".ttf"),))

    for path in candidates:
        if Path(path).is_file():
            return path

    # Windows 上再扫一遍字体目录，兜住非默认文件名
    fonts_dir = Path(r"C:\Windows\Fonts")
    if fonts_dir.is_dir():
        for pattern in ("simhei*.ttf", "msyh*.ttc", "simsun*.ttc"):
            for path in sorted(fonts_dir.glob(pattern)):
                try:
                    if _looks_like_cjk_font(path):
                        return str(path)
                except Exception:                   # noqa: BLE001 - 探测失败就当这个字体不可用
                    continue
    return None


def _looks_like_cjk_font(path: Path) -> bool:
    """粗略判断字体里有没有中文字形：能解析出「中」字的 glyph 就算"""
    font = _TrueTypeFont(str(path))
    return font.gid("中") != 0


# --------------------------------------------------------------------------
# 简历 → 版面块（后端无关的中间格式）
# --------------------------------------------------------------------------

def _to_lines(value) -> list:
    """把字段值统一成列表：list 原样返回，字符串按换行/分号切"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [v for v in value if v not in (None, "")]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [line.strip() for line in re.split(r"[\n;；]+", text) if line.strip()]
    return [value]


def _pick(item: dict, *keys, default=""):
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return default


def normalize_resume(resume) -> dict:
    """把各种形态的简历统一成结构化 dict。

    支持：
    - storage 里的简历记录：{"id","name","content","created_at"} → 取 content
    - 结构化简历 dict：{"name","skills","experience","projects","education","city"}
    - content 是字符串：先按 JSON 解析，失败则当纯文本（返回 {"_plain": 文本}）
    """
    if resume is None:
        return {}

    if isinstance(resume, str):
        text = resume.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return {"_plain": text}
        return normalize_resume(parsed)

    if not isinstance(resume, dict):
        return {"_plain": str(resume)}

    # storage 的简历记录：优先用 content，名字另存备用
    if "content" in resume:
        record_name = str(resume.get("name") or "").strip()
        inner = normalize_resume(resume.get("content"))
        if not inner.get("name") and record_name:
            inner["name"] = record_name
        return inner

    if "_plain" in resume:
        return resume

    return dict(resume)


def _experience_blocks(item) -> list:
    if isinstance(item, dict):
        company = _pick(item, "company", "公司")
        role = _pick(item, "role", "position", "title", "岗位")
        months = _pick(item, "months", "duration", "月数")
        desc = _pick(item, "description", "desc", "detail", "描述")

        head = " | ".join(str(p) for p in (company, role) if p)
        if months not in ("", None):
            months_text = f"{months} 个月" if str(months).strip().isdigit() else str(months)
            head = f"{head}（{months_text}）" if head else months_text

        blocks = [{"kind": "body", "text": head}] if head else []
        if desc:
            blocks.append({"kind": "bullet", "text": f"· {desc}"})
        return blocks
    return [{"kind": "body", "text": str(item)}]


def _project_blocks(item) -> list:
    if isinstance(item, dict):
        name = _pick(item, "name", "项目名", "title")
        tech = _pick(item, "tech", "stack", "技术栈")
        desc = _pick(item, "desc", "description", "描述")

        if isinstance(tech, (list, tuple)):
            tech_text = "、".join(str(t) for t in tech if str(t).strip())
        else:
            tech_text = str(tech or "").strip()

        head = str(name or "").strip()
        if tech_text:
            head = f"{head}　|　技术栈：{tech_text}" if head else f"技术栈：{tech_text}"

        blocks = [{"kind": "body", "text": head}] if head else []
        if desc:
            blocks.append({"kind": "bullet", "text": f"· {desc}"})
        return blocks
    return [{"kind": "body", "text": str(item)}]


def resume_blocks(resume) -> list:
    """把简历转成版面块列表：[{"kind": "h1"/"h2"/"body"/"bullet"/"space", "text": ...}]"""
    data = normalize_resume(resume)
    if not data:
        return [{"kind": "body", "text": "（空简历）"}]

    # 纯文本简历：第一行当标题，其余原样排
    if "_plain" in data:
        lines = [ln.strip() for ln in str(data["_plain"]).splitlines() if ln.strip()]
        if not lines:
            return [{"kind": "body", "text": "（空简历）"}]
        blocks = [{"kind": "h1", "text": lines[0][:40]}]
        blocks += [{"kind": "body", "text": ln} for ln in lines[1:]]
        return blocks

    blocks = []

    name = str(_pick(data, "name", "姓名", default="简历")).strip() or "简历"
    blocks.append({"kind": "h1", "text": name})

    contact = []
    city = str(_pick(data, "city", "城市")).strip()
    education = str(_pick(data, "education", "学历")).strip()
    if city:
        contact.append(f"城市：{city}")
    if education:
        contact.append(f"教育：{education}")
    for extra in ("phone", "电话", "email", "邮箱"):
        value = str(_pick(data, extra)).strip()
        if value:
            contact.append(value)
    if contact:
        blocks.append({"kind": "contact", "text": "　|　".join(contact)})

    blocks.append({"kind": "space", "size": 6})

    skills = _to_lines(_pick(data, "skills", "技能"))
    if skills:
        blocks.append({"kind": "h2", "text": "技能"})
        flat = []
        for item in skills:
            if isinstance(item, (list, tuple)):
                flat.extend(str(x) for x in item if str(x).strip())
            else:
                flat.append(str(item))
        blocks.append({"kind": "body", "text": "、".join(flat)})
        blocks.append({"kind": "space", "size": 4})

    experience = _to_lines(_pick(data, "experience", "实习经历", "experiences"))
    if experience:
        blocks.append({"kind": "h2", "text": "实习经历"})
        for item in experience:
            blocks.extend(_experience_blocks(item))
            blocks.append({"kind": "space", "size": 2})
        blocks.append({"kind": "space", "size": 2})

    projects = _to_lines(_pick(data, "projects", "项目经历", "project"))
    if projects:
        blocks.append({"kind": "h2", "text": "项目经历"})
        for item in projects:
            blocks.extend(_project_blocks(item))
            blocks.append({"kind": "space", "size": 2})

    if len(blocks) <= 3:                        # 只有名字/联系方式，说明字段名不认识
        blocks.append({"kind": "space", "size": 4})
        blocks.append({"kind": "note", "text": "（未识别到技能/经历/项目字段，原始内容如下）"})
        blocks.append({"kind": "body", "text": json.dumps(data, ensure_ascii=False)[:2000]})

    return blocks


# ==========================================================================
# 后端 3：纯标准库兜底实现
# ==========================================================================

# ---------- 3.1 TrueType 字体解析 ----------

def _checksum(data: bytes) -> int:
    """sfnt 表校验和：按 4 字节大端累加（不足补 0）"""
    padded = data + b"\x00" * ((4 - len(data) % 4) % 4)
    total = 0
    for i in range(0, len(padded), 4):
        total = (total + struct.unpack(">I", padded[i:i + 4])[0]) & 0xFFFFFFFF
    return total


def _sfnt_base(data: bytes) -> int:
    if data[:4] == b"ttcf":
        if len(data) < 16:
            raise ValueError("字体集合文件损坏")
        num_fonts = struct.unpack(">I", data[8:12])[0]
        if num_fonts < 1:
            raise ValueError("字体集合里没有字体")
        return struct.unpack(">I", data[12:16])[0]
    return 0


def _read_tables(data: bytes, base: int):
    version = data[base:base + 4]
    if version == b"OTTO":
        raise ValueError("不支持 CFF/OTF 字体，请用 TrueType 的 .ttf/.ttc")
    num_tables = struct.unpack(">H", data[base + 4:base + 6])[0]
    tables = {}
    for i in range(num_tables):
        off = base + 12 + i * 16
        tag = data[off:off + 4]
        _sum, offset, length = struct.unpack(">III", data[off + 4:off + 16])
        tables[tag] = (offset, length)
    if b"glyf" not in tables or b"loca" not in tables:
        raise ValueError("字体缺少 glyf/loca 表，不是 TrueType 轮廓字体")
    return version, tables


def _rebuild_sfnt_from_blobs(version: bytes, blobs: dict) -> bytes:
    """把若干张表拼成一个独立字体文件（重排偏移 + 重算校验和）。

    .ttc 字体集合不能直接塞进 PDF，必须先这样「拍平」成单个字体。
    """
    tags = sorted(blobs)
    num = len(tags)
    entry_selector = max(0, num.bit_length() - 1)
    search_range = (1 << entry_selector) * 16
    range_shift = num * 16 - search_range

    offset = 12 + num * 16
    directory = bytearray()
    bodies = []
    head_abs = None
    for tag in tags:
        body = bytearray(blobs[tag])
        if tag == b"head" and len(body) >= 12:
            body[8:12] = b"\x00\x00\x00\x00"      # checkSumAdjustment 先清零，最后回填
            head_abs = offset
        pad = (4 - len(body) % 4) % 4
        directory += tag + struct.pack(">III", _checksum(bytes(body)), offset, len(body))
        bodies.append(bytes(body) + b"\x00" * pad)
        offset += len(body) + pad

    header = version + struct.pack(">HHHH", num, search_range, entry_selector, range_shift)
    font = bytearray(header + bytes(directory) + b"".join(bodies))

    if head_abs is not None:
        adjustment = (0xB1B0AFBA - _checksum(bytes(font))) & 0xFFFFFFFF
        font[head_abs + 8:head_abs + 12] = struct.pack(">I", adjustment)
    return bytes(font)


def _rebuild_sfnt(version: bytes, tables: dict, data: bytes) -> bytes:
    """按「表名 → (偏移, 长度)」从原文件里取表并拍平成一个独立字体文件"""
    blobs = {tag: data[off:off + length] for tag, (off, length) in tables.items()}
    return _rebuild_sfnt_from_blobs(version, blobs)


def _glyph_offsets(font) -> list:
    """读 loca 表，返回 numGlyphs+1 个 glyf 表内偏移"""
    loca = font._table(b"loca")
    head = font._table(b"head")
    long_format = struct.unpack(">h", head[50:52])[0] if len(head) >= 52 else 1

    offsets = []
    for i in range(font.num_glyphs + 1):
        if long_format:
            if i * 4 + 4 > len(loca):
                break
            offsets.append(struct.unpack(">I", loca[i * 4:i * 4 + 4])[0])
        else:
            if i * 2 + 2 > len(loca):
                break
            offsets.append(struct.unpack(">H", loca[i * 2:i * 2 + 2])[0] * 2)
    while len(offsets) < font.num_glyphs + 1:
        offsets.append(offsets[-1] if offsets else 0)
    return offsets


def _component_gids(glyph_data: bytes) -> list:
    """复合字形引用的部件 GID 列表（简单字形返回空）"""
    if len(glyph_data) < 10 or struct.unpack(">h", glyph_data[0:2])[0] >= 0:
        return []
    ids = []
    p = 10
    while p + 4 <= len(glyph_data):
        flags, gid = struct.unpack(">HH", glyph_data[p:p + 4])
        p += 4
        p += 4 if flags & 0x0001 else 2
        if flags & 0x0008:                          # WE_HAVE_A_SCALE
            p += 2
        elif flags & 0x0040:                        # X_AND_Y_SCALE
            p += 4
        elif flags & 0x0080:                        # TWO_BY_TWO
            p += 8
        ids.append(gid)
        if not flags & 0x0020:                      # MORE_COMPONENTS
            break
    return ids


# 这些表只服务于「设备度量 / 竖排 / 数字签名」，横排文字渲染用不到，
# 子集化时直接丢掉（hdmx 一个表就有 500KB+），能显著减小 PDF 体积。
_DROPPABLE_TABLES = {
    b"hdmx", b"LTSH", b"VDMX", b"vmtx", b"vhea", b"DSIG",
    b"meta", b"FFTM", b"PCLT", b"EBDT", b"EBLC", b"EBSC",
}


def _subset_font(font, used_gids) -> bytes:
    """只保留用到的字形，把字体从 ~19MB 压到几百 KB。

    做法是「把没用的字形清空」而不是重排 GID：cmap / hmtx / post 全部保持原样，
    复合字形引用到的部件（比如带声调的字母）也会一并保留，
    这样字体结构改动最小，最不容易被阅读器判为损坏。
    """
    offsets = _glyph_offsets(font)
    glyf_off = font.tables[b"glyf"][0]

    keep = {0}
    stack = [g for g in used_gids if 0 <= g < font.num_glyphs]
    while stack:
        gid = stack.pop()
        if gid in keep or not (0 <= gid < font.num_glyphs):
            continue
        keep.add(gid)
        start, end = offsets[gid], offsets[gid + 1]
        if end > start:
            stack.extend(_component_gids(font.data[glyf_off + start:glyf_off + end]))

    new_glyf = bytearray()
    loca = []
    for gid in range(font.num_glyphs):
        loca.append(len(new_glyf))
        if gid in keep:
            start, end = offsets[gid], offsets[gid + 1]
            if end > start:
                body = font.data[glyf_off + start:glyf_off + end]
                new_glyf += body
                new_glyf += b"\x00" * ((4 - len(body) % 4) % 4)
    loca.append(len(new_glyf))

    blobs = {
        tag: font.data[off:off + length]
        for tag, (off, length) in font.tables.items()
        if tag not in _DROPPABLE_TABLES
    }
    blobs[b"glyf"] = bytes(new_glyf)
    blobs[b"loca"] = b"".join(struct.pack(">I", off) for off in loca)

    head = bytearray(blobs[b"head"])
    head[50:52] = struct.pack(">h", 1)             # indexToLocFormat = 1（长格式 loca）
    blobs[b"head"] = bytes(head)

    return _rebuild_sfnt_from_blobs(font.version, blobs)


class _CmapLookup:
    """cmap 子表查询：unicode 码点 → glyph id（按需查，不展开成大字典）"""

    def __init__(self, data: bytes, tables: dict):
        self.data = data
        self.fmt = None
        self.sub = 0
        self.groups = []
        self.seg_count = 0
        self.ends = []
        self.start_codes = b""
        self.id_deltas = b""
        self.id_range_offsets = b""

        if b"cmap" not in tables:
            return
        base, _length = tables[b"cmap"]
        _version, num_tables = struct.unpack(">HH", data[base:base + 4])

        best = None
        for i in range(num_tables):
            rec = base + 4 + i * 8
            pid, eid, sub_off = struct.unpack(">HHI", data[rec:rec + 8])
            sub = base + sub_off
            fmt = struct.unpack(">H", data[sub:sub + 2])[0]
            score = {
                (3, 10): 5, (0, 4): 4, (0, 6): 4,
                (3, 1): 3, (0, 3): 3, (0, 2): 2, (0, 1): 2, (3, 0): 1,
            }.get((pid, eid), 0)
            if fmt in (4, 12) and (best is None or score > best[0]):
                best = (score, fmt, sub)
        if best is None:
            return

        _score, self.fmt, self.sub = best
        if self.fmt == 12:
            n_groups = struct.unpack(">I", data[self.sub + 12:self.sub + 16])[0]
            for i in range(n_groups):
                off = self.sub + 16 + i * 12
                start, end, gid = struct.unpack(">III", data[off:off + 12])
                self.groups.append((start, end, gid))
        else:
            seg_x2 = struct.unpack(">H", data[self.sub + 6:self.sub + 8])[0]
            self.seg_count = seg_x2 // 2
            p = self.sub + 14
            self.ends = [
                struct.unpack(">H", data[p + i * 2:p + i * 2 + 2])[0]
                for i in range(self.seg_count)
            ]
            p += seg_x2 + 2                        # +2 跳过 reservedPad
            self.start_codes = data[p:p + seg_x2]
            p += seg_x2
            self.id_deltas = data[p:p + seg_x2]
            p += seg_x2
            self.id_range_offsets = data[p:p + seg_x2]

    def gid(self, code: int) -> int:
        import bisect

        if self.fmt == 12:
            if not self.groups:
                return 0
            starts = [g[0] for g in self.groups]
            i = bisect.bisect_right(starts, code) - 1
            if i < 0:
                return 0
            start, end, gid = self.groups[i]
            return gid + (code - start) if code <= end else 0

        if self.fmt == 4 and self.seg_count:
            i = bisect.bisect_left(self.ends, code)
            if i >= self.seg_count:
                return 0
            start = struct.unpack(">H", self.start_codes[i * 2:i * 2 + 2])[0]
            if code < start:
                return 0
            delta = struct.unpack(">h", self.id_deltas[i * 2:i * 2 + 2])[0]
            range_off = struct.unpack(">H", self.id_range_offsets[i * 2:i * 2 + 2])[0]
            if range_off == 0:
                return (code + delta) & 0xFFFF
            addr = self.sub + 14 + self.seg_count * 6 + 2 + range_off + (code - start) * 2
            if addr + 2 > len(self.data):
                return 0
            glyph = struct.unpack(">H", self.data[addr:addr + 2])[0]
            return 0 if glyph == 0 else (glyph + delta) & 0xFFFF
        return 0


class _TrueTypeFont:
    """够用就好的 TrueType 读取器：字形 id、字宽、字体度量"""

    def __init__(self, path: str):
        self.path = str(path)
        self.data = Path(self.path).read_bytes()
        base = _sfnt_base(self.data)
        self.version, self.tables = _read_tables(self.data, base)

        head = self._table(b"head")
        if len(head) < 54:
            raise ValueError("head 表损坏")
        self.units_per_em = struct.unpack(">H", head[18:20])[0] or 1000
        self.bbox = struct.unpack(">hhhh", head[36:44])

        hhea = self._table(b"hhea")
        self.ascent = struct.unpack(">h", hhea[4:6])[0]
        self.descent = struct.unpack(">h", hhea[6:8])[0]
        self.num_h_metrics = struct.unpack(">H", hhea[34:36])[0] or 1

        maxp = self._table(b"maxp")
        self.num_glyphs = struct.unpack(">H", maxp[4:6])[0] if len(maxp) >= 6 else 0

        os2 = self._table(b"OS/2")
        if len(os2) >= 90 and struct.unpack(">H", os2[0:2])[0] >= 2:
            self.cap_height = struct.unpack(">h", os2[88:90])[0] or self.ascent
        else:
            self.cap_height = self.ascent

        post = self._table(b"post")
        self.italic_angle = (
            struct.unpack(">i", post[4:8])[0] / 65536.0 if len(post) >= 8 else 0.0
        )

        self.hmtx_off = self.tables[b"hmtx"][0]
        self.cmap = _CmapLookup(self.data, self.tables)
        self.used: dict[int, str] = {}             # glyph id -> 字符（用于生成 ToUnicode）

        stem = re.sub(r"[^A-Za-z0-9]", "", Path(self.path).stem) or "EmbeddedCJK"
        self.ps_name = stem[:40]

    def _table(self, tag: bytes) -> bytes:
        if tag not in self.tables:
            return b""
        off, length = self.tables[tag]
        return self.data[off:off + length]

    def gid(self, ch: str) -> int:
        try:
            return self.cmap.gid(ord(ch))
        except (TypeError, ValueError):
            return 0

    def advance_gid(self, gid: int) -> float:
        """字形宽度，换算成 1/1000 em"""
        if self.num_glyphs == 0:
            return 500.0
        index = min(gid, self.num_h_metrics - 1)
        off = self.hmtx_off + index * 4
        if off + 2 > len(self.data):
            return 500.0
        return struct.unpack(">H", self.data[off:off + 2])[0] * 1000.0 / self.units_per_em

    def advance(self, ch: str) -> float:
        return self.advance_gid(self.gid(ch))

    def encode(self, text: str) -> str:
        """把文本编码成 Identity-H 的 2 字节 glyph 序列（返回十六进制字符串）"""
        parts = []
        for ch in text:
            gid = self.gid(ch)
            if gid == 0 and ch not in (" ", "\u3000", "\t"):
                fallback = self.gid("?")        # 字体里缺这个字，降级成 ?
                if fallback:
                    gid = fallback
                    self.used[fallback] = "?"
                else:
                    gid = 0
            elif gid:
                self.used[gid] = ch
            parts.append("%04X" % (gid & 0xFFFF))
        return "".join(parts)

    def to_unicode_cmap(self) -> bytes:
        """生成 ToUnicode CMap，保证 pypdf/Acrobat 能抽出正确的中文"""
        items = sorted((g, c) for g, c in self.used.items() if g != 0)
        lines = [
            "/CIDInit /ProcSet findresource begin",
            "12 dict begin",
            "begincmap",
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
            "/CMapName /Adobe-Identity-UCS def",
            "/CMapType 2 def",
            "1 begincodespacerange",
            "<0000> <FFFF>",
            "endcodespacerange",
        ]
        for i in range(0, len(items), 100):
            chunk = items[i:i + 100]
            lines.append(f"{len(chunk)} beginbfchar")
            for gid, ch in chunk:
                try:
                    hex_text = ch.encode("utf-16-be").hex().upper()
                except UnicodeEncodeError:
                    continue
                lines.append(f"<{gid:04X}> <{hex_text}>")
            lines.append("endbfchar")
        lines += [
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
        return ("\n".join(lines) + "\n").encode("latin-1")


class _Base14Font:
    """没有中文字体时的降级字体：Helvetica + WinAnsi，非 ASCII 一律变 ?"""

    ps_name = "Helvetica"

    # Helvetica 标准字宽（1/1000 em），ASCII 32..126
    _WIDTHS = [
        278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
        556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
        1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
        667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
        333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
        556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
    ]

    def __init__(self):
        self.used = {}

    def _byte(self, ch: str) -> int:
        try:
            encoded = ch.encode("cp1252")
        except UnicodeEncodeError:
            return ord("?")
        return encoded[0]

    def advance(self, ch: str) -> float:
        code = self._byte(ch)
        if 32 <= code <= 126:
            return float(self._WIDTHS[code - 32])
        return 556.0

    def encode(self, text: str) -> str:
        return "".join("%02X" % self._byte(ch) for ch in text)


# ---------- 3.2 极简 PDF 写入器 ----------

class _PdfBuilder:
    """只做「拼对象 + 写 xref」，够画文字和线条就行"""

    def __init__(self):
        self.objects: dict[int, bytes] = {}
        self._next = 1

    def reserve(self) -> int:
        num = self._next
        self._next += 1
        return num

    def set(self, num: int, body: bytes) -> None:
        self.objects[num] = body

    def add(self, body: bytes) -> int:
        num = self.reserve()
        self.set(num, body)
        return num

    def build(self, root_ref: int) -> bytes:
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = {}
        for num in sorted(self.objects):
            offsets[num] = len(out)
            out += b"%d 0 obj\n" % num
            out += self.objects[num]
            out += b"\nendobj\n"

        start_xref = len(out)
        size = self._next
        out += b"xref\n0 %d\n" % size
        out += b"0000000000 65535 f \n"
        for num in range(1, size):
            if num in offsets:
                out += b"%010d 00000 n \n" % offsets[num]
            else:
                out += b"0000000000 65535 f \n"
        out += (b"trailer\n<< /Size %d /Root %d 0 R "
                b"/Producer (agent-intern-roadmap pdf_export) >>\n" % (size, root_ref))
        out += b"startxref\n%d\n%%%%EOF\n" % start_xref
        return bytes(out)


def _stream(dict_extra: str, data: bytes) -> bytes:
    head = "<< %s /Length %d >>\nstream\n" % (dict_extra.strip(), len(data))
    return head.encode("latin-1") + data + b"\nendstream"


def _wrap_text(text: str, font, size: float, max_width: float) -> list:
    """按字宽折行：拉丁词优先在空格断，中文逐字断"""
    lines = []
    for para in str(text).split("\n"):
        if not para.strip():
            lines.append("")
            continue
        current = ""
        width = 0.0
        for ch in para:
            char_w = font.advance(ch) * size / 1000.0
            if current and width + char_w > max_width:
                cut = current.rfind(" ")
                if cut > len(current) * 0.6:        # 空格靠后 → 在空格处断
                    lines.append(current[:cut])
                    current = current[cut + 1:]
                    width = sum(font.advance(c) * size / 1000.0 for c in current)
                else:
                    lines.append(current)
                    current = ""
                    width = 0.0
            current += ch
            width += char_w
        lines.append(current)
    return lines


def _stdlib_layout(blocks: list, font) -> list:
    """把版面块排进若干页，返回每页的 content stream（字节）"""
    pages: list[str] = []
    ops: list[str] = []

    def flush():
        nonlocal ops
        if ops:
            pages.append("\n".join(ops))
        ops = []

    y = PAGE_H - MARGIN
    for block in blocks:
        kind = block.get("kind", "body")

        if kind == "space":
            y -= float(block.get("size", 6))
            continue

        size = _SIZES.get(kind, 10.5)
        color = _COLORS.get(kind, (0, 0, 0))
        line_h = size * 1.55
        indent = _INDENTS.get(kind, 0.0)
        max_width = PAGE_W - 2 * MARGIN - indent

        text = block.get("text", "")
        lines = _wrap_text(text, font, size, max_width) if text else [""]

        for line in lines:
            if y - line_h < MARGIN:                 # 空间不够就换页
                flush()
                y = PAGE_H - MARGIN
            y -= line_h
            if not line:
                continue
            hex_text = font.encode(line)
            x = MARGIN + indent
            if kind == "h1":                        # 姓名居中
                line_w = sum(font.advance(c) * size / 1000.0 for c in line)
                x = max(MARGIN, (PAGE_W - line_w) / 2)
            ops.append("%.3f %.3f %.3f rg" % color)
            ops.append("BT /F1 %.2f Tf 1 0 0 1 %.2f %.2f Tm <%s> Tj ET" % (size, x, y, hex_text))

        if kind == "h2":                            # 小节标题下的横线
            rule_y = y - 4
            if rule_y < MARGIN:
                flush()
                y = PAGE_H - MARGIN
                rule_y = y - 4
            ops.append("0.75 0.78 0.85 rg")
            ops.append("%.2f %.2f %.2f 0.7 re f" % (MARGIN, rule_y, PAGE_W - 2 * MARGIN))

    flush()
    if not pages:
        pages.append("")
    return [page.encode("latin-1") for page in pages]


def _embed_font(builder: _PdfBuilder, font) -> int:
    """把字体写进 PDF，返回 /F1 指向的对象号"""
    if isinstance(font, _Base14Font):
        return builder.add(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding >>"
        )

    font_ref = builder.reserve()
    cid_ref = builder.reserve()
    desc_ref = builder.reserve()
    file_ref = builder.reserve()
    tou_ref = builder.reserve()

    program = _subset_font(font, list(font.used))
    builder.set(file_ref, _stream(f"/Length1 {len(program)}", program))

    scale = 1000.0 / font.units_per_em
    x_min, y_min, x_max, y_max = [int(round(v * scale)) for v in font.bbox]
    ascent = int(round(font.ascent * scale))
    descent = int(round(font.descent * scale))
    cap_height = int(round(font.cap_height * scale))

    builder.set(desc_ref, (
        "<< /Type /FontDescriptor /FontName /%s /Flags 4 "
        "/FontBBox [%d %d %d %d] /ItalicAngle %.1f /Ascent %d /Descent %d "
        "/CapHeight %d /StemV 80 /FontFile2 %d 0 R >>"
        % (font.ps_name, x_min, y_min, x_max, y_max, font.italic_angle,
           ascent, descent, cap_height, file_ref)
    ).encode("latin-1"))

    # /W：只声明用到的字形宽度（其他走 /DW）
    widths = []
    for gid in sorted(font.used):
        if gid == 0:
            continue
        widths.append("%d [%d]" % (gid, int(round(font.advance_gid(gid)))))
    w_array = "[%s]" % " ".join(widths)

    builder.set(cid_ref, (
        "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /%s "
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        "/FontDescriptor %d 0 R /DW 1000 /W %s /CIDToGIDMap /Identity >>"
        % (font.ps_name, desc_ref, w_array)
    ).encode("latin-1"))

    builder.set(tou_ref, _stream("", font.to_unicode_cmap()))
    builder.set(font_ref, (
        "<< /Type /Font /Subtype /Type0 /BaseFont /%s /Encoding /Identity-H "
        "/DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>"
        % (font.ps_name, cid_ref, tou_ref)
    ).encode("latin-1"))
    return font_ref


def _render_stdlib(blocks: list, out_path: Path) -> str:
    font = None
    font_error = ""
    font_path = find_cjk_font()
    if font_path:
        try:
            font = _TrueTypeFont(font_path)
        except Exception as e:                      # noqa: BLE001 - 字体有问题就降级 base14
            font_error = f"{type(e).__name__}: {e}"
    if font is None:
        if font_path:
            print(f"[pdf] 中文字体 {font_path} 不可用（{font_error}），降级 Helvetica，中文显示为 ?")
        else:
            print("[pdf] 未找到中文字体，降级 Helvetica，中文显示为 ?（可用 PDF_CJK_FONT 指定）")
        font = _Base14Font()

    page_streams = _stdlib_layout(blocks, font)

    builder = _PdfBuilder()
    catalog_ref = builder.reserve()
    pages_ref = builder.reserve()
    font_ref = _embed_font(builder, font)

    page_refs = []
    for stream in page_streams:
        page_ref = builder.reserve()
        content_ref = builder.reserve()
        builder.set(content_ref, _stream("", stream))
        page_refs.append((page_ref, content_ref))

    for page_ref, content_ref in page_refs:
        builder.set(page_ref, (
            "<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.2f %.2f] "
            "/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (pages_ref, PAGE_W, PAGE_H, font_ref, content_ref)
        ).encode("latin-1"))

    kids = " ".join("%d 0 R" % ref for ref, _ in page_refs)
    builder.set(pages_ref, (
        "<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_refs))
    ).encode("latin-1"))
    builder.set(catalog_ref, b"<< /Type /Catalog /Pages %d 0 R >>" % pages_ref)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(builder.build(catalog_ref))
    return str(out_path)


# ==========================================================================
# 后端 1 / 2：reportlab、fpdf2（本机未安装，属于「有就用」的优先路径）
# ==========================================================================

def _render_reportlab(blocks: list, out_path: Path) -> str:
    from xml.sax.saxutils import escape

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    font_name = "STSong-Light"                      # reportlab 内置的中文 CID 字体
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    except Exception:                               # noqa: BLE001 - 注册失败就退回 Helvetica
        font_name = "Helvetica"

    styles = {
        kind: ParagraphStyle(
            name=kind,
            fontName=font_name,
            fontSize=_SIZES[kind],
            leading=_SIZES[kind] * 1.55,
            textColor=_COLORS[kind],
            spaceAfter=2,
            leftIndent=_INDENTS.get(kind, 0),
        )
        for kind in ("h1", "contact", "h2", "body", "bullet", "note")
    }
    styles["h1"].alignment = 1                      # 居中

    story = []
    for block in blocks:
        kind = block.get("kind", "body")
        if kind == "space":
            story.append(Spacer(1, float(block.get("size", 6))))
            continue
        story.append(Paragraph(escape(str(block.get("text", ""))) or "&nbsp;", styles[kind]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title="简历",
    ).build(story)
    return str(out_path)


def _render_fpdf(blocks: list, out_path: Path) -> str:
    from fpdf import FPDF

    font_path = find_cjk_font(prefer_ttf=True)
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=MARGIN / 2.834645)
    pdf.set_margins(MARGIN / 2.834645, MARGIN / 2.834645, MARGIN / 2.834645)
    pdf.add_page()

    family = "helvetica"
    if font_path:
        try:
            pdf.add_font("cjk", "", font_path)
            family = "cjk"
        except Exception as e:                      # noqa: BLE001 - 字体加不进去就退回 helvetica
            print(f"[pdf] fpdf2 加载字体 {font_path} 失败（{e}），改用 helvetica")

    def write(text: str, size: float, color, indent: float = 0.0):
        pdf.set_font(family, size=size)
        pdf.set_text_color(*[int(c * 255) for c in color])
        pdf.set_x(pdf.l_margin + indent)
        try:
            pdf.multi_cell(0, size * 1.55, text=text, new_x="LMARGIN", new_y="NEXT")
        except TypeError:                           # 旧版 fpdf2 没有 new_x/new_y
            pdf.multi_cell(0, size * 1.55, txt=text)
            pdf.ln(size * 0.2)

    for block in blocks:
        kind = block.get("kind", "body")
        if kind == "space":
            pdf.ln(float(block.get("size", 6)))
            continue
        write(str(block.get("text", "")), _SIZES[kind], _COLORS[kind], _INDENTS.get(kind, 0.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return str(out_path)


# ==========================================================================
# 对外入口
# ==========================================================================

def export_resume_pdf(resume: dict, output_path: str) -> str:
    """把简历 dict 导出为 PDF，返回实际写入的文件路径。

    resume 可以是结构化简历、storage 的简历记录（含 content）、
    或一段纯文本/JSON 字符串。output_path 的父目录会自动创建。
    首选后端失败时自动降级到下一个可用后端。
    """
    path = Path(output_path)
    blocks = resume_blocks(resume)

    order = []
    for name in (available_backend(), "stdlib"):
        if name not in order:
            order.append(name)

    errors = []
    for backend in order:
        renderer = {
            "reportlab": _render_reportlab,
            "fpdf2": _render_fpdf,
            "stdlib": _render_stdlib,
        }[backend]
        try:
            return renderer(blocks, path)
        except Exception as e:                      # noqa: BLE001 - 换下一个后端再试
            errors.append(f"{backend} 失败：{type(e).__name__}: {e}")

    raise RuntimeError("所有 PDF 后端都失败了：" + "；".join(errors))


if __name__ == "__main__":
    import sys
    import tempfile

    sample = {
        "name": "张三",
        "city": "广州",
        "education": "本科",
        "skills": ["Python", "PyTorch", "RAG", "LangChain", "SQL"],
        "experience": [
            {"company": "某科技公司", "role": "算法实习生", "months": 3,
             "description": "参与 RAG 问答系统开发与评测，负责检索召回优化。"},
            "2025.03-至今 某创业公司 后端实习生：用 FastAPI 写了岗位检索接口",
        ],
        "projects": [
            {"name": "基于 ReAct 的求职助手 Agent",
             "tech": ["Python", "LLM", "工具调用"],
             "desc": "实现工具调用 + 评估集，10 个任务的自动化评测。"},
        ],
    }

    out = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(tempfile.gettempdir()) / "resume_sample.pdf"
    )
    print(f"后端：{available_backend()}　字体：{find_cjk_font()}")
    print("导出：", export_resume_pdf(sample, out))
    print("大小：", Path(out).stat().st_size, "bytes")
