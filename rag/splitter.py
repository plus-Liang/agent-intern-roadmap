import re

# 头部元信息行（公司/岗位/城市/薪资/来源链接/发布时间、分隔线）——只用于
# _prune_header 在极端情况下缩短头部，不参与切块判定
_META_LINE_RE = re.compile(r"^【\d+】公司[:：]|^(公司|岗位|城市|薪资|来源链接|发布时间)[:：]")

SECTION_TITLES = [
    "岗位职责", "工作职责", "任职要求", "职位要求",
    "加分项", "岗位福利", "实习时长", "岗位要求",
]

_TITLE_ALT = "|".join(SECTION_TITLES)
PATTERN = re.compile(
    r"(【(?:" + _TITLE_ALT + r")】|^(?:" + _TITLE_ALT + r")\s*[:：]?)",
    re.MULTILINE,
)


def split_text(text: str, chunk_size: int = 600) -> list[str]:
    """按标题把一条 JD 切成若干 chunk。

    第一个标题之前的部分是头部元信息（【n】公司：/岗位：/城市：/薪资：/
    来源链接：/发布时间：/分隔线），它不单独成 chunk，而是合并到第一个
    正文块（通常是【岗位职责】，没有【岗位职责】时就是【任职要求】等
    紧随其后的段落）。因此「头部 + 【岗位职责】」是一个 chunk，
    【任职要求】【加分项】各自独立成 chunk。
    """
    parts = PATTERN.split(text)
    chunks = []
    # 头部元信息：留给第一个正文块合并；如果整条 JD 没有任何标题段落，
    # 就直接作为唯一的 chunk 输出
    head = parts[0].strip() if parts else ""

    i = 1
    while i < len(parts) - 1:
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        block = f"{title}\n{content}".strip()
        if len(block) < 20:
            i += 2
            continue
        if head:
            block = f"{head}\n{block}"
            head = ""
        if len(block) > chunk_size * 1.5:
            chunks.extend(_split_by_sentence(block, chunk_size))
        else:
            chunks.append(block)
        i += 2

    if not chunks:
        # 没有任何分段：整条文本原样返回（保持旧行为）
        return [text.strip()]
    if head:
        # 所有段落块都因太短被丢弃，头部仍然保留下来
        chunks.insert(0, _prune_header(head, chunk_size))
    return chunks


def _prune_header(head: str, chunk_size: int) -> str:
    """头部过长时逐行砍掉末尾的元信息，保证头部不超过 chunk_size。

    正常情况下头部只有 5～7 行（约 220 字），走不到这里；这只是在超长
    头部下避免把正文挤掉的兜底。
    """
    if len(head) <= chunk_size:
        return head
    lines = [line for line in head.splitlines() if line.strip()]
    meta = [line for line in lines if _META_LINE_RE.match(line.strip())]
    if len(meta) < len(lines):
        # 头部里混着正文，砍行会丢正文，原样返回
        return head
    while len(meta) > 1 and len("\n".join(meta)) > chunk_size:
        meta.pop()
    return "\n".join(meta)


def _split_by_sentence(text: str, chunk_size: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？\n])", text)
    sentences = [s for s in sentences if s.strip()]
    result = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += sent
        else:
            if current:
                result.append(current.strip())
            current = sent
    if current:
        result.append(current.strip())
    return result


def split_jds(jds: list[dict], chunk_size: int = 600) -> list[dict]:
    all_chunks = []
    for jd in jds:
        chunks = split_text(jd["content"], chunk_size)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "company": jd["company"],
                "title": jd["title"],
                "city": jd["city"],
                "chunk_index": i,
                "text": chunk,
            })
    return all_chunks


if __name__ == "__main__":
    from shared.config import DATA_DIR
    from rag.loader import load_jd_file

    jds = load_jd_file(str(DATA_DIR / "scraped_jd.txt"))
    chunks = split_jds(jds)
    print(f"共 {len(jds)} 条 JD，切成 {len(chunks)} 个 chunk\n")
    for c in chunks:
        first_line = c["text"].split("\n", 1)[0][:30]
        print(f"  {c['company']:<12} chunk#{c['chunk_index']} "
              f"({len(c['text']):>4}字)  {first_line}")