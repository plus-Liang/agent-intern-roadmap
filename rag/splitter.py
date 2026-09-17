import re

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
    parts = PATTERN.split(text)
    chunks = []
    if parts and parts[0].strip():
        head = parts[0].strip()
        if len(head) >= 20:
            chunks.append(head)

    i = 1
    while i < len(parts) - 1:
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        block = f"{title}\n{content}".strip()
        if len(block) < 20:
            i += 2
            continue
        if len(block) > chunk_size * 1.5:
            chunks.extend(_split_by_sentence(block, chunk_size))
        else:
            chunks.append(block)
        i += 2

    if not chunks:
        return [text.strip()]
    return chunks


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