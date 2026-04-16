import os
import re

DATA_FOLDER = "/Users/dedeepyaguntaka/Library/CloudStorage/GoogleDrive-dg4489@nyu.edu/My Drive/Checkit_Analytics_RAG"

CHUNK_SIZE = 800   # larger chunks = more financial context per vector
OVERLAP = 150

# SEC section headers for 10-K / 10-Q — used to split before sliding window
SECTION_HEADERS = re.compile(
    r"(?im)^(?:item\s+\d+[a-z]?\.?\s+)?"
    r"("
    r"management.{0,10}discussion|"
    r"results of operations|"
    r"risk factors|"
    r"financial statements|"
    r"quantitative and qualitative|"
    r"liquidity and capital|"
    r"critical accounting|"
    r"business overview|"
    r"selected financial data|"
    r"controls and procedures|"
    r"legal proceedings"
    r")[^\n]*$"
)

MONTH_TO_QUARTER = {
    1: "Q1", 2: "Q1", 3: "Q1",
    4: "Q2", 5: "Q2", 6: "Q2",
    7: "Q3", 8: "Q3", 9: "Q3",
    10: "Q4", 11: "Q4", 12: "Q4",
}

SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z\s\-'\.]{2,40}):\s", re.MULTILINE)


def derive_quarter(date_str):
    try:
        parts = date_str.split("-")
        month = int(parts[1])
        year = parts[0]
        return f"{MONTH_TO_QUARTER[month]} {year}"
    except Exception:
        return "Unknown"


def split_sentences(text):
    """Split text into sentences on . ! ? boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p for p in parts if p.strip()]


def chunk_8k(text):
    """Chunk 8-K by speaker turns, max 600 chars, tag speaker."""
    chunks = []
    lines = text.splitlines()
    current_speaker = "N/A"
    current_block = []

    def flush_block(speaker, block_lines):
        block_text = " ".join(" ".join(block_lines).split())
        if not block_text.strip():
            return []
        result = []
        # Split into max 600-char sentence-respecting chunks
        sentences = split_sentences(block_text)
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) + 1 <= CHUNK_SIZE:
                buf = (buf + " " + sent).strip()
            else:
                if buf:
                    result.append((speaker, buf))
                buf = sent[:CHUNK_SIZE]
        if buf:
            result.append((speaker, buf))
        return result

    for line in lines:
        m = SPEAKER_RE.match(line)
        if m:
            # flush current block
            chunks.extend(flush_block(current_speaker, current_block))
            current_speaker = m.group(1).strip()
            rest = line[m.end():].strip()
            current_block = [rest] if rest else []
        else:
            current_block.append(line)

    chunks.extend(flush_block(current_speaker, current_block))
    return chunks  # list of (speaker, text)


def chunk_sliding(text):
    """Sliding window chunking."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += CHUNK_SIZE - OVERLAP
    return chunks


def chunk_by_section(text):
    """
    Split 10-K/10-Q by SEC section headers, then apply sliding window within
    each section. Prepends the section title to every chunk for context.
    Falls back to plain sliding window if no sections are found.
    """
    boundaries = [(m.start(), m.group(0).strip()) for m in SECTION_HEADERS.finditer(text)]

    if not boundaries:
        return chunk_sliding(text)

    sections = []
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()
        sections.append((title, section_text))

    chunks = []
    for title, section_text in sections:
        for chunk in chunk_sliding(section_text):
            chunks.append(f"[{title}] {chunk}")
    return chunks


def parse_filename(fname):
    """
    Parse both filename formats:
      Old: AAPL_8-K_2024-11-01.txt       → (AAPL, 8-K, 2024-11-01)
      New: AAPL_8-K_2025_Q3_2025-08-01.txt → (AAPL, 8-K, 2025-08-01)
    Returns (ticker, form_type, date_str) or None if unparseable.
    """
    base = fname[:-4]
    parts = base.split("_")
    if len(parts) < 3:
        return None
    ticker = parts[0]
    form_type = parts[1]
    # New format: TICKER_FORMTYPE_YEAR_Q#_DATE  (5 parts, parts[3] starts with Q)
    if len(parts) == 5 and parts[3].startswith("Q"):
        date_str = parts[4]
    # Old format: TICKER_FORMTYPE_DATE (3 parts)
    elif len(parts) == 3:
        date_str = parts[2]
    # Legacy edge case: TICKER_10_K_DATE or TICKER_8_K_DATE
    elif len(parts) == 4 and parts[1] in ("10", "8"):
        form_type = f"{parts[1]}-{parts[2]}"
        date_str = parts[3]
    else:
        # Fallback: last part as date
        date_str = parts[-1]
    return ticker, form_type, date_str


def load_all_chunks():
    all_chunks = []
    file_count = 0

    for root, dirs, files in os.walk(DATA_FOLDER):
        dirs.sort()
        for fname in sorted(files):
            if not fname.endswith(".txt"):
                continue
            parsed = parse_filename(fname)
            if parsed is None:
                continue
            ticker, form_type, date_str = parsed

            quarter = derive_quarter(date_str)
            fpath = os.path.join(root, fname)

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                print(f"  Could not read {fname}: {e}")
                continue

            # 10-K and 10-Q: section-aware chunking for better retrieval coherence
            # 8-K: plain sliding window (press releases have no standard sections)
            if form_type in ("10-K", "10-Q"):
                raw_chunks = chunk_by_section(text)
            else:
                raw_chunks = chunk_sliding(text)
            file_chunks = []
            for idx, chunk_text in enumerate(raw_chunks):
                file_chunks.append({
                    "ticker": ticker,
                    "form_type": form_type,
                    "date": date_str,
                    "quarter": quarter,
                    "speaker": "N/A",
                    "chunk_index": idx,
                    "source_file": fname,
                    "text": chunk_text,
                })

            print(f"  {ticker} {form_type} {date_str} → {len(file_chunks)} chunks")
            all_chunks.extend(file_chunks)
            file_count += 1

    print(f"\nTotal: {len(all_chunks):,} chunks from {file_count} files")
    return all_chunks


if __name__ == "__main__":
    load_all_chunks()
