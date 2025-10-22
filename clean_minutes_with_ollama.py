#!/usr/bin/env python3
"""
Batch-clean FOMC minutes using a local LLM via Ollama.

- Input:  minutes_text/<year>/*.txt        (your existing corpus)
- Output: minutes_text_clean/<year>/*.txt  (cleaned text)
- Works chunk-by-chunk to stay within model context.
- Requires: Ollama running at http://127.0.0.1:11434 (default), and a pulled model.

Install deps:
  pip install requests tqdm
Start server:
  ollama serve
Pull a model (examples):
  ollama pull llama3.1:8b-instruct-q4_K_M
  ollama pull phi3:instruct

Examples:
  # clean everything
  python clean_minutes_with_ollama.py --in-root minutes_text --out-root minutes_text_clean --model llama3.1:8b-instruct-q4_K_M

  # clean one year
  python clean_minutes_with_ollama.py --years 2008

  # clean a range (hyphen or two dots both work)
  python clean_minutes_with_ollama.py --years 2008-2012
  python clean_minutes_with_ollama.py --years 2008..2012

  # multiple ranges/years (commas/spaces both ok)
  python clean_minutes_with_ollama.py --years "1936-1940,1951 2008..2010"
"""
from config_paths import resolve_path, ensure_all_dirs
ensure_all_dirs()


from __future__ import annotations
import argparse, os, sys, json, time, re
from pathlib import Path
from typing import Iterable, List, Optional, Set
import requests
from tqdm import tqdm

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

PROMPT_TEMPLATE = """You are a meticulous text normalizer for historical FOMC minutes.

GOAL: Produce ONE SINGLE LINE of clean text (no line breaks at all), preserving the original wording and meaning.

RULES (follow exactly; do NOT summarize, rephrase, or drop content):
- Unwrap hard line breaks so sentences flow, BUT PRESERVE SENTENCE BOUNDARIES: ensure every sentence ends with a period (or ?/!) followed by a single space before the next sentence.
- Preserve and normalize punctuation; do NOT delete periods, commas, semicolons, colons, or parentheses.
- Fix broken words across line breaks and hyphenation splits (e.g., "govern-\\nment" -> "government").
- Fix run-together words (e.g., "ofthe" -> "of the", "marketaccount" -> "market account").
- Preserve list structure by inserting commas or semicolons where the source clearly enumerates names/titles (e.g., attendee rolls). Do NOT merge names without separators.
- Remove page headers/footers, navigation chrome, and “Print” controls (e.g., lines starting with "FRB:", "Print", page numbers like "-5-" or isolated date stamps not in sentences).
- Collapse all sequences of whitespace to single spaces. Remove all line breaks.
- Keep capitalization, numbers, acronyms, and content intact. Do NOT add commentary.

Return ONLY the cleaned text as a single line (no code fences, no notes).

RAW:
<<<
{chunk}
>>>
CLEAN:
"""

def iter_txt_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.txt")):
        if p.is_file():
            yield p

def split_into_chunks(text: str, target_chars: int = 6000) -> List[str]:
    """Chunk by paragraphs so we don't cut sentences mid-stream."""
    paras = re.split(r"\n\s*\n", text)  # split on blank lines
    chunks, buf, size = [], [], 0
    for para in paras:
        p = para.strip("\n")
        add = (("\n\n" if buf else "") + p) if p else ""
        if size + len(add) > target_chars and buf:
            chunks.append("".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(add if add else "")
            size += len(add)
    if buf:
        chunks.append("".join(buf))
    return chunks

def call_ollama_generate(model: str, prompt: str, temperature: float = 0.0, num_ctx: int = 8192, timeout: int = 0) -> str:
    """Call Ollama /api/generate with streaming and collect the response."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx
        }
    }
    with requests.post(url, json=payload, stream=True, timeout=None if timeout == 0 else timeout) as r:
        r.raise_for_status()
        out = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in obj:
                out.append(obj["response"])
            if obj.get("done"):
                break
        return "".join(out)

def clean_chunk(model: str, chunk: str) -> str:
    prompt = PROMPT_TEMPLATE.format(chunk=chunk)
    return call_ollama_generate(model, prompt, temperature=0.0, num_ctx=8192)

def clean_text_with_llm(model: str, raw: str, chunk_chars: int = 6000) -> str:
    chunks = split_into_chunks(raw, target_chars=chunk_chars)
    cleaned_parts: List[str] = []
    for i, ch in enumerate(chunks, 1):
        cleaned = clean_chunk(model, ch)
        cleaned = cleaned.strip()
        cleaned_parts.append(cleaned)
    cleaned_text = "\n\n".join(part.strip() for part in cleaned_parts if part.strip())
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip() + "\n"
    return cleaned_text

def parse_years_arg(s: Optional[str]) -> Optional[Set[int]]:
    """Parse --years arg like '2008', '2008-2012', '2008..2012', or comma/space mixed."""
    if not s:
        return None
    tokens = re.split(r"[,\s]+", s.strip())
    years: Set[int] = set()
    for tok in tokens:
        if not tok:
            continue
        m = re.fullmatch(r"(\d{4})\s*(?:-|\.{2})\s*(\d{4})", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            step = 1 if a <= b else -1
            for y in range(a, b + step, step):
                years.add(y)
            continue
        if re.fullmatch(r"\d{4}", tok):
            years.add(int(tok))
            continue
        raise ValueError(f"Unrecognized year token: '{tok}'. Use 4-digit years and '-' or '..' for ranges.")
    if not years:
        return None
    return years

def year_from_relative_path(p: Path) -> Optional[int]:
    """Try to infer a 4-digit year from any part of the relative path."""
    for part in p.parts:
        if re.fullmatch(r"\d{4}", part):
            try:
                return int(part)
            except ValueError:
                pass
    return None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root",  default="minutes_text", help="Input root with raw .txt files")
    ap.add_argument("--out-root", default="minutes_text_clean", help="Output root for cleaned .txt files")
    ap.add_argument("--model",    default="llama3.1:8b-instruct-q4_K_M", help="Ollama model name")
    ap.add_argument("--limit",    type=int, default=0, help="Max files to process (debug)")
    ap.add_argument("--chunk-chars", type=int, default=6000, help="Approx input chars per chunk")
    ap.add_argument("--years",    type=str, default="", help="Filter to specific years/ranges, e.g. '1936-1940,1951 2008..2010'")
    args = ap.parse_args()

    # Resolve and create output root
    in_root  = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover files
    all_files = list(iter_txt_files(in_root))

    # Filter by years if requested
    selected_years = parse_years_arg(args.years)
    if selected_years:
        files = []
        for p in all_files:
            rel = p.relative_to(in_root)
            y = year_from_relative_path(rel)
            if y is not None and y in selected_years:
                files.append(p)
        print(f"[INFO] Year filter: {sorted(selected_years)} -> {len(files)} files matched")
    else:
        files = all_files

    if args.limit:
        files = files[:args.limit]

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Input : {in_root}  (files: {len(files)})")
    print(f"[INFO] Output: {out_root}")
    if not files:
        print("[WARN] No .txt files found after filtering. Check --in-root / --years.")
        return 0

    errors = 0
    for p in tqdm(files, desc="Cleaning files", unit="file"):
        rel = p.relative_to(in_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text_with_llm(args.model, raw, chunk_chars=args.chunk_chars)
            out_path.write_text(cleaned, encoding="utf-8")
        except Exception as e:
            errors += 1
            print(f"\n[ERR] {p}: {e}")
            continue
        time.sleep(0.02)  # small breather
    print(f"[DONE] Cleaned {len(files)-errors}/{len(files)} files → {out_root}")
    return 0

if __name__ == "__main__":
    sys.exit(main())