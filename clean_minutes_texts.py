#!/usr/bin/env python3
"""
Clean formatting of FOMC minutes text files with safe mid-sentence joins.

- Input: minutes_text/<year>/*.txt  (or --in-root elsewhere)
- Output (default): minutes_text_clean/<year>/*.txt  (or --inplace to overwrite)
- Default behavior:
    * Unicode normalize (NFKC), remove control chars/form-feeds
    * Remove page/date artifacts (e.g., 3/17/41, -5-)
    * Fix hyphenated linebreaks (eco-\nnomy -> economy)
    * Join BLANK-LINE breaks mid-sentence (e.g., "in\n\n the" -> "in the")
    * Rebuild paragraphs (single newlines inside paragraphs -> spaces)
    * DOES NOT glue "word splits" like "of the" -> "ofthe"

- Optional:
    --glue-splits   : cautiously glue true splits like "Mar ket"->"Market"
                      (protected by a stopword list to avoid "of the" -> "ofthe")
    --aggressive    : join blank-line breaks before Titlecase words as well

Usage:
  python clean_minutes_texts.py 1941
  python clean_minutes_texts.py 1936 1941 2008 --aggressive
  # PowerShell range:
  #   $years = 2008..2025; python clean_minutes_texts.py $years --inplace
"""

from __future__ import annotations
import argparse
import re
import unicodedata
from pathlib import Path
from typing import List

# -------- heuristics / regexes --------

FF = "\f"

# Date headers like 3/17/41, 03/17/1941, even truncated 3/17/1 on its own line
DATE_LINE_RX = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{1,4}\s*$", re.MULTILINE)

# Page numbers like "-2", "-5-", "—7—", "-- 12 --"
DASHED_PAGENO_RX = re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]?\s*$", re.MULTILINE)

# Bare page numbers (just "7") — remove only if isolated between blanks
BARE_PAGENO_RX = re.compile(r"^\s*\d+\s*$", re.MULTILINE)

# Hyphen at end of line joining to next word (also across accidental blank line)
HYPHEN_BREAK_ACROSS_BLANK_RX = re.compile(r"(\w)-\n\s*\n(\w)")
HYPHEN_BREAK_RX = re.compile(r"(\w)-\n(\w)")

# Collapse 3+ blank lines
MULTIBLANK_RX = re.compile(r"\n{3,}")

# Collapse runs of spaces/tabs
SPACES_RX = re.compile(r"[ \t]{2,}")

# Control chars except \n and \t
CTRL_RX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Join blank-line breaks mid-sentence:
BLANK_JOIN_STRICT_RX = re.compile(r"([^\.\?\!\:;\)”’\]\}])\n\s*\n(?=[a-z])")
BLANK_JOIN_AGGR_RX = re.compile(r"([^\.\?\!\:;\)”’\]\}])\n\s*\n(?=[A-Z][a-z])")

# OPTIONAL: cautious gluing of true split words (off by default)
STOPWORDS = {
    # common function words to NEVER glue across
    "a","an","and","or","of","the","to","in","on","at","by","for","from","with","without",
    "as","if","than","then","that","this","these","those","is","are","was","were","be",
    "been","being","not","no","nor","but","so","such","into","upon","over","under","out",
    "up","down","off","per","via"
}
GLUE_SPLIT_RX = re.compile(r"\b([A-Za-z]{3,})\s+([a-z]{3,})\b")  # both parts >=3; second must be lowercase


def _strip_bare_page_numbers_safely(text: str) -> str:
    """Remove bare page number lines only when isolated (blank above/below)."""
    lines = text.splitlines()
    out: List[str] = []
    n = len(lines)
    for i, ln in enumerate(lines):
        if BARE_PAGENO_RX.match(ln):
            prev_blank = (i == 0) or (not lines[i - 1].strip())
            next_blank = (i == n - 1) or (not lines[i + 1].strip())
            if prev_blank and next_blank:
                continue  # drop isolated page number
        out.append(ln)
    return "\n".join(out)


def _join_blanklines_mid_sentence(s: str, aggressive: bool) -> str:
    """Join blank-line breaks likely splitting a sentence."""
    for _ in range(3):
        new_s = BLANK_JOIN_STRICT_RX.sub(r"\1 ", s)
        if aggressive:
            new_s = BLANK_JOIN_AGGR_RX.sub(r"\1 ", new_s)
        if new_s == s:
            break
        s = new_s
    return s


def _cautious_glue_splits(s: str) -> str:
    """
    VERY conservative glue: only when both fragments are >=3 letters,
    the second is lowercase, and neither side is a stopword.
    This avoids 'of the' -> 'ofthe', etc.
    """
    def repl(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        if a.lower() in STOPWORDS or b in STOPWORDS:
            return m.group(0)  # keep as-is
        # Don't glue if looks like proper phrase (Title + word), e.g., 'New York'
        if a[0].isupper():
            return m.group(0)
        # Safe glue
        return a + b

    for _ in range(2):
        s2 = GLUE_SPLIT_RX.sub(repl, s)
        if s2 == s:
            break
        s = s2
    return s


def clean_text(raw: str, aggressive: bool = False, glue_splits: bool = False) -> str:
    # Normalize unicode and line endings; drop control chars
    s = unicodedata.normalize("NFKC", raw).replace("\xa0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = CTRL_RX.sub("", s)

    # Convert form feeds to paragraph breaks
    s = s.replace(FF, "\n\n")

    # Remove clear page/date artifacts
    s = DATE_LINE_RX.sub("", s)
    s = DASHED_PAGENO_RX.sub("", s)
    s = _strip_bare_page_numbers_safely(s)

    # Fix hyphenation across line and across blank line
    s = HYPHEN_BREAK_ACROSS_BLANK_RX.sub(r"\1\2", s)
    s = HYPHEN_BREAK_RX.sub(r"\1\2", s)

    # Collapse long blank runs first
    s = MULTIBLANK_RX.sub("\n\n", s)

    # Join blank-line breaks mid-sentence
    s = _join_blanklines_mid_sentence(s, aggressive=aggressive)

    # Rebuild paragraphs: keep blank-line breaks, join single line breaks into spaces
    blocks = re.split(r"\n\s*\n+", s.strip())
    merged_blocks: List[str] = []
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        para = " ".join(lines)
        para = SPACES_RX.sub(" ", para)
        merged_blocks.append(para)
    s = "\n\n".join(merged_blocks).strip()

    # OPTIONAL: cautiously glue true split words (disabled by default)
    if glue_splits:
        s = _cautious_glue_splits(s)

    # Final tidy
    s = SPACES_RX.sub(" ", s).strip()
    return s


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean FOMC minutes text files (safe joins; no word gluing by default).")
    ap.add_argument("years", nargs="+", type=int, help="Year(s) to process (e.g., 1941 2008)")
    ap.add_argument("--in-root", default="minutes_text", help="Input root containing <year>/*.txt")
    ap.add_argument("--out-root", default="minutes_text_clean", help="Output root (default: minutes_text_clean)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input files in-place instead of writing to out-root")
    ap.add_argument("--aggressive", action="store_true", help="Aggressively join blank-line breaks before Titlecase words")
    ap.add_argument("--glue-splits", action="store_true", help="Conservatively glue true word splits (stopword-protected)")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N files per year (0 = no limit)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    in_root = Path(args.in_root)
    if not in_root.exists():
        print(f"[ERR] Input root not found: {in_root}")
        return 1

    out_root = in_root if args.inplace else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total = ok = 0
    for year in args.years:
        in_year = in_root / str(year)
        if not in_year.exists():
            print(f"[WARN] Missing folder for year {year}: {in_year}")
            continue

        files = sorted(p for p in in_year.rglob("*.txt") if p.is_file())
        if args.limit > 0:
            files = files[:args.limit]

        print(f"[INFO] Year {year}: {len(files)} files")
        for f in files:
            total += 1
            rel = f.relative_to(in_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
                cleaned = clean_text(raw, aggressive=args.aggressive, glue_splits=args.glue_splits)
                out_path.write_text(cleaned, encoding="utf-8")
                ok += 1
                print(f"[OK]   {rel} -> {out_path.relative_to(out_root)} (chars {len(cleaned)})")
            except Exception as e:
                print(f"[ERR]  {rel}: {e}")

    print(f"[DONE] cleaned={ok}/{total} -> {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
