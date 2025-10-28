from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional
import argparse
import csv
import re

import spacy

from config_paths import resolve_path, ensure_all_dirs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze FOMC minutes sentiments by summing lexicon scores over lemmatized tokens.

Inputs (defaults; override with flags):
  - minutes_text_clean/<year>/*.txt     (--in-root)
  - Economic_Lexicon.csv                (--lexicon)

Output:
  - sentiment_scores.csv                (--out-csv)
  - sentiment_hits/ per-doc token hits  (--hits-dir, optional)

Behavior:
  * Uses spaCy lemmatization only (no other tokenizers).
  * Matches lexicon entries on lemma (lowercased).
  * Sums 'sentiment' values; tracks polarity counts and coverage metrics.
  * Extracts date from filename (flexible: supports YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD anywhere in the name).
  * Filters documents by year range (--start-year to --end-year, inclusive).

Usage:
  python analyze_sentiments.py
  python analyze_sentiments.py --in-root minutes_text_clean --lexicon Economic_Lexicon.csv --out-csv sentiment_scores.csv
"""
ensure_all_dirs()



def load_lexicon(path: Path) -> Dict[str, Tuple[float, int]]:
    """
    Return dict: lemma -> (sentiment: float, polarity: int{-1,0,1}).
    CSV header must have: token,sentiment,polarity
    """
    lex: Dict[str, Tuple[float, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tok = (row.get("token") or "").strip().lower()
            if not tok:
                continue
            try:
                s = float((row.get("sentiment") or "0").strip())
            except Exception:
                s = 0.0
            try:
                p = int(float((row.get("polarity") or "0").strip()))
            except Exception:
                p = 0
            lex[tok] = (s, p)
    return lex


def iter_doc_files(in_root: Path) -> Iterable[Path]:
    # Search ALL .txt files, not just ones starting with a particular prefix.
    for p in sorted(in_root.rglob("*.txt")):
        if p.is_file():
            yield p


# Accept a wide variety of filename styles:
# ...YYYYMMDD..., ...YYYY-MM-DD..., ...YYYY_MM_DD...
DATE_ANY_RE = re.compile(
    r"(?P<y>19\d{2}|20\d{2})[-_]? (?P<m>\d{2}) [-_]? (?P<d>\d{2})",
    re.IGNORECASE | re.VERBOSE,
)


def extract_date_parts(p: Path) -> Optional[tuple]:
    """
    Return (year, month, day) as ints if found in filename; else None.
    """
    m = DATE_ANY_RE.search(p.name)
    if not m:
        return None
    try:
        y = int(m.group("y"))
        mo = int(m.group("m"))
        d = int(m.group("d"))
        # Basic sanity check for month/day ranges
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return None
        return (y, mo, d)
    except Exception:
        return None


def extract_date_iso(p: Path) -> str:
    parts = extract_date_parts(p)
    if not parts:
        return ""
    y, mo, d = parts
    return f"{y:04d}-{mo:02d}-{d:02d}"


def extract_year(p: Path) -> Optional[int]:
    parts = extract_date_parts(p)
    return parts[0] if parts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=Path, default=Path("minutes_text_clean"))
    ap.add_argument("--lexicon", type=Path, default=Path(resolve_path("Economic_Lexicon.csv")))
    ap.add_argument("--out-csv", type=Path, default=Path(resolve_path("sentiment_scores.csv")))
    ap.add_argument("--hits-dir", type=Path, default=None, help="Optional directory to write per-doc hit lists")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    # Inclusive year range; defaults widened for 1960–2025
    ap.add_argument("--start-year", type=int, default=1960, help="Earliest year to include (default 1960)")
    ap.add_argument("--end-year", type=int, default=2025, help="Latest year to include (default 2025)")
    args = ap.parse_args()

    in_root: Path = args.in_root
    lex_path: Path = args.lexicon
    out_csv: Path = args.out_csv
    hits_dir: Optional[Path] = args.hits_dir if args.hits_dir else None

    if hits_dir:
        hits_dir.mkdir(parents=True, exist_ok=True)

    if args.start_year > args.end_year:
        raise SystemExit(f"--start-year ({args.start_year}) cannot be greater than --end-year ({args.end_year}).")

    nlp = spacy.load(args.spacy_model, disable=["ner", "parser", "textcat"])
    if "lemmatizer" not in nlp.pipe_names:
        nlp.add_pipe("lemmatizer", config={"mode": "rule"}, first=True)

    lex = load_lexicon(lex_path)

    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow([
            "date_iso", "file_path",
            "token_count", "lexicon_hits",
            "sentiment_sum",
            "pos_hits", "neg_hits", "neu_hits",
            "lexicon_coverage_pct"
        ])

        for p in iter_doc_files(in_root):
            yr = extract_year(p)
            # Skip files we can't date or outside the allowed range
            if yr is None or yr < args.start_year or yr > args.end_year:
                continue

            text = p.read_text(encoding="utf-8", errors="ignore")
            doc = nlp(text)

            sent_sum = 0.0
            hits = 0
            pos_hits = neg_hits = neu_hits = 0

            freq: Dict[str, int] = {}
            for tok in doc:
                if tok.is_space or tok.is_punct or tok.is_stop or tok.like_num:
                    continue
                lemma = tok.lemma_.lower().strip()
                if not lemma:
                    continue
                if lemma in lex:
                    s, pval = lex[lemma]
                    freq[lemma] = freq.get(lemma, 0) + 1
                    sent_sum += s
                    hits += 1
                    if pval > 0:
                        pos_hits += 1
                    elif pval < 0:
                        neg_hits += 1
                    else:
                        neu_hits += 1

            if hits_dir and freq:
                hit_path = (hits_dir / p.name.replace(".txt", "_hits.csv"))
                with hit_path.open("w", encoding="utf-8", newline="") as fh:
                    wh = csv.writer(fh)
                    wh.writerow(["lemma", "count", "sentiment", "polarity"])
                    for lemma, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
                        s, pol = lex[lemma]
                        wh.writerow([lemma, c, s, pol])

            token_count = sum(1 for t in doc if not (t.is_space or t.is_punct))
            coverage = (hits / token_count * 100.0) if token_count > 0 else 0.0

            w.writerow([
                extract_date_iso(p),
                str(p),
                token_count, hits,
                round(sent_sum, 6),
                pos_hits, neg_hits, neu_hits,
                round(coverage, 4),
            ])

    print(f"Wrote {out_csv} (years {args.start_year}–{args.end_year})")


if __name__ == "__main__":
    main()
