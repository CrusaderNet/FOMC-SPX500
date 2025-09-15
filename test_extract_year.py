#!/usr/bin/env python3
"""
Test extractor for one or more years of FOMC minutes.

- Scans minutes_html/<year>/** for .htm/.html/.pdf
- Extracts text (BeautifulSoup+lxml for HTML; pdfminer for PDF)
- Prints per-file status + summary
- Optional CSV report and on-screen preview

Usage examples:
  python test_extract_year.py 2007
  python test_extract_year.py 2007 2008 --preview 200
  python test_extract_year.py 1993 --root minutes_html --out report_1993.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup, Tag
from pdfminer.high_level import extract_text as pdf_extract_text

# ---------------- Config ----------------

HTML_EXTS = {".htm", ".html"}
PDF_EXTS = {".pdf"}

DATE_RX = re.compile(r"(?<!\d)(\d{8})(?!\d)")
NONWS_RX = re.compile(r"\S")
HYPHEN_BREAK_RX = re.compile(r"(\w)-\n(\w)")
MULTIBLANK_RX = re.compile(r"\n{3,}")

DROP_TAGS = ["script", "style", "noscript"]
LIKELY_NAV_CLASSES = (
    "nav", "menu", "breadcrumb", "footer", "header", "skip",
    "toolbar", "sidebar", "share", "social", "search"
)

# ---------------- Utils ----------------

def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\xa0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = HYPHEN_BREAK_RX.sub(r"\1\2", s)
    s = MULTIBLANK_RX.sub("\n\n", s)
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()

def yyyymmdd_to_iso(yyyymmdd: str) -> Optional[str]:
    try:
        return datetime.strptime(yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return None

def detect_date_from_name(path: Path) -> Tuple[Optional[str], Optional[str]]:
    m = DATE_RX.search(path.name)
    if not m:
        return None, None
    ymd = m.group(1)
    return ymd, yyyymmdd_to_iso(ymd)

# ---------------- Extractors ----------------

def extract_text_html(path: Path) -> Tuple[str, Optional[str]]:
    """Extract readable text + title from an HTML file (hardened)."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")  # falls back to std parser if lxml unavailable

    # Drop obvious non-content
    for tag in list(soup.find_all(DROP_TAGS)):
        if isinstance(tag, Tag):
            tag.decompose()

    # Remove common nav/boilerplate by class/id hints
    to_remove: List[Tag] = []
    for el in soup.find_all(True):  # any tag
        if not isinstance(el, Tag):
            continue
        cls_attr = el.get("class", [])
        if isinstance(cls_attr, str):
            cls_attr = [cls_attr]
        cls_s = " ".join(cls_attr).lower()
        id_s = str(el.get("id") or "").lower()
        if any(k in cls_s for k in LIKELY_NAV_CLASSES) or any(k in id_s for k in LIKELY_NAV_CLASSES):
            to_remove.append(el)
    for el in to_remove:
        el.decompose()

    title = None
    if soup.title and soup.title.string is not None:
        title = " ".join(str(soup.title.string).split())

    text = soup.get_text(separator="\n")
    text = norm_text(text)

    # Drop leading trivial lines
    lines = text.split("\n")
    while lines and len(lines[0].strip()) < 3:
        lines.pop(0)
    text = "\n".join(lines).strip()

    return text, title

def extract_text_pdf(path: Path) -> str:
    """Extract text from a PDF via pdfminer.six (no OCR in this test script)."""
    try:
        text = pdf_extract_text(str(path)) or ""
    except Exception:
        text = ""
    return norm_text(text)

# ---------------- Data model ----------------

@dataclass
class FileResult:
    rel_path: str
    year: int
    yyyymmdd: Optional[str]
    date_iso: Optional[str]
    ext: str
    title: Optional[str]
    char_count: int
    word_count: int
    status: str       # OK | EMPTY | ERR
    error: Optional[str] = None

# ---------------- CLI & Main ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Test HTML/PDF text extraction for selected years.")
    ap.add_argument("years", nargs="+", type=int, help="Year(s) to test (e.g., 2007 2008)")
    ap.add_argument("--root", default="minutes_html", help="Root folder containing the downloaded files")
    ap.add_argument("--out", default=None, help="Optional CSV report path")
    ap.add_argument("--preview", type=int, default=0, help="Print the first N characters of extracted text")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N files per year (0 = no limit)")
    return ap.parse_args()

def process_file(root: Path, f: Path) -> FileResult:
    rel = str(f.relative_to(root)).replace("\\", "/")
    year = int(f.parts[-2]) if f.parent.name.isdigit() else None
    ymd, iso = detect_date_from_name(f)
    ext = f.suffix.lower()
    try:
        if ext in HTML_EXTS:
            text, title = extract_text_html(f)
        elif ext in PDF_EXTS:
            text = extract_text_pdf(f)
            title = None
        else:
            return FileResult(rel, year or 0, ymd, iso, ext, None, 0, 0, "ERR", f"unsupported ext {ext}")

        if not NONWS_RX.search(text):
            return FileResult(rel, year or 0, ymd, iso, ext, title, 0, 0, "EMPTY", None)
        return FileResult(rel, year or 0, ymd, iso, ext, title, len(text), len(text.split()), "OK", None)

    except Exception as e:
        return FileResult(rel, year or 0, ymd, iso, ext, None, 0, 0, "ERR", str(e))

def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] Root not found: {root}")
        return 1

    all_rows: List[FileResult] = []
    for year in args.years:
        folder = root / str(year)
        if not folder.exists():
            print(f"[WARN] Missing folder for year {year}: {folder}")
            continue

        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (HTML_EXTS | PDF_EXTS)]
        files.sort()
        if args.limit > 0:
            files = files[:args.limit]

        print(f"[INFO] Year {year}: {len(files)} files to test")
        for f in files:
            res = process_file(root, f)
            all_rows.append(res)
            msg = f"[{res.status}] {res.rel_path} | chars={res.char_count} words={res.word_count}"
            if res.title:
                msg += f" | title={res.title}"
            if res.error:
                msg += f" | error={res.error}"
            print(msg)
            if args.preview and res.status == "OK":
                try:
                    text = (Path(args.root).parent / "minutes_text" / Path(res.rel_path).with_suffix(".txt")).read_text(encoding="utf-8")  # unlikely to exist here
                except Exception:
                    # For preview, re-extract quickly:
                    if f.suffix.lower() in HTML_EXTS:
                        txt, _ = extract_text_html(f)
                    else:
                        txt = extract_text_pdf(f)
                    text = txt
                sample = (text[:args.preview] + "…") if len(text) > args.preview else text
                print("---- preview ----")
                print(sample)
                print("-----------------")

        # Year summary
        ok = sum(1 for r in all_rows if r.year == year and r.status == "OK")
        empty = sum(1 for r in all_rows if r.year == year and r.status == "EMPTY")
        err = sum(1 for r in all_rows if r.year == year and r.status == "ERR")
        print(f"[SUMMARY {year}] OK={ok} EMPTY={empty} ERR={err}")

    # Optional CSV
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rel_path","year","yyyymmdd","date_iso","ext","title","char_count","word_count","status","error"])
            for r in all_rows:
                w.writerow([r.rel_path, r.year, r.yyyymmdd or "", r.date_iso or "", r.ext, r.title or "",
                            r.char_count, r.word_count, r.status, r.error or ""])
        print(f"[OK] Wrote CSV: {outp.resolve()}")

    # Final overall summary
    ok = sum(1 for r in all_rows if r.status == "OK")
    empty = sum(1 for r in all_rows if r.status == "EMPTY")
    err = sum(1 for r in all_rows if r.status == "ERR")
    print(f"[DONE] Overall: OK={ok} EMPTY={empty} ERR={err} (files={len(all_rows)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
