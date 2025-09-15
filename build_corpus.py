#!/usr/bin/env python3
"""
Build a plain-text corpus from downloaded FOMC minutes (HTML + PDF).

Inputs:
  - minutes_html/  (your raw files from both scrapers)
  - (optional) minutes_manifest.csv and minutes_historical_manifest.csv for richer metadata

Outputs:
  - minutes_text/<year>/<basename>.txt
  - minutes_corpus_manifest.jsonl  (1 JSON per document with metadata)
  - minutes_corpus_index.csv       (CSV summary)

Usage:
  python build_corpus.py
  # or:
  python build_corpus.py --root minutes_html --out minutes_text --manifests minutes_manifest.csv minutes_historical_manifest.csv --enable-ocr
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from bs4 import BeautifulSoup, Tag

# ------------- Config -------------

HTML_EXTS = {".htm", ".html"}
PDF_EXTS = {".pdf"}

# Heuristic: if PDF extract yields too little readable text, treat as scanned and try OCR (if enabled)
MIN_TEXT_CHARS = 300
MIN_ALNUM_RATIO = 0.15

# ------------- CLI -------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a text corpus from FOMC minutes.")
    ap.add_argument("--root", default="minutes_html", help="Root folder containing downloaded minutes (default: minutes_html)")
    ap.add_argument("--out", default="minutes_text", help="Output folder for plain text files (default: minutes_text)")
    ap.add_argument("--manifests", nargs="*", default=["minutes_manifest.csv", "minutes_historical_manifest.csv"],
                    help="Optional manifest CSVs to enrich metadata (default: both manifests if present)")
    ap.add_argument("--enable-ocr", action="store_true", help="Enable OCR for PDFs that look scanned (requires ocrmypdf OR pytesseract+pdf2image)")
    ap.add_argument("--prefer-ocrmypdf", action="store_true", help="Prefer ocrmypdf CLI over pytesseract if both available")
    return ap.parse_args()

# ------------- Utilities -------------

DATE_RX = re.compile(r"(?<!\d)(\d{8})(?!\d)")
NONWS_RX = re.compile(r"\S")
HYPHEN_BREAK_RX = re.compile(r"(\w)-\n(\w)")
MULTIBLANK_RX = re.compile(r"\n{3,}")

def yyyymmdd_to_iso(yyyymmdd: str) -> Optional[str]:
    try:
        return datetime.strptime(yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return None

def detect_date_from_name(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return (yyyymmdd, iso) parsed from filename if present."""
    m = DATE_RX.search(path.name)
    if not m:
        return None, None
    ymd = m.group(1)
    return ymd, yyyymmdd_to_iso(ymd)

def norm_text(s: str) -> str:
    # Unicode normalize, de-NBSP, normalize whitespace and hyphenated line breaks.
    s = unicodedata.normalize("NFKC", s).replace("\xa0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Join hyphenated line breaks: e.g., "eco-\nnomy" -> "economy"
    s = HYPHEN_BREAK_RX.sub(r"\1\2", s)
    # Collapse long blank runs
    s = MULTIBLANK_RX.sub("\n\n", s)
    # Trim trailing spaces per line
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()

def alnum_ratio(s: str) -> float:
    if not s:
        return 0.0
    alnum = sum(ch.isalnum() for ch in s)
    return alnum / max(len(s), 1)

# ------------- Manifest loading (optional) -------------

def load_manifests(paths: List[Path]) -> Dict[str, Dict]:
    """
    Load manifests into a dict keyed by normalized relative path under root (with / separators).
    We’ll try to match later by comparing relative path strings.
    """
    idx: Dict[str, Dict] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                rel = (row.get("saved_path") or "").strip()
                if not rel:
                    continue
                # Normalize slashes for robust matching
                rel_norm = rel.replace("\\", "/")
                idx[rel_norm] = row
    return idx

# ------------- HTML extraction -------------

DROP_TAGS = ["script", "style", "noscript"]
LIKELY_NAV_CLASSES = ("nav", "menu", "breadcrumb", "footer", "header", "skip", "toolbar", "sidebar", "share", "social", "search")


def extract_text_html(path: Path) -> Tuple[str, Optional[str]]:
    """
    Extract readable text from HTML. Returns (text, title).
    Hardened to avoid calling .get(...) on non-Tag nodes.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")  # falls back to html.parser if lxml not present

    # Drop obviously non-content tags
    for tag in list(soup.find_all(DROP_TAGS)):
        if isinstance(tag, Tag):
            tag.decompose()

    # Remove common nav/boilerplate by class/id hints (collect first, then decompose)
    to_remove: list[Tag] = []
    for el in soup.find_all(True):  # True => any tag
        if not isinstance(el, Tag):
            continue
        # classes may be list/str/None
        cls_attr = el.attrs.get("class", [])
        if isinstance(cls_attr, str):
            cls_attr = [cls_attr]
        cls_s = " ".join(cls_attr).lower()
        id_s = str(el.attrs.get("id") or "").lower()

        if any(key in cls_s for key in LIKELY_NAV_CLASSES) or any(key in id_s for key in LIKELY_NAV_CLASSES):
            to_remove.append(el)

    for el in to_remove:
        el.decompose()

    title = None
    if soup.title and soup.title.string:
        # soup.title.string can be None; guard with str(...)
        title = " ".join(str(soup.title.string).split())

    # Extract text with line breaks between blocks
    text = soup.get_text(separator="\n")
    text = norm_text(text)

    # Light de-boilerplate: drop leading empty/near-empty lines
    lines = [ln for ln in text.split("\n")]
    while lines and len(lines[0].strip()) < 3:
        lines.pop(0)
    text = "\n".join(lines).strip()

    return text, title

# ------------- PDF extraction & optional OCR -------------

def have_cmd(name: str) -> bool:
    try:
        subprocess.run([name, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def try_ocrmypdf(in_pdf: Path, out_pdf: Path) -> bool:
    """Run ocrmypdf if available. Returns True on success."""
    if not have_cmd("ocrmypdf"):
        return False
    try:
        # --skip-text avoids double OCR; --quiet keeps logs clean
        subprocess.run(
            ["ocrmypdf", "--skip-text", "--quiet", str(in_pdf), str(out_pdf)],
            check=True
        )
        return out_pdf.exists() and out_pdf.stat().st_size > 0
    except Exception:
        return False

def try_pytesseract(in_pdf: Path) -> Optional[str]:
    """Fallback OCR: pdf2image + pytesseract (if installed). Returns text or None."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return None
    try:
        pages = convert_from_path(str(in_pdf), dpi=300)
        chunks = []
        for img in pages:
            chunks.append(pytesseract.image_to_string(img))
        return "\n".join(chunks)
    except Exception:
        return None

def extract_text_pdf(path: Path, enable_ocr: bool, prefer_ocrmypdf: bool, tmp_dir: Path) -> str:
    # First pass: pdfminer (fast, works on digital PDFs)
    try:
        text = pdf_extract_text(str(path)) or ""
    except Exception:
        text = ""
    text = norm_text(text)

    # If looks empty/scanned and OCR is enabled, try OCR.
    if enable_ocr:
        need_ocr = (len(text) < MIN_TEXT_CHARS) or (alnum_ratio(text) < MIN_ALNUM_RATIO)
        if need_ocr:
            if prefer_ocrmypdf and have_cmd("ocrmypdf"):
                ocr_pdf = tmp_dir / (path.stem + ".ocr.pdf")
                ocr_pdf.parent.mkdir(parents=True, exist_ok=True)
                if try_ocrmypdf(path, ocr_pdf):
                    try:
                        text2 = pdf_extract_text(str(ocr_pdf)) or ""
                    except Exception:
                        text2 = ""
                    text2 = norm_text(text2)
                    if len(text2) > len(text):
                        return text2
            # Fallback pytesseract route
            ocr_text = try_pytesseract(path)
            if ocr_text:
                ocr_text = norm_text(ocr_text)
                if len(ocr_text) > len(text):
                    return ocr_text

    return text

# ------------- Metadata -------------

@dataclass
class DocMeta:
    date_iso: Optional[str]
    yyyymmdd: Optional[str]
    year: Optional[int]
    kind: Optional[str]          # minutes_html | hist_minutes_pdf | minutes_of_actions_pdf | unknown
    title: Optional[str]
    char_count: int
    word_count: int
    rel_raw_path: str
    rel_txt_path: str
    url: Optional[str]

def guess_kind_from_name(name: str) -> str:
    n = name.lower()
    if "fomcminutes" in n or "fomcminutes-" in n or "minutes" in n and n.endswith(".htm"):
        return "minutes_html"
    if "fomchistmin" in n:
        return "hist_minutes_pdf"
    if "fomcmoa" in n:
        return "minutes_of_actions_pdf"
    return "unknown"

# ------------- Main -------------

def main() -> int:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.out)
    if not root.exists():
        print(f"[ERR] root not found: {root}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    tmpdir = outdir / "_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Optional manifests for URL/kind enrichment
    manifest_idx = load_manifests([Path(p) for p in args.manifests])

    jsonl_path = outdir / "minutes_corpus_manifest.jsonl"
    csv_path = outdir / "minutes_corpus_index.csv"
    n_ok = n_empty = n_err = 0

    with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open("w", encoding="utf-8", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["date_iso","yyyymmdd","year","kind","title","char_count","word_count","rel_raw_path","rel_txt_path","url"])

        for raw_path in sorted(root.rglob("*")):
            if not raw_path.is_file():
                continue
            ext = raw_path.suffix.lower()
            if ext not in HTML_EXTS and ext not in PDF_EXTS:
                continue

            # Build relative references for manifest matching
            rel_from_root = str(raw_path.relative_to(root)).replace("\\", "/")

            # Determine year subfolder for output
            # Keep same directory tree under minutes_text/
            txt_rel = Path(rel_from_root).with_suffix(".txt")
            txt_out_path = outdir / txt_rel
            txt_out_path.parent.mkdir(parents=True, exist_ok=True)

            # Date & kind
            ymd, iso = detect_date_from_name(raw_path)
            kind = None
            url = None
            # Enrich from manifests if available
            mrow = manifest_idx.get(str((Path(args.root) / rel_from_root).as_posix()))
            if mrow:
                url = (mrow.get("url") or "").strip() or None
                kind = (mrow.get("kind") or "").strip() or None
            if not kind:
                kind = guess_kind_from_name(raw_path.name)
            year = int(ymd[:4]) if ymd else None

            # Extract
            try:
                if ext in HTML_EXTS:
                    text, title = extract_text_html(raw_path)
                else:
                    text = extract_text_pdf(raw_path, enable_ocr=args.enable_ocr, prefer_ocrmypdf=args.prefer_ocrmypdf, tmp_dir=tmpdir)
                    title = None
            except Exception as e:
                print(f"[ERR] extract failed: {raw_path} -> {e}")
                n_err += 1
                continue

            if not NONWS_RX.search(text or ""):
                # no visible text
                n_empty += 1
                # still write an empty file for traceability
                txt_out_path.write_text("", encoding="utf-8")
            else:
                txt_out_path.write_text(text, encoding="utf-8")
                n_ok += 1

            meta = DocMeta(
                date_iso=iso,
                yyyymmdd=ymd,
                year=year,
                kind=kind,
                title=title,
                char_count=len(text),
                word_count=len(text.split()),
                rel_raw_path=rel_from_root,
                rel_txt_path=str(txt_rel).replace("\\","/"),
                url=url,
            )
            jf.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
            cw.writerow([meta.date_iso, meta.yyyymmdd, meta.year, meta.kind, meta.title or "",
                         meta.char_count, meta.word_count, meta.rel_raw_path, meta.rel_txt_path, meta.url or ""])

    print(f"[DONE] Text files in: {outdir.resolve()}")
    print(f"[DONE] JSONL manifest: {jsonl_path.resolve()}")
    print(f"[DONE] CSV index: {csv_path.resolve()}")
    print(f"[STATS] ok={n_ok} empty={n_empty} errors={n_err}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
