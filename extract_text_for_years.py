from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import csv
import json
import re

from bs4 import BeautifulSoup, Tag

from __future__ import annotations
from config_paths import resolve_path, ensure_all_dirs
from dataclasses import asdict, dataclass
from pdfminer.high_level import extract_text as pdf_extract_text
import unicodedata

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resume corpus building for specific years:
- Reads minutes_html/<year>/** (HTM/HTML/PDF)
- Writes minutes_text/<year>/**.txt
- Updates minutes_text/minutes_corpus_index.csv and
  minutes_text/minutes_corpus_manifest.json (array) OR .jsonl (line-delimited),
  whichever exists (creates JSONL if neither exists)

Usage examples:
  python resume_corpus_for_years.py 2008
  $years = 2008..2025; python resume_corpus_for_years.py $years
"""
ensure_all_dirs()

HTML_EXTS = {".htm", ".html"}
PDF_EXTS = {".pdf"}

DATE_RX = re.compile(r"(?<!\d)(\d{8})(?!\d)")
NONWS_RX = re.compile(r"\S")
HYPHEN_BREAK_RX = re.compile(r"(\w)-\n(\w)")
MULTIBLANK_RX = re.compile(r"\n{3,}")

DROP_TAGS = ["script", "style", "noscript"]
LIKELY_NAV_CLASSES = ("nav","menu","breadcrumb","footer","header","skip","toolbar","sidebar","share","social","search")

# default locations/names
DEFAULT_IN_ROOT = "minutes_html"
DEFAULT_OUT_ROOT = "minutes_text"
DEFAULT_MANIFESTS = [resolve_path("minutes_manifest.csv"), resolve_path("minutes_historical_manifest.csv")]
INDEX_CSV_NAME = "minutes_corpus_index.csv"
MANIFEST_JSONL_NAME = "minutes_corpus_manifest.jsonl"
MANIFEST_JSON_NAME  = "minutes_corpus_manifest.json"


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


def extract_text_html(path: Path) -> Tuple[str, Optional[str]]:
    """Extract readable text + title from HTML (hardened)."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    for tag in list(soup.find_all(DROP_TAGS)):
        if isinstance(tag, Tag):
            tag.decompose()

    to_remove: List[Tag] = []
    for el in soup.find_all(True):
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

    lines = text.split("\n")
    while lines and len(lines[0].strip()) < 3:
        lines.pop(0)
    text = "\n".join(lines).strip()
    return text, title

def extract_text_pdf(path: Path) -> str:
    try:
        text = pdf_extract_text(str(path)) or ""
    except Exception:
        text = ""
    return norm_text(text)


def load_minutes_manifests(paths: List[Path]) -> Dict[str, Dict]:
    """
    Returns mapping from multiple keys -> row:
      - exact saved_path normalized
      - suffix after minutes_html/ (to match relative)
    """
    idx: Dict[str, Dict] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                saved = (row.get("saved_path") or "").strip().replace("\\", "/")
                if not saved:
                    continue
                idx[saved] = row
                # also index by suffix under minutes_html if present
                parts = saved.split("minutes_html/")
                if len(parts) > 1:
                    idx[parts[-1]] = row
    return idx

def lookup_manifest_row(idx: Dict[str, Dict], rel_from_root: str, in_root: Path) -> Optional[Dict]:
    key1 = (in_root / rel_from_root).as_posix()
    key2 = rel_from_root
    return idx.get(key1) or idx.get(key2)


@dataclass
class DocMeta:
    date_iso: Optional[str]
    yyyymmdd: Optional[str]
    year: Optional[int]
    kind: Optional[str]
    title: Optional[str]
    char_count: int
    word_count: int
    rel_raw_path: str
    rel_txt_path: str
    url: Optional[str]

def guess_kind_from_name(name: str) -> str:
    n = name.lower()
    if "fomchistmin" in n:
        return "hist_minutes_pdf"
    if "fomcmoa" in n:
        return "minutes_of_actions_pdf"
    if n.endswith(".htm") or n.endswith(".html"):
        return "minutes_html"
    return "unknown"


def load_existing_rel_txt_from_csv(csv_path: Path) -> set[str]:
    s: set[str] = set()
    if not csv_path.exists():
        return s
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        # detect column name
        colnames = [c.lower() for c in (rdr.fieldnames or [])]
        rel_col = None
        for nm in ("rel_txt_path", "rel txt path", "rel_text_path"):
            if nm in colnames:
                rel_col = rdr.fieldnames[colnames.index(nm)]
                break
        if rel_col is None:
            # fallback: try last column
            rel_col = (rdr.fieldnames or [""])[-1]
        for r in rdr:
            v = (r.get(rel_col) or "").strip().replace("\\", "/")
            if v:
                s.add(v)
    return s

def load_existing_rel_txt_from_jsonl(jsonl_path: Path) -> set[str]:
    s: set[str] = set()
    if not jsonl_path.exists():
        return s
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rp = (obj.get("rel_txt_path") or "").strip().replace("\\", "/")
                if rp:
                    s.add(rp)
            except Exception:
                continue
    return s

def load_existing_rel_txt_from_json(json_path: Path) -> Tuple[set[str], List[dict]]:
    s: set[str] = set()
    arr: List[dict] = []
    if not json_path.exists():
        return s, arr
    try:
        arr = json.loads(json_path.read_text(encoding="utf-8"))
        for obj in arr:
            rp = (obj.get("rel_txt_path") or "").strip().replace("\\", "/")
            if rp:
                s.add(rp)
    except Exception:
        arr = []
    return s, arr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resume corpus building and update manifests for specific years.")
    ap.add_argument("years", nargs="+", type=int, help="Year(s) to process (e.g., 2008 2009 ...)")
    ap.add_argument("--root", default=DEFAULT_IN_ROOT, help="Input root (minutes_html)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Output root (minutes_text)")
    ap.add_argument("--manifests", nargs="*", default=DEFAULT_MANIFESTS, help="Minutes manifests to enrich metadata")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files (default: skip)")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    in_root = Path(args.root)
    out_root = Path(args.out_root)
    if not in_root.exists():
        print(f"[ERR] Input root not found: {in_root}")
        return 1
    out_root.mkdir(parents=True, exist_ok=True)

    # Determine which manifest file we’ll update (prefer existing)
    index_csv = out_root / INDEX_CSV_NAME
    manifest_jsonl = out_root / MANIFEST_JSONL_NAME
    manifest_json  = out_root / MANIFEST_JSON_NAME

    # Load existing indices to avoid duplicates
    existing_csv = load_existing_rel_txt_from_csv(index_csv)
    existing_jsonl = load_existing_rel_txt_from_jsonl(manifest_jsonl)
    existing_json, json_array = load_existing_rel_txt_from_json(manifest_json)

    existing_all = set()
    existing_all |= existing_csv
    existing_all |= existing_jsonl
    existing_all |= existing_json

    # Prepare writers
    need_header = not index_csv.exists()
    csv_f = index_csv.open("a", encoding="utf-8", newline="")
    csv_w = csv.writer(csv_f)
    if need_header:
        csv_w.writerow(["date_iso","yyyymmdd","year","kind","title","char_count","word_count","rel_raw_path","rel_txt_path","url"])

    # Load minutes manifests (for url/kind enrichment)
    manifest_idx = load_minutes_manifests([Path(p) for p in args.manifests])

    # Decide JSON mode: if .json exists, update array; elif .jsonl exists, append lines; else create .jsonl
    use_json_array = manifest_json.exists()
    jsonl_f = None
    if not use_json_array:
        jsonl_f = manifest_jsonl.open("a", encoding="utf-8")

    total_ok = total_empty = total_skip = total_err = 0

    try:
        for year in args.years:
            in_year = in_root / str(year)
            if not in_year.exists():
                print(f"[WARN] Missing input folder for year {year}: {in_year}")
                continue

            files = [p for p in in_year.rglob("*") if p.is_file() and p.suffix.lower() in (HTML_EXTS | PDF_EXTS)]
            files.sort()
            print(f"[INFO] Year {year}: {len(files)} source files")

            for f in files:
                rel_in = str(f.relative_to(in_root)).replace("\\", "/")
                rel_out = str(Path(rel_in).with_suffix(".txt")).replace("\\", "/")
                txt_path = out_root / rel_out
                txt_path.parent.mkdir(parents=True, exist_ok=True)

                # Skip if already recorded in any existing manifest/index
                if rel_out in existing_all and txt_path.exists() and not args.overwrite:
                    total_skip += 1
                    print(f"[SKIP] {rel_in} -> {rel_out} (already indexed)")
                    continue

                # Extract
                ext = f.suffix.lower()
                try:
                    if ext in HTML_EXTS:
                        text, title = extract_text_html(f)
                    else:
                        text = extract_text_pdf(f)
                        title = None
                except Exception as e:
                    total_err += 1
                    print(f"[ERR]  {rel_in} -> {rel_out} | {e}")
                    continue

                # Write .txt
                if not NONWS_RX.search(text or ""):
                    txt_path.write_text("", encoding="utf-8")
                    char_count = word_count = 0
                    status = "EMPTY"
                else:
                    txt_path.write_text(text, encoding="utf-8")
                    char_count = len(text)
                    word_count = len(text.split())
                    status = "OK"

                # Metadata
                ymd, iso = detect_date_from_name(f)
                year_val = int(ymd[:4]) if ymd else year
                mrow = lookup_manifest_row(manifest_idx, rel_in, in_root)
                url = (mrow.get("url").strip() if mrow and mrow.get("url") else None)
                kind = (mrow.get("kind").strip() if mrow and mrow.get("kind") else None) or guess_kind_from_name(f.name)

                meta = DocMeta(
                    date_iso=iso,
                    yyyymmdd=ymd,
                    year=year_val,
                    kind=kind,
                    title=title,
                    char_count=char_count,
                    word_count=word_count,
                    rel_raw_path=rel_in,
                    rel_txt_path=rel_out,
                    url=url,
                )

                # Append to CSV (always)
                csv_w.writerow([meta.date_iso or "", meta.yyyymmdd or "", meta.year or "",
                                meta.kind or "", meta.title or "", meta.char_count, meta.word_count,
                                meta.rel_raw_path, meta.rel_txt_path, meta.url or ""])
                existing_all.add(rel_out)

                # Append to JSON(L)
                if use_json_array:
                    json_array.append(asdict(meta))
                else:
                    assert jsonl_f is not None
                    jsonl_f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")

                if status == "OK":
                    total_ok += 1
                    print(f"[OK]   {rel_in} -> {rel_out} | chars={char_count} words={word_count}")
                else:
                    total_empty += 1
                    print(f"[EMPTY] {rel_in} -> {rel_out}")

        # If we were updating a JSON array, write it back once at the end
        if use_json_array:
            manifest_json.write_text(json.dumps(json_array, ensure_ascii=False, indent=2), encoding="utf-8")

    finally:
        csv_f.close()
        if jsonl_f:
            jsonl_f.close()

    print(f"[DONE] OK={total_ok} EMPTY={total_empty} SKIP={total_skip} ERR={total_err}")
    print(f"[OUT]  CSV index: {(out_root / INDEX_CSV_NAME).resolve()}")
    tgt = manifest_json if use_json_array else manifest_jsonl
    print(f"[OUT]  Manifest:  {tgt.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
