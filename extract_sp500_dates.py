#!/usr/bin/env python3
"""
Extract S&P 500 dates directly from the FOMC minutes manifests.

Inputs (defaults; override with --manifests):
  - minutes_manifest.csv
  - minutes_historical_manifest.csv

Behavior:
  - Reads rows, prefers the 'date' column (YYYYMMDD). If missing/invalid,
    falls back to extracting YYYYMMDD from 'saved_path' or 'url'.
  - Filters to status == 200 unless --allow-non200 is set.
  - Deduplicates by date with priority:
      minutes_html > hist_minutes_pdf (fomchistmin) > minutes_of_actions_pdf (fomcmoa)
    (You can change with --prefer-order.)
  - Optionally exclude Minutes of Actions with --exclude-moa.

Outputs:
  - sp500_dates_from_manifest_full.csv
  - sp500_dates_from_manifest_unique.csv
  - sp500_dates_from_manifest_unique.txt
"""
from config_paths import resolve_path, ensure_all_dirs
ensure_all_dirs()


from __future__ import annotations
import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Default inputs/outputs
DEFAULT_MANIFESTS = [resolve_path("minutes_manifest.csv"), resolve_path("minutes_historical_manifest.csv")]
DEFAULT_FULL = resolve_path("sp500_dates_from_manifest_full.csv")
DEFAULT_UNIQUE_CSV = resolve_path("sp500_dates_from_manifest_unique.csv")
DEFAULT_UNIQUE_TXT = resolve_path("sp500_dates_from_manifest_unique.txt")

# Recognize date in paths/urls as fallback
DATE_RX = re.compile(r"(?<!\d)(\d{8})(?!\d)")

# Map "kind" values to a stable vocabulary
KIND_ALIASES = {
    # historical script kinds
    "html_modern": "minutes_html",
    "html_old_deep": "minutes_html",
    "html_old_flat": "minutes_html",
    "pdf_histmin": "hist_minutes_pdf",
    "pdf_moa": "minutes_of_actions_pdf",
    # explicit names we might set for modern script
    "minutes_html": "minutes_html",
    "hist_minutes_pdf": "hist_minutes_pdf",
    "minutes_of_actions_pdf": "minutes_of_actions_pdf",
    # unknown -> minutes_html as a sensible default for modern manifest
    "": "minutes_html",
    None: "minutes_html",
}

# Default kind priority (higher wins on dedup)
DEFAULT_PRIORITY = ["minutes_html", "hist_minutes_pdf", "minutes_of_actions_pdf"]


@dataclass
class Row:
    yyyymmdd: str
    date_iso: str
    kind: str
    saved_path: str
    url: str
    status: Optional[int]
    source_manifest: str


def to_iso(yyyymmdd: str) -> Optional[str]:
    try:
        return datetime.strptime(yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def normalize_kind(raw: Optional[str]) -> str:
    raw = (raw or "").strip()
    return KIND_ALIASES.get(raw, KIND_ALIASES.get(raw.lower(), "minutes_html"))


def extract_date_any(row: dict) -> Optional[str]:
    # 1) Prefer explicit 'date' column
    d = (row.get("date") or "").strip()
    if DATE_RX.fullmatch(d):
        return d
    # 2) Try saved_path then url
    for key in ("saved_path", "url", "saved path", "path"):
        v = (row.get(key) or "").strip()
        if not v:
            continue
        m = DATE_RX.search(v)
        if m:
            return m.group(1)
    return None


def parse_status(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(str(val).strip())
    except Exception:
        return None


def read_manifest(path: Path, allow_non200: bool, exclude_moa: bool, prefer_order: List[str]) -> List[Row]:
    out: List[Row] = []
    if not path.exists():
        # Silent skip if missing; user may only have one manifest
        return out

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            yyyymmdd = extract_date_any(r)
            if not yyyymmdd:
                continue
            date_iso = to_iso(yyyymmdd)
            if not date_iso:
                continue

            status = parse_status(r.get("status"))
            if not allow_non200 and status != 200:
                continue

            kind = normalize_kind(r.get("kind"))
            if exclude_moa and kind == "minutes_of_actions_pdf":
                continue

            saved_path = (r.get("saved_path") or "").strip()
            url = (r.get("url") or "").strip()
            out.append(Row(yyyymmdd, date_iso, kind, saved_path, url, status, path.name))
    return out


def dedup_rows(rows: List[Row], prefer_order: List[str]) -> List[Row]:
    priority: Dict[str, int] = {k: i for i, k in enumerate(reversed(prefer_order), start=1)}
    best: Dict[str, Tuple[int, Row]] = {}
    for row in rows:
        pr = priority.get(row.kind, 0)
        prev = best.get(row.yyyymmdd)
        if prev is None or pr > prev[0]:
            best[row.yyyymmdd] = (pr, row)
    # Sort by yyyymmdd ascending
    return [v[1] for v in sorted(best.values(), key=lambda x: x[1].yyyymmdd)]


def write_full_csv(rows: List[Row], path: Path) -> None:
    rows_sorted = sorted(rows, key=lambda r: r.yyyymmdd)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date_iso", "yyyymmdd", "kind", "status", "saved_path", "url", "source_manifest"])
        for r in rows_sorted:
            w.writerow([r.date_iso, r.yyyymmdd, r.kind, r.status, r.saved_path, r.url, r.source_manifest])


def write_unique_csv(rows: List[Row], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date_iso", "yyyymmdd", "kind"])
        for r in rows:
            w.writerow([r.date_iso, r.yyyymmdd, r.kind])


def write_unique_txt(rows: List[Row], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r.date_iso}\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract S&P 500 dates from FOMC manifests.")
    ap.add_argument("--manifests", nargs="*", default=DEFAULT_MANIFESTS,
                    help="Manifest CSV paths (default: minutes_manifest.csv minutes_historical_manifest.csv)")
    ap.add_argument("--full_csv", default=DEFAULT_FULL, help="Full (non-deduped) output CSV")
    ap.add_argument("--unique_csv", default=DEFAULT_UNIQUE_CSV, help="Unique dates CSV output")
    ap.add_argument("--unique_txt", default=DEFAULT_UNIQUE_TXT, help="Unique ISO dates TXT output")
    ap.add_argument("--allow-non200", action="store_true", help="Include rows with non-200 status")
    ap.add_argument("--exclude-moa", action="store_true", help="Exclude Minutes of Actions PDFs from output")
    ap.add_argument("--prefer-order", default=",".join(DEFAULT_PRIORITY),
                    help="Comma-separated kind priority, high→low. "
                         "Defaults to: " + ",".join(DEFAULT_PRIORITY))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    prefer_order = [k.strip() for k in args.prefer_order.split(",") if k.strip()]
    rows: List[Row] = []

    for mf in args.manifests:
        rows.extend(read_manifest(Path(mf), allow_non200=args.allow_non200,
                                  exclude_moa=args.exclude_moa,
                                  prefer_order=prefer_order))

    if not rows:
        print("[WARN] No rows found. Check manifest paths or statuses.")
        return 0

    # Write full (non-dedup) list
    write_full_csv(rows, Path(args.full_csv))
    print(f"[OK] Wrote full list: {args.full_csv} ({len(rows)} rows)")

    # Deduplicate by date with priority
    uniq = dedup_rows(rows, prefer_order)
    write_unique_csv(uniq, Path(args.unique_csv))
    print(f"[OK] Wrote unique dates CSV: {args.unique_csv} ({len(uniq)} rows)")

    write_unique_txt(uniq, Path(args.unique_txt))
    print(f"[OK] Wrote unique dates TXT: {args.unique_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())