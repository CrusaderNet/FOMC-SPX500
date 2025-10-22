from __future__ import annotations
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import csv
import io
import json
import os
import sys
import time

import requests

from config_paths import resolve_path, ensure_all_dirs
from dataclasses import dataclass
import bisect

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch SP500 closes for a list of dates (YYYY-MM-DD), without yfinance/Yahoo.

Default source: Stooq (^SPX) via single CSV download (no API key).
Optional fallback: FRED SP500 (requires env FRED_API_KEY).

Inputs (searched in this order unless --dates* provided):
  - sp500_dates_from_manifest_unique.csv (date_iso column)
  - sp500_dates_unique.csv (date_iso column)
  - sp500_dates_from_manifest_unique.txt (YYYY-MM-DD, one per line)
  - sp500_dates_unique.txt (YYYY-MM-DD, one per line)

Outputs:
  - sp500_prices.csv        (requested_date, matched_trading_date, close, source, align)
  - sp500_prices_missing.csv (requested_date, reason)

Usage examples:
  python fetch_sp500_prices.py
  python fetch_sp500_prices.py --align next     # default
  python fetch_sp500_prices.py --align prev
  python fetch_sp500_prices.py --source stooq
  python fetch_sp500_prices.py --source auto    # stooq then FRED fallback if missing
  FRED_API_KEY=... python fetch_sp500_prices.py --source fred
"""
ensure_all_dirs()



STOOQ_URL = "https://stooq.com/q/d/l/?s=%5Espx&i=d"   # ^SPX daily CSV
STOOQ_CACHE = Path("data/spx_stooq.csv")
FRED_SERIES = "SP500"
FRED_CACHE = Path("data/sp500_fred.json")  # JSON cache of observations

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SPXFetcher/1.0; +https://example.org/)",
    "Accept": "*/*",
}

DEFAULT_ALIGN = "next"            # {exact, next, prev, nearest}
DEFAULT_SOURCE = "stooq"          # {stooq, fred, auto}

CANDIDATE_DATE_FILES = [
    ("csv", resolve_path("sp500_dates_from_manifest_unique.csv")),
    ("csv", resolve_path("sp500_dates_unique.csv")),
    ("txt", resolve_path("sp500_dates_from_manifest_unique.txt")),
    ("txt", resolve_path("sp500_dates_unique.txt")),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch SP500 closes for given dates.")
    ap.add_argument("--align", choices=["exact", "next", "prev", "nearest"], default=DEFAULT_ALIGN,
                    help="How to align non-trading dates (default: next)")
    ap.add_argument("--source", choices=["stooq", "fred", "auto"], default=DEFAULT_SOURCE,
                    help="Data source: stooq (no key), fred (needs FRED_API_KEY), or auto fallback (default: stooq)")
    ap.add_argument("--dates_csv", help="CSV with a 'date_iso' column")
    ap.add_argument("--dates_txt", help="TXT with YYYY-MM-DD one per line")
    ap.add_argument("--out_csv", default=resolve_path("sp500_prices.csv"), help="Output CSV")
    ap.add_argument("--missing_csv", default=resolve_path("sp500_prices_missing.csv"), help="Missing/failed CSV")
    ap.add_argument("--force_refresh", action="store_true", help="Redownload/cold-refresh caches")
    return ap.parse_args()

def to_date(s: str) -> Optional[date]:
    s = s.strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None

def http_get(url: str, max_attempts=4, backoff=1.6, timeout=30) -> requests.Response:
    last_exc = None
    for i in range(1, max_attempts + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}")
            return r
        except Exception as e:
            last_exc = e
            if i < max_attempts:
                time.sleep(backoff ** i)
            else:
                raise
    raise last_exc


def load_dates_from_csv(p: Path) -> List[date]:
    dates: List[date] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        # try exact 'date_iso', else first column named like 'date'
        col = "date_iso" if "date_iso" in rdr.fieldnames else None
        if not col:
            for candidate in rdr.fieldnames or []:
                if candidate.lower().startswith("date"):
                    col = candidate; break
        if not col:
            raise ValueError(f"No date column found in {p}. Expected 'date_iso'.")
        for row in rdr:
            d = to_date(row[col])
            if d:
                dates.append(d)
    return sorted(set(dates))

def load_dates_from_txt(p: Path) -> List[date]:
    dates: List[date] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            d = to_date(line)
            if d:
                dates.append(d)
    return sorted(set(dates))

def autodetect_dates() -> Tuple[List[date], str]:
    for kind, fname in CANDIDATE_DATE_FILES:
        p = Path(fname)
        if not p.exists():
            continue
        if kind == "csv":
            return load_dates_from_csv(p), str(p)
        else:
            return load_dates_from_txt(p), str(p)
    raise FileNotFoundError("No input date list found. Provide --dates_csv or --dates_txt.")


def load_stooq_series(force_refresh=False) -> Dict[date, float]:
    """
    Returns {date: close} from Stooq ^SPX CSV (cached to disk).
    CSV columns: Date,Open,High,Low,Close,Volume
    """
    STOOQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if force_refresh or not STOOQ_CACHE.exists():
        r = http_get(STOOQ_URL)
        r.raise_for_status()
        STOOQ_CACHE.write_bytes(r.content)

    series: Dict[date, float] = {}
    with STOOQ_CACHE.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            ds = row.get("Date") or row.get("date")
            cs = row.get("Close") or row.get("close")
            if not ds or not cs:
                continue
            try:
                d = datetime.strptime(ds, "%Y-%m-%d").date()
                c = float(cs)
            except Exception:
                continue
            series[d] = c
    return series



@dataclass(order=True)
class TradingSeries:
    dates_sorted: List[date]
    closes_by_date: Dict[date, float]
    source_name: str

    def find(self, target: date, align: str) -> Tuple[Optional[date], Optional[float]]:
        # exact match
        if target in self.closes_by_date:
            return target, self.closes_by_date[target]

        ds = self.dates_sorted
        i = bisect.bisect_left(ds, target)
        prev_d = ds[i - 1] if i > 0 else None
        next_d = ds[i] if i < len(ds) else None

        if align == "exact":
            return None, None

        if align == "next":
            # normal case
            if next_d is not None:
                return next_d, self.closes_by_date.get(next_d)
            # edge fallback: past the last available date -> use prev
            return (prev_d, self.closes_by_date.get(prev_d)) if prev_d else (None, None)

        if align == "prev":
            # normal case
            if prev_d is not None:
                return prev_d, self.closes_by_date.get(prev_d)
            # edge fallback: before the first available date -> use next
            return (next_d, self.closes_by_date.get(next_d)) if next_d else (None, None)

        # "nearest": choose closer of prev/next; tie -> next
        if prev_d and next_d:
            return (prev_d, self.closes_by_date.get(prev_d)) \
                if (target - prev_d) <= (next_d - target) \
                else (next_d, self.closes_by_date.get(next_d))
        return (prev_d, self.closes_by_date.get(prev_d)) if prev_d else \
               ((next_d, self.closes_by_date.get(next_d)) if next_d else (None, None))



def main() -> int:
    args = parse_args()

    # Load requested dates
    if args.dates_csv:
        dates = load_dates_from_csv(Path(args.dates_csv)); src_descr = args.dates_csv
    elif args.dates_txt:
        dates = load_dates_from_txt(Path(args.dates_txt)); src_descr = args.dates_txt
    else:
        dates, src_descr = autodetect_dates()
    if not dates:
        print("[ERR] No input dates found.")
        return 1
    print(f"[INFO] Loaded {len(dates)} dates from {src_descr}")

    # Load data sources
    sources: List[TradingSeries] = []

    if args.source in ("stooq", "auto"):
        stooq = load_stooq_series(force_refresh=args.force_refresh)
        if stooq:
            sources.append(TradingSeries(sorted(stooq.keys()), stooq, "stooq"))

    if not sources:
        print("[ERR] No data sources available. For FRED set FRED_API_KEY or use --source stooq.")
        return 1

    # Resolve prices
    out_rows: List[Tuple[str, str, Optional[float], str, str]] = []  # requested_date, matched_date, close, source, align
    missing: List[Tuple[str, str]] = []  # requested_date, reason

    for d in dates:
        matched = None
        for ts in sources:
            md, pv = ts.find(d, args.align)
            if md and pv is not None:
                matched = (md, pv, ts.source_name)
                break
        if matched:
            md, pv, src = matched
            out_rows.append((d.isoformat(), md.isoformat(), pv, src, args.align))
        else:
            # try to craft reason
            min_cov = min(s.dates_sorted[0] for s in sources if s.dates_sorted)
            max_cov = max(s.dates_sorted[-1] for s in sources if s.dates_sorted)
            if d < min_cov or d > max_cov:
                reason = f"outside coverage ({min_cov}..{max_cov})"
            else:
                reason = "no trading day match"
            missing.append((d.isoformat(), reason))

    # Write outputs
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["requested_date", "matched_trading_date", "close", "source", "align"])
        for row in out_rows: w.writerow(row)

    with open(args.missing_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["requested_date", "reason"])
        for row in missing: w.writerow(row)

    print(f"[OK] Wrote prices: {args.out_csv} ({len(out_rows)} rows)")
    if missing:
        print(f"[WARN] Missing: {args.missing_csv} ({len(missing)} rows)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
