#!/usr/bin/env python3
"""
Scrape historical FOMC meeting minutes from 2019 back to the earliest available year.

Formats handled:
- Modern HTML minutes:           /monetarypolicy/fomcminutesYYYYMMDD.htm
- Old HTML minutes (deep):       /fomc/minutes/YYYY/YYYYMMDDmin.htm   (case-insensitive)
- Old HTML minutes (flat):       /fomc/minutes/YYYYMMDD.htm
- Historical minutes PDFs:       /monetarypolicy/files/fomchistminYYYYMMDD.pdf   <-- NEW
- Minutes of actions PDFs (MOA): /monetarypolicy/files/fomcmoaYYYYMMDD.pdf

Discovery:
- Year index: https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm
- Year pages: https://www.federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm

Outputs:
- minutes_html/<year>/fomcminutes-YYYYMMDD.htm
- minutes_html/<year>/fomchistmin-YYYYMMDD.pdf
- minutes_html/<year>/fomcmoa-YYYYMMDD.pdf
- minutes_historical_manifest.csv

Usage:
  python scrape_fomc_historical_minutes.py
"""
from config_paths import resolve_path, ensure_all_dirs
ensure_all_dirs()


from __future__ import annotations
import csv
import hashlib
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
from urllib.parse import urljoin, urlparse, urlunparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

# ------------------- Config -------------------

BASE = "https://www.federalreserve.gov"
HIST_INDEX_URL = f"{BASE}/monetarypolicy/fomc_historical_year.htm"

# Year page links
YEAR_PAGE_RE = re.compile(r"^/monetarypolicy/fomchistorical(\d{4})\.htm$", re.IGNORECASE)

# Minutes patterns
NEW_HTML_RE        = re.compile(r"^/monetarypolicy/fomcminutes(\d{8})\.htm$", re.IGNORECASE)
OLD_HTML_DEEP_RE   = re.compile(r"^/fomc/minutes/(\d{4})/(\d{8})min\.htm$", re.IGNORECASE)
OLD_HTML_FLAT_RE   = re.compile(r"^/fomc/minutes/(\d{8})\.htm$", re.IGNORECASE)
HIST_MIN_PDF_RE    = re.compile(r"^/monetarypolicy/files/fomchistmin(\d{8})\.pdf$", re.IGNORECASE)  # NEW
MOA_PDF_RE         = re.compile(r"^/monetarypolicy/files/fomcmoa(\d{8})\.pdf$", re.IGNORECASE)

# We want everything up to and including 2019 (newer handled by your other script)
YEAR_MAX = 2019

# Output
OUT_DIR = Path("minutes_html")
MANIFEST = Path(resolve_path("minutes_historical_manifest.csv"))

# HTTP
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FOMCMinutesScraper/1.1; +https://example.org/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
TIMEOUT = 25
PER_REQUEST_SLEEP = 1.0

# Robots: bypass for research scraping (set False to re-enable)
SKIP_ROBOTS = True

# ------------------- Types -------------------

@dataclass
class Item:
    date_str: str      # YYYYMMDD
    year: int
    url: str
    kind: str          # "html_modern" | "html_old_deep" | "html_old_flat" | "pdf_histmin" | "pdf_moa"
    rel_path: str      # path under OUT_DIR

# ------------------- Helpers -------------------

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def can_fetch(url: str, user_agent: str) -> bool:
    if SKIP_ROBOTS:
        return True
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url); rp.read()
    except Exception:
        return True
    return rp.can_fetch(user_agent, url)

def get_with_retries(url: str, max_attempts=4, backoff=1.6) -> requests.Response:
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"Server {resp.status_code}")
            return resp
        except Exception as e:
            last_exc = e
            if attempt < max_attempts:
                time.sleep(backoff ** attempt)
            else:
                raise
    if last_exc:
        raise last_exc

def strip_query_and_fragment(href: str) -> str:
    """Return href without ?query or #fragment; preserve absolute form if provided."""
    p = urlparse(href)
    clean = p._replace(query="", fragment="")
    if p.scheme and p.netloc:
        return urlunparse(clean)
    return clean.path

# ------------------- Extraction -------------------

def extract_year_pages(index_html: str) -> List[Tuple[int, str]]:
    soup = BeautifulSoup(index_html, "html.parser")
    found: Dict[int, str] = {}
    for a in soup.find_all("a", href=True):
        href = strip_query_and_fragment(a["href"])
        m = YEAR_PAGE_RE.match(urlparse(href).path)
        if not m:
            continue
        year = int(m.group(1))
        if year > YEAR_MAX:
            continue
        abs_url = urljoin(BASE, href)
        found.setdefault(year, abs_url)
    return sorted(found.items(), key=lambda t: t[0], reverse=True)

def extract_items_from_year_page(year_html: str, default_year: int) -> List[Item]:
    soup = BeautifulSoup(year_html, "html.parser")
    items: Dict[str, Item] = {}  # keyed by date_str; preference order: html_modern > html_old_deep > html_old_flat > pdf_histmin > pdf_moa

    for a in soup.find_all("a", href=True):
        href = strip_query_and_fragment(a["href"])
        path = urlparse(href).path
        abs_url = urljoin(BASE, href)

        # Modern HTML minutes
        m = NEW_HTML_RE.match(path)
        if m:
            yyyymmdd = m.group(1)
            year = int(yyyymmdd[:4])
            rel = f"{year}/fomcminutes-{yyyymmdd}.htm"
            it = Item(yyyymmdd, year, abs_url, "html_modern", rel)
            prev = items.get(yyyymmdd)
            if prev is None or prev.kind != "html_modern":
                items[yyyymmdd] = it
            continue

        # Old deep structure: /fomc/minutes/YYYY/YYYYMMDDmin.htm
        m = OLD_HTML_DEEP_RE.match(path)
        if m:
            url_year = int(m.group(1))
            yyyymmdd = m.group(2)
            rel = f"{url_year}/fomcminutes-{yyyymmdd}.htm"
            it = Item(yyyymmdd, url_year, abs_url, "html_old_deep", rel)
            prev = items.get(yyyymmdd)
            if prev is None or prev.kind in ("html_old_flat", "pdf_histmin", "pdf_moa"):
                items[yyyymmdd] = it
            continue

        # Old flat structure: /fomc/minutes/YYYYMMDD.htm
        m = OLD_HTML_FLAT_RE.match(path)
        if m:
            yyyymmdd = m.group(1)
            year = int(yyyymmdd[:4])
            rel = f"{year}/fomcminutes-{yyyymmdd}.htm"
            it = Item(yyyymmdd, year, abs_url, "html_old_flat", rel)
            prev = items.get(yyyymmdd)
            if prev is None or prev.kind in ("pdf_histmin", "pdf_moa"):
                items[yyyymmdd] = it
            continue

        # Historical minutes PDFs: /monetarypolicy/files/fomchistminYYYYMMDD.pdf
        m = HIST_MIN_PDF_RE.match(path)
        if m:
            yyyymmdd = m.group(1)
            year = int(yyyymmdd[:4])
            rel = f"{year}/fomchistmin-{yyyymmdd}.pdf"
            it = Item(yyyymmdd, year, abs_url, "pdf_histmin", rel)
            prev = items.get(yyyymmdd)
            if prev is None:
                items[yyyymmdd] = it
            continue

        # Minutes of Actions PDFs: /monetarypolicy/files/fomcmoaYYYYMMDD.pdf
        m = MOA_PDF_RE.match(path)
        if m:
            yyyymmdd = m.group(1)
            year = int(yyyymmdd[:4])
            rel = f"{year}/fomcmoa-{yyyymmdd}.pdf"
            it = Item(yyyymmdd, year, abs_url, "pdf_moa", rel)
            # Only keep if we don't already have a more specific minutes document
            if yyyymmdd not in items:
                items[yyyymmdd] = it
            continue

    return sorted(items.values(), key=lambda x: x.date_str)

# ------------------- Manifest -------------------

def write_manifest_header(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["date", "year", "url", "saved_path", "status", "bytes", "sha256", "kind"])

def append_manifest(row: Tuple[str, int, str, str, int, int, str, str]) -> None:
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ------------------- Main -------------------

def main() -> int:
    print(f"[INFO] Fetching historical index: {HIST_INDEX_URL}")
    if SKIP_ROBOTS:
        print("[INFO] Robots.txt check is DISABLED for this run.")
    if not can_fetch(HIST_INDEX_URL, HEADERS["User-Agent"]):
        print("[WARN] robots.txt disallows fetching the historical index; exiting.")
        return 1

    idx_resp = get_with_retries(HIST_INDEX_URL)
    idx_resp.raise_for_status()

    year_pages = extract_year_pages(idx_resp.text)
    if not year_pages:
        print(f"[WARN] No historical year pages found (<= {YEAR_MAX}).")
        return 0

    print(f"[INFO] Found {len(year_pages)} year pages (<= {YEAR_MAX}).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest_header(MANIFEST)

    seen_urls: set[str] = set()
    total_items = 0

    for year, year_url in year_pages:
        print(f"[INFO] Year {year}: {year_url}")
        if not can_fetch(year_url, HEADERS["User-Agent"]):
            print(f"[SKIP] robots.txt disallows: {year_url}")
            continue
        try:
            yresp = get_with_retries(year_url); yresp.raise_for_status()
        except Exception as e:
            print(f"[ERR]  Fetch year page {year_url}: {e}")
            continue

        items = extract_items_from_year_page(yresp.text, year)
        if not items:
            print(f"[INFO] No minutes links found on {year_url}")
            continue

        print(f"[INFO]   Found {len(items)} minutes entries on year page {year}.")
        total_items += len(items)

        for i, it in enumerate(items, 1):
            if it.url in seen_urls:
                print(f"[SKIP]   ({i}/{len(items)}) already downloaded: {it.url}")
                continue

            save_path = OUT_DIR / it.rel_path
            ensure_parent(save_path)

            if save_path.exists():
                data = save_path.read_bytes()
                append_manifest((it.date_str, it.year, it.url, str(save_path), 200, len(data), sha256_bytes(data), it.kind))
                print(f"[SKIP]   ({i}/{len(items)}) exists: {save_path}")
                seen_urls.add(it.url)
                continue

            if not can_fetch(it.url, HEADERS["User-Agent"]):
                print(f"[SKIP]   robots.txt disallows: {it.url}")
                append_manifest((it.date_str, it.year, it.url, str(save_path), 0, 0, "", it.kind))
                continue

            try:
                resp = get_with_retries(it.url)
                status = resp.status_code
                if status == 200:
                    data = resp.content
                    save_path.write_bytes(data)
                    append_manifest((it.date_str, it.year, it.url, str(save_path), status, len(data), sha256_bytes(data), it.kind))
                    print(f"[OK]     ({i}/{len(items)}) Saved -> {save_path} [{it.kind}] ({len(data)} bytes)")
                    seen_urls.add(it.url)
                else:
                    print(f"[WARN]   ({i}/{len(items)}) Unexpected status {status} for {it.url}")
                    append_manifest((it.date_str, it.year, it.url, str(save_path), status, 0, "", it.kind))
            except Exception as e:
                print(f"[ERR]    ({i}/{len(items)}) {it.url}: {e}")
                append_manifest((it.date_str, it.year, it.url, str(save_path), 0, 0, "", it.kind))

            time.sleep(PER_REQUEST_SLEEP)

    print(f"[DONE] Downloaded/processed items from {len(year_pages)} year pages. Total found: {total_items}.")
    print(f"[DONE] Files in: {OUT_DIR.resolve()}")
    print(f"[DONE] Manifest: {MANIFEST.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())