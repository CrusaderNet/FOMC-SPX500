from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import csv
import os
import re
import sys
import time

from bs4 import BeautifulSoup
import requests


from config_paths import resolve_path, ensure_all_dirs
from dataclasses import dataclass
from urllib import robotparser
from urllib.parse import urljoin, urlparse
import hashlib

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape FOMC meeting minutes HTML pages (2020–2025) and save them locally.

Outputs:
- minutes_html/<year>/fomcminutes-YYYYMMDD.htm (raw HTML)
- minutes_manifest.csv (URL, path, status, size, sha256)

Usage:
  python scrape_fomc_minutes.py
"""
ensure_all_dirs()





BASE = "https://www.federalreserve.gov"
CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
MINUTES_RE = re.compile(r"^/monetarypolicy/fomcminutes(\d{8})\.htm$", re.IGNORECASE)

# Year range to include
YEAR_MIN, YEAR_MAX = 2020, 2025

# Output locations
OUT_DIR = Path("minutes_html")
MANIFEST = Path(resolve_path("minutes_manifest.csv"))

# HTTP settings
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FOMCMinutesScraper/1.0; +https://example.org/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Politeness
PER_REQUEST_SLEEP = 1.0  # seconds between requests

# Robots handling: set True to bypass robots.txt everywhere
SKIP_ROBOTS = True


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

@dataclass
class MinutesLink:
    date_str: str     # YYYYMMDD from the URL
    year: int
    url: str
    rel_path: str     # e.g., "2024/fomcminutes-20240131.htm"

def ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def within_year_range(yyyymmdd: str) -> bool:
    try:
        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        return YEAR_MIN <= dt.year <= YEAR_MAX
    except ValueError:
        return False


def can_fetch(url: str, user_agent: str) -> bool:
    """Return True if fetching is allowed, or if SKIP_ROBOTS=True."""
    if SKIP_ROBOTS:
        return True
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        # If robots can't be read, default to cautious allow
        return True
    return rp.can_fetch(user_agent, url)


def get_with_retries(url: str, headers=None, max_attempts=4, backoff=1.5, timeout=20) -> requests.Response:
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=headers or HEADERS, timeout=timeout)
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


def extract_minutes_links(html: str) -> List[MinutesLink]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[MinutesLink] = []
    for a in soup.find_all("a", href=True):
        m = MINUTES_RE.match(a["href"])
        if not m:
            continue
        yyyymmdd = m.group(1)
        if not within_year_range(yyyymmdd):
            continue
        year = int(yyyymmdd[:4])
        full_url = urljoin(BASE, a["href"])
        rel = f"{year}/fomcminutes-{yyyymmdd}.htm"
        links.append(MinutesLink(date_str=yyyymmdd, year=year, url=full_url, rel_path=rel))
    # Deduplicate while preserving order
    unique, seen = [], set()
    for x in links:
        if x.url not in seen:
            unique.append(x)
            seen.add(x.url)
    # Sort by date ascending
    unique.sort(key=lambda x: x.date_str)
    return unique

def write_manifest_header(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "year", "url", "saved_path", "status", "bytes", "sha256"])

def append_manifest(row: Tuple[str, int, str, str, int, int, str]) -> None:
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)


def main() -> int:
    print(f"[INFO] Fetching calendar: {CAL_URL}")
    if SKIP_ROBOTS:
        print("[INFO] Robots.txt check is DISABLED for this run.")

    # Calendar fetch
    if not can_fetch(CAL_URL, HEADERS["User-Agent"]):
        print("[WARN] robots.txt disallows fetching the calendar page; exiting.")
        return 1

    cal_resp = get_with_retries(CAL_URL)
    cal_resp.raise_for_status()

    links = extract_minutes_links(cal_resp.text)
    if not links:
        print("[WARN] No minutes links found in the specified range.")
        return 0

    print(f"[INFO] Found {len(links)} minute links (years {YEAR_MIN}-{YEAR_MAX}).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest_header(MANIFEST)

    for i, item in enumerate(links, 1):
        save_path = OUT_DIR / item.rel_path
        ensure_dirs(save_path)

        if save_path.exists():
            # Already downloaded: log current file info for completeness
            data = save_path.read_bytes()
            append_manifest((item.date_str, item.year, item.url, str(save_path), 200, len(data), sha256_bytes(data)))
            print(f"[SKIP] ({i}/{len(links)}) {item.date_str} already on disk: {save_path}")
            continue

        if not can_fetch(item.url, HEADERS["User-Agent"]):
            print(f"[SKIP] robots.txt disallows: {item.url}")
            append_manifest((item.date_str, item.year, item.url, str(save_path), 0, 0, ""))
            continue

        try:
            resp = get_with_retries(item.url)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if status == 200 and "text/html" in ctype:
                data = resp.content
                save_path.write_bytes(data)
                checksum = sha256_bytes(data)
                print(f"[OK]   ({i}/{len(links)}) Saved {item.date_str} -> {save_path} ({len(data)} bytes)")
                append_manifest((item.date_str, item.year, item.url, str(save_path), status, len(data), checksum))
            else:
                print(f"[WARN] ({i}/{len(links)}) Unexpected response ({status}, {ctype}) for {item.url}")
                append_manifest((item.date_str, item.year, item.url, str(save_path), status, 0, ""))
        except Exception as e:
            print(f"[ERR]  ({i}/{len(links)}) {item.url}: {e}")
            append_manifest((item.date_str, item.year, item.url, str(save_path), 0, 0, ""))

        time.sleep(PER_REQUEST_SLEEP)

    print(f"[DONE] Files in: {OUT_DIR.resolve()}")
    print(f"[DONE] Manifest: {MANIFEST.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
