#!/usr/bin/env python3
"""
Centralized paths for the FOMC-SPX500 project.

- Keep pipeline scripts at repo root.
- Keep these three as-is and their year subfolders:
    minutes_html/, minutes_text/, minutes_text_clean/
- All other inputs/outputs are routed into organized subfolders.
"""
from __future__ import annotations
import os
from pathlib import Path

# Root = current working directory of the repo (script location's parent)
ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
PATHS = {
    # FOMC artifacts (excluding the three canonical folders you already use)
    "fomc": DATA / "fomc",
    "manifests": DATA / "fomc" / "manifests",
    "features": DATA / "fomc" / "features",
    "sentiments": DATA / "fomc" / "sentiments",
    "artifacts": DATA / "fomc" / "artifacts",

    # S&P 500 related
    "sp500": DATA / "sp500",
    "sp500_dates": DATA / "sp500" / "dates",
    "sp500_prices": DATA / "sp500" / "prices",

    # Lexicon
    "lexicon": DATA / "lexicon",

    # Models & metrics
    "models": ROOT / "models",
    "model_metrics": ROOT / "models" / "metrics",
    "model_preds": ROOT / "models" / "predictions",

    # Reports & logs
    "reports": ROOT / "reports",
    "predictions": ROOT / "reports" / "predictions",
    "logs": ROOT / "logs",
}

# Map well-known basenames to target directories
BASENAME_TO_DIRKEY = {
    # CSVs
    "minutes_manifest.csv": "manifests",
    "minutes_historical_manifest.csv": "manifests",
    "sentiment_scores.csv": "sentiments",
    "Economic_Lexicon.csv": "lexicon",
    "sp500_dates_from_manifest_full.csv": "sp500_dates",
    "sp500_dates_from_manifest_unique.csv": "sp500_dates",
    "sp500_dates_unique.csv": "sp500_dates",
    "sp500_prices.csv": "sp500_prices",
    "sp500_prices_missing.csv": "sp500_prices",

    # TXT / JSON
    "sp500_dates_from_manifest_unique.txt": "sp500_dates",
    "sp500_dates_unique.txt": "sp500_dates",
    "next_meeting_prediction.txt": "predictions",
    "next_meeting_features.json": "features",

    # Model artifacts
    "model.pkl": "models",
    "model_metrics.txt": "model_metrics",
    "model_train_eval_predictions.csv": "model_metrics",
}

def ensure_all_dirs() -> None:
    """Create the folder tree if missing."""
    for p in PATHS.values():
        p.mkdir(parents=True, exist_ok=True)

def resolve_dir(key: str) -> Path:
    return PATHS[key]

def resolve_path(basename: str, *, fallback_dir_key: str | None = None) -> Path:
    """
    Return the full path for a known output/input basename according to
    the new layout. If unknown, place into the provided fallback_dir_key
    (or data/ if not set).
    """
    dir_key = BASENAME_TO_DIRKEY.get(basename, fallback_dir_key or "data")
    base_dir = PATHS.get(dir_key, DATA)
    return base_dir / basename
