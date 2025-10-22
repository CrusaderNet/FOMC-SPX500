#!/usr/bin/env python3
"""
Train a simple model to predict the NEXT FOMC-meeting S&P 500 close.

Inputs:
  - sentiment_scores.csv (from analyze_sentiments.py)
      Columns required: date_iso, sentiment_sum
  - sp500_prices.csv     (from fetch_sp500_prices.py)
      Columns required: requested_date, matched_trading_date, close

Outputs:
  - model_metrics.txt
  - model.pkl                          (sklearn LinearRegression)
  - model_train_eval_predictions.csv   (per-date y_true, y_pred)
  - next_meeting_prediction.txt        (single-line: predicted_close)
  - next_meeting_features.json         (features used for the last->next prediction)

Target definition:
  For each FOMC date t, predict S&P close at the NEXT meeting date t+1.
  Features (minimal baseline): sentiment_sum_t, close_t

Notes:
  - Supports --start-year / --end-year (inclusive), e.g. 1960–2025.
  - We train/evaluate only on rows that have a known next_close (i.e., we drop the
    most recent row), but we *still* use the latest available features to produce
    the "next meeting" prediction for deployment.
"""
from config_paths import resolve_path, ensure_all_dirs
ensure_all_dirs()


import argparse
import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


# ----------------------------
# Data loading / preparation
# ----------------------------

def load_data(
    sentiment_csv: Path,
    prices_csv: Path,
    start_year: int,
    end_year: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      (df_trainable, last_features)

      df_trainable: meeting rows with columns [date_iso, sentiment_sum, close, next_close],
                    filtered to the requested year window and with next_close defined.
      last_features: the latest available meeting's (date_iso, sentiment_sum, close),
                     even if it has no next_close yet (used for next-meeting prediction).
    """
    # Sentiment
    s = pd.read_csv(sentiment_csv)
    if not {"date_iso", "sentiment_sum"}.issubset(s.columns):
        cols = ", ".join(s.columns)
        raise ValueError(
            f"{sentiment_csv} must have columns ['date_iso','sentiment_sum'] (got: {cols})"
        )
    s = s[["date_iso", "sentiment_sum"]].dropna()
    s["date_iso"] = pd.to_datetime(s["date_iso"], errors="coerce")
    s = s.dropna(subset=["date_iso"])

    # Prices
    p = pd.read_csv(prices_csv)
    required = {"requested_date", "matched_trading_date", "close"}
    if not required.issubset(p.columns):
        cols = ", ".join(p.columns)
        raise ValueError(
            f"{prices_csv} must have columns {sorted(required)} (got: {cols})"
        )
    p = p[["requested_date", "matched_trading_date", "close"]].dropna()
    p["requested_date"] = pd.to_datetime(p["requested_date"], errors="coerce")
    p["matched_trading_date"] = pd.to_datetime(p["matched_trading_date"], errors="coerce")
    p = p.dropna(subset=["requested_date", "matched_trading_date"])

    # Merge on meeting date
    df = s.merge(p, left_on="date_iso", right_on="requested_date", how="inner")

    # Keep only the requested year window on the meeting date itself
    df = df[(df["date_iso"].dt.year >= start_year) & (df["date_iso"].dt.year <= end_year)]
    df = df.sort_values("date_iso").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No merged meetings in the year window {start_year}-{end_year}.")

    # Capture the last available features BEFORE dropping rows without next_close
    last_features = df[["date_iso", "sentiment_sum", "close"]].iloc[-1].copy()

    # Create the target (next meeting's close) WITHIN the filtered window
    df["next_close"] = df["close"].shift(-1)

    # Drop rows without next_close (e.g., last row)
    df_trainable = df.dropna(subset=["next_close"]).reset_index(drop=True)

    # Final columns for training
    df_trainable = df_trainable[["date_iso", "sentiment_sum", "close", "next_close"]]
    return df_trainable, last_features


# ----------------------------
# Training / evaluation
# ----------------------------

def train_and_eval(df: pd.DataFrame, holdout_n: int = 5):
    # Guard: tiny datasets
    if len(df) <= holdout_n + 3:
        holdout_n = max(1, min(2, len(df) - 3))

    train = df.iloc[:-holdout_n, :].copy()
    test = df.iloc[-holdout_n:, :].copy()

    X_train = train[["sentiment_sum", "close"]].values
    y_train = train["next_close"].values
    X_test = test[["sentiment_sum", "close"]].values
    y_test = test["next_close"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return pipe, train, test.assign(y_pred=y_pred), mae, r2


def write_outputs(model, train_df, eval_df, mae: float, r2: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / resolve_path("model.pkl"))

    with (out_dir / resolve_path("model_metrics.txt")).open("w", encoding="utf-8") as f:
        f.write(f"MAE: {mae:.4f}\nR2: {r2:.6f}\n")
        f.write(f"Train size: {len(train_df)}  Eval size: {len(eval_df)}\n")
        f.write("Features: ['sentiment_sum','close']  Target: next_close\n")
        if len(train_df) > 0:
            f.write(f"Train span: {train_df['date_iso'].min().date()} -> {train_df['date_iso'].max().date()}\n")
        if len(eval_df) > 0:
            f.write(f"Eval  span: {eval_df['date_iso'].min().date()} -> {eval_df['date_iso'].max().date()}\n")

    eval_out = eval_df[["date_iso", "sentiment_sum", "close", "next_close", "y_pred"]].copy()
    eval_out.to_csv(out_dir / resolve_path("model_train_eval_predictions.csv"), index=False)


# ----------------------------
# Next-meeting prediction
# ----------------------------

def predict_next(model, last_features: pd.Series, out_dir: Path) -> None:
    # Use the last *available* meeting's features (even though it lacks next_close)
    X_last = [[float(last_features["sentiment_sum"]), float(last_features["close"])]]
    pred = float(model.predict(X_last)[0])

    (out_dir / resolve_path("next_meeting_prediction.txt")).write_text(f"{pred:.4f}\n", encoding="utf-8")

    with (out_dir / resolve_path("next_meeting_features.json")).open("w", encoding="utf-8") as f:
        json.dump({
            "last_meeting_date_iso": str(pd.to_datetime(last_features["date_iso"]).date()),
            "sentiment_sum": float(last_features["sentiment_sum"]),
            "close": float(last_features["close"]),
            "predicted_next_close": pred
        }, f, indent=2)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentiment", type=Path, default=Path(resolve_path("sentiment_scores.csv")))
    ap.add_argument("--prices", type=Path, default=Path(resolve_path("sp500_prices.csv")))
    ap.add_argument("--out-dir", type=Path, default=Path("model_out"))
    ap.add_argument("--holdout-n", type=int, default=5)
    ap.add_argument("--start-year", type=int, default=1960, help="Earliest meeting year to include (inclusive)")
    ap.add_argument("--end-year", type=int, default=2025, help="Latest meeting year to include (inclusive)")
    args = ap.parse_args()

    if args.start_year > args.end_year:
        raise SystemExit(f"--start-year ({args.start_year}) cannot be greater than --end-year ({args.end_year}).")

    df, last_features = load_data(args.sentiment, args.prices, args.start_year, args.end_year)
    if len(df) < 5:
        raise SystemExit(f"Not enough meetings with next_close in range {args.start_year}-{args.end_year} after merge to train (got {len(df)}).")

    model, train_df, eval_df, mae, r2 = train_and_eval(df, holdout_n=args.holdout_n)
    write_outputs(model, train_df, eval_df, mae, r2, args.out_dir)
    predict_next(model, last_features, args.out_dir)

    print(
        f"Done. Years {args.start_year}-{args.end_year}. "
        f"Metrics at {args.out_dir/resolve_path("model_metrics.txt")}. "
        f"Next prediction at {args.out_dir/resolve_path("next_meeting_prediction.txt")}."
    )


if __name__ == "__main__":
    main()