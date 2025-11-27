#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward SP500 next-day prediction from FOMC sentiment.

Differences from the previous version:
- same expanding / walk-forward training
- BUT the plot now only:
    * plots the original true series once
    * plots predictions ONLY for validation rows (split startswith "val_")
so you don't get the fan-of-lines effect.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================= Arg Parsing ===================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sent",   required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year",   type=int, default=2025)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr",     type=float, default=0.01)
    ap.add_argument("--l2",     type=float, default=1e-4)
    ap.add_argument("--no-plot", action="store_true")
    # walk-forward knobs
    ap.add_argument("--min-train", type=int, default=80,
                    help="rows in first training window")
    ap.add_argument("--val-window", type=int, default=12,
                    help="rows to validate per walk-forward step")
    return ap.parse_args()


# ============================= Loaders =======================================

def load_sent(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date_iso" not in df.columns or "sentiment_sum" not in df.columns:
        raise ValueError("sentiment CSV must have columns: date_iso, sentiment_sum")
    df = df[["date_iso", "sentiment_sum"]].copy()
    df["date_iso"] = pd.to_datetime(df["date_iso"]).dt.date.astype(str)
    return df


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date_iso" not in cols:
        if "date" in cols:
            df.rename(columns={cols["date"]: "date_iso"}, inplace=True)
        else:
            raise ValueError("prices CSV must have column date_iso (or date)")
    if "close" not in cols:
        raise ValueError("prices CSV must have column close")
    df.rename(columns={
        cols.get("date_iso", "date_iso"): "date_iso",
        cols["close"]: "close"
    }, inplace=True)
    df = df[["date_iso", "close"]].copy()
    df["date_iso"] = pd.to_datetime(df["date_iso"]).dt.date.astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


# ============================= Feature Builder ===============================

def build_supervised(sent_df: pd.DataFrame, px_df: pd.DataFrame,
                     y0: int, y1: int) -> pd.DataFrame:
    df = sent_df.merge(px_df, on="date_iso", how="inner")
    df["date"] = pd.to_datetime(df["date_iso"])
    yrs = df["date"].dt.year
    mask = (yrs >= y0) & (yrs <= y1)
    df = df.loc[mask].sort_values("date").reset_index(drop=True)

    df["x"] = df["sentiment_sum"].astype(float)
    df["x_delta"] = df["x"].diff().fillna(0.0)
    df["year_norm"] = (df["date"].dt.year - y0) / float(max(y1 - y0, 1))
    df["t_norm"] = np.linspace(0.0, 1.0, len(df))

    df["next_close"] = df["close"].shift(-1)
    df["target_ratio"] = df["next_close"] / df["close"]

    df = df.iloc[:-1].reset_index(drop=True)
    return df[[
        "date_iso", "x", "x_delta", "year_norm", "t_norm",
        "close", "next_close", "target_ratio"
    ]].copy()


# ============================= MLP core ======================================

def _relu(a):
    return np.maximum(a, 0.0)


def train_mlp_ratio(X_tr, y_tr_ratio, hidden, epochs, lr, l2, seed=42):
    x_mu = X_tr.mean(axis=0)
    x_sd = X_tr.std(axis=0)
    x_sd[x_sd < 1e-8] = 1e-8

    y_mu = float(y_tr_ratio.mean())
    y_sd = float(max(y_tr_ratio.std(), 1e-8))

    Xz = (X_tr - x_mu) / x_sd
    yz = (y_tr_ratio - y_mu) / y_sd

    rng = np.random.default_rng(seed)
    d_in = Xz.shape[1]
    h = int(hidden)
    W1 = rng.normal(0, 0.1, size=(d_in, h))
    b1 = np.zeros(h)
    W2 = rng.normal(0, 0.1, size=(h, 1))
    b2 = np.zeros(1)

    for ep in range(int(epochs)):
        Z1 = Xz @ W1 + b1
        H = _relu(Z1)
        pred = (H @ W2 + b2).squeeze(-1)
        err = pred - yz

        mse = float(np.mean(err**2))
        reg = l2 * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = mse + reg

        n = Xz.shape[0]
        dloss_dpred = (2.0 / n) * err
        dW2 = H.T @ dloss_dpred[:, None] + 2 * l2 * W2
        db2 = np.sum(dloss_dpred)
        dH = dloss_dpred[:, None] @ W2.T
        dZ1 = dH * (Z1 > 0)
        dW1 = Xz.T @ dZ1 + 2 * l2 * W1
        db1 = np.sum(dZ1, axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if ep % 500 == 0 or ep == epochs - 1:
            print(f"[ep {ep:5d}] loss={loss:.6f} (mse={mse:.6f})")

    model = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "x_mu": x_mu.tolist(),
        "x_sd": x_sd.tolist(),
        "y_mu_ratio": y_mu,
        "y_sd_ratio": y_sd,
    }

    yhat_tr_ratio = infer_ratio_from_model(model, X_tr)
    return model, yhat_tr_ratio


def infer_ratio_from_model(model: dict, X: np.ndarray) -> np.ndarray:
    W1 = np.array(model["W1"])
    b1 = np.array(model["b1"])
    W2 = np.array(model["W2"])
    b2 = np.array(model["b2"])
    x_mu = np.array(model["x_mu"])
    x_sd = np.array(model["x_sd"])
    y_mu = float(model["y_mu_ratio"])
    y_sd = float(model["y_sd_ratio"])

    Xz = (X - x_mu) / x_sd
    Z1 = Xz @ W1 + b1
    H = _relu(Z1)
    z = (H @ W2 + b2).squeeze(-1)
    ratio = z * y_sd + y_mu
    return ratio


# ============================= Walk-forward ==================================

def walk_forward_train(df: pd.DataFrame,
                       hidden: int,
                       epochs: int,
                       lr: float,
                       l2: float,
                       min_train: int,
                       val_window: int):
    n = len(df)
    all_rows = []
    metrics = []
    train_end = min_train
    fold = 0

    while train_end < n:
        val_end = min(train_end + val_window, n)

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)

        X_tr = train_df[["x", "x_delta", "year_norm", "t_norm"]].values.astype("float64")
        y_tr_ratio = train_df["target_ratio"].values.astype("float64")

        model, yhat_tr_ratio = train_mlp_ratio(
            X_tr, y_tr_ratio,
            hidden=hidden, epochs=epochs, lr=lr, l2=l2,
            seed=42 + fold
        )

        # validate
        if len(val_df) > 0:
            X_va = val_df[["x", "x_delta", "year_norm", "t_norm"]].values.astype("float64")
            va_close = val_df["close"].values.astype("float64")
            yhat_va_ratio = infer_ratio_from_model(model, X_va)
            yhat_va_next = va_close * yhat_va_ratio
            ytrue_va_next = val_df["next_close"].values.astype("float64")

            mae_va = float(np.mean(np.abs(ytrue_va_next - yhat_va_next)))
            rmse_va = float(np.sqrt(np.mean((ytrue_va_next - yhat_va_next) ** 2)))
        else:
            yhat_va_next = np.array([])
            ytrue_va_next = np.array([])
            yhat_va_ratio = np.array([])
            mae_va = float("nan")
            rmse_va = float("nan")

        metrics.append({
            "fold": fold,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "mae_va": mae_va,
            "rmse_va": rmse_va,
        })

        # store ONLY val rows for plotting / oos inspection
        if len(val_df) > 0:
            va_rows = pd.DataFrame({
                "date_iso": val_df["date_iso"].values,
                "x": val_df["x"].values,
                "x_delta": val_df["x_delta"].values,
                "close": val_df["close"].values,
                "y_true_next_close": ytrue_va_next,
                "y_pred_next_close": yhat_va_next,
                "split": f"val_{fold}",
                "y_true_ratio": val_df["target_ratio"].values,
                "y_pred_ratio": yhat_va_ratio,
            })
            all_rows.append(va_rows)

        fold += 1
        train_end = val_end
        if val_end >= n:
            break

    if all_rows:
        return pd.concat(all_rows).reset_index(drop=True), metrics
    else:
        return pd.DataFrame(), metrics


# ============================= Main ==========================================

def main():
    args = parse_args()

    sent_df = load_sent(args.sent)
    price_df = load_prices(args.prices)
    base_df = build_supervised(sent_df, price_df, args.start_year, args.end_year)
    print(f"[INFO] rows={len(base_df)}  range={base_df['date_iso'].iloc[0]} → {base_df['date_iso'].iloc[-1]}")

    # walk-forward to get OUT-OF-SAMPLE predictions
    pred_df, fold_metrics = walk_forward_train(
        base_df,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        min_train=args.min_train,
        val_window=args.val_window,
    )

    # train final model on ALL data so you still get model.json
    X_all = base_df[["x", "x_delta", "year_norm", "t_norm"]].values.astype("float64")
    y_all_ratio = base_df["target_ratio"].values.astype("float64")
    final_model, _ = train_mlp_ratio(
        X_all, y_all_ratio,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=999,
    )

    outdir = Path("artifacts_spx_from_sent_delta")
    outdir.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(outdir / "predictions.csv", index=False)
    print("[OK] wrote:", outdir / "predictions.csv")

    summary = {
        "folds": fold_metrics,
        "avg_mae_va": float(np.nanmean([m["mae_va"] for m in fold_metrics])) if fold_metrics else None,
        "avg_rmse_va": float(np.nanmean([m["rmse_va"] for m in fold_metrics])) if fold_metrics else None,
    }
    (outdir / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "model.json").write_text(json.dumps(final_model, indent=2))
    print("[OK] wrote:", outdir / "fit_summary.json")
    print("[OK] wrote:", outdir / "model.json")

    if not args.no_plot:
        # make everything bigger for PPT
        plt.rcParams["font.size"] = 18      # base font
        plt.rcParams["axes.titlesize"] = 18   # title
        plt.rcParams["axes.labelsize"] = 16   # x/y labels
        plt.rcParams["legend.fontsize"] = 14
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["ytick.labelsize"] = 12

        # true line from base_df (clean, no duplicates)
        base_df["date"] = pd.to_datetime(base_df["date_iso"])
        plt.figure(figsize=(14, 6))
        plt.plot(
            base_df["date"],
            base_df["next_close"],
            label="true next close",
            linewidth=2.5,      # thicker
        )

        # predicted points from validation rows only
        if not pred_df.empty:
            pred_df["date"] = pd.to_datetime(pred_df["date_iso"])
            plt.plot(
                pred_df["date"],
                pred_df["y_pred_next_close"],
                label="pred next close (OOS)",
                linewidth=2.5,
            )

            # shade each validation window
            unique_splits = pred_df["split"].unique()
            for i, s in enumerate(unique_splits):
                mask = pred_df["split"] == s
                plt.axvspan(
                    pred_df.loc[mask, "date"].min(),
                    pred_df.loc[mask, "date"].max(),
                    alpha=0.08,
                    color="grey",
                    label="validation" if i == 0 else None,
                )

        plt.title("SP500 next-close: walk-forward true vs predicted")
        plt.xlabel("date")
        plt.ylabel("next close level")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
