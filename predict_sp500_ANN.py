#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict next SP500 close using sentiment and delta sentiment (no returns, no current price).

Inputs:
  --sent    path to sentiment scores CSV (must have: date_iso, sentiment_sum)
  --prices  path to SP500 prices CSV (must have: date_iso, close)
  --start-year  First year to keep (default 2000)
  --end-year    Last year to keep  (default 2025)

Outputs:
  artifacts_spx_from_sent_delta/predictions.csv
  artifacts_spx_from_sent_delta/fit_summary.json
  artifacts_spx_from_sent_delta/model.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

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
    return ap.parse_args()

def load_sent(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date_iso" not in df.columns or "sentiment_sum" not in df.columns:
        raise ValueError("sentiment CSV must have columns: date_iso, sentiment_sum")
    df = df[["date_iso", "sentiment_sum"]].copy()
    df["date_iso"] = pd.to_datetime(df["date_iso"]).dt.date.astype(str)
    return df

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # accept any case/extra cols; map
    cols = {c.lower(): c for c in df.columns}
    if "date_iso" not in cols:
        # allow 'date' fallback
        if "date" in cols:
            df.rename(columns={cols["date"]:"date_iso"}, inplace=True)
        else:
            raise ValueError("prices CSV must have column date_iso (or date)")
    if "close" not in cols:
        raise ValueError("prices CSV must have column close")
    # normalize column names
    df.rename(columns={cols.get("date_iso", "date_iso"):"date_iso",
                       cols["close"]:"close"}, inplace=True)
    df = df[["date_iso","close"]].copy()
    df["date_iso"] = pd.to_datetime(df["date_iso"]).dt.date.astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df

def build_supervised(sent_df: pd.DataFrame, px_df: pd.DataFrame,
                     y0: int, y1: int) -> pd.DataFrame:
    # inner join on meeting dates (your prices are curated for meeting days)
    df = sent_df.merge(px_df, on="date_iso", how="inner")
    # filter years
    years = pd.to_datetime(df["date_iso"]).dt.year
    mask = (years >= y0) & (years <= y1)
    df = df.loc[mask].sort_values("date_iso").reset_index(drop=True)

    # features: x_t and delta x_t
    df["x"] = df["sentiment_sum"].astype(float)
    df["x_delta"] = df["x"].diff().fillna(0.0)

    # target: next meeting close (level)
    df["next_close"] = df["close"].shift(-1)
    df = df.iloc[:-1].reset_index(drop=True)  # drop last (no target)

    return df[["date_iso","x","x_delta","close","next_close"]].copy()

def zfit_train_val_split(df: pd.DataFrame, hidden: int, epochs: int, lr: float, l2: float,
                         seed: int = 42):
    # chronological split
    n = len(df)
    n_val = max(1, int(round(0.2 * n)))
    tr = slice(0, n - n_val)
    va = slice(n - n_val, n)

    X = df[["x","x_delta"]].values.astype("float64")
    y = df["next_close"].values.astype("float64")
    dates = df["date_iso"].values
    cur_close = df["close"].values

    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]
    dates_tr, dates_va = dates[tr], dates[va]
    curc_tr, curc_va   = cur_close[tr], cur_close[va]

    # standardize on TRAIN only
    x_mu = X_tr.mean(axis=0); x_sd = X_tr.std(axis=0); x_sd[x_sd < 1e-8] = 1e-8
    y_mu = float(y_tr.mean()); y_sd = float(y_tr.std()); y_sd = max(y_sd, 1e-8)

    X_trz = (X_tr - x_mu) / x_sd
    X_vaz = (X_va - x_mu) / x_sd
    y_trz = (y_tr - y_mu) / y_sd
    y_vaz = (y_va - y_mu) / y_sd

    # tiny MLP: 2 -> hidden -> 1, ReLU
    rng = np.random.default_rng(seed)
    d = X_trz.shape[1]
    h = int(hidden)
    W1 = rng.normal(0, 0.1, size=(d, h))
    b1 = np.zeros(h)
    W2 = rng.normal(0, 0.1, size=(h, 1))
    b2 = np.zeros(1)

    def relu(a): return np.maximum(a, 0)
    def fwd(A):
        Z1 = A @ W1 + b1
        H = relu(Z1)
        Z2 = H @ W2 + b2
        return Z1, H, Z2.squeeze(-1)

    lr = float(lr); l2 = float(l2); epochs = int(epochs)

    for ep in range(epochs):
        # forward (train)
        Z1, H, yhat = fwd(X_trz)
        err = yhat - y_trz
        mse = float(np.mean(err**2))
        reg = l2*(np.sum(W1*W1)+np.sum(W2*W2))
        loss = mse + reg

        # grads
        ntr = X_trz.shape[0]
        dy = (2.0/ntr) * err
        dW2 = H.T @ dy[:,None] + 2*l2*W2
        db2 = np.sum(dy)
        dH = dy[:,None] @ W2.T
        dZ1 = dH * (Z1 > 0)
        dW1 = X_trz.T @ dZ1 + 2*l2*W1
        db1 = np.sum(dZ1, axis=0)

        # update
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

        # val
        _, H_va, yhat_va = fwd(X_vaz)
        vmse = float(np.mean((yhat_va - y_vaz)**2))

        if ep % 500 == 0 or ep == epochs-1:
            print(f"[ep {ep:5d}] loss={loss:.6f} val={vmse:.6f} (mse={mse:.6f} vmse={vmse:.6f})")

    # de-standardize
    def inv_y(z): return z * y_sd + y_mu
    _, _, yhat_trz = fwd(X_trz)
    _, _, yhat_vaz = fwd(X_vaz)
    yhat_tr = inv_y(yhat_trz)
    yhat_va = inv_y(yhat_vaz)

    # metrics (levels)
    def metrics(y_true, y_pred):
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        return mae, rmse

    mae_tr, rmse_tr = metrics(y_tr, yhat_tr)
    mae_va, rmse_va = metrics(y_va, yhat_va)

    model = {
        "W1": W1.tolist(), "b1": b1.tolist(),
        "W2": W2.tolist(), "b2": b2.tolist(),
        "x_mu": x_mu.tolist(), "x_sd": x_sd.tolist(),
        "y_mu": y_mu, "y_sd": y_sd,
    }

    pack = {
        "train": (dates_tr, X_tr, curc_tr, y_tr, yhat_tr),
        "val":   (dates_va, X_va, curc_va, y_va, yhat_va),
    }
    return model, pack, {"mae_tr":mae_tr,"rmse_tr":rmse_tr,"mae_va":mae_va,"rmse_va":rmse_va}

def main():
    args = parse_args()
    sent = load_sent(args.sent)
    px   = load_prices(args.prices)
    df = build_supervised(sent, px, args.start_year, args.end_year)
    print(f"[INFO] rows: {len(df)}  years: {df['date_iso'].iloc[0]} → {df['date_iso'].iloc[-1]}")

    model, pack, summ = zfit_train_val_split(
        df, hidden=args.hidden, epochs=args.epochs, lr=args.lr, l2=args.l2
    )

    outdir = Path("artifacts_spx_from_sent_delta")
    outdir.mkdir(parents=True, exist_ok=True)

    # write predictions
    frames = []
    for split, tup in pack.items():
        dates, X, curc, y_true, y_pred = tup
        tmp = pd.DataFrame({
            "date_iso": dates,
            "x": X[:,0].astype(float),
            "x_delta": X[:,1].astype(float),
            "close": curc.astype(float),
            "y_true_next_close": y_true.astype(float),
            "y_pred_next_close": y_pred.astype(float),
            "split": split
        })
        frames.append(tmp)
    pred = pd.concat(frames).reset_index(drop=True)
    pred.to_csv(outdir/"predictions.csv", index=False)
    print("[OK] Wrote:", outdir/"predictions.csv")

    # write summary + model
    with open(outdir/"fit_summary.json","w") as f:
        json.dump(summ, f, indent=2)
    with open(outdir/"model.json","w") as f:
        json.dump(model, f, indent=2)

    print("[SUMMARY]", summ)

if __name__ == "__main__":
    main()
