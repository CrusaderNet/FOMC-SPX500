#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANN (single neuron) sentiment trainer + scorer for FOMC minutes.

Model:
    y_hat = sigmoid( bias + Σ (x_i * w_i) ), where x_i = token count or freq
Targets:
    labels in [-1, +1] mapped to [0,1] via t01 = (label + 1) / 2

Inputs:
  - Documents: minutes_text_clean/<year>/fomcminutes-YYYYMMDD*.txt
  - Lexicon init: Economic_Lexicon.csv (columns: token, sentiment[, polarity])
  - Labels CSV: labels_ann.csv (see formats below)

Labels CSV accepted formats:
  1) date_iso,label
       2002-03-19,-0.4
       2007-05-09,0.3
  2) date_iso,class   # class ∈ {dovish, neutral, hawkish}
       2008-09-16,hawkish
     Mapping: dovish=-1, neutral=0, hawkish=+1

Outputs (default ./ann_out):
  - learned_lexicon.csv       token, weight_init, weight_learned
  - model_weights.json        {"bias": float, "weights": {token: float, ...}}
  - training_metrics.txt      MAE/R2 (in [-1,1] space), losses, span, counts
  - scored_documents.csv      (if --score-only or after training with --score-after)
  - debug_feature_dims.txt    snapshot of vocab size + doc counts

Typical usage:
  # Train on 1960-2025 window
  python train_ann_sentiment.py \
      --in-root minutes_text_clean \
      --lexicon Economic_Lexicon.csv \
      --labels labels_ann.csv \
      --start-year 1960 --end-year 2025 \
      --epochs 800 --lr 0.001 --l2 1e-6 --tf count --balance none \
      --score-after

  # Score only (uses saved model_weights.json)
  python train_ann_sentiment.py --score-only --in-root minutes_text_clean --start-year 1960 --end-year 2025
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import spacy

# ------------------------------ I/O utils ------------------------------

DATE_RE = re.compile(r"fomcminutes-(\d{8})", re.IGNORECASE)

def extract_date_iso(fname: str) -> Optional[str]:
    m = DATE_RE.search(fname)
    if not m:
        return None
    ymd = m.group(1)
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"

def iter_doc_files(in_root: Path, start_year: int, end_year: int) -> Iterable[Path]:
    # Traverse minutes_text_clean/<year>/* .txt
    for p in sorted(in_root.rglob("fomcminutes-*.txt")):
        if not p.is_file():
            continue
        d = extract_date_iso(p.name)
        if not d:
            continue
        y = int(d[:4])
        if start_year <= y <= end_year:
            yield p

# ------------------------------ Lexicon ------------------------------

def load_lexicon(path: Path) -> Dict[str, float]:
    """
    Returns lemma->weight (init) from Economic_Lexicon.csv.
    Expects 'token' and 'sentiment' columns. Ignores missing/invalid.
    """
    lex: Dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"{path} has no header")
        if "token" not in rdr.fieldnames or "sentiment" not in rdr.fieldnames:
            raise ValueError(f"{path} must have columns: token,sentiment (got: {rdr.fieldnames})")
        for row in rdr:
            tok = (row.get("token") or "").strip().lower()
            if not tok:
                continue
            try:
                w = float((row.get("sentiment") or "0").strip())
            except ValueError:
                w = 0.0
            # clamp to [-1,1]
            w = max(-1.0, min(1.0, w))
            lex[tok] = w
    return lex

# ------------------------------ Labels ------------------------------

def load_labels(path: Path) -> Dict[str, float]:
    """
    date_iso -> label in [-1,1]
    Accepts columns:
      - date_iso,label           (float in [-1,1])
      - date_iso,class           (dovish|neutral|hawkish)
    """
    lab: Dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None or "date_iso" not in rdr.fieldnames:
            raise ValueError("labels file must include column 'date_iso'")
        has_label = "label" in rdr.fieldnames
        has_class = "class" in rdr.fieldnames
        if not has_label and not has_class:
            raise ValueError("labels file must have 'label' (float) or 'class' (dovish|neutral|hawkish)")
        for row in rdr:
            d = (row.get("date_iso") or "").strip()
            if not d:
                continue
            if has_label:
                try:
                    v = float((row.get("label") or "").strip())
                except ValueError:
                    continue
                v = max(-1.0, min(1.0, v))
            else:
                c = (row.get("class") or "").strip().lower()
                if c == "dovish":
                    v = -1.0
                elif c == "neutral":
                    v = 0.0
                elif c == "hawkish":
                    v = 1.0
                else:
                    continue
            lab[d] = v
    return lab

# ------------------------------ NLP ------------------------------

def build_nlp(model: str = "en_core_web_sm"):
    nlp = spacy.load(model, disable=["ner", "parser", "textcat"])
    if "lemmatizer" not in nlp.pipe_names:
        nlp.add_pipe("lemmatizer", config={"mode": "rule"}, first=True)
    return nlp

# ------------------------------ Features ------------------------------

@dataclass
class DocVec:
    date_iso: str
    x: np.ndarray
    norm: float  # optional scaling info (e.g., token_count) for diagnostics

def vectorize_docs(
    files: List[Path],
    nlp,
    vocab: List[str],
    tf_mode: str = "count",
    stopwords: bool = True
) -> List[DocVec]:
    """
    Build X from document lemma counts over a fixed vocab.
    tf_mode: 'count' (raw counts) or 'freq' (count/sum_counts)
    """
    idx = {t: i for i, t in enumerate(vocab)}
    out: List[DocVec] = []

    for p in files:
        d = extract_date_iso(p.name)
        if not d:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        doc = nlp(text)

        vec = np.zeros(len(vocab), dtype=np.float64)
        total = 0.0

        for tok in doc:
            if tok.is_space or tok.is_punct or tok.like_num:
                continue
            if stopwords and tok.is_stop:
                continue
            lemma = tok.lemma_.lower().strip()
            if not lemma:
                continue
            if lemma in idx:
                vec[idx[lemma]] += 1.0
            total += 1.0

        if tf_mode == "freq" and total > 0:
            vec /= total

        out.append(DocVec(date_iso=d, x=vec, norm=total))
    return out

# ------------------------------ Model ------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z_clip = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z_clip))

@dataclass
class AnnModel:
    w: np.ndarray  # shape [V]
    b: float       # bias

    def predict01(self, X: np.ndarray) -> np.ndarray:
        # returns in [0,1]
        return sigmoid(X @ self.w + self.b)

    def predict_signed(self, X: np.ndarray) -> np.ndarray:
        # map back to [-1,1]
        p = self.predict01(X)
        return 2.0 * p - 1.0

# ------------------------------ Training ------------------------------

def mse_loss(p: np.ndarray, t: np.ndarray) -> float:
    # mean squared error in [0,1] space
    return float(np.mean((p - t) ** 2))

def train_ann(
    X: np.ndarray,           # [N, V]
    y_signed: np.ndarray,    # [N] in [-1,1]
    w_init: np.ndarray,      # [V]
    b_init: float,
    epochs: int,
    lr: float,
    l2: float,
    balance: str = "none"    # 'none' | 'mean' | 'median'
) -> Tuple[AnnModel, List[float]]:
    """
    Batch gradient descent with optional class balancing.
    balance:
      - none: uniform
      - mean: weight by inverse freq of sign buckets [-1,0,1] using mean counts
      - median: same using median counts
    """
    w = w_init.copy().astype(np.float64)
    b = float(b_init)

    # targets in [0,1]
    y = (y_signed + 1.0) / 2.0  # [-1,1] -> [0,1]

    # class weights
    if balance == "none":
        cw = np.ones_like(y)
    else:
        # 3 buckets by sign
        sgn = np.sign(y_signed)  # -1, 0, +1
        buckets = [-1.0, 0.0, 1.0]
        counts = [np.sum(sgn == k) for k in buckets]
        counts = [c if c > 0 else 1 for c in counts]
        inv = [1.0 / c for c in counts]
        scale = np.mean(counts) if balance == "mean" else np.median(counts)
        wt = {k: (scale * (1.0 / c)) for k, c in zip(buckets, counts)}
        cw = np.array([wt[kk] for kk in sgn], dtype=np.float64)

    losses: List[float] = []
    for _ in range(epochs):
        # forward
        z = X @ w + b
        p = sigmoid(z)              # [0,1]
        # weighted MSE
        err = (p - y)
        loss = float(np.sum(cw * (err ** 2)) / np.sum(cw))
        # L2
        loss += l2 * float(w @ w)
        losses.append(loss)

        # gradients
        # dL/dp = 2 * cw * (p - y) / sum(cw)
        denom = np.sum(cw)
        dLdp = (2.0 * cw * err) / denom
        # dp/dz = p*(1-p)
        dLdz = dLdp * p * (1.0 - p)   # [N]
        # dL/dw = X^T dLdz + 2*l2*w
        grad_w = X.T @ dLdz + 2.0 * l2 * w
        grad_b = float(np.sum(dLdz))

        # update
        w -= lr * grad_w
        b -= lr * grad_b

    return AnnModel(w=w, b=b), losses

# ------------------------------ Scoring ------------------------------

def score_docs(model: AnnModel, X: np.ndarray, dates: List[str]) -> List[Tuple[str, float]]:
    """
    Returns list of (date_iso, score_signed in [-1,1])
    """
    y_pred = model.predict_signed(X)
    out: List[Tuple[str, float]] = []
    for d, v in zip(dates, y_pred):
        out.append((d, float(v)))
    return out

# ------------------------------ CLI ------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train/Score an ANN (single neuron) for FOMC sentiment using a custom lexicon.")
    ap.add_argument("--in-root", type=Path, default=Path("minutes_text_clean"))
    ap.add_argument("--lexicon", type=Path, default=Path("Economic_Lexicon.csv"))
    ap.add_argument("--labels", type=Path, default=Path("labels_ann.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("ann_out"))

    ap.add_argument("--start-year", type=int, default=1960)
    ap.add_argument("--end-year", type=int, default=2025)

    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--tf", choices=["count", "freq"], default="count", help="count = raw counts; freq = normalized by doc length")

    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--balance", choices=["none", "mean", "median"], default="none")

    # Modes
    ap.add_argument("--score-only", action="store_true", help="Load model_weights.json and score documents only.")
    ap.add_argument("--score-after", action="store_true", help="After training, score the full window.")

    return ap.parse_args()

# ------------------------------ Main ------------------------------

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    nlp = build_nlp(args.spacy_model)

    # Gather documents in window
    files = list(iter_doc_files(args.in_root, args.start_year, args.end_year))
    if not files:
        raise SystemExit(f"No documents found in {args.in_root} for {args.start_year}-{args.end_year}")

    # Load/init vocab
    lex = load_lexicon(args.lexicon)              # lemma -> init weight ([-1,1])
    vocab = sorted(lex.keys())
    V = len(vocab)
    if V == 0:
        raise SystemExit("Lexicon is empty after load.")

    # Build features (for all docs in window)
    docvecs = vectorize_docs(files, nlp, vocab, tf_mode=args.tf)
    # Sort docvecs by date_iso for stable alignment
    docvecs.sort(key=lambda d: d.date_iso)
    dates_all = [d.date_iso for d in docvecs]
    X_all = np.stack([d.x for d in docvecs], axis=0) if docvecs else np.zeros((0, V), dtype=np.float64)

    # Persist feature dims for debugging
    (args.out_dir / "debug_feature_dims.txt").write_text(
        f"docs={len(dates_all)} vocab={V} tf={args.tf}\n"
        f"span={dates_all[0] if dates_all else 'NA'} -> {dates_all[-1] if dates_all else 'NA'}\n",
        encoding="utf-8"
    )

    # Score-only path
    if args.score_only:
        mw = json.loads((args.out_dir / "model_weights.json").read_text(encoding="utf-8"))
        w_vec = np.array([mw["weights"].get(tok, 0.0) for tok in vocab], dtype=np.float64)
        b = float(mw.get("bias", 0.0))
        model = AnnModel(w=w_vec, b=b)
        scored = score_docs(model, X_all, dates_all)
        with (args.out_dir / "scored_documents.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["date_iso", "score_ann_signed"])
            for d, v in scored: w.writerow([d, f"{v:.6f}"])
        print(f"[OK] Scored {len(scored)} documents -> {args.out_dir/'scored_documents.csv'}")
        return

    # Load labels and make supervised dataset
    labels = load_labels(args.labels)             # date_iso -> [-1,1]
    # Align X,y by date intersection
    idx = [i for i, d in enumerate(dates_all) if d in labels]
    if not idx:
        raise SystemExit(f"No labeled documents overlap with files in {args.start_year}-{args.end_year}.")
    X = X_all[idx, :]
    y_signed = np.array([labels[dates_all[i]] for i in idx], dtype=np.float64)

    # Initial weights/bias
    w_init = np.array([lex[tok] for tok in vocab], dtype=np.float64)  # start from lexicon weights ([-1,1])
    b_init = 0.0

    # Train
    model, losses = train_ann(
        X=X,
        y_signed=y_signed,
        w_init=w_init,
        b_init=b_init,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        balance=args.balance
    )

    # Train metrics (in signed space)
    y_pred_signed = model.predict_signed(X)
    mae_signed = float(np.mean(np.abs(y_pred_signed - y_signed)))
    # R2 in signed space; guard for constant labels
    y_bar = float(np.mean(y_signed))
    ss_tot = float(np.sum((y_signed - y_bar) ** 2))
    ss_res = float(np.sum((y_signed - y_pred_signed) ** 2))
    r2_signed = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Persist model weights
    weights_dict = {tok: float(model.w[i]) for i, tok in enumerate(vocab)}
    (args.out_dir / "model_weights.json").write_text(
        json.dumps({"bias": float(model.b), "weights": weights_dict}, indent=2),
        encoding="utf-8"
    )

    # Learned lexicon dump
    with (args.out_dir / "learned_lexicon.csv").open("w", encoding="utf-8", newline="") as f:
        wcsv = csv.writer(f); wcsv.writerow(["token", "weight_init", "weight_learned"])
        for i, tok in enumerate(vocab):
            wcsv.writerow([tok, f"{w_init[i]:.6f}", f"{model.w[i]:.6f}"])

    # Training report
    with (args.out_dir / "training_metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"Docs (window): {len(dates_all)}\n")
        f.write(f"Labeled docs:  {len(idx)}\n")
        if idx:
            f.write(f"Labeled span: {dates_all[idx[0]]} -> {dates_all[idx[-1]]}\n")
        f.write(f"Vocab size:    {V}\n")
        f.write(f"TF mode:       {args.tf}\n")
        f.write(f"Epochs/LR/L2:  {args.epochs}/{args.lr}/{args.l2}\n")
        f.write(f"Balance:       {args.balance}\n")
        f.write(f"MAE (signed):  {mae_signed:.6f}\n")
        f.write(f"R2  (signed):  {r2_signed:.6f}\n")
        if losses:
            f.write(f"Final loss:    {losses[-1]:.8f}\n")
        if dates_all:
            f.write(f"Window span:   {dates_all[0]} -> {dates_all[-1]}\n")

    print(f"[OK] Trained ANN | MAE={mae_signed:.4f} R2={r2_signed:.4f}")
    print(f"[OK] Weights -> {args.out_dir/'model_weights.json'}")
    print(f"[OK] Learned lexicon -> {args.out_dir/'learned_lexicon.csv'}")

    # Optional: score full window after training
    if args.score_after:
        scored = score_docs(model, X_all, dates_all)
        with (args.out_dir / "scored_documents.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["date_iso", "score_ann_signed"])
            for d, v in scored: w.writerow([d, f"{v:.6f}"])
        print(f"[OK] Scored {len(scored)} documents -> {args.out_dir/'scored_documents.csv'}")


if __name__ == "__main__":
    main()
