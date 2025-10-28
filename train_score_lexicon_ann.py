#!/usr/bin/env python3
# train_score_lexicon_ann.py
"""
Single-neuron ANN to score FOMC minutes using a compact, hand-crafted
hawkish/dovish/positive/negative lexicon. Trains on supervised document
scores produced by your sentence-matching pipeline.

Inputs
  - data/fomc/sentiments/sentiment_document_scores.csv
      columns (at least): doc_path, score_doc   # score in [-1, 1]
  - minutes_text_clean/<YYYY>/fomcminutes-*_singleline.txt

Outputs
  - model_out_ann/ann_model.json         # weights, bias, vocab, mu, sigma
  - model_out_ann/ann_weights.csv        # token, weight
  - model_out_ann/train_eval_predictions.csv (doc_path, y_true, y_pred)
  - sentiment_scores.csv                 # full-run (2000..2025) document scores

Notes
  - Features: per-token counts (no TF), then log1p(counts), then standardize
    using training-set mean/std (saved and reused at inference).
  - Prediction range: [-1, 1] via y = 2*sigmoid(z) - 1
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------- Fixed paths & years -----------------------------

SUPERVISED_CSV = Path("data/fomc/sentiments/sentiment_document_scores.csv")
MINUTES_ROOT   = Path("minutes_text_clean")
MODEL_DIR      = Path("model_out_ann")
OUT_SCORES     = Path("sentiment_scores.csv")

START_YEAR = 2000
END_YEAR   = 2025

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------- Seed lexicon (lemmas) --------------------------

HAWKISH = {
    "business","businesses",
    "demand","economic","economy","employment","energy","equities","equity",
    "expansion","financial","growth","housing","income","indicators","inflation","inflationary",
    "investment","investments","labor","manufacturing","outlook","output","price","prices",
    "production","recovery","resource","resources","securities","slack","spending","target",
    "toll","wage","wages",
}
DOVISH = {"accommodation","devastation","downturn","recession","unemployment"}
POSITIVE = {
    "abate","abating","accelerated","accelerate","add","advance","advanced",
    "augmented","balanced","better","bolster","bolstered","bolsters","boom","booming",
    "boost","boosted","ease","eased","easing","elevate","elevated","elevating","elevates",
    "expand","expanded","expanding","expansionary","extend","extended","fast","faster","firm","firmer",
    "gain","gains","growing","heightened","high","higher","improve","improved","improving","increase",
    "increased","increases","increasing","more","raise","rapid","rebound","rebounded",
    "recover","recovered","recovering","rise","risen","rising","robust",
    "rose","significant","solid","sooner","spike","spikes","spiking","stable","stability",
    "strength","strengthen","strengthened","strengthening","strengthens","strong","stronger",
    "supportive","up","upside","upswing","uptick"
}
NEGATIVE = {
    "adverse","back","below","constrain","constrained","contract","contracting",
    "contraction","cooling","correction","dampen","decelerate","decelerated","decline",
    "declined","declines","declining","decrease","decreased","decreases","decreasing",
    "deepening","depress","depressed","deteriorate","deteriorated","deterioration",
    "diminish","diminished","disappointing","dislocation","disruptions","down","downbeat",
    "downside","drop","dropped","dropping","ebbed","erosion","fade","faded","fading",
    "fall","fallen","falling","insufficient","less","limit","low","lower","moderate",
    "moderated","moderating","moderation","reduce","reduced","reduction","reluctant",
    "removed","restrain","restrained","restraining","restraint","reversal","reversed",
    "slow","slowed","slower","slowing","slowly","sluggish","sluggishness","slumped","soft",
    "soften","softened","softening","stress","subdued","tragic","turmoil","underutilization",
    "volatile","vulnerable","wary","weak","weaken","weakened","weakness"
}

VOCAB: List[str] = sorted(HAWKISH | DOVISH | POSITIVE | NEGATIVE)
VOCAB_INDEX: Dict[str, int] = {t: i for i, t in enumerate(VOCAB)}
TEMP = 1.5  # 1.0 = original; >1 softens extremes

# ------------------------------ NLP (spaCy) ---------------------------------

def _build_nlp():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner","parser","textcat"])
        if "lemmatizer" not in nlp.pipe_names:
            nlp.add_pipe("lemmatizer", config={"mode": "rule"}, first=True)
        return nlp
    except Exception:
        # Simple fallback tokenizer/lemmatizer (lowercase split); not ideal but robust
        return None

def _lemmatize(nlp, text: str) -> List[str]:
    if nlp is None:
        # crude fallback: alpha-only, lowercase
        import re
        return [w.lower() for w in re.findall(r"[A-Za-z]+", text)]
    doc = nlp(text)
    out = []
    for t in doc:
        if t.is_space or t.is_punct or t.like_num:
            continue
        lemma = t.lemma_.lower().strip()
        if lemma:
            out.append(lemma)
    return out

# --------------------------- Feature construction ---------------------------

def vectorize_doc(nlp, text: str) -> np.ndarray:
    """
    Build per-vocab COUNT vector (no TF). Then apply log1p to tame large counts.
    """
    counts = np.zeros(len(VOCAB), dtype=np.float64)
    for lemma in _lemmatize(nlp, text):
        j = VOCAB_INDEX.get(lemma)
        if j is not None:
            counts[j] += 1.0
    # tame heavy-hitters but keep rare words informative
    count = np.log1p(counts)
    counts = np.minimum(counts, 5.0)  # tame extreme counts
    return counts

def minutes_files_in_range(root: Path, y0: int, y1: int) -> List[Path]:
    files: List[Path] = []
    for y in range(y0, y1 + 1):
        d = root / f"{y}"
        if d.exists():
            files.extend(sorted(d.glob("fomcminutes-*_singleline.txt")))
    return files

# ------------------------------- Model core ---------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def _scale_to_neg1_pos1(y01: np.ndarray) -> np.ndarray:
    return 2.0 * y01 - 1.0

def train_single_neuron(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    l2: float = 1e-3,
    epochs: int = 4000,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Minimize MSE for y_hat = 2*sigmoid(Xw + b) - 1 with L2 on w.
    X must already be standardized.
    """
    n, d = X.shape
    rng = np.random.default_rng(42)
    w = rng.normal(scale=0.01, size=(d,))
    b = 0.0

    for ep in range(epochs):
        z = X @ w + b                 # (n,)
        y01 = _sigmoid(z / TEMP)
        yhat = 2*y01 - 1
        diff = yhat - y               # (n,)

        mse = float(np.mean(diff**2))
        reg = float(l2 * np.sum(w*w))
        loss = mse + reg

        # gradients
        dL_dyhat = (2.0 / n) * diff
        dyhat_dz = (2.0 / TEMP) * (y01 * (1.0 - y01))
        dL_dz = dL_dyhat * dyhat_dz              # (n,)
        grad_w = X.T @ dL_dz + 2.0 * l2 * w
        grad_b = float(np.sum(dL_dz))

        # update
        w -= lr * grad_w
        b -= lr * grad_b

        if verbose and (ep % 500 == 0 or ep == epochs - 1):
            print(f"[ep {ep:4d}] loss={loss:.6f}  mse={mse:.6f}  reg={reg:.6f}  "
                  f"z_mu={z.mean():.3f} z_sd={z.std():.3f}  yhat_mu={yhat.mean():.3f}")

    stats = {"w_mean": float(w.mean()), "w_std": float(w.std()), "b": float(b)}
    return w, b, stats

# ------------------------------ Data loading --------------------------------

def load_supervised(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Accept your schema: doc_path, ..., score_doc
    req = {"doc_path", "score_doc"}
    if not req.issubset(df.columns):
        raise SystemExit(f"[ERR] {csv_path} missing columns {req} (got {list(df.columns)})")
    df = df[["doc_path", "score_doc"]].dropna().copy()
    return df

# ------------------------------- Main script --------------------------------

def _extract_date_iso_from_name(name: str) -> str:
    # fomcminutes-YYYYMMDD_singleline.txt
    try:
        s = name.split("fomcminutes-")[1][:8]
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    except Exception:
        return ""

def main():
    print(f"[INFO] Vocab size: {len(VOCAB)}")

    supervised = load_supervised(SUPERVISED_CSV)
    print(f"[INFO] Supervised docs: {len(supervised)}")

    nlp = _build_nlp()

    # --- Build training design matrix (raw counts -> log1p) ---
    X_rows, y_rows, used_docs = [], [], []
    for _, r in supervised.iterrows():
        p = Path(r["doc_path"])
        if not p.exists():
            print(f"[WARN] Missing supervised file: {p}")
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        X_rows.append(vectorize_doc(nlp, txt))
        y_rows.append(float(r["score_doc"]))
        used_docs.append(str(p))

    if not X_rows:
        raise SystemExit("[ERR] No supervised documents could be vectorized")

    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=np.float64)

    # --- Standardize features (save μ, σ) ---
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-6] = 1e-6
    Xz = (X - mu) / sigma

    #clip z-scores to avoid extreme outliers
    Xz = (X - mu) / sigma
    Xz = np.clip(Xz, -3.0, 3.0)

    # --- Train ---
    w, b, stats = train_single_neuron(
    Xz, y, lr=0.05, l2=1e-2, epochs=1500, verbose=True
    )
    print(f"[DEBUG] w mean={stats['w_mean']:.4f}  std={stats['w_std']:.4f}  b={stats['b']:.4f}")

    # --- Save model ---
    (MODEL_DIR / "ann_model.json").write_text(
        json.dumps(
            {
                "bias": float(b),
                "weights": w.tolist(),
                "vocab": VOCAB,
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "scaling": "Xz=(X-mu)/sigma; y=2*sigmoid(w·Xz + b)-1",
                "train_docs": used_docs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"token": VOCAB, "weight": w}).to_csv(MODEL_DIR / "ann_weights.csv", index=False)
    print(f"[OK] Saved model -> {MODEL_DIR/'ann_model.json'}; weights -> {MODEL_DIR/'ann_weights.csv'}")

    # --- Train predictions dump (for sanity) ---
    z_train = Xz @ w + b
    yhat_train = _scale_to_neg1_pos1(_sigmoid(z_train))
    pd.DataFrame(
        {"doc_path": used_docs, "y_true": y, "y_pred": yhat_train}
    ).to_csv(MODEL_DIR / "train_eval_predictions.csv", index=False)
    print(f"[OK] Wrote train eval -> {MODEL_DIR/'train_eval_predictions.csv'}")

    # --- Score ALL minutes 2000..2025 with saved μ/σ ---
    all_files = minutes_files_in_range(MINUTES_ROOT, START_YEAR, END_YEAR)
    print(f"[INFO] Scoring {len(all_files)} minutes files ({START_YEAR}..{END_YEAR})")

    out_rows = []
    for p in all_files:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        x_raw = vectorize_doc(nlp, txt)           # log1p(counts)
        xz = (x_raw - mu) / sigma
        xz = np.clip(xz, -3.0, 3.0)
        z = float(xz @ w + b)
        y01 = float(_sigmoid(z / TEMP))
        y11 = float(_scale_to_neg1_pos1(np.array([y01]))[0])

        out_rows.append(
            {
                "date_iso": _extract_date_iso_from_name(p.name),
                "file_path": str(p),
                "token_count": int(np.sum(x_raw > 0)),
                "sentiment_sum": round(y11, 6),
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_df = out_df[out_df["date_iso"] != ""].sort_values("date_iso").reset_index(drop=True)
    out_df.to_csv(OUT_SCORES, index=False)
    print(f"[OK] Wrote document scores -> {OUT_SCORES}")

if __name__ == "__main__":
    main()
