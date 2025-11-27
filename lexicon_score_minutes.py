#!/usr/bin/env python3
# lexicon_score_minutes.py
"""
===============================================================================
Score selected FOMC minutes using a simple lexicon scheme (per paper's §4.1).

Sentence scoring rule (k = sentence keyword type; p = #positive terms; n = #negative terms):
    score(sent, k) =  +1 if k == "hawk" and p > n
                      -1 if k == "hawk" and p < n
                      -1 if k == "dove" and p > n
                      +1 if k == "dove" and p < n
                       0 otherwise
Only sentences that contain at least one hawkish or dovish keyword are scored;
others are ignored (as in the paper).

Document score = mean of sentence scores (range [-1, +1]).

Outputs:
  data/fomc/sentiments/sentiment_supervised_index.csv
    columns: doc_path,year,n_sentences,count_hawkish,count_dovish,count_neutral,score_doc

Usage (PowerShell one-line example):
  python .\\lexicon_score_minutes.py `
    --files minutes_text_clean\\2007\\fomcminutes-20070509_singleline.txt `
            minutes_text_clean\\2008\\fomcminutes-20080916_singleline.txt
===============================================================================
"""

# ================================ Imports ====================================

import argparse
from pathlib import Path
import re
from typing import List, Dict, Tuple
from statistics import mean

# ================================ Lexicons ===================================

# Seeded from paper tables; extend as needed (keep lowercased tokens)
HAWKISH = {
    "business","businesses",
    "demand","economic","economy","employment","energy","equities","equity",
    "expansion","financial","growth","housing","income","indicators","inflation","inflationary",
    "investment","investments","labor","manufacturing","outlook","output","price","prices",
    "production","recovery","resource","resources","securities","slack","spending","target",
    "toll","wage","wages",
}
DOVISH = {
    "accommodation","devastation","downturn","recession","unemployment"
}
POSITIVE = {
    "abate","abating","accelerated","accelerate","add","advance","advanced",
    "augmented","balanced","better","bolster","bolstered","bolsters","boom","booming",
    "boost","boosted","ease","eased","easing","elevate","elevated",
    "elevating","elevates","expand","expanded","expanding","expansionary",
    "extend","extended","fast","faster","firm","firmer","gain","gains",
    "growing","heightened","high","higher","improve","improved","improving","increase",
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

# ================================ Helpers ====================================

def simple_sent_tokenize(text: str) -> List[str]:
    """
    Split on common sentence enders while keeping reasonable chunks.
    Many docs are a single line; treat literal '\n' defensively.
    """
    parts = re.split(r'(?<=[\.\?\!;:])\s+', text.replace('\\n', ' '))
    return [seg.strip() for seg in parts if seg.strip()]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def normalize_tokens(s: str) -> List[str]:
    """Lowercase and keep simple word tokens (drop punctuation/numbers)."""
    return [tok.lower() for tok in WORD_RE.findall(s)]

def count_polarity(tokens: List[str]) -> tuple[int, int]:
    """Count positive/negative token hits."""
    pos_count = sum(1 for tok in tokens if tok in POSITIVE)
    neg_count = sum(1 for tok in tokens if tok in NEGATIVE)
    return pos_count, neg_count

def which_keyword(tokens: List[str]) -> str:
    """
    Determine sentence keyword tag:
      - 'hawk' if any hawkish term and no dovish term
      - 'dove' if any dovish term and no hawkish term
      - 'both' if both present (we drop these later)
      - '' if none present
    """
    has_hawk = any(tok in HAWKISH for tok in tokens)
    has_dove = any(tok in DOVISH for tok in tokens)
    if has_hawk and not has_dove:
        return "hawk"
    if has_dove and not has_hawk:
        return "dove"
    if has_hawk and has_dove:
        return "both"
    return ""

def score_sentence(tokens: List[str]) -> int:
    """
    Apply the sign rule from the paper:
      hawk: +1 if p>n, -1 if p<n, 0 otherwise
      dove: -1 if p>n, +1 if p<n, 0 otherwise
    """
    k = which_keyword(tokens)
    if not k or k == "both":
        return 0
    p, n = count_polarity(tokens)
    if k == "hawk":
        if p > n:  return +1
        if p < n:  return -1
        return 0
    # k == "dove"
    if p > n:      return -1
    if p < n:      return +1
    return 0

def score_document(text: str) -> tuple[float, int, int, int, int]:
    """
    Score a document:
      - keep only sentences with a single-side keyword signal
      - compute mean over sentence scores
    Returns: (doc_score, cnt_hawkish, cnt_dovish, cnt_neutral, matched_sentences)
    """
    sents = simple_sent_tokenize(text)
    sent_scores: List[int] = []
    cnt_hawkish = cnt_dovish = cnt_neutral = 0

    for sent in sents:
        toks = normalize_tokens(sent)
        k = which_keyword(toks)
        if not k or k == "both":
            # Ignore sentences without a clear hawk/dove signal
            continue
        sc = score_sentence(toks)
        sent_scores.append(sc)
        if sc > 0:
            cnt_hawkish += 1
        elif sc < 0:
            cnt_dovish += 1
        else:
            cnt_neutral += 1

    matched = len(sent_scores)
    doc_score = float(mean(sent_scores)) if sent_scores else 0.0
    return doc_score, cnt_hawkish, cnt_dovish, cnt_neutral, matched

def extract_year_from_path(path_obj: Path) -> int:
    """
    Prefer directory name as year; fallback to first 4-digit year in filename.
    Return -1 if not found.
    """
    try:
        return int(path_obj.parent.name)
    except Exception:
        m = re.search(r'(19|20)\\d{2}', path_obj.name)
        return int(m.group(0)) if m else -1

# ================================= Main ======================================

def main():
    # ---- Args ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="Minutes files to score")
    parser.add_argument("--out-dir", default="data/fomc/sentiments", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sentiment_document_scores.csv"

    records: List[Dict[str, object]] = []

    for file_str in args.files:
        path_obj = Path(file_str)
        if not path_obj.exists():
            print(f"[WARN] missing file: {path_obj}")
            continue

        text = path_obj.read_text(encoding="utf-8", errors="ignore")
        doc_score, cnt_h, cnt_d, cnt_0, matched = score_document(text)
        year_val = extract_year_from_path(path_obj)

        records.append({
            "doc_path": str(path_obj),
            "year": year_val,
            "n_sentences": matched,
            "count_hawkish": cnt_h,
            "count_dovish": cnt_d,
            "count_neutral": cnt_0,
            "score_doc": round(doc_score, 6),
        })

        print(f"[OK] {path_obj} -> score={doc_score:.3f}  matched={matched} (+{cnt_h}/-{cnt_d}/0:{cnt_0})")

    # ---- Write CSV ----
    import pandas as pd  # keep import local as in original
    df = pd.DataFrame(records).sort_values(["year", "doc_path"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote supervised index -> {out_csv}")

if __name__ == "__main__":
    main()
