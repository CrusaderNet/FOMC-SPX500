#!/usr/bin/env python3
# lexicon_score_minutes.py
"""
Score selected FOMC minutes using a simple lexicon scheme (per paper's Section 4.1 idea).

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
"""

import argparse
from pathlib import Path
import re
from typing import List, Dict, Tuple
from statistics import mean

# ---------------- Lexicons (seeded from paper tables; can be extended) ---------------

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

# ---------------- Helpers ----------------

def simple_sent_tokenize(text: str) -> List[str]:
    # Split on sentence enders while keeping reasonable chunks
    # Also handle that many docs are a single line.
    parts = re.split(r'(?<=[\.\?\!;:])\s+', text.replace('\\n', ' '))
    return [p.strip() for p in parts if p.strip()]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def normalize_tokens(s: str) -> List[str]:
    # lowercase, remove punctuation/numbers; keep simple word tokens
    return [t.lower() for t in WORD_RE.findall(s)]

def count_polarity(tokens: List[str]) -> tuple[int,int]:
    p = sum(1 for t in tokens if t in POSITIVE)
    n = sum(1 for t in tokens if t in NEGATIVE)
    return p, n

def which_keyword(tokens: List[str]) -> str:
    has_hawk = any(t in HAWKISH for t in tokens)
    has_dove = any(t in DOVISH for t in tokens)
    if has_hawk and not has_dove:
        return "hawk"
    if has_dove and not has_hawk:
        return "dove"
    if has_hawk and has_dove:
        # if both, prefer "neutral" for safety (paper doesn't state; we drop it)
        return "both"
    return ""  # none

def score_sentence(tokens: List[str]) -> int:
    k = which_keyword(tokens)
    if not k or k == "both":
        return 0
    p, n = count_polarity(tokens)
    if k == "hawk":
        if p > n: 
            return +1
        if p < n:
            return -1
        return 0
    else:  # dove
        if p > n:
            return -1
        if p < n:
            return +1
        return 0

def score_document(text: str) -> tuple[float, int, int, int, int]:
    sents = simple_sent_tokenize(text)
    scores = []
    cnt_hawk = cnt_dove = cnt_neu = 0
    for s in sents:
        toks = normalize_tokens(s)
        k = which_keyword(toks)
        if not k or k == "both":
            # Ignore sentences without a keyword signal or conflicting ones
            continue
        sc = score_sentence(toks)
        scores.append(sc)
        if sc > 0: cnt_hawk += 1
        elif sc < 0: cnt_dove += 1
        else: cnt_neu += 1
    matched = len(scores)
    doc_score = float(mean(scores)) if scores else 0.0
    return doc_score, cnt_hawk, cnt_dove, cnt_neu, matched

def extract_year_from_path(p: Path) -> int:
    try:
        return int(p.parent.name)
    except Exception:
        # fallback: try to find 4-digit year in filename
        m = re.search(r'(19|20)\\d{2}', p.name)
        return int(m.group(0)) if m else -1

def main():
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="Minutes files to score")
    ap.add_argument("--out-dir", default="data/fomc/sentiments", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sentiment_document_scores.csv"

    rows = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"[WARN] missing file: {p}")
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        doc_score, c_h, c_d, c_z, matched = score_document(txt)
        year = extract_year_from_path(p)
        rows.append({
            "doc_path": str(p),
            "year": year,
            "n_sentences": matched,
            "count_hawkish": c_h,
            "count_dovish": c_d,
            "count_neutral": c_z,
            "score_doc": round(doc_score, 6),
        })
        print(f"[OK] {p} -> score={doc_score:.3f}  matched={matched} (+{c_h}/-{c_d}/0:{c_z})")

    # Write CSV
    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["year","doc_path"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote supervised index -> {out_csv}")

if __name__ == "__main__":
    main()
