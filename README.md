# FOMC Meeting Minutes Sentiment Analysis & S&P 500 Prediction (2000–2025)

This repository contains an end-to-end pipeline that:

1. Scrapes **FOMC meeting minutes**
2. Extracts and cleans the text into a consistent corpus
3. Builds a **small supervised sentiment set** (manual scoring)
4. Trains an ANN to **score sentiment across the entire corpus**
5. Aligns minutes with **S&P 500** prices
6. Predicts **next-day S&P 500 close** using **walk-forward (expanding-window) evaluation**

The overall goal is to see how **monetary-policy language** correlates with market behavior, and how far sentiment signals alone can go in predicting next-day movement.

---

## Pipeline Overview

### 1) Scrape minutes (modern + historical)
- **Scripts:** `scrape_fomc_minutes.py`, `scrape_fomc_historical_minutes.py`
- **Output (download target):** `minutes_html/<year>/...`
- **Manifests:**
  - `data/fomc/manifests/minutes_manifest.csv`
  - `data/fomc/manifests/minutes_historical_manifest.csv`

### 2) Extract text (HTML/PDF → .txt)
- **Script:** `extract_text_for_years.py`
- **Output:** `minutes_text/<year>/...`
- Also produces:
  - `minutes_text/minutes_corpus_index.csv`
  - `minutes_text/minutes_corpus_manifest.jsonl`

### 3) Clean minutes (optional, but recommended)
- **Script:** `clean_minutes_with_ollama.py`
- **Output:** `minutes_text_clean/<year>/...`
- Uses a local LLM (via **Ollama**) to normalize formatting, spacing, and OCR artifacts.

### 4) Create a supervised sentiment set (manual scoring + lexicon scoring)
- **Script:** `lexicon_score_minutes.py`
- **Input:** selected files from `minutes_text_clean/<year>/...`
- **Output:** `data/fomc/sentiments/sentiment_document_scores.csv`

### 5) Train sentiment ANN + score the entire corpus
- **Script:** `train_score_lexicon_ann.py`
- **Output:**
  - `model_out_ann/ann_model.json`
  - `model_out_ann/ann_weights.csv`
  - `model_out_ann/train_eval_predictions.csv`
  - `sentiment_scores.csv` (final per-document sentiment scores in chronological order)

### 6) Align meeting dates to S&P 500 close prices
- **Scripts:** `extract_sp500_dates.py`, `fetch_sp500_prices.py`
- **Outputs:**
  - `data/sp500/dates/sp500_dates_from_manifest_full.csv`
  - `data/sp500/dates/sp500_dates_from_manifest_unique.csv`
  - `data/sp500/dates/sp500_dates_from_manifest_unique.txt`
  - `data/sp500/prices/sp500_prices.csv`
  - `data/sp500/prices/sp500_prices_missing.csv`

### 7) Predict next-day S&P 500 close (ANN + walk-forward validation)
- **Script:** `predict_sp500_ANN.py`
- **Inputs:**
  - `sentiment_scores.csv`
  - `data/sp500/prices/sp500_prices.csv`
- **Outputs:**
  - `artifacts_spx_from_sent_delta/fit_summary.json`
  - `artifacts_spx_from_sent_delta/model.json`
  - `artifacts_spx_from_sent_delta/predictions.csv`

This prediction model uses **only**:
- sentiment level (`x`)
- sentiment delta (`x_delta`)
- simple time normalization features
- **walk-forward / expanding-window** evaluation to preserve chronology

To improve stability across price regimes, the network predicts a ratio target:

\[
\text{target\_ratio}=\frac{\text{next\_close}}{\text{close}}
\qquad\Rightarrow\qquad
\widehat{\text{next\_close}}=\text{close}\cdot \widehat{\text{target\_ratio}}
\]

---

## Repository Structure (current)

```text
FOMC-SPX500/
├── README.md
├── .gitignore
├── config_paths.py
│
├── scrape_fomc_minutes.py
├── scrape_fomc_historical_minutes.py
├── extract_text_for_years.py
├── clean_minutes_with_ollama.py
├── lexicon_score_minutes.py
├── train_score_lexicon_ann.py
├── extract_sp500_dates.py
├── fetch_sp500_prices.py
├── predict_sp500_ANN.py
│
├── minutes_html/                        # (download target; may be empty in repo snapshots)
├── minutes_text/
│   ├── minutes_corpus_index.csv
│   ├── minutes_corpus_manifest.jsonl
│   └── <year>/*.txt
├── minutes_text_clean/
│   └── <year>/*_singleline.txt
│
├── data/
│   ├── fomc/
│   │   ├── manifests/
│   │   │   ├── minutes_manifest.csv
│   │   │   └── minutes_historical_manifest.csv
│   │   ├── sentiments/
│   │   │   └── sentiment_document_scores.csv
│   │   ├── features/
│   │   └── artifacts/
│   ├── sp500/
│   │   ├── dates/
│   │   │   ├── sp500_dates_from_manifest_full.csv
│   │   │   ├── sp500_dates_from_manifest_unique.csv
│   │   │   └── sp500_dates_from_manifest_unique.txt
│   │   └── prices/
│   │       ├── sp500_prices.csv
│   │       └── sp500_prices_missing.csv
│   └── lexicon/                         # (reserved)
│
├── model_out_ann/
│   ├── ann_model.json
│   ├── ann_weights.csv
│   └── train_eval_predictions.csv
│
├── artifacts_spx_from_sent_delta/
│   ├── fit_summary.json
│   ├── model.json
│   └── predictions.csv
│
├── models/
│   ├── metrics/
│   └── predictions/
└── reports/
```

---

## Requirements

### Python
- Python 3.10+ recommended

### Packages
```bash
pip install requests beautifulsoup4 pandas numpy matplotlib tqdm pdfminer.six spacy
python -m spacy download en_core_web_sm
```

### Optional: Ollama (for the cleaning step)
The cleaning script calls a local Ollama endpoint. Install Ollama and pull a model (default in code is `llama3.1:8b-instruct-q4_K_M`).

---

## Running the Project

All commands assume execution from the repo root.

### 1) Scrape minutes
```bash
python scrape_fomc_minutes.py
python scrape_fomc_historical_minutes.py --year-min 2000 --year-max 2019
```

### 2) Extract text for the years of interest
Example: 2000–2025
```bash
python extract_text_for_years.py 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
```

### 3) Clean minutes (optional)
```bash
python clean_minutes_with_ollama.py --in-root minutes_text --out-root minutes_text_clean --years "2000-2025"
```

### 4) Create supervised sentiment labels (manual-scored subset)
`lexicon_score_minutes.py` requires explicit file paths to score:
```bash
python lexicon_score_minutes.py --out-dir data/fomc/sentiments --files \
  minutes_text_clean/2000/fomcminutes-20000202_singleline.txt \
  minutes_text_clean/2008/fomcminutes-20081216_singleline.txt
```

### 5) Train sentiment ANN + score the full corpus
```bash
python train_score_lexicon_ann.py
```

This generates `sentiment_scores.csv`.

### 6) Extract meeting dates & fetch S&P 500 prices
```bash
python extract_sp500_dates.py
python fetch_sp500_prices.py --source stooq --align next \
  --dates_csv data/sp500/dates/sp500_dates_from_manifest_unique.csv
```

### 7) Predict next-day close (walk-forward ANN)
```bash
python predict_sp500_ANN.py \
  --sent sentiment_scores.csv \
  --prices data/sp500/prices/sp500_prices.csv \
  --start-year 2000 --end-year 2025 \
  --hidden 32 --epochs 4000 --lr 0.01 --l2 1e-4
```

Optional walk-forward controls:
- `--min-train` (default: 80)
- `--val-window` (default: 12)
- `--no-plot`

---

## Key Outputs

### Sentiment scoring
- `sentiment_scores.csv` — chronological document sentiment scores
- `model_out_ann/ann_weights.csv` — token-level weights (interpretability)

### Market prediction
- `artifacts_spx_from_sent_delta/predictions.csv` — true vs predicted next close (by split)
- `artifacts_spx_from_sent_delta/fit_summary.json` — walk-forward split metrics
- `artifacts_spx_from_sent_delta/model.json` — trained model parameters

---

## Notes / Observations

- Walk-forward evaluation is used because the data is **chronological** and look-ahead leakage must be avoided.
- The prediction model tends to **underpredict** the true close level, but has been more informative on **direction** than on exact price.

---

## References

1. “Meeting calendars and information,” The Federal Reserve, FOMC calendars. (accessed Nov. 10, 2025).
2. R. C. Tadle, “FOMC minutes sentiments and their impact on financial markets,” *Journal of Economics and Business*, vol. 118, p. 106021, Jan. 2022. doi:10.1016/j.jeconbus.2021.106021
3. Stooq, “^SPX - S&P 500 - U.S.,” historical data. (accessed Nov. 10, 2025).

---

## Author

Seth Tourish  
CS 4395 — Senior Project (Fall 2025)  
University of Houston — Downtown
