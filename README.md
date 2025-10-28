# **FOMC–S&P 500 Sentiment Correlation Pipeline**

This project implements a fully automated NLP and data analysis pipeline that quantifies the relationship between **Federal Open Market Committee (FOMC)** meeting sentiment and subsequent **S&P 500** market performance. It integrates data scraping, text preprocessing, lexicon-based sentiment scoring, and statistical modeling to explore how monetary policy language influences market trends.

---

## 🧭 **Overview**

FOMC meeting minutes often signal changes in monetary policy, which in turn drive stock market behavior.  
This pipeline measures that effect through a repeatable, data-driven process that:

1. **Scrapes** FOMC meeting minutes from the Federal Reserve website.  
2. **Cleans and normalizes** the text into a structured corpus.  
3. **Calculates sentiment scores** using a custom economic lexicon.  
4. **Aligns** meeting sentiments with historical S&P 500 prices.  
5. **Trains a regression model** to predict post-meeting market movements.

All scripts are designed to be re-runnable as new FOMC meetings occur, automatically incorporating new data and updating predictions.

---

## ⚙️ **Pipeline Architecture**

### **1. Scraping**
- **Scripts:** `scrape_fomc_minutes.py`, `scrape_fomc_historical_minutes.py`  
- **Output:** `minutes_html/<year>/...`
- Downloads and indexes all FOMC minutes (modern and legacy formats).
- Generates manifests (`minutes_manifest.csv`, `minutes_historical_manifest.csv`) detailing URLs, file paths, and metadata.

### **2. Text Extraction**
- **Script:** `extract_text_for_years.py`  
- **Output:** `minutes_text/<year>/...`
- Converts HTML/PDF minutes to plain text for NLP processing, preserving per-year folder organization.

### **3. Cleaning**
- **Script:** `clean_minutes_with_ollama.py`  
- **Output:** `minutes_text_clean/<year>/...`
- Uses a local LLM via **Ollama** to refine formatting, spacing, and remove OCR artifacts.

### **4. Sentiment Analysis** -- UPDATE TO REFLECT CHANGES
- **Script:** `lexicon_score_minutes.py`, `train_score_lexicon_ann.py`
- **Inputs:**  
  - `minutes_text_clean/<year>/...`   
- **Output:** `data/fomc/sentiments/sentiment_scores.csv`  
- Uses **spaCy** for tokenization and lemmatization.  
  Applies a macroeconomic lexicon to compute an aggregate sentiment score for each document.

### **5. Market Data Integration**
- **Scripts:** `extract_sp500_dates.py`, `fetch_sp500_prices.py`  
- **Outputs:**  
  - `data/sp500/dates/sp500_dates_from_manifest_*.csv`  
  - `data/sp500/prices/sp500_prices.csv`  
- Extracts meeting dates from manifests and retrieves corresponding or nearest S&P 500 close prices from **Stooq** (and optionally **FRED** if configured).

### **6. Modeling and Prediction** -- UPDATE TO REFLECT CHANGES
- **Script:** `predict_sp500.py`  
- **Inputs:**  
  - `data/fomc/sentiments/sentiment_scores.csv`  
  - `data/sp500/prices/sp500_prices.csv`  
- **Outputs:**  
  - `models/model.pkl`  
  - `models/metrics/model_metrics.txt`  
  - `reports/predictions/next_meeting_prediction.txt`  
- Trains a simple **Linear Regression model** to predict the next S&P 500 close based on cumulative FOMC sentiment.

---

## 🧩 **Repository Structure**

```
FOMC-SPX500/
├── config_paths.py
├── scrape_fomc_minutes.py
├── scrape_fomc_historical_minutes.py
├── extract_text_for_years.py
├── clean_minutes_with_ollama.py
├── analyze_sentiments.py
├── extract_sp500_dates.py
├── fetch_sp500_prices.py
├── predict_sp500.py
│
├── data/
│   ├── fomc/
│   │   ├── manifests/
│   │   ├── features/
│   │   ├── sentiments/
│   │   └── artifacts/
│   ├── sp500/
│   │   ├── dates/
│   │   └── prices/
│   └── lexicon/
│
├── models/
│   ├── metrics/
│   └── predictions/
│
├── reports/
│   └── predictions/
│
├── minutes_html/
├── minutes_text/
└── minutes_text_clean/
```

---

## 🚀 **Usage**

### **Step 1 — Scrape FOMC minutes**
```bash
python scrape_fomc_minutes.py
python scrape_fomc_historical_minutes.py
```

### **Step 2 — Extract text**
```bash
python extract_text_for_years.py
```

### **Step 3 — Clean with Ollama**
```bash
python clean_minutes_with_ollama.py
```

### **Step 4 — Analyze sentiment**
```bash
python analyze_sentiments.py
```

### **Step 5 — Retrieve S&P 500 data**
```bash
python extract_sp500_dates.py
python fetch_sp500_prices.py
```

### **Step 6 — Train and predict**
```bash
python predict_sp500.py
```

All output paths are automatically organized using `config_paths.py`.

---

## 📊 **Technical Summary**

| Component | Technology |
|------------|-------------|
| Web scraping | `requests`, `BeautifulSoup4` |
| Text extraction | HTML parsing, PDF conversion |
| NLP processing | `spaCy` (lemmatization, tokenization) |
| Sentiment analysis | Lexicon-based scoring (custom CSV) |
| Data retrieval | Stooq & FRED APIs |
| Modeling | `scikit-learn` LinearRegression |
| Storage format | CSV + JSON |
| Reproducibility | Script-driven, modular pipeline |

---

## 🎯 **Project Objectives**

- Quantify the link between FOMC tone and S&P 500 performance.  
- Automate the full pipeline from raw text to statistical modeling.  
- Provide an extendable framework for future economic sentiment studies.  
- Ensure reproducibility and transparency through open-source design.

---

## 🔮 **Planned Enhancements**

- Integrate transformer-based sentiment models (FinBERT, Llama 3).  
- Add visualization dashboard (e.g., Streamlit) for meeting sentiment trends.  
- Expand to include other indices or macroeconomic variables.  
- Compare lexicon-based vs ML-driven sentiment correlation.

---

## 🧾 **Author**
**Seth T.**  
Full-Stack & Data Science Developer  
📧 Contact: [GitHub Issues or project email if applicable]
