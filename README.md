# 🔍 Fake News Detection using BiLSTM-Attention & BERT Hybrid

A full-stack AI-powered fake news detection system that combines **Deep Learning (BiLSTM with Self-Attention)** and **Transformer-based NLP (DistilBERT)** to classify news articles as real or fake. The project features an end-to-end pipeline — from data preprocessing and model training to a live web interface powered by a FastAPI backend.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Frontend](#frontend)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Misinformation spreads rapidly on digital platforms. This project tackles the problem by training and comparing multiple NLP approaches on benchmark fake news datasets:

| Approach | Description |
|---|---|
| **Classical NLP** | TF-IDF vectorization + POS tagging + Sentiment features |
| **BiLSTM** | Bidirectional LSTM for sequential text modeling |
| **BiLSTM + Attention** | Self-attention layer on top of BiLSTM for explainability |
| **DistilBERT (Transformer)** | Fine-tuned DistilBERT for sequence classification |
| **Hybrid BERT-Linguistic** | DistilBERT embeddings combined with handcrafted linguistic features (TF-IDF, POS, Sentiment) via a meta-classifier |

The best-performing model is served through a **FastAPI** backend and consumed by a modern **glassmorphism-styled web UI**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend (HTML/CSS/JS)             │
│  Glassmorphism UI · Particle Background · Attention Map │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP POST /predict
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend (app.py)               │
│  DistilBERT Tokenizer → Model Inference → Prediction    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Trained Models (models/)                    │
│  fake_news_model/ (DistilBERT)  ·  bert_model/          │
│  hybrid_clf.pkl  ·  tfidf.pkl                           │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
NLP.DL.Prjct/
├── backend/
│   └── app.py                  # FastAPI server with /predict endpoint
├── data/
│   ├── ISOT/
│   │   ├── True.csv            # ISOT Real news (~53 MB)
│   │   └── Fake.csv            # ISOT Fake news (~62 MB)
│   ├── train.tsv               # LIAR dataset – training split
│   ├── valid.tsv               # LIAR dataset – validation split
│   ├── test.tsv                # LIAR dataset – test split
│   └── README                  # LIAR dataset documentation
├── frontend/
│   ├── index.html              # Main web interface
│   ├── style.css               # Dark-themed glassmorphism styles
│   └── script.js               # Client-side logic & particle animation
├── models/
│   ├── fake_news_model/        # Fine-tuned DistilBERT (primary)
│   ├── bert_model/             # Fine-tuned DistilBERT (BERT-only)
│   ├── hybrid_clf.pkl          # Hybrid meta-classifier (sklearn)
│   └── tfidf.pkl               # Fitted TF-IDF vectorizer
├── notebooks/
│   ├── 01_prepocessing.ipynb   # Data loading, cleaning, EDA
│   ├── 02_bilstm_attention.ipynb  # BiLSTM + Attention training
│   └── 03_bert_hybrid.ipynb    # DistilBERT fine-tuning & hybrid model
├── outputs/
│   ├── train_processed.csv     # Preprocessed training data
│   ├── test_processed.csv      # Preprocessed test data
│   └── misclassified_20.csv    # Error analysis samples
├── src/                        # (Reserved for modular source code)
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## Datasets

This project uses two widely-cited fake news benchmarks:

### 1. LIAR Dataset
- **Source:** [ACL 2017 – William Yang Wang](https://aclanthology.org/P17-2067/)
- **Format:** TSV with 14 columns (statement, label, speaker, context, etc.)
- **Labels:** 6-class (pants-fire, false, barely-true, half-true, mostly-true, true) — binarized to `FAKE` / `REAL` during preprocessing
- **Splits:** Train / Valid / Test provided

### 2. ISOT Fake News Dataset
- **Source:** [University of Victoria – ISOT Research Lab](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
- **Format:** CSV with title, text, subject, and date columns
- **Labels:** Binary (True / Fake) based on source file
- **Size:** ~44,000 articles combined

Both datasets are **merged and preprocessed** in `01_prepocessing.ipynb` to create a unified binary classification corpus.

---

## Models

### BiLSTM + Self-Attention
- Embedding → Bidirectional LSTM → Self-Attention → Dense → Sigmoid
- Trained with PyTorch on tokenized & padded sequences
- Attention weights enable token-level explainability

### DistilBERT (Fine-Tuned)
- `distilbert-base-uncased` fine-tuned for binary sequence classification
- Hugging Face `Transformers` library
- Max sequence length: 64 tokens

### Hybrid BERT-Linguistic
- Extracts `[CLS]` embeddings from DistilBERT
- Concatenates with handcrafted features: **TF-IDF scores**, **POS tag ratios**, **Sentiment polarity**
- Meta-classifier (sklearn) on the combined feature vector

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Ananjay07/Fake-News-Detection-using-BiLSTM-and-BERT.git
cd Fake-News-Detection-using-BiLSTM-and-BERT

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for preprocessing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

> **Note:** Trained model weights and datasets are excluded from the repository via `.gitignore` due to file size. You will need to train the models by running the notebooks or download them separately.

---

## Usage

### 1. Run the Notebooks (Training Pipeline)

Open and execute the notebooks in order:

```
notebooks/01_prepocessing.ipynb    → Data cleaning & feature engineering
notebooks/02_bilstm_attention.ipynb → BiLSTM + Attention model training
notebooks/03_bert_hybrid.ipynb      → DistilBERT & Hybrid model training
```

### 2. Start the Backend API

```bash
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The API exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/predict` | POST | Classify news text as REAL or FAKE |

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA confirms water found on Mars surface"}'
```

**Example response:**
```json
{
  "prediction": "REAL",
  "confidence": 94.37
}
```

### 3. Open the Frontend

Open `frontend/index.html` in a browser. Make sure the backend is running on port `8000` for live predictions. If the backend is offline, the UI falls back to a demo mode with simulated results.

---

## Frontend

The web interface features a modern, dark-themed design with:

- ✨ **Particle background animation** (canvas-based)
- 🔮 **Glassmorphism cards** with frosted-glass effects
- 📊 **Confidence gauge** with animated SVG arc
- 🔥 **Attention heatmap** highlighting influential tokens
- 📈 **NLP signal indicators** (Sentiment, POS ratio, TF-IDF, Credibility)
- 📉 **Model comparison dashboard** with horizontal bar charts
- ⚠️ **Error analysis table** for known edge cases

---

## Results

| Model | Accuracy |
|---|---|
| Classical NLP (TF-IDF + POS) | 78% |
| BiLSTM (Deep Learning) | 60% |
| BiLSTM + Self-Attention | 62% |
| DistilBERT (Transformer) | **80%** |
| Hybrid BERT-Linguistic | 79% |

> Metrics are reported on the combined LIAR + ISOT test set. The DistilBERT model achieves the best standalone accuracy, while the Hybrid approach demonstrates the value of fusing semantic and linguistic features.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Deep Learning** | PyTorch, Transformers (Hugging Face) |
| **NLP & Feature Engineering** | NLTK, TextBlob, scikit-learn (TF-IDF) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JavaScript |
| **Data Science** | Pandas, NumPy, Matplotlib, Seaborn, WordCloud |
| **Notebooks** | Jupyter, IPyKernel |

---

## License

This project is developed for academic and research purposes as part of the NLP course at Lovely Professional University (LPU), Semester 6.

The LIAR dataset is provided under the terms described by [William Yang Wang (UCSB)](https://www.cs.ucsb.edu/~william/) and is for **research use only**. The ISOT dataset is provided by the [ISOT Research Lab, University of Victoria](https://onlineacademiccommunity.uvic.ca/isot/).

---

<p align="center">
  Built with ❤️ using PyTorch, Transformers & FastAPI
</p>
