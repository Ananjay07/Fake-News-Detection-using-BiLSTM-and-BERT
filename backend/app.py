from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DEVICE
# =========================

device = torch.device("cpu")

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "../models/fake_news_model"

tokenizer = DistilBertTokenizer.from_pretrained(
    MODEL_PATH
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(device)

model.eval()

# =========================
# INPUT SCHEMA
# =========================

class NewsInput(BaseModel):
    text: str

# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():
    return {"message": "Fake News Detection API Running"}

# =========================
# PREDICTION ROUTE
# =========================

@app.post("/predict")
def predict(news: NewsInput):

    inputs = tokenizer(
        news.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()

    label = "REAL" if pred == 1 else "FAKE"

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }