from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.features import build_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
STATIC_DIR = Path(__file__).resolve().parent / "static"

REVIEW_LOW = 0.30
REVIEW_HIGH = 0.70

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
threshold = bundle["threshold"]
metrics = bundle["metrics"]

tfidf = pipeline.named_steps["tfidf"]
clf = pipeline.named_steps["clf"]
feature_names = tfidf.get_feature_names_out()
weights = clf.coef_[0]

app = FastAPI(
    title="Fake Job Posting Detector",
    description="Paste a job posting and get a FAKE/REAL prediction with the words that influenced it.",
    version="1.1.0",
)


class JobPosting(BaseModel):
    description: str = Field(..., min_length=20, max_length=20000)
    title: str = Field("", max_length=300)
    company_profile: str = Field("", max_length=10000)
    requirements: str = Field("", max_length=10000)
    benefits: str = Field("", max_length=10000)


class Reason(BaseModel):
    word: str
    weight: float


class Prediction(BaseModel):
    label: str
    verdict: str
    fake_probability: float
    threshold: float
    review_range: list[float]
    red_flags: list[Reason]
    trust_signals: list[Reason]


def verdict_for(prob):
    if prob >= REVIEW_HIGH:
        return "likely_scam"
    if prob >= REVIEW_LOW:
        return "needs_review"
    return "looks_genuine"


def explain(x, top_k=8):
    contrib = x.toarray()[0] * weights
    order = np.argsort(contrib)

    red_flags = [
        Reason(word=str(feature_names[i]), weight=round(float(contrib[i]), 3))
        for i in order[::-1][:top_k]
        if contrib[i] > 0
    ]
    trust_signals = [
        Reason(word=str(feature_names[i]), weight=round(float(contrib[i]), 3))
        for i in order[:top_k]
        if contrib[i] < 0
    ]
    return red_flags, trust_signals


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model": "TF-IDF + Logistic Regression (C=8, class-weighted)",
        "threshold": round(threshold, 4),
        "review_range": [REVIEW_LOW, REVIEW_HIGH],
        "test_metrics": metrics,
    }


@app.post("/predict", response_model=Prediction)
def predict(posting: JobPosting):
    text = build_text(posting.model_dump())
    if not text:
        raise HTTPException(status_code=422, detail="No usable text after cleaning.")

    x = tfidf.transform([text])
    prob = float(clf.predict_proba(x)[0, 1])
    red_flags, trust_signals = explain(x)

    return Prediction(
        label="FAKE" if prob >= threshold else "REAL",
        verdict=verdict_for(prob),
        fake_probability=round(prob, 4),
        threshold=round(threshold, 4),
        review_range=[REVIEW_LOW, REVIEW_HIGH],
        red_flags=red_flags,
        trust_signals=trust_signals,
    )