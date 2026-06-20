import os
import joblib
import numpy as np
import torch
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using PyTorch neural network",
    version="1.0.0",
)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
PREDICTIONS      = Counter("fraud_predictions_total", "Total predictions", ["result"])
LATENCY          = Histogram("fraud_prediction_latency_seconds", "Prediction latency")
HIGH_RISK_SCORES = Histogram("fraud_risk_score", "Distribution of fraud scores",
                             buckets=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])


# ── Model loading ──────────────────────────────────────────────────────────────
class FraudNet(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128), torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 1), torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


MODEL  = None
SCALER = None
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))


@app.on_event("startup")
async def load_model():
    global MODEL, SCALER
    model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/fraud-detection/Production")
    scaler_path = os.getenv("SCALER_PATH", "artifacts/scaler.pkl")

    logger.info("Loading model from %s", model_uri)
    MODEL = mlflow.pytorch.load_model(model_uri)
    MODEL.eval()

    logger.info("Loading scaler from %s", scaler_path)
    SCALER = joblib.load(scaler_path)
    logger.info("Model and scaler loaded successfully")


# ── Request / Response schemas ─────────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=29,
        max_items=30,
        description="V1–V28 PCA features + Amount (29 or 30 values)",
        example=[0.1] * 29,
    )
    transaction_id: str = Field(default="", description="Optional transaction ID for logging")


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    model_version: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionFeatures):
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.time()

    features = np.array(transaction.features).reshape(1, -1)
    features_scaled = SCALER.transform(features)

    with torch.no_grad():
        tensor = torch.tensor(features_scaled, dtype=torch.float32)
        prob = MODEL(tensor).item()

    latency = time.time() - start
    LATENCY.observe(latency)
    HIGH_RISK_SCORES.observe(prob)

    is_fraud = prob >= THRESHOLD
    risk_level = (
        "HIGH"   if prob >= 0.7 else
        "MEDIUM" if prob >= 0.4 else
        "LOW"
    )
    PREDICTIONS.labels(result="fraud" if is_fraud else "legitimate").inc()

    logger.info(
        "txn=%s  prob=%.4f  fraud=%s  latency=%.3fs",
        transaction.transaction_id, prob, is_fraud, latency,
    )

    return PredictionResponse(
        transaction_id=transaction.transaction_id,
        fraud_probability=round(prob, 4),
        is_fraud=is_fraud,
        risk_level=risk_level,
        model_version=os.getenv("MODEL_VERSION", "1.0.0"),
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy" if MODEL is not None else "loading",
        "model_loaded": MODEL is not None,
        "threshold": THRESHOLD,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)