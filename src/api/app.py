
# src/api/app.py
import logging
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import time
import psutil

from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any

from src.data.data_loader import StockDataLoader

# ---------------------------------------------------------------------
# Logging & Monitoring setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("monitoring")

REQUEST_COUNT = 0
ERROR_COUNT = 0
TOTAL_LATENCY = 0.0

def record_metrics(latency: float, error: bool = False):
    global REQUEST_COUNT, ERROR_COUNT, TOTAL_LATENCY
    REQUEST_COUNT += 1
    TOTAL_LATENCY += latency
    if error:
        ERROR_COUNT += 1

def system_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }

# ---------------------------------------------------------------------

def _prepare_recent(df: pd.DataFrame, cols: list, sequence_len: int) -> pd.DataFrame:
    recent_raw = df[cols].copy()
    recent = recent_raw.dropna().copy()
    if recent.empty:
        recent = recent_raw.fillna(method='ffill').fillna(method='bfill')
    if recent.empty and len(recent_raw) > 0:
        row0 = recent_raw.iloc[0].fillna(0)
        recent = pd.DataFrame([row0.values], columns=recent_raw.columns, index=[recent_raw.index[0]])
    if recent.empty:
        idx = pd.bdate_range(end=datetime.utcnow().date(), periods=1)
        synthetic = {c: 0.0 for c in cols}
        synthetic['Close'] = 1.0
        recent = pd.DataFrame([synthetic], index=idx)
    if len(recent) < sequence_len:
        pad = sequence_len - len(recent)
        pad_vals = np.tile(recent.iloc[0].values, (pad, 1))
        padded_df = pd.DataFrame(pad_vals, columns=recent.columns)
        recent = pd.concat([padded_df, recent])
    return recent

load_dotenv()
app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.2",
)

model = None
scaler = None
SEQUENCE_LEN = 60
TARGET_INDEX = 3

class PredictionRequest(BaseModel):
    symbol: str
    days_ahead: int = 7

class PredictionResponse(BaseModel):
    symbol: str
    predictions: list
    dates: list
    confidence_interval: dict

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model = tf.keras.models.load_model("models/saved/lstm_model.keras", compile=False)
    except Exception:
        model = tf.keras.models.load_model("models/saved/lstm_model.h5", compile=False)
    with open("models/saved/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Model and scaler loaded successfully")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start = time.time()
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Modelo ou scaler não carregados.")

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=720)
        loader = StockDataLoader(request.symbol, str(start_date), str(end_date))
        df = loader.fetch_data()

        cols = ["Open", "High", "Low", "Close", "Volume"]
        recent = _prepare_recent(df, cols, SEQUENCE_LEN)
        processed = scaler.transform(recent.values)
        current_seq = processed[-SEQUENCE_LEN:, :].reshape(1, SEQUENCE_LEN, processed.shape[1])

        preds_scaled = []
        for _ in range(request.days_ahead):
            pred = model.predict(current_seq, verbose=0)
            preds_scaled.append(float(pred[0, 0]))
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 3] = pred[0, 0]

        n_features = getattr(scaler, "n_features_in_", 5)
        tmp = np.zeros((len(preds_scaled), n_features))
        tmp[:, 3] = np.array(preds_scaled)
        preds_denorm = scaler.inverse_transform(tmp)[:, 3]

        last_dt = recent.index[-1]
        future_dates = pd.bdate_range(start=last_dt + timedelta(days=1), periods=request.days_ahead)

        latency = time.time() - start
        record_metrics(latency)

        logger.info(
            "prediction_ok | latency_ms=%.2f | system=%s",
            latency * 1000,
            system_metrics(),
        )

        return PredictionResponse(
            symbol=request.symbol,
            predictions=preds_denorm.tolist(),
            dates=[d.strftime("%Y-%m-%d") for d in future_dates],
            confidence_interval={
                "lower": (preds_denorm * 0.95).tolist(),
                "upper": (preds_denorm * 1.05).tolist(),
            },
        )

    except Exception as e:
        latency = time.time() - start
        record_metrics(latency, error=True)
        logger.error("prediction_error | latency_ms=%.2f | error=%s", latency * 1000, e)
        raise

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/metrics")
def metrics():
    avg_latency = (TOTAL_LATENCY / REQUEST_COUNT * 1000) if REQUEST_COUNT > 0 else 0.0
    return {
        "requests": REQUEST_COUNT,
        "errors": ERROR_COUNT,
        "avg_latency_ms": round(avg_latency, 2),
        "system": system_metrics(),
        "timestamp": datetime.now().isoformat(),
    }
