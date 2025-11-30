# src/api/app.py
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.data_loader import StockDataLoader

load_dotenv()
app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.1",
)

# Globais
model = None
scaler = None
SEQUENCE_LEN = 60
TARGET_INDEX = 3  # 'Close' na ordem [Open, High, Low, Close, Volume]


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
    # Modelo (.keras preferível; .h5 como fallback)
    try:
        model = tf.keras.models.load_model("models/saved/lstm_model.keras", compile=False)
    except Exception:
        model = tf.keras.models.load_model("models/saved/lstm_model.h5", compile=False)

    with open("models/saved/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)


@app.get("/")
def root():
    return {
        "message": "Stock Prediction API",
        "endpoints": {
            "/predict": "POST - Fazer previsão",
            "/health": "GET - Status da API",
            "/metrics": "GET - Métricas do modelo",
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo ou scaler não carregados.")

    # 1) Buscar bastante histórico para garantir janela (≈ 720 dias corridos ~ 500 B-days)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=720)

    try:
        loader = StockDataLoader(request.symbol, str(start_date), str(end_date))
        df = loader.fetch_data()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2) Seleciona colunas na ordem treinada
    cols = ["Open", "High", "Low", "Close", "Volume"]
    try:
        recent = df[cols].dropna().copy()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Dados retornados não possuem colunas esperadas [Open, High, Low, Close, Volume].",
        )

    if len(recent) < SEQUENCE_LEN:
        raise HTTPException(status_code=400, detail=f"Dados insuficientes para janela de {SEQUENCE_LEN} passos.")

    # 3) Escala e cria sequência
    processed = scaler.transform(recent.values)  # (N, 5)
    current_seq = processed[-SEQUENCE_LEN:, :].reshape(1, SEQUENCE_LEN, processed.shape[1])

    # 4) Previsões iterativas
    preds_scaled = []
    for _ in range(request.days_ahead):
        pred = model.predict(current_seq, verbose=0)  # (1,1)
        preds_scaled.append(float(pred[0, 0]))
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 3] = pred[0, 0]  # TARGET_INDEX = 3

    # 5) Desnormalizar apenas a coluna alvo
    n_features = getattr(scaler, "n_features_in_", 5)
    tmp = np.zeros((len(preds_scaled), n_features))
    tmp[:, 3] = np.array(preds_scaled).reshape(-1)
    preds_denorm = scaler.inverse_transform(tmp)[:, 3].reshape(-1, 1)

    # 6) Datas de saída (dias úteis)
    last_dt = recent.index[-1]
    future_dates = pd.bdate_range(start=last_dt + timedelta(days=1), periods=request.days_ahead, freq="B")

    preds_list = preds_denorm.flatten().tolist()
    return PredictionResponse(
        symbol=request.symbol,
        predictions=preds_list,
        dates=[d.strftime("%Y-%m-%d") for d in future_dates],
        confidence_interval={
            "lower": (np.array(preds_list) * 0.95).tolist(),
            "upper": (np.array(preds_list) * 1.05).tolist(),
        },
    )


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
    # Tenta carregar métricas geradas pelo treinamento: models/saved/metrics.json
    metrics_path = "models/saved/metrics.json"
    try:
        import json, os
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)

            training_metrics = {
                "mae": data.get("mae"),
                "rmse": data.get("rmse"),
                "mape": data.get("mape"),
            }

            # usa a hora de modificação do arquivo como referência de atualização
            mtime = os.path.getmtime(metrics_path)
            last_updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%S")

            return {
                "model_version": data.get("model_version") or getattr(app, "version", "unknown"),
                "training_metrics": training_metrics,
                "last_updated": last_updated,
                "raw": data,
            }

        # Fallback: alguns deploys não incluem artefatos gerados em runtime (models/saved/)
        # Tenta carregar um relatório de avaliação já comitado (`evaluation_report.json`) e extrair métricas
        eval_path = "models/saved/evaluation_report.json"
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                eval_data = json.load(f)

            metrics_block = eval_data.get("metrics", {})
            training_metrics = {
                "mae": metrics_block.get("mae"),
                "rmse": metrics_block.get("rmse"),
                "mape": metrics_block.get("mape"),
            }

            mtime = os.path.getmtime(eval_path)
            last_updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%S")

            return {
                "model_version": getattr(app, "version", "unknown"),
                "training_metrics": training_metrics,
                "last_updated": last_updated,
                "raw": eval_data,
                "detail": "Usando fallback evaluation_report.json (artifact comitado).",
            }

        # Nenhum arquivo encontrado
        return {
            "model_version": getattr(app, "version", "unknown"),
            "training_metrics": None,
            "last_updated": None,
            "detail": "Métricas não encontradas em models/saved/metrics.json nem em models/saved/evaluation_report.json",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler métricas: {e}")
