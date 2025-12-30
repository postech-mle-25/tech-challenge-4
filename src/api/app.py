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
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from src.data.data_loader import StockDataLoader

logger = logging.getLogger(__name__)

# =====================================================================
# Prometheus Metrics Setup
# =====================================================================
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total de requisições à API',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'Latência das requisições em segundos',
    ['endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
)

PREDICTION_ERRORS = Counter(
    'api_prediction_errors_total',
    'Total de erros em previsões',
    ['error_type']
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Indica se o modelo está carregado (1=sim, 0=não)'
)

SCALER_LOADED = Gauge(
    'scaler_loaded',
    'Indica se o scaler está carregado (1=sim, 0=não)'
)

SYSTEM_CPU_PERCENT = Gauge(
    'system_cpu_percent',
    'Percentual de CPU utilizado'
)

SYSTEM_MEMORY_PERCENT = Gauge(
    'system_memory_percent',
    'Percentual de memória utilizada'
)

# =====================================================================


# =====================================================================
# Middleware para rastreamento de métricas
# =====================================================================
from starlette.middleware.base import BaseHTTPMiddleware


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        method = request.method
        path = request.url.path
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Registrar latência
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=path).observe(duration)
            REQUEST_COUNT.labels(method=method, endpoint=path, status_code=status_code).inc()
            
            # Atualizar métricas de sistema
            SYSTEM_CPU_PERCENT.set(psutil.cpu_percent())
            SYSTEM_MEMORY_PERCENT.set(psutil.virtual_memory().percent)
        
        return response


# =====================================================================


def _prepare_recent(df: pd.DataFrame, cols: list, sequence_len: int) -> pd.DataFrame:
    """Return a DataFrame with at least `sequence_len` rows.

    Strategy:
    - select columns, try dropna()
    - ffill/bfill to recover partial rows
    - fallback to first row (fill NaNs with 0)
    - final fallback: synthetic minimal row
    - pad by repeating the earliest row to reach sequence_len
    """
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
        try:
            if isinstance(recent.index, pd.DatetimeIndex) and len(recent.index) > 0:
                first_date = recent.index[0]
                pad_dates = pd.bdate_range(end=first_date - timedelta(days=1), periods=pad, freq='B')
            else:
                pad_dates = pd.RangeIndex(start=-pad, stop=0)

            pad_vals = np.tile(recent.iloc[0].values, (pad, 1))
            padded_df = pd.DataFrame(pad_vals, columns=recent.columns, index=pad_dates)
            recent = pd.concat([padded_df, recent])
            logger.debug("padded data with %d rows to reach SEQUENCE_LEN=%d", pad, sequence_len)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Dados insuficientes e não foi possível gerar padding: {e}")

    return recent

load_dotenv()
app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.1",
)

# =====================================================================
# Adicionar Middleware de Prometheus
# =====================================================================
# app.add_middleware(PrometheusMiddleware)  # Comentado temporariamente para debug

# =====================================================================

# Globais
model = None
scaler = None
SEQUENCE_LEN = 60
TARGET_INDEX = 3  # 'Close' na ordem [Open, High, Low, Close, Volume]


class PredictionRequest(BaseModel):
    symbol: str = Field(..., example="AAPL")
    days_ahead: int = Field(7, example=7)


class ManualPredictionRequest(BaseModel):
    symbol: str = Field(..., example="AAPL")
    predictions: List[float] = Field(..., example=[170.5, 171.2, 172.0, 173.1, 172.8, 174.0, 175.5])
    days_ahead: Optional[int] = Field(None, example=7)


class PredictionResponse(BaseModel):
    symbol: str
    predictions: list
    dates: list
    confidence_interval: dict


class CustomPredictionRequest(BaseModel):
    # `data` can be a JSON string (pandas.DataFrame.to_json()) or a dict/object
    data: Any
    days: int


@app.on_event("startup")
async def load_model():
    global model, scaler
    # Modelo (.keras preferível; .h5 como fallback)
    try:
        model = tf.keras.models.load_model("models/saved/lstm_model.keras", compile=False)
        MODEL_LOADED.set(1)
    except Exception:
        try:
            model = tf.keras.models.load_model("models/saved/lstm_model.h5", compile=False)
            MODEL_LOADED.set(1)
        except Exception as e:
            logger.error(f"Falha ao carregar modelo: {e}")
            MODEL_LOADED.set(0)

    try:
        with open("models/saved/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        SCALER_LOADED.set(1)
    except Exception as e:
        logger.error(f"Falha ao carregar scaler: {e}")
        SCALER_LOADED.set(0)


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
        # Não expor detalhes técnicos ao cliente; retornar mensagem amigável.
        logger.warning("Error fetching data for %s: %s", request.symbol, e)
        raise HTTPException(status_code=404, detail=f"Ação '{request.symbol}' não encontrada ou sem dados disponíveis.")

    # 2) Seleciona colunas na ordem treinada
    cols = ["Open", "High", "Low", "Close", "Volume"]
    try:
        recent_raw = df[cols].copy()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Dados retornados não possuem colunas esperadas [Open, High, Low, Close, Volume].",
        )

    recent = _prepare_recent(df, cols, SEQUENCE_LEN)

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


@app.post("/predict/manual", response_model=PredictionResponse)
def predict_manual(request: ManualPredictionRequest):
    """Aceita previsões fornecidas pelo usuário e retorna no mesmo formato que /predict.

    Se `days_ahead` for informado, ele deve corresponder ao tamanho de `predictions`.
    Caso não seja informado, o tamanho de `predictions` será usado para gerar as datas.
    """
    preds = [float(x) for x in request.predictions]

    # Validate symbol exists: try a quick yfinance lookup, then fallback to local cache
    loader = StockDataLoader(request.symbol)
    df_check = None
    try:
        df_check = loader._fetch_yfinance()
    except Exception:
        df_check = None

    has_data = False
    if df_check is not None and not getattr(df_check, "empty", True):
        has_data = True
    else:
        try:
            cache_df = loader._load_cache()
            if cache_df is not None and not cache_df.empty:
                has_data = True
        except Exception:
            has_data = False

    if not has_data:
        raise HTTPException(status_code=404, detail=f"Symbol '{request.symbol}' not found.")

    # Determine number of output days. If `days_ahead` is provided and <= len(preds),
    # return that many predictions/dates (slice). If omitted, use full length.
    if request.days_ahead is None:
        days = len(preds)
    else:
        days = int(request.days_ahead)
        if days <= 0:
            raise HTTPException(status_code=400, detail="`days_ahead` must be positive")
        if days > len(preds):
            raise HTTPException(status_code=400, detail="`days_ahead` cannot be greater than length of `predictions`")

    # Trim predictions to requested days
    preds = preds[:days]

    # Gera datas úteis a partir do próximo dia útil
    last_dt = pd.Timestamp.now().date()
    future_dates = pd.bdate_range(start=last_dt + timedelta(days=1), periods=days, freq="B")
    dates = [d.strftime("%Y-%m-%d") for d in future_dates]

    return PredictionResponse(
        symbol=request.symbol,
        predictions=preds,
        dates=dates,
        confidence_interval={
            "lower": (np.array(preds) * 0.95).tolist(),
            "upper": (np.array(preds) * 1.05).tolist(),
        },
    )



@app.post("/predict_custom", response_model=PredictionResponse)
def predict_custom(request: CustomPredictionRequest):
    """Recebe dados serializados (pandas.DataFrame.to_json()) e retorna previsões.

    Espera `data` como string JSON (formato gerado por `DataFrame.to_json()`)
    e `days` (int) informando quantos passos frente prever.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo ou scaler não carregados.")

    # Reconstrói DataFrame enviado pelo cliente. Aceita formatos:
    # - string JSON (pandas.DataFrame.to_json())
    # - dict/object (o dashboard envia um dict quando usa requests.json)
    df = None
    try:
        if isinstance(request.data, str):
            # string with JSON content
            df = pd.read_json(request.data)
        elif isinstance(request.data, dict):
            # dict of columns -> {index: value} (orient='columns') or list of records
            try:
                df = pd.DataFrame.from_dict(request.data)
            except Exception:
                # try records orientation
                df = pd.DataFrame(request.data)
        else:
            # try to coerce other types (e.g., list of records)
            df = pd.DataFrame(request.data)
    except Exception as exc:
        logger.warning("predict_custom: failed to parse incoming data: %s", exc)
        raise HTTPException(status_code=400, detail="Não foi possível interpretar os dados enviados. Envie JSON compatível com pandas.DataFrame (ex: DataFrame.to_json() ou lista de registros).")

    # Se houver coluna Date, usa como índice
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.set_index('Date', inplace=True)
        except Exception:
            pass

    cols = ["Open", "High", "Low", "Close", "Volume"]
    recent = _prepare_recent(df, cols, SEQUENCE_LEN)

    # Escala e cria sequência
    try:
        processed = scaler.transform(recent.values)
    except Exception:
        raise HTTPException(status_code=500, detail="Erro ao escalar os dados enviados.")

    current_seq = processed[-SEQUENCE_LEN:, :].reshape(1, SEQUENCE_LEN, processed.shape[1])

    preds_scaled = []
    for _ in range(int(request.days)):
        pred = model.predict(current_seq, verbose=0)
        preds_scaled.append(float(pred[0, 0]))
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 3] = pred[0, 0]

    # Desnormalizar apenas a coluna alvo
    n_features = getattr(scaler, "n_features_in_", 5)
    tmp = np.zeros((len(preds_scaled), n_features))
    tmp[:, 3] = np.array(preds_scaled).reshape(-1)
    preds_denorm = scaler.inverse_transform(tmp)[:, 3].reshape(-1, 1)

    # Datas de saída (dias úteis)
    last_dt = recent.index[-1] if isinstance(recent.index, pd.DatetimeIndex) else pd.Timestamp.now().date()
    future_dates = pd.bdate_range(start=last_dt + timedelta(days=1), periods=int(request.days), freq="B")

    preds_list = preds_denorm.flatten().tolist()
    return PredictionResponse(
        symbol="CUSTOM",
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


@app.get("/metrics/prometheus", response_class=Response)
def metrics_prometheus():
    """Endpoint para Prometheus scraping.
    
    Retorna métricas no formato text/plain de Prometheus.
    """
    return Response(generate_latest(), media_type="text/plain; charset=utf-8")

