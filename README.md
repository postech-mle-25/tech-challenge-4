# 📈 Stock Price Prediction (LSTM + FastAPI)

Sistema completo para previsão de preços de ações usando redes LSTM, com coleta de dados robusta (múltiplos provedores), API REST e containerização.

## ✨ Principais Recursos
- LSTM para séries temporais (Close)
- Coleta resiliente com múltiplos provedores
  - yfinance (opcional)
  - Stooq (CSV público)
  - Alpha Vantage (API key)
  - BRAPI (B3) – token opcional
- API REST (FastAPI) com `/predict`, `/health`, `/metrics`
- Docker/Docker Compose
- Métricas (MAE, RMSE, MAPE) e exemplo de dashboard
- Cache/aquecimento de dados e execução “sem sintético” ou “destravado”

## 🧱 Arquitetura (alto nível)
```

Client -> FastAPI (/predict) -> Modelo LSTM (H5) + Scaler (pkl)
|
-> Loader de dados (yfinance/stooq/alpha/brapi)

````

## 🛠️ Tecnologias
- Python 3.10+
- TensorFlow / Keras
- FastAPI + Uvicorn
- NumPy / Pandas / scikit-learn
- Requests
- Docker / Docker Compose

## 📦 Instalação (Local)

### 1) Clonar e instalar
```bash
git clone https://github.com/postech-mle-25/tech-challenge-4
cd tech-challenge-4
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
````

### 2) Variáveis de ambiente (recomendado)

[Alpha Vantage](https://www.alphavantage.co/support/#api-key)

[BRAPI](https://brapi.dev/)

```bash
# Desligar yfinance se estiver instável na sua rede
export DISABLE_YFINANCE=1

# Alpha Vantage (opcional; melhora cobertura internacional)
export ALPHAVANTAGE_API_KEY="4IZBG8P0THL3QJ6S"

# BRAPI (opcional; melhora cobertura B3)
export BRAPI_TOKEN="5tpGtjwENCfNDBagKjBm6k"

# Suprimir logs de TF, rodar sempre em CPU
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
```

## 🧊 Aquecer Cache (dados reais)

Aquece e valida o pipeline de dados por ~8 anos:

```bash
python -m src.tools.warmup_cache --years 8
```

**Dica (B3):** tente `ITUB4.SA`, `PETR4.SA`, `VALE3.SA`.
Se falhar, use **ADRs**: `ITUB`, `PBR`, `VALE`.

## 🏋️ Treinamento

### 1) Treino com **dados reais apenas** (sem sintético)

```bash
python -m src.train --symbol AAPL --epochs 10 --no_synthetic
```

### 2) Treino “destravado” (pode cair em stooq/cache/sintético se necessário)

```bash
python -m src.train --symbol AAPL --epochs 10
```

Durante o treino, as métricas são exibidas e os artefatos são salvos em:

```
models/saved/lstm_model.h5
models/saved/scaler.pkl
```

## 🚀 API

### Rodar local

```bash
uvicorn src.api.app:app --reload --port 8000
# ou, para forçar previsões só com dados reais:
ALLOW_SYNTHETIC=false uvicorn src.api.app:app --reload --port 8000
```

Acesse a documentação: `http://localhost:8000/docs`

### Exemplo de request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","days_ahead":7}'
```

### Endpoints

* `GET /health` – status da API (e do modelo)
* `GET /metrics` – métricas do modelo (dummy/treino)
* `POST /predict` – previsão multi-step com intervalo “ingênuo” (±5%)

## 🐳 Docker

### Subir com Docker Compose

```bash
docker-compose up -d
```

> Ajuste o `docker-compose.yml` para definir as variáveis de ambiente (ALPHAVANTAGE_API_KEY/BRAPI_TOKEN/DISABLE_YFINANCE).

## 📊 Métricas (exemplo)

Após um treino de referência:

* MAE: 2.45
* RMSE: 3.12
* MAPE: 4.8%

> Esses valores variam conforme período/símbolo/seed. Use o `/metrics` e/ou gere gráficos de evolução (veja `src/utils/metrics.py`).

## 🧪 Estrutura do Projeto

```
tech-challenge-4/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved/
├── notebooks/                # (recomendado) EDA/treino/avaliação
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py
│   │   └── trainer.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── predictor.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── visualizer.py
├── tests/
├── docker/
│   └── Dockerfile
├── requirements.txt
├── docker-compose.yml
└── README.md


## 🧾 Entregáveis

* Código + README (este documento)
* Dockerfile / Docker Compose funcionais
* Vídeo (5–10 min) demonstrando dados → treino → API → previsões
* (Opcional) Link de deploy na nuvem
