# Stock Price Prediction with LSTM

## 📋 Descrição
Sistema de previsão de preços de ações utilizando redes neurais LSTM, 
com API REST para consumo das previsões.

## 🚀 Features
- Previsão de preços para múltiplas ações
- API REST com documentação automática
- Containerização com Docker
- Monitoramento e logging
- Dashboard de visualização

## 🛠️ Tecnologias
- Python 3.9
- TensorFlow/Keras
- FastAPI
- Docker
- yfinance

## 📦 Instalação

### Local
\`\`\`bash
git clone https://github.com/postech-mle-25/tech-challenge-4
pip install -r requirements.txt

export DISABLE_YFINANCE=1
export ALPHAVANTAGE_API_KEY="4IZBG8P0THL3QJ6S"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
\`\`\`

### Criar virtualenv python
\`\`\`bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
\`\`\`bash

### Docker
\`\`\`bash
docker-compose up -d
\`\`\`

## 🔧 Uso

### Treinamento

- aquecer cache real (sem sintético)

\`\`\`bash
python -m src.tools.warmup_cache --years 8
\`\`\`


- treino com dados reais ou falha (bom pra validação)

\`\`\`bash
python -m src.train --symbol AAPL --epochs 10 --no_synthetic
\`\`\`

- treino destravado (pode cair pra stooq/cache/sintético)

\`\`\`bash
python -m src.train --symbol AAPL --epochs 10
\`\`\`

- API sem sintético

\`\`\`bash
ALLOW_SYNTHETIC=false uvicorn src.api.app:app --reload --port 8000
\`\`\`

\`\`\`bash
python -m src.train --symbol AAPL --epochs 100
\`\`\`

### API
\`\`\`bash
uvicorn src.api.app:app --reload
\`\`\`

Acesse: http://localhost:8000/docs

### Exemplo de Request
\`\`\`bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 7}'
\`\`\`

## 📊 Resultados
- MAE: 2.45
- RMSE: 3.12
- MAPE: 4.8%

## 📁 Estrutura do Projeto
stock-prediction-lstm/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved/
├── notebooks/
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
├── README.md
└── .gitignore