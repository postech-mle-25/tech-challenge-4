# 📈 Stock Price Prediction - LSTM + FastAPI

Sistema de previsão de preços de ações com Deep Learning (LSTM), API REST, monitoramento e deploy em produção.

## 📊 Resultados

| Métrica | Valor |
|---------|-------|
| **MAE** | 5.17 |
| **RMSE** | 5.81 |
| **MAPE** | 2.85% |

## 🎯 Funcionalidades

✅ **Modelo LSTM** - 3 camadas com dropout + batch normalization  
✅ **Coleta de dados** - yfinance com cache local  
✅ **API REST** - FastAPI com Swagger UI interativo  
✅ **Monitoramento** - Prometheus + Grafana (7 métricas)  
✅ **Dashboard** - Streamlit com visualizações  
✅ **Containerizado** - Docker ready para deploy  

## 🚀 Começar

### Instalação
```bash
git clone https://github.com/postech-mle-25/tech-challenge-4
cd tech-challenge-4

# Criar ambiente virtual
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### Executar Localmente

**Terminal 1 - API:**
```bash
uvicorn src.api.app:app --reload --port 8000
```

**Terminal 2 - Dashboard (opcional):**
```bash
streamlit run src/utils/dashboard.py
```

**Terminal 3 - Monitoramento (opcional):**
```bash
docker-compose up -d
```

## 🌐 Acessar

| Componente | URL |
|-----------|-----|
| **Swagger UI (API)** | http://localhost:8000/docs |
| **Dashboard** | http://localhost:8501 |
| **Prometheus** | http://localhost:9090 |
| **Grafana** | http://localhost:3000 |
| **API em Produção** | https://tech-challenge-4-production.up.railway.app/docs |

## 📡 Endpoints da API

- `GET /health` - Status da API e modelo
- `GET /metrics` - Métricas do modelo (JSON)
- `GET /metrics/prometheus` - Métricas em formato Prometheus
- `POST /predict` - Previsão com dados históricos
- `POST /predict/manual` - Previsão com valores manuais

### Exemplo - Fazer uma previsão

```bash
curl -X POST "http://localhost:8000/predict/manual" \
  -H "Content-Type: application/json" \
  -d '{"predictions": [180.5, 181.2, 182.0, 183.1, 184.5]}'
```

## 📦 Tecnologias

- **ML**: TensorFlow/Keras, scikit-learn
- **API**: FastAPI, Uvicorn
- **Dados**: pandas, numpy, yfinance
- **Monitoramento**: Prometheus, Grafana, prometheus-client
- **Visualização**: Streamlit
- **Deploy**: Docker, Railway

## 📂 Estrutura do Projeto

```
tech-challenge-4/
├── src/
│   ├── api/               # API FastAPI
│   ├── data/              # Carregamento e preprocessamento
│   ├── models/            # Arquitetura LSTM
│   └── utils/             # Métricas e dashboard
├── models/saved/          # Modelo treinado + scaler
├── notebooks/             # 4 notebooks: EDA → Treino → Avaliação
├── docker/                # Dockerfile para API
├── docker-compose.yml     # Orquestração (API + Prometheus + Grafana)
├── requirements.txt       # Dependências Python
└── README.md
```

## 🏋️ Treinar o Modelo

```bash
python src/train.py --symbol AAPL --epochs 50
```

Artefatos salvos em `models/saved/`:
- `lstm_model.keras` - Modelo treinado
- `scaler.pkl` - Scaler para normalização
- `metrics.json` - Métricas de avaliação
- `training_history.csv` - Histórico de treinamento

## 🐳 Deploy com Docker

```bash
# Build
docker build -f docker/Dockerfile -t stock-api .

# Run
docker run -p 8000:8000 stock-api
```

## ☁️ Deploy em Produção (Railway)

```bash
# 1. Fazer push para GitHub
git add .
git commit -m "ready for deploy"
git push origin monitoring-prometheus-test

# 2. Conectar no Railway (https://railway.app)
# - Selecionar repositório
# - Railway faz deploy automático
```

## 📊 Métricas de Monitoramento

| Métrica | Descrição |
|---------|-----------|
| `api_requests_total` | Total de requisições |
| `api_request_latency_seconds` | Latência por endpoint |
| `api_prediction_errors_total` | Erros em previsões |
| `model_loaded` | Status do modelo |
| `system_cpu_percent` | CPU utilizado |
| `system_memory_percent` | Memória utilizada |

## 📚 Documentação Adicional

- [MONITORING.md](MONITORING.md) - Setup completo do Prometheus + Grafana
- [notebooks/](notebooks/) - 4 notebooks documentados (EDA, preprocessing, treinamento, avaliação)

## 👤 Autor

Postech MLE - Tech Challenge 4

## 📄 Licença

MIT
