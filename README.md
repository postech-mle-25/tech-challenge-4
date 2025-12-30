# 📈 Stock Price Prediction (LSTM + FastAPI)

Sistema completo para previsão de preços de ações usando redes LSTM, com coleta de dados robusta (múltiplos provedores), API REST e containerização.

## Principais Recursos
- LSTM para séries temporais (Close)
- Coleta resiliente com múltiplos provedores
  - yfinance (opcional)
  - Stooq (CSV público)
  - BRAPI (B3) – token opcional
- API REST (FastAPI) com `/predict`, `/health`, `/metrics`
- Docker/Docker Compose
- Métricas (MAE, RMSE, MAPE) e exemplo de dashboard
- Cache/aquecimento de dados e execução “sem sintético” ou “destravado”

## Arquitetura (alto nível)
```
Client -> FastAPI (/predict) -> Modelo LSTM (keras/h5) + Scaler (pkl)
                                   |
                                   V
                             Loader de dados 
                      (yfinance/stooq/brapi)
```

## Tecnologias
- Python 3.10+
- TensorFlow / Keras
- FastAPI + Uvicorn
- NumPy / Pandas / scikit-learn
- Requests
- Docker / Docker Compose

## Instalação (Local)

### 1) Clonar e instalar
```bash
git clone https://github.com/postech-mle-25/tech-challenge-4
cd tech-challenge-4
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows (Git Bash / bash.exe):
# source venv/Scripts/activate
# PowerShell: .\venv\Scripts\Activate.ps1
# CMD: venv\Scripts\activate.bat
pip install -r requirements.txt
````

### 2) Variáveis de ambiente (recomendado)

[BRAPI](https://brapi.dev/)

Crie suas chaves de api e adicione-as ao arquivo .env, seguindo o formato:

```
DISABLE_YFINANCE=0
BRAPI_TOKEN="SUA_CHAVE"
TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=""
```




## Aquecer Cache (dados reais)

Aquece e valida o pipeline de dados por ~8 anos:

```bash
python -m src.tools.warmup_cache --years 8
```

**Dica (B3):** tente `ITUB4.SA`, `PETR4.SA`, `VALE3.SA`.
Se falhar, use **ADRs**: `ITUB`, `PBR`, `VALE`.

## Treinamento

### 1) Treino com **dados reais**

```bash
python -m src.train --symbol AAPL --epochs 10
```

```

Durante o treino, as métricas são exibidas e os artefatos são salvos em:

```
models/saved/lstm_model.keras    # Modelo treinado (ou .h5 para compatibilidade)
models/saved/scaler.pkl          # Scaler para normalização
models/saved/metrics.json        # Métricas de treinamento (MAE, RMSE, MAPE, etc.)
models/saved/training_history.csv # Histórico completo do treinamento
```

## API

### Rodar local

```bash
uvicorn src.api.app:app --reload --port 8000
```

Acesse a documentação: `http://localhost:8000/docs`

No Swagger (OpenAPI) você verá exemplos automáticos para os corpos de requisição. Exemplos úteis:

- `POST /predict` (exemplo):

```json
{
  "symbol": "AAPL",
  "days_ahead": 7
}
```

- `POST /predict/manual` (exemplo):

```json
{
  "symbol": "AAPL",
  "predictions": [170.5, 171.2, 172.0, 173.1, 172.8, 174.0, 175.5]
}
```

Use os exemplos no Swagger para preencher automaticamente o corpo de request e testar os endpoints interativamente.

### Exemplo de request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","days_ahead":7}'
```

### Inserir previsões manuais

Você pode testar o formato de resposta enviando previsões manuais (útil para QA ou demonstrações). Exemplo com o símbolo `AAPL`:

```bash
curl -X POST "http://localhost:8000/predict/manual" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","predictions":[170.5,171.2,172.0,173.1,172.8,174.0,175.5], "days_ahead":7}'
```

O campo `days_ahead` é opcional; se informado, seu valor deve ser igual ao tamanho de `predictions`. Se omitido, o comprimento de `predictions` será usado para gerar as datas.

### Endpoints

* `GET /health` – status da API (e do modelo)
* `GET /metrics` – métricas do modelo (lê `models/saved/metrics.json` quando presente; se ausente, usa fallback `models/saved/evaluation_report.json` comitado)
* `POST /predict` – previsão multi-step com intervalo de confiança (calculado com base no erro do modelo)

### API Pública

A API está disponível publicamente em:

```
https://tech-challenge-4-production.up.railway.app/docs
```

Use esse endpoint para acessar a documentação (Swagger UI) e testar os endpoints sem rodar localmente.

## Docker

### Subir com Docker Compose

```bash
docker-compose up -d
```

> Ajuste o `docker-compose.yml` para definir as variáveis de ambiente (ALPHAVANTAGE_API_KEY/BRAPI_TOKEN/DISABLE_YFINANCE).

### Docker - Dashboard (opcional)

Um Dockerfile pronto para o dashboard Streamlit foi adicionado em `docker/Dockerfile.dashboard`. Ele cria uma imagem contendo o app em `src/utils/dashboard.py` e expõe a porta definida por `PORT` (padrão 8501).

Build e teste local:

```bash
# build
docker build -f docker/Dockerfile.dashboard -t tc4-dashboard:latest .

# rodar apontando para a API local (se a API estiver no host):
docker run --rm -e API_URL="http://host.docker.internal:8000" -p 8501:8501 tc4-dashboard:latest
```

No Railway, você pode deployar o serviço do dashboard apontando o `Dockerfile` ou usando o editor de deploy do repositório. Não esqueça de adicionar a variável de ambiente `API_URL` no serviço do dashboard com a URL pública da API.

## Monitoramento com Prometheus + Grafana

### Setup Local com Docker Compose

O projeto inclui suporte completo para monitoramento de aplicação com Prometheus e Grafana.

**Componentes:**
- **API**: exporta métricas no endpoint `/metrics/prometheus`
- **Prometheus**: scrapa métricas a cada 15 segundos
- **Grafana**: visualiza as métricas em dashboards

**Iniciar tudo:**

```bash
docker-compose up -d
```

Isso vai subir:
1. **API** em `http://localhost:8000`
2. **Prometheus** em `http://localhost:9090`
3. **Grafana** em `http://localhost:3000`

### Acessar as Ferramentas

**Prometheus:**
- URL: `http://localhost:9090`
- Consulte métricas direto no explorador
- Verifique targets em `Status -> Targets`

**Grafana:**
- URL: `http://localhost:3000`
- **Usuário**: `admin`
- **Senha**: `admin123`

**Adicionar Prometheus como data source:**
1. Acesse `Configuration -> Data Sources`
2. Clique `Add data source`
3. Selecione `Prometheus`
4. URL: `http://prometheus:9090`
5. Salve e teste

### Métricas Disponíveis

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `api_requests_total` | Counter | Total de requisições (método, endpoint, status) |
| `api_request_latency_seconds` | Histogram | Latência das requisições em segundos |
| `api_prediction_errors_total` | Counter | Total de erros em previsões |
| `model_loaded` | Gauge | 1 se modelo está carregado, 0 caso contrário |
| `scaler_loaded` | Gauge | 1 se scaler está carregado, 0 caso contrário |
| `system_cpu_percent` | Gauge | Percentual de CPU utilizado |
| `system_memory_percent` | Gauge | Percentual de memória utilizada |

### Criar Dashboard no Grafana

1. Acesse `Dashboards -> New Dashboard`
2. Clique `Add an empty panel`
3. No editor, execute uma query Prometheus, exemplo:
   ```
   rate(api_requests_total[5m])
   ```
4. Configure título e visualização
5. Salve o dashboard

**Exemplo de queries úteis:**
```promql
# Taxa de requisições por segundo
rate(api_requests_total[1m])

# Latência P95
histogram_quantile(0.95, api_request_latency_seconds_bucket)

# Taxa de erros
rate(api_prediction_errors_total[5m])

# CPU utilizado
system_cpu_percent

# Memória utilizada
system_memory_percent
```

### Testar Monitoramento

Faça requisições para gerar métricas:

```bash
# Fazer uma previsão
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","days_ahead":7}'

# Verificar métricas brutos no Prometheus
curl http://localhost:8000/metrics/prometheus
```

No Grafana, você verá os gráficos sendo atualizados em tempo real.

### Parar o Monitoramento

```bash
docker-compose down
```

Adiciona a opção `-v` para também remover volumes (dados do Prometheus/Grafana):
```bash
docker-compose down -v
```

### Rodar o Dashboard (Streamlit) localmente

Passo a passo para executar o dashboard sem Docker.

```bash
# 1) ativar o ambiente (se já criou o venv acima)
# Windows (Git Bash / bash.exe):
source venv/Scripts/activate
# Linux/Mac:
# source venv/bin/activate

# 2) instalar dependências
pip install -r requirements.txt

# 3) subir a API (escolha UMA opção)
# via Docker Compose
docker-compose up -d
# ou local com uvicorn
uvicorn src.api.app:app --reload --port 8000

# 4) (opcional) aquecer o cache de dados
python -m src.tools.warmup_cache --years 8 --workers 5

# 5) iniciar o Streamlit do dashboard
streamlit run src/utils/dashboard.py --server.port 8501
```

- Acesse: `http://localhost:8501`.
- Por padrão, o dashboard usa `http://localhost:8000` como API. Para apontar para a API pública, defina `API_URL` antes de iniciar o Streamlit:

```bash
export API_URL="https://tech-challenge-4-production.up.railway.app"
streamlit run src/utils/dashboard.py --server.port 8501
```

- A página "Informações do Modelo" utiliza `models/saved/evaluation_report.json` e `models/saved/training_history.csv` (incluídos para demonstração).

## Métricas (exemplo)

Após um treino de referência:

* MAE: 2.45
* RMSE: 3.12
* MAPE: 4.8%

> Esses valores variam conforme período/símbolo/seed. Use o `/metrics` e/ou gere gráficos de evolução (veja `src/utils/metrics.py`).

## Estrutura do Projeto

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
