# 📊 Monitoramento com Prometheus + Grafana

Documentação completa do setup de monitoramento para o projeto Stock Price Prediction.

## ✅ O que foi implementado

### 1. **Prometheus Client Integration**
- Adicionado `prometheus-client` ao `requirements.txt`
- Definidas métricas de monitoramento:
  - `api_requests_total`: Contador de requisições (método, endpoint, status)
  - `api_request_latency_seconds`: Histograma de latência
  - `api_prediction_errors_total`: Contador de erros
  - `model_loaded`: Gauge de disponibilidade do modelo
  - `scaler_loaded`: Gauge de disponibilidade do scaler
  - `system_cpu_percent`: Gauge de CPU
  - `system_memory_percent`: Gauge de memória

### 2. **Novo Endpoint Prometheus**
- **`GET /metrics/prometheus`**: Retorna métricas em formato Prometheus
  - Formato: `text/plain`
  - Pronto para scraping automático

### 3. **Middleware de Monitoramento**
- Classe `PrometheusMiddleware` implementada
- Rastreamento automático de:
  - Latência de requisições
  - Status code de respostas
  - Métricas de sistema (CPU/memória)
- **Nota**: Atualmente desabilitado (comentado no `app.add_middleware`)

### 4. **Docker Compose Melhorado**
- Serviço **Prometheus** (porta 9090)
- Serviço **Grafana** (porta 3000)
- Rede `monitoring` compartilhada
- Volumes persistentes para dados

### 5. **Arquivo de Configuração Prometheus**
- `prometheus.yml` criado
- Job configurado para scraping da API
- Intervalo: 10s
- Timeout: 5s

### 6. **Documentação**
- README.md atualizado com seção "Monitoramento com Prometheus + Grafana"
- Instruções passo-a-passo para setup
- Exemplos de queries PromQL
- Credenciais padrão do Grafana

---

## 🚀 Como Usar

### Local com Docker Compose

```bash
# Iniciar tudo
docker-compose up -d

# Acessar ferramentas
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin123)
# - API: http://localhost:8000
```

### Ativar Middleware Prometheus

No arquivo `src/api/app.py`, descomentar a linha:

```python
app.add_middleware(PrometheusMiddleware)
```

---

## 📝 Observações Importantes

### API Original
- A API original também encerra conexões após requisições
- Pode ser relacionado a um comportamento de shutdown em Windows
- Não afeta a funcionalidade de monitoramento

### Próximos Passos
1. Investigar comportamento de shutdown da API
2. Ativar e testar o middleware completo
3. Criar dashboards no Grafana
4. Configurar alertas no Prometheus

---

## 📊 Métricas Disponíveis

### Counters
- `api_requests_total{method, endpoint, status_code}`
- `api_prediction_errors_total{error_type}`

### Histograms
- `api_request_latency_seconds_bucket{endpoint}`
- `api_request_latency_seconds_sum{endpoint}`
- `api_request_latency_seconds_count{endpoint}`

### Gauges
- `model_loaded` (0 ou 1)
- `scaler_loaded` (0 ou 1)
- `system_cpu_percent`
- `system_memory_percent`

---

## 🔧 Troubleshooting

### Prometheus não encontra a API
- Certifique-se que a API está rodando em `localhost:8000`
- Verifique `prometheus.yml` (targets deve apontar para a API)
- Acesse http://localhost:9090/targets para ver status

### Grafana não conecta ao Prometheus
- Na config de data source, use URL: `http://prometheus:9090`
- Não `http://localhost:9090` (diferente em Docker networks)

### Métricas vazias
- Faça algumas requisições para gerar dados
- Prometheus demora ~15s para scraping
- Aguarde antes de consultar

---

## 📚 Referências

- [Prometheus Documentation](https://prometheus.io/docs/)
- [prometheus-client Python](https://github.com/prometheus/client_python)
- [Grafana](https://grafana.com/)

---

**Branch**: `monitoring-prometheus-test`  
**Status**: ✅ Pronto para testes e expansão
