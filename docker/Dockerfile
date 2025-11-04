# docker/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY src/ ./src/
COPY models/ ./models/

# Expor porta
EXPOSE 8000

# Comando para iniciar
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]