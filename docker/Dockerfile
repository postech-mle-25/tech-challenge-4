FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    PORT=8501

WORKDIR /app

# Install system deps (minimal) and python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

EXPOSE ${PORT}

# Run streamlit dashboard (use env PORT provided by platform)
CMD streamlit run src/utils/dashboard.py --server.port $PORT --server.address 0.0.0.0
