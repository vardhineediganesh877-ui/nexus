FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null || pip install --no-cache-dir \
    ccxt>=4.0.0 \
    tradingview-ta>=3.3.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.20.0 \
    matplotlib>=3.7.0 \
    python-dotenv>=1.0.0

COPY src/ src/

# Data directory
RUN mkdir -p /root/.nexus/data

ENV NEXUS_PAPER=true
ENV NEXUS_DATA_DIR=/root/.nexus/data
ENV NEXUS_LOG_LEVEL=INFO

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
