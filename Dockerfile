FROM python:3.12-slim

LABEL maintainer="QuantCore Team"
LABEL description="QuantCore algorithmic trading system"

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[dev,stats,data]" 2>/dev/null || \
    pip install --no-cache-dir numpy scipy pandas polars pydantic dynaconf structlog \
    exchange-calendars fastapi uvicorn schedule pyyaml alpaca-py requests psutil
# pyarrow is required for parquet caching in live data feed
RUN pip install --no-cache-dir pyarrow

# Copy source
COPY quantcore/ quantcore/
COPY config/ config/
COPY scripts/ scripts/

# Create reports directory (also mountable as volume)
RUN mkdir -p /app/reports /app/data /app/logs

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV QUANTCORE_LIVE_CONFIG=config/live.yaml

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "quantcore.live.production_runner"]
