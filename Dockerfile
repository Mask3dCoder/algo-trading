# Multi-stage build for QuantCore
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY quantcore/ quantcore/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[all]"

# Runtime stage
FROM python:3.12-slim AS runtime

RUN groupadd -r quantcore && useradd -r -g quantcore quantcore

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /app

WORKDIR /app
USER quantcore

EXPOSE 8000

CMD ["uvicorn", "quantcore.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
