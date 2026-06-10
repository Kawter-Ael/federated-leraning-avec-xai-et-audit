FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /app/requirements-docker.txt
RUN python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements-docker.txt

COPY audit /app/audit
COPY config /app/config
COPY dashboard /app/dashboard
COPY explainability /app/explainability
COPY shared /app/shared

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
