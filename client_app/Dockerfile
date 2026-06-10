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
COPY client /app/client
COPY client_app /app/client_app
COPY config /app/config
COPY explainability /app/explainability
COPY server /app/server
COPY shared /app/shared

EXPOSE 8502

CMD ["streamlit", "run", "client_app/app.py", "--server.address=0.0.0.0", "--server.port=8502"]
