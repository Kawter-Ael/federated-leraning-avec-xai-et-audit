FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /app/requirements-docker.txt
RUN python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements-docker.txt

FROM base AS server

COPY audit /app/audit
COPY client /app/client
COPY config /app/config
COPY explainability /app/explainability
COPY server /app/server
COPY shared /app/shared

EXPOSE 8080

CMD ["python", "-m", "server.run_training"]

FROM base AS dashboard

COPY audit /app/audit
COPY config /app/config
COPY dashboard /app/dashboard
COPY explainability /app/explainability
COPY shared /app/shared

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

FROM docker:cli AS docker-cli

FROM base AS client-app

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

COPY audit /app/audit
COPY client /app/client
COPY client_app /app/client_app
COPY config /app/config
COPY explainability /app/explainability
COPY scripts /app/scripts
COPY server /app/server
COPY shared /app/shared

EXPOSE 8502

CMD ["streamlit", "run", "client_app/app.py", "--server.address=0.0.0.0", "--server.port=8502"]

FROM base AS fl-client

COPY client /app/client
COPY config /app/config
COPY explainability /app/explainability
COPY shared /app/shared

CMD ["python", "-m", "client.runtime"]
