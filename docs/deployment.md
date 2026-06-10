# Deployment

## Overview

The project runs via Docker Compose. The default mode is **`docker_per_user`**: the client portal spawns dedicated FL client containers on demand for each training run. No static FL server is started by default.

## Services

| Service | Image | Port | Started by default |
|---|---|---|---|
| `mongodb` | `mongo:7` | 27017 | Yes |
| `client-app` | `ensaj-client-app` | 8502 | Yes |
| `dashboard` | `ensaj-dashboard` | 8501 | Yes |
| `server` | `ensaj-server` | 8080 | No (`--profile fl` only) |
| `fl-client-*` | `ensaj-fl-client` | — | No (`--profile fl` only) |

The `ensaj-fl-client` image is also used as the dynamically-spawned per-user container. The `client-app` container spawns sibling containers via the Docker socket mounted at `/var/run/docker.sock`.

## Setup

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — mandatory host path settings:

```
FL_ARTIFACT_ROOT_HOST=D:/ZKR/kawter/artifacts   # absolute path on Docker host
FL_DATA_ROOT_HOST=D:/ZKR/kawter/data
FL_CONFIG_ROOT_HOST=D:/ZKR/kawter/config
```

All other variables have working defaults. See `.env.example` for the full list.

### 2. Build all images

```bash
docker compose --profile fl build
```

`--profile fl` ensures the `ensaj-fl-client` image is built even though the static server is not started by default.

### 3. Start default services

```bash
docker compose up -d
```

Starts: `mongodb` + `dashboard` + `client-app`.

### 4. Seed client accounts (one-time)

```bash
docker compose exec client-app python -m scripts.seed_clients
```

Creates `client_1` … `client_5` with default passwords. See `docs/client-credentials.md`.

### 5. Access portals

| Portal | URL |
|---|---|
| Client portal (FL training) | http://localhost:8502 |
| Admin dashboard | http://localhost:8501 |

Login → upload dataset → configure → **Train**.

## Docker Compose profiles

| Profile | What starts |
|---|---|
| *(default)* | `mongodb`, `dashboard`, `client-app` |
| `fl` | All default services **plus** static `server` + `fl-client-1/2/3` |

Use `--profile fl` only when testing the legacy static-server mode.

## Multi-stage Dockerfile

```
FROM python:3.11-slim AS base
  └─ installs all Python deps (CPU-only PyTorch)

FROM docker:cli AS docker-cli
  └─ provides the docker CLI binary

FROM base AS server          → CMD: python -m server.run_training
FROM base AS dashboard       → CMD: streamlit run dashboard/app.py --server.port=8501
FROM base AS client-app      → COPY docker binary from docker-cli stage
                               CMD: streamlit run client_app/app.py --server.port=8502
FROM base AS fl-client       → CMD: python -m client.runtime
```

Key design choices:
- `data/` is mounted read-only (`./data:/app/data:ro`), never baked into images.
- `config/` is mounted read-only (`./config:/app/config:ro`).
- `artifacts/` is a shared writable volume (`./artifacts:/app/artifacts`).
- The Docker CLI binary is copied from the official `docker:cli` image (not `docker.io` from apt, which only provides the daemon).

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEPLOYMENT_MODE` | `docker_per_user` | Training spawn mode for the client-app |
| `PORTAL_HOSTNAME` | `ensaj-client-app` | Container name used as FL server host for spawned clients |
| `FL_ARTIFACT_ROOT_HOST` | — | **Required.** Absolute host path to `artifacts/` for `-v` mounts |
| `FL_DATA_ROOT_HOST` | — | **Required.** Absolute host path to `data/` |
| `FL_CONFIG_ROOT_HOST` | — | **Required.** Absolute host path to `config/` |
| `FL_DOCKER_IMAGE` | `ensaj-fl-client:latest` | Image to use for spawned client containers |
| `FL_DOCKER_NETWORK` | `kawter_fl-net` | Docker network for spawned client containers |
| `MONGODB_URI` | `mongodb://mongodb:27017` | MongoDB connection string |
| `DASHBOARD_PORT` | `8501` | Host port for admin dashboard |
| `CLIENT_APP_PORT` | `8502` | Host port for client portal |

Full list in `.env.example`.

## Per-run artifact path translation

Inside the `client-app` container, artifacts live at `/app/artifacts/user_runs/<run_id>/`.
When spawning a client container with `-v`, this path must be translated to the Docker host path using `FL_ARTIFACT_ROOT_HOST`. The `_to_host_path()` helper in `shared/user_client.py` handles this translation for both `/app/artifacts/...` and relative `artifacts/...` prefixes.

## Rebuild after code changes

```bash
docker compose --profile fl build && docker compose up -d
```

## Logs

```bash
docker compose logs -f               # all services
docker compose logs -f client-app    # portal + training spawn logs
docker compose logs -f dashboard
docker compose logs -f mongodb
```

Per-run server and client logs are also written to:
```
artifacts/user_runs/<run_id>/logs/server_stdout.log
artifacts/user_runs/<run_id>/logs/server_stderr.log
artifacts/user_runs/<run_id>/logs/client_0_stdout.log   # client_1
artifacts/user_runs/<run_id>/logs/client_0_stderr.log
artifacts/user_runs/<run_id>/logs/client_1_stdout.log   # client_2 (if num_clients >= 2)
artifacts/user_runs/<run_id>/logs/client_1_stderr.log
```

## Limitations

- Docker socket mounting is required for `docker_per_user` mode — not available in all CI/cloud environments.
- All services share the same `artifacts/` volume — no per-tenant isolation at the filesystem level.
- Designed for single-machine academic use, not for multi-host or production deployment.
