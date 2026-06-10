# ENSAJ FL — Federated Learning Platform

Academic prototype Federated Learning system for binary health risk assessment. Combines Flower, PyTorch, SHAP, AI audit, admin dashboard, and authenticated client portal.

The pipeline is **dataset-agnostic**: any tabular binary-classification CSV/XLSX/Parquet can be uploaded via the client portal. The diabetes dataset is the default example; it is not hardcoded. See `docs/dataset-onboarding.md` for the upload checklist.

## What is implemented

- Federated learning with a Flower server and per-user Docker client containers (`docker_per_user` mode)
- Dataset-agnostic tabular preprocessing — schema derived per run from the uploaded CSV
- Binary PyTorch tabular model with numeric + categorical embedding inputs
- Local SHAP explanations and global aggregated SHAP summary
- Client-side IF-THEN rules and server-side aggregated rules
- 5-dimension AI audit: privacy, accuracy, fairness, data drift, explainability
- Client case history in MongoDB
- Authenticated client portal with business-language prediction output
- Admin dashboard with run selector and business/technical views
- PDF report export per prediction

## What is not implemented

- Multi-tenant production deployment
- Kubernetes orchestration
- Advanced IAM
- Real-time monitoring beyond the prototype scope
- Multiclass classification, time-series, NLP, image-based diagnosis

## Deployment mode — `docker_per_user`

Each authenticated user who clicks **Train** triggers the portal container to:
1. Spawn one **dedicated `ensaj-fl-client` Docker container per FL client** (sibling containers via Docker socket).
2. Launch a Flower server subprocess internally.
3. Connect each client container to the server via gRPC over `ensaj_fl-net`.
4. Store all artifacts under `artifacts/user_runs/<run_id>/`.

When `num_clients=2`, two containers run simultaneously: `ensaj-fl-client_1-<suffix>` and `ensaj-fl-client_2-<suffix>`. Each has its own data partition and logs. FedAvg aggregates their updates weighted by `num_examples`.

The static `server` service is **not started by default**. It is only needed for the legacy `--profile fl` mode.

### Validated multi-client run

Run `principal-20260506220641-2ff86b85` with `num_clients=2` produced:
- `ensaj-fl-client_1-client_1` and `ensaj-fl-client_2-client_2` simultaneously in `docker ps`
- `observed_client_ids: ["client_1", "client_2"]`
- `per_client_metrics`: client_1 = 269 examples, client_2 = 268 examples
- Full pipeline: server → FL rounds → XAI → audit

## Quick start

```bash
# 1. Configure host paths
cp .env.example .env
# Edit .env — set FL_ARTIFACT_ROOT_HOST, FL_DATA_ROOT_HOST, FL_CONFIG_ROOT_HOST

# 2. Build all images
docker compose --profile fl build

# 3. Start default services (mongodb + dashboard + client-app)
docker compose up -d

# 4. Seed client accounts (one-time)
docker compose exec client-app python -m scripts.seed_clients

# 5. Open portals
#   Client portal (FL training):  http://localhost:8502
#   Admin dashboard:               http://localhost:8501
```

Login → upload dataset → configure columns → click **Train**.
Each training run automatically spawns a dedicated `ensaj-fl-client` container.

See `run.md` for the full setup guide.

## `config/project-config.json` — intentionally neutral defaults

The base config ships with **empty** dataset-specific fields:
- `data.target_column = ""`
- `data.positive_class = ""`
- `data.numeric_columns = []`
- `data.fairness_attribute = ""`

These are filled at run time by the client portal (column mapping UI) or by loading a preset from `config/presets/`. The diabetes preset is at `config/presets/diabetes.json`. Do **not** hardcode PIMA column names back into the base config.

## Default example dataset

`data/diabetes.csv` — PIMA Indians Diabetes, 768 rows, 8 numeric features, binary `Outcome` target.

Any CSV/XLSX/Parquet with a binary label column is supported.

## Risk levels

| Probability | Label |
|---|---|
| `< 0.33` | Low risk |
| `0.33 – 0.66` | Medium risk |
| `≥ 0.66` | High risk |

## Docker commands

```bash
# Start default services
docker compose up -d

# Start with static FL server (legacy mode)
docker compose --profile fl up

# View logs
docker compose logs -f
docker compose logs -f client-app
docker compose logs -f dashboard

# Stop
docker compose stop

# Remove containers and network
docker compose down

# Remove containers + volumes
docker compose down -v

# Rebuild without cache
docker compose --profile fl build --no-cache
```

## Per-run artifact layout

```
artifacts/user_runs/<run_id>/
├── effective_config.json
├── run_manifest.json
├── logs/
│   ├── server_stdout.log
│   ├── server_stderr.log
│   ├── client_stdout.log
│   └── client_stderr.log
├── data/
│   ├── metadata/preprocessing_metadata.json
│   └── splits/{train,validation,test}.npz
├── input/<run_id>-<original_filename>
├── models/global_model.pt
├── metrics/federated_training_metrics.json
├── xai/{global_shap_summary,global_rules_summary,local_shap_summaries,
│        instance_explanations,xai_validation_report}.json
└── audit/{client_payload_audit,server_audit_report,
│          audit_validation_summary}.json + audit_log.jsonl
```

## Tests

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests/unit -v
python -m unittest discover -s tests/integration -v
```

## Documentation

- [Index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Data processing](docs/data-processing.md)
- [Federated learning](docs/federated-learning.md)
- [Explainability](docs/explainability.md)
- [Audit and traceability](docs/audit-and-traceability.md)
- [Deployment](docs/deployment.md)
- [Client credentials](docs/client-credentials.md)
- [Dataset onboarding](docs/dataset-onboarding.md)
- [Tests](docs/testing.md)

## Diagrams

Visual Mermaid diagrams in `diagrams/`:

| File | Content |
|---|---|
| `01-system-architecture.md` | C4 container view |
| `02-fl-training-sequence.md` | Full FL sequence from login to stored model |
| `03-data-pipeline.md` | Phase 2 preprocessing flowchart |
| `04-fedavg-aggregation.md` | Weight + SHAP protocol, FedAvg formula |
| `05-docker-deployment.md` | Container topology, env var table |
| `06-module-dependencies.md` | Full module import graph |
| `07-audit-pipeline.md` | 5-dimension audit flowchart |
| `08-user-journey.md` | User journey + portal state machine |
| `09-artifact-lifecycle.md` | Who creates/reads each artifact |
| `10-database-schema.md` | MongoDB ER diagram + auth sequence |
| `11-model-architecture.md` | PyTorch model + threshold selection |
