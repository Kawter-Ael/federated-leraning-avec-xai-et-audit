# Architecture

## Overview

The project implements a dataset-agnostic Federated Learning prototype for binary risk assessment. It combines a Flower FL server, PyTorch tabular model, SHAP explainability, 5-dimension AI audit, and two Streamlit portals (admin dashboard + authenticated client portal).

The system is organized as a sequential pipeline in which each stage consumes persisted artifacts produced by the previous stage. All artifacts are isolated per user run under `artifacts/user_runs/<run_id>/`.

## Design / Methodology

### Five pipeline phases

| Phase | Module | What it does |
|---|---|---|
| 2 — Preprocessing | `shared/data_preparation.py` | Load CSV, clean, split, fit imputer/scaler, encode categoricals, save NPZ splits + metadata |
| 3 — FL Training | `server/federated_trainer.py` + `client/federated_client.py` | Flower FedAvg rounds; client trains locally and returns weights + safe SHAP metrics |
| 4 — Explainability | `explainability/shap_explainer.py` | Aggregate SHAP summaries from stored client round data; generate IF-THEN rules |
| 5 — Audit | `audit/audit_pipeline.py` | Validate 5 dimensions: privacy, accuracy, fairness, data drift, explainability |
| Portals | `dashboard/app.py`, `client_app/` | Display results; client portal triggers phases 2-5 on demand |

### Deployment mode — `docker_per_user`

Each authenticated user's training run:
1. Spawns an FL server subprocess inside the `client-app` container.
2. Spawns a dedicated `ensaj-fl-client` Docker container via the Docker socket.
3. Client container connects to server via gRPC.
4. After the final round, server saves model + runs Phase 4 + Phase 5 inline.
5. All outputs land in `artifacts/user_runs/<run_id>/`.

The static `server` service is **not started by default** in this mode.

### Dataset-agnostic schema

Each run derives its model architecture from the uploaded dataset at Phase 2 time:
- Column roles (numeric vs. categorical) set by the user in the portal.
- `preprocessing_metadata.json` captures the full schema contract (column list, cardinalities, transformer state).
- All downstream phases read this metadata — the model architecture, SHAP feature names, and audit checks all follow the schema.

## Component Roles

| Component | Tech | Responsibility |
|---|---|---|
| `server/` | Python + Flower | FL server; FedAvg aggregation; Phase 4 + 5 auto-trigger after final round |
| `client/` | Python + Flower | FL client; local training; SHAP summary + rules computed locally, only aggregates sent |
| `client_app/` | Streamlit | Client portal: auth, dataset upload, pipeline trigger, prediction, PDF export |
| `dashboard/` | Streamlit | Admin dashboard: run selector, metrics, SHAP, rules, audit, prediction form |
| `shared/` | Python | All shared business logic: preprocessing, modeling, XAI, aggregation, auth, workflow |
| `audit/` | Python | 5-dimension AI audit pipeline |
| `explainability/` | Python + SHAP | Global SHAP aggregation and validation |
| `config/` | JSON | Central project configuration + presets |
| `artifacts/` | Files | All generated outputs (Docker volume) |
| MongoDB | Docker | User accounts, run history, case history |

## Artifact Flow

```
artifacts/user_runs/<run_id>/
├── data/metadata/preprocessing_metadata.json   ← Phase 2 output; schema contract
├── data/splits/{train,validation,test}.npz      ← Phase 2 output; FL client input
├── models/global_model.pt                       ← Phase 3 output; dashboard + portal read
├── metrics/federated_training_metrics.json      ← Phase 3 output; Phase 4+5 input
├── xai/                                         ← Phase 4 output; dashboard reads
└── audit/                                       ← Phase 5 output; dashboard reads
```

## Privacy Contract

Clients never transmit raw feature data, per-instance SHAP values, or individual data rows. Only model weight updates and aggregated SHAP means are sent. This is enforced by `FORBIDDEN_PAYLOAD_KEYS` checked in `client/federated_client.py` and validated by the Phase 5 privacy audit.

## Outputs / Artifacts

Per completed run under `artifacts/user_runs/<run_id>/`:
- `data/`: splits, preprocessing metadata
- `models/`: trained global model bundle (weights + schema + threshold)
- `metrics/`: per-round accuracy, recall, F1, ROC-AUC, SHAP summaries
- `xai/`: global SHAP, rules, local summaries, instance explanations, XAI validation report
- `audit/`: client payload audit, server audit report, audit validation summary, audit log

## Limitations

- Single-client mode warning is shown but not blocked — SHAP aggregation is trivial with 1 client.
- No production IAM or CSRF protection — Streamlit session isolation only.
- All components communicate via shared artifact directory — no API contract.
- Fairness audit uses Age as proxy and a fixed 0.15 gap threshold.
