# Documentation Index

## Overview

This documentation describes the ENSAJ FL project — a dataset-agnostic Federated Learning prototype for binary risk assessment. The system combines Flower FL, PyTorch, SHAP explainability, 5-dimension AI audit, authenticated client portal, and admin dashboard.

Default deployment mode: `docker_per_user` — each training run spawns a dedicated `ensaj-fl-client` container automatically. No static FL server runs by default.

## Documents

- `architecture.md` — global architecture, component roles, artifact flow, privacy contract
- `data-processing.md` — dataset preparation, schema generation, preprocessing pipeline
- `federated-learning.md` — Flower-based federated workflow, FedAvg, threshold selection
- `explainability.md` — SHAP and IF-THEN explanation pipeline
- `audit-and-traceability.md` — 5-dimension audit and case history
- `deployment.md` — Docker Compose setup, `docker_per_user` mode, env variables
- `client-credentials.md` — pre-provisioned client accounts and seeding
- `dataset-onboarding.md` — upload checklist, format table, PII guidelines
- `testing.md` — automated test infrastructure
- `dvc.md` — DVC data versioning: pipeline stages, commands, remote setup

## Diagrams

Visual Mermaid diagrams are in `../diagrams/`. See `../README.md` for the full index.

## Pipeline Summary

1. Phase 2 — Data preparation (client portal upload → NPZ splits + metadata)
2. Phase 3 — Federated training (Flower rounds → global model)
3. Phase 4 — Explainability (aggregate SHAP summaries + IF-THEN rules)
4. Phase 5 — AI Audit (5-dimension validation)
5. Portals — Admin dashboard (run selector) + Client portal (prediction + PDF export)

## Artifact layout (per run)

```
artifacts/user_runs/<run_id>/
├── data/metadata/preprocessing_metadata.json
├── data/splits/{train,validation,test}.npz
├── models/global_model.pt
├── metrics/federated_training_metrics.json
├── xai/
└── audit/
```
