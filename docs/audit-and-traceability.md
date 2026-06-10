# Audit and Traceability

## Overview

The audit layer validates the federated artifacts after training and explainability. It now evaluates five dimensions: privacy, accuracy, fairness, data drift, and explainability.

## Design / Methodology

The audit operates on persisted artifacts and reconstructed payloads. It does not inspect raw network traffic, but it verifies that server-side retained data remains consistent with the privacy contract.

## Implementation Summary

The audit checks:

- privacy: forbidden raw keys, absence of row-level data leakage
- accuracy: global quality thresholds from persisted metrics
- fairness: basic group comparison over the `Age` attribute
- data drift: distribution shift against the diabetes reference dataset
- explainability: presence and coherence of SHAP and rule outputs

The project also stores prediction case history in MongoDB for client traceability.

## Outputs / Artifacts

- `artifacts/audit/client_payload_audit.json`
- `artifacts/audit/server_audit_report.json`
- `artifacts/audit/audit_validation_summary.json`
- `artifacts/audit/audit_log.jsonl`

## Limitations

- the audit is artifact-based
- fairness and drift checks are intentionally lightweight for an academic prototype
