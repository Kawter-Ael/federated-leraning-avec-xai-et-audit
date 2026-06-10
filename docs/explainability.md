# Explainability

## Overview

The explainability layer combines local SHAP computation on clients with server-side aggregation. The objective is to keep detailed local explanations private while exposing readable global insights.

## Design / Methodology

Local SHAP values are computed with `KernelExplainer` on the client side. Following the SHAP formulation used in this project:

- local SHAP explains an individual case
- global SHAP is the mean absolute contribution aggregated across clients

The system also derives simplified IF-THEN rules from the most influential local features and aggregates those rules on the server.

## Implementation Summary

The explainability pipeline now produces:

- global SHAP visual summaries
- local SHAP summaries kept compact
- aggregated rule summaries
- an XAI validation report containing
  - model information
  - input/output structure
  - generated IF-THEN rules
  - explanation consistency checks
  - privacy checks
  - explainability quality assessment

Client-facing interfaces show business-language explanations rather than raw numeric SHAP tables.

## Outputs / Artifacts

- `artifacts/xai/global_shap_summary.json`
- `artifacts/xai/local_shap_summaries.json`
- `artifacts/xai/global_rules_summary.json`
- `artifacts/xai/xai_validation_report.json`

## Limitations

- SHAP remains sampling-based and computationally expensive
- IF-THEN rules are explanatory approximations, not causal guarantees
