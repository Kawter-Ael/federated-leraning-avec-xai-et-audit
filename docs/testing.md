# Testing

## Overview

This document describes the validation strategy used to assess the consistency of the final project. Testing focuses on reproducibility, artifact integrity, and compatibility between pipeline stages.

## Design / Methodology

The project uses `unittest` to validate the system from a software and artifact perspective. The tests are designed to run from a clean state: if generated artifacts are missing, the required stages are executed automatically before assertions are evaluated.

The validation strategy covers:
- raw data and preprocessing contracts
- trained model loading and prediction
- explainability generation
- five-dimension audit validation
- client workflow and case persistence
- dashboard loading and prediction path
- Docker Compose configuration

## Implementation Summary

The test suite verifies:
- target definition and schema stability for the diabetes dataset
- generation of artifacts from scratch
- successful loading of the persisted PyTorch model
- coherence of SHAP summaries with the feature schema
- successful execution of audit validators
- client authentication and case history persistence
- dashboard compatibility with generated artifacts
- syntactic validity of `docker-compose.yml`

## Outputs / Artifacts

The tests do not introduce a new artifact family. Their role is to confirm that the existing pipeline can be regenerated and consumed consistently.

The test entrypoint is:
- `python -m unittest discover -s tests -v`

## Limitations

- tests validate a Flower-based federated flow on the local runtime but not a full production deployment
- Docker runtime validation remains dependent on the local Docker daemon
- evaluation focuses on correctness and reproducibility rather than benchmark performance
