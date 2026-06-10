# Data Processing

## Overview

This stage prepares the diabetes dataset used by the federated workflow. The system now relies on the compact tabular schema stored in `data/diabetes.csv`.

## Design / Methodology

The raw dataset contains eight numeric predictors:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

The target column is `Outcome`:

- class `1` -> diabetic case
- class `0` -> non-diabetic case

Missing markers are normalized, the target is converted to the internal binary `target` field, numeric values are imputed with the median and scaled, and stratified train/validation/test splits are generated. Client partitions are derived from the training split only.

## Implementation Summary

The phase produces:

- train, validation and test `.npz` splits
- client-specific training partitions
- JSON preprocessing metadata
- numeric transformer state for runtime reconstruction

The current schema is numeric-only. The embedding-capable model is still reused, but categorical inputs remain empty for this dataset.

## Outputs / Artifacts

- `artifacts/data/splits/train.npz`
- `artifacts/data/splits/validation.npz`
- `artifacts/data/splits/test.npz`
- `artifacts/data/clients/client_*.npz`
- `artifacts/data/metadata/preprocessing_metadata.json`

## Limitations

- preprocessing is tailored to the diabetes dataset
- drift and fairness checks depend on the available columns in this schema
