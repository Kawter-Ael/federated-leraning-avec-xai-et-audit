# Dataset Onboarding Guide

> Any tabular binary-classification dataset can be used with this pipeline.
> Follow this checklist before uploading.

---

## Supported file formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | UTF-8 or Latin-1; comma or semicolon separator auto-detected |
| TSV | `.tsv` | Tab-separated |
| Excel | `.xlsx`, `.xls` | First sheet is loaded |
| Parquet | `.parquet` | Fast columnar format |

**Unsupported:** JSON, XML, SPSS (.sav), SAS (.sas7bdat), free-text notes, image files.

---

## Required inputs on upload

| Field | What to enter | Example (diabetes) | Example (TB) |
|-------|--------------|-------------------|--------------|
| **Target column** | Column containing the binary label | `Outcome` | `tb_label` |
| **Positive class value** | The value that means "positive case" | `1` | `1` |
| **Numeric columns** | Columns treated as continuous numbers | `Glucose, BMI, Age, …` | `age, cough_days, …` |
| **Excluded columns** | Columns to drop (PII, IDs, free text) | *(none)* | `name` |
| **Fairness attribute** *(optional)* | One numeric column for group-fairness split | `Age` | `age` |

---

## Auto-detection defaults

The portal auto-proposes values when you upload:

- **Target column** — last column if it has ≤ 5 unique values, otherwise the column with the fewest unique values.
- **Numeric columns** — all `int`/`float` columns except the target.
- **Excluded columns** — columns whose name is in `{name, full_name, patient, patient_name, id, identifier, uuid, email, phone}`, or any column where > 90 % of values are unique (likely identifiers).
- **Fairness attribute** — first numeric column whose name (lowercased) is in `{age, bmi, weight, height}`.

You can override any of these before clicking **Validate and prepare run**.

---

## What the pipeline supports

- Binary classification only (exactly 2 target classes).
- One target column.
- Tabular rows (one patient/observation per row).
- Numeric and low-to-medium cardinality categorical features.
- No free-text modeling (NLP clinical notes are not supported).
- No automatic multiclass → multi-label conversion.

---

## PII and privacy guidelines

Remove or exclude columns that identify individuals:

| Column type | Examples | Action |
|-------------|---------|--------|
| Direct identifiers | `name`, `patient_id`, `email`, `phone` | Add to **Excluded columns** |
| Quasi-identifiers | `postcode` + `birthdate` combination | Exclude or coarsen |
| Free text | `notes`, `clinical_summary` | Exclude — not supported |

The portal warns if it detects quasi-unique columns that are not excluded, but does not block submission.

---

## Class balance requirements

| Minority class count | Pipeline reaction |
|---------------------|------------------|
| < 3 rows | **Blocked** — stratified split impossible |
| 3–9 rows | Warning — training and XAI may be unstable |
| ≥ 10 rows | Safe |

If your dataset is severely imbalanced, enable **Data Augmentation** (SMOTE) in the Preprocessing step.

---

## Fairness attribute guidelines

The audit computes a positive-rate gap between two groups (low vs high value of the fairness attribute, split at median).

- Works only on numeric columns.
- Leave empty (`(none)`) if no meaningful fairness attribute exists — the audit will mark the fairness dimension as `not_applicable` instead of `warning`.
- An `age` column (any casing) is matched automatically.

---

## Out-of-scope scenarios

The following are **not supported by design**, not bugs:

- Multiclass classification (3+ classes).
- Sequence / time-series data.
- NLP / free-text clinical notes.
- Image-based diagnosis.
- Fairness claims without a valid numeric fairness attribute.
- Federated training across physically separate devices (this system simulates FL locally).

---

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---------|-------------|-----------|
| "Target column missing" at validation | Wrong column name selected | Re-select target column in Dataset form |
| Training stalls > 5 min | Client subprocess failed | Check `logs/client_stderr.log` in the run artifact folder |
| "Federated training in progress…" spinner never ends | Server subprocess failed to start | Check `logs/server_stderr.log` |
| Audit shows `not_applicable` for fairness | No fairness attribute set | Select a numeric fairness column (e.g. `age`) in Dataset form |
| XAI runs very slowly | Many features or high-cardinality categoricals | Reduce features, increase SHAP `background_size` to 3 |
| Prediction form defaults to 0.0 | Missing imputer statistics | Rerun Phase 2 preprocessing |
| CSV works, XLSX fails | `openpyxl` not installed | Run `pip install openpyxl` |
| "non_binary_target" blocking error | More than 2 unique values in target | Check target column — may need recoding |

---

## Quick-start command (CLI, non-Docker)

```bash
# 1. Preprocess your dataset
python -m shared.run_phase2

# 2. Train (server + client auto-spawned)
python -m server.run_training

# 3. Launch admin dashboard
streamlit run dashboard/app.py
```

For the client portal (upload + predict):
```bash
streamlit run client_app/app.py
```
