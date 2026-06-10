from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from shared.data_preparation import prepare_phase2
from shared.federated_data import build_artifact_paths, load_project_config
from shared.user_client import run_local_training
from shared.utils import apply_config_override, utc_now, PipelineTracer
from audit.audit_pipeline import run_phase5_audit
from explainability.shap_explainer import run_phase4_explainability

_logger = logging.getLogger("shared.client_workflow")
_PII_NAMES = {
    "name",
    "full_name",
    "patient",
    "patient_name",
    "id",
    "identifier",
    "uuid",
    "email",
    "phone",
    "ssn",
}
_FAIRNESS_NAMES = {"age", "bmi", "weight", "height"}
_LEAKAGE_NAMES = {
    "label",
    "labels",
    "target",
    "targets",
    "prediction",
    "predictions",
    "diagnosis",
    "diagnostic",
    "outcome",
    "result",
    "results",
}


def _normalize_label_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def _build_issue(
    severity: str, code: str, message: str, columns: list[str] | None = None
) -> dict[str, Any]:
    issue = {"severity": severity, "code": code, "message": message}
    if columns:
        issue["columns"] = list(columns)
    return issue


def _estimate_shap_cost(
    *, feature_count: int, max_categorical_cardinality: int, num_rows: int
) -> str:
    if feature_count >= 40 or max_categorical_cardinality >= 100 or num_rows >= 10000:
        return "high"
    if feature_count >= 20 or max_categorical_cardinality >= 30 or num_rows >= 3000:
        return "medium"
    return "low"


def _estimate_split_risk(class_counts: dict[str, int], num_rows: int) -> str:
    if not class_counts or num_rows <= 0:
        return "blocking"
    min_class = min(class_counts.values())
    if min_class < 3:
        return "blocking"
    if min_class < 10:
        return "warning"
    return "safe"


def summarize_dataset_complexity(
    frame: pd.DataFrame,
    *,
    target_column: str,
    numeric_columns: list[str],
    excluded_columns: list[str],
) -> dict[str, Any]:
    detected_numeric = frame.select_dtypes(include=["number"]).columns.tolist()
    candidate_columns = [column for column in frame.columns if column != target_column]
    categorical_columns = [
        column
        for column in candidate_columns
        if column not in set(numeric_columns) and column not in set(excluded_columns)
    ]
    categorical_cardinality = {
        column: int(frame[column].nunique(dropna=True)) for column in categorical_columns
    }
    target_values = (
        frame[target_column].dropna().map(str).tolist()
        if target_column in frame.columns
        else []
    )
    class_counts = (
        {
            str(key): int(value)
            for key, value in frame[target_column].value_counts(dropna=False).items()
        }
        if target_column in frame.columns
        else {}
    )
    class_balance = (
        {
            str(key): round(int(value) / max(len(frame), 1), 4)
            for key, value in frame[target_column].value_counts(dropna=False).items()
        }
        if target_column in frame.columns
        else {}
    )
    max_cardinality = max(categorical_cardinality.values(), default=0)
    feature_count = len(numeric_columns) + len(categorical_columns)
    return {
        "num_rows": int(frame.shape[0]),
        "num_columns": int(frame.shape[1]),
        "target_cardinality": int(frame[target_column].dropna().nunique())
        if target_column in frame.columns
        else 0,
        "target_values": sorted(set(target_values)),
        "class_counts": class_counts,
        "class_balance": class_balance,
        "detected_numeric_count": len(detected_numeric),
        "detected_categorical_count": len(
            [column for column in frame.columns if column not in detected_numeric]
        ),
        "configured_numeric_count": len(numeric_columns),
        "configured_categorical_count": len(categorical_columns),
        "max_categorical_cardinality": int(max_cardinality),
        "categorical_cardinality": categorical_cardinality,
        "feature_count": int(feature_count),
        "shap_cost_level": _estimate_shap_cost(
            feature_count=feature_count,
            max_categorical_cardinality=max_cardinality,
            num_rows=int(frame.shape[0]),
        ),
        "split_risk": _estimate_split_risk(class_counts, int(frame.shape[0])),
    }


def autodetect_dataset_roles(frame: pd.DataFrame) -> dict[str, Any]:
    columns = frame.columns.tolist()
    numeric_columns_all = frame.select_dtypes(include=["number"]).columns.tolist()
    auto_pii: list[str] = []
    for column in columns:
        unique_ratio = frame[column].nunique(dropna=True) / max(len(frame), 1)
        is_object_like = pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column])
        looks_free_text = is_object_like and unique_ratio > 0.9
        name_match = column.lower().strip() in _PII_NAMES
        if looks_free_text or name_match:
            auto_pii.append(column)

    auto_target = ""
    for column in reversed(columns):
        if frame[column].dropna().nunique() == 2:
            auto_target = column
            break
    if not auto_target and columns:
        unique_counts = {column: frame[column].nunique(dropna=True) for column in columns}
        auto_target = min(unique_counts, key=unique_counts.get)

    numeric_columns = [
        column
        for column in numeric_columns_all
        if column != auto_target and column not in auto_pii
    ]
    candidate_columns = [column for column in columns if column != auto_target]
    fairness_attribute = ""
    fairness_priority = ["age", "bmi", "weight", "height"]
    normalized_numeric = {column.lower().strip(): column for column in numeric_columns}
    for candidate in fairness_priority:
        if candidate in normalized_numeric:
            fairness_attribute = normalized_numeric[candidate]
            break

    return {
        "target_column": auto_target,
        "numeric_columns": numeric_columns,
        "excluded_columns": [column for column in auto_pii if column in candidate_columns],
        "fairness_attribute": fairness_attribute or None,
        "candidate_columns": candidate_columns,
        "target_values": sorted(str(value) for value in frame[auto_target].dropna().unique())
        if auto_target
        else [],
    }


def _run_docker_training(
    runtime_config: dict[str, Any], user_context: dict[str, Any]
) -> dict[str, Any]:
    import subprocess

    config = load_effective_config(runtime_config)
    num_clients = int(
        runtime_config["config_override"]["federated_learning"]["num_clients"]
    )
    artifact_root = runtime_config.get("global_artifact_root", "artifacts")

    from scripts.partition_data import partition_for_docker

    partition_for_docker(num_clients, artifact_root)

    compose_env = {
        **dict(os.environ),
        "FL_ARTIFACT_ROOT_HOST": Path(str(artifact_root)).as_posix(),
        "CONFIG_OVERRIDE_JSON": json.dumps(runtime_config["config_override"]),
        "FL_FORCE_EXIT_ON_COMPLETE": "0",
    }
    cmd = [
        "docker",
        "compose",
        "--profile",
        "fl",
        "up",
        "server",
        *[f"fl-client-{i}" for i in range(1, num_clients + 1)],
    ]
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,
        env=compose_env,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Docker FL training failed: {process.stderr or process.stdout}"
        )

    metrics_path = build_artifact_paths(artifact_root)["metrics"]
    if not metrics_path.exists():
        raise RuntimeError(
            "Docker FL training completed without producing global metrics."
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def validate_uploaded_dataset(
    dataset_path: str | Path,
    *,
    target_column: str,
    positive_class: str | int | float | None,
    excluded_columns: list[str],
    numeric_columns: list[str],
) -> dict[str, Any]:
    from shared.dataset_io import load_tabular

    frame = load_tabular(dataset_path)
    blocking: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    info: list[dict[str, Any]] = []

    if frame.empty:
        blocking.append(
            _build_issue(
                "blocking",
                "empty_dataset",
                "Dataset is empty. Upload a tabular file with at least one row.",
            )
        )

    duplicate_columns = sorted(
        column for column in frame.columns if int((frame.columns == column).sum()) > 1
    )
    if duplicate_columns:
        blocking.append(
            _build_issue(
                "blocking",
                "duplicate_columns",
                "Dataset contains duplicated column names after normalization.",
                duplicate_columns,
            )
        )

    if target_column not in frame.columns:
        blocking.append(
            _build_issue(
                "blocking",
                "missing_target_column",
                f"Target column '{target_column}' is missing from the dataset.",
                [target_column],
            )
        )

    missing_numeric = sorted(
        column for column in numeric_columns if column not in frame.columns
    )
    if missing_numeric:
        blocking.append(
            _build_issue(
                "blocking",
                "missing_numeric_columns",
                "Some selected numeric columns are missing from the dataset.",
                missing_numeric,
            )
        )

    detected_numeric = frame.select_dtypes(include=["number"]).columns.tolist()
    inferred_categorical = [
        column for column in frame.columns if column not in detected_numeric
    ]
    target_values: list[str] = []
    normalized_target_values: list[str] = []
    class_counts: dict[str, int] = {}
    class_balance: dict[str, float] = {}

    if target_column in frame.columns:
        raw_target = frame[target_column]
        non_null_target = raw_target.dropna()
        target_values = sorted(str(value) for value in non_null_target.unique())
        normalized_target = non_null_target.map(_normalize_label_token)
        normalized_target_values = sorted(value for value in normalized_target.unique() if value)
        normalized_raw_count = len(target_values)
        normalized_unique_count = len(normalized_target_values)
        if normalized_raw_count != normalized_unique_count:
            warnings.append(
                _build_issue(
                    "warning",
                    "target_values_normalize_to_same_token",
                    "Some target labels differ only by casing or whitespace. They will be treated as the same class.",
                    [target_column],
                )
            )
        if normalized_unique_count != 2:
            blocking.append(
                _build_issue(
                    "blocking",
                    "non_binary_target",
                    f"Target column '{target_column}' must contain exactly 2 classes after normalization.",
                    [target_column],
                )
            )
        else:
            class_counts = {
                str(key): int(value)
                for key, value in non_null_target.value_counts(dropna=False).items()
            }
            class_balance = {
                key: round(value / max(int(non_null_target.shape[0]), 1), 4)
                for key, value in class_counts.items()
            }
            if class_counts and min(class_counts.values()) < 3:
                blocking.append(
                    _build_issue(
                        "blocking",
                        "too_few_examples_per_class",
                        "At least one class has fewer than 3 rows. Stratified train/validation/test split is unsafe.",
                        [target_column],
                    )
                )
            elif class_counts and min(class_counts.values()) < 10:
                warnings.append(
                    _build_issue(
                        "warning",
                        "small_minority_class",
                        "At least one class has fewer than 10 rows. Training, fairness and XAI may be unstable.",
                        [target_column],
                    )
                )

    positive_class_normalized = _normalize_label_token(positive_class)
    if target_column in frame.columns and normalized_target_values:
        if (
            positive_class is not None
            and positive_class_normalized not in normalized_target_values
        ):
            blocking.append(
                _build_issue(
                    "blocking",
                    "positive_class_not_in_target",
                    "Selected positive class does not match the normalized target values in the dataset.",
                    [target_column],
                )
            )

    high_cardinality_columns: list[str] = []
    quasi_unique_columns: list[str] = []
    leakage_columns: list[str] = []
    for column in frame.columns:
        if column == target_column:
            continue
        series = frame[column]
        unique_ratio = series.nunique(dropna=True) / max(len(frame), 1)
        lowered = column.lower().strip()
        if unique_ratio > 0.9:
            quasi_unique_columns.append(column)
        if (
            column not in numeric_columns
            and series.nunique(dropna=True) >= max(30, int(len(frame) * 0.5))
        ):
            high_cardinality_columns.append(column)
        if lowered in _LEAKAGE_NAMES or (
            target_column and target_column.lower().strip() in lowered and lowered != target_column.lower().strip()
        ):
            leakage_columns.append(column)

    if quasi_unique_columns:
        warnings.append(
            _build_issue(
                "warning",
                "quasi_unique_columns",
                "Some columns are nearly unique per row and are likely identifiers or free text.",
                sorted(quasi_unique_columns),
            )
        )
    if high_cardinality_columns:
        warnings.append(
            _build_issue(
                "warning",
                "high_cardinality_categorical_columns",
                "Some categorical columns have very high cardinality and may slow down training and XAI.",
                sorted(high_cardinality_columns),
            )
        )
    if leakage_columns:
        warnings.append(
            _build_issue(
                "warning",
                "possible_label_leakage_columns",
                "Some columns look like labels, predictions or outcomes and may leak the target.",
                sorted(leakage_columns),
            )
        )

    complexity = summarize_dataset_complexity(
        frame,
        target_column=target_column,
        numeric_columns=numeric_columns,
        excluded_columns=excluded_columns,
    )
    if complexity["shap_cost_level"] == "high":
        warnings.append(
            _build_issue(
                "warning",
                "high_expected_shap_cost",
                "This dataset is expected to produce expensive SHAP computations. XAI may be degraded or slowed down.",
            )
        )
    if complexity["split_risk"] == "blocking":
        blocking.append(
            _build_issue(
                "blocking",
                "split_risk_blocking",
                "Class distribution is too small for a safe stratified split.",
                [target_column] if target_column else None,
            )
        )
    elif complexity["split_risk"] == "warning":
        warnings.append(
            _build_issue(
                "warning",
                "split_risk_warning",
                "The smallest class is small. Validation/test metrics may be noisy.",
                [target_column] if target_column else None,
            )
        )

    info.append(
        _build_issue(
            "info",
            "product_scope",
            "Supported pipeline contract: binary classification, one target column, tabular rows only, no free-text modeling, no automatic multiclass support.",
        )
    )

    return {
        "valid": not blocking,
        "issues": [issue["message"] for issue in blocking + warnings + info],
        "blocking": blocking,
        "warning": warnings,
        "info": info,
        "num_rows": int(frame.shape[0]),
        "num_columns": int(frame.shape[1]),
        "columns": frame.columns.tolist(),
        "preview": frame.head(10),
        "detected_numeric_columns": detected_numeric,
        "detected_categorical_columns": inferred_categorical,
        "target_values": target_values,
        "normalized_target_values": normalized_target_values,
        "class_counts": class_counts,
        "class_balance": class_balance,
        "severity_counts": {
            "blocking": len(blocking),
            "warning": len(warnings),
            "info": len(info),
        },
        "excluded_columns_present": [
            column for column in excluded_columns if column in frame.columns
        ],
        "complexity": complexity,
    }


def build_runtime_config(
    *,
    dataset_path: str | Path,
    mode: str,
    target_column: str,
    positive_class: str,
    negative_classes: list[str],
    excluded_columns: list[str],
    numeric_columns: list[str],
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    original_dataset_name: str | None = None,
    fairness_attribute: str | None = None,
    condition_name: str | None = None,
    positive_label: str | None = None,
    negative_label: str | None = None,
) -> dict[str, Any]:
    if mode not in {"principal", "temporaire"}:
        raise ValueError(f"Unsupported mode: {mode}")

    timestamp_utc = utc_now()
    timestamp_token = (
        timestamp_utc.replace("-", "").replace(":", "").replace("+00:00", "Z")
    )
    timestamp_token = timestamp_token.replace("T", "").replace(".", "")
    run_id = f"{mode}-{timestamp_token[:14]}-{uuid.uuid4().hex[:8]}"
    artifact_root = Path("artifacts") / "user_runs" / run_id
    dataset_name = original_dataset_name or Path(dataset_path).name

    return {
        "run_id": run_id,
        "run_mode": mode,
        "dataset_name": dataset_name,
        "dataset_path": Path(dataset_path).as_posix(),
        "artifact_root": artifact_root.as_posix(),
        # Portal-triggered runs are dataset-specific. The Flower server, the
        # logical client and the downstream XAI/audit phases must all read the
        # exact same per-run metadata/schema unless the caller explicitly
        # overrides the server artifact root later (tests and custom orchestration
        # still do this on purpose).
        "global_artifact_root": artifact_root.as_posix(),
        "timestamp_utc": timestamp_utc,
        "config_override": {
            "data": {
                "raw_dataset_path": str(Path(dataset_path)),
                "target_column": target_column,
                "positive_class": positive_class,
                "negative_classes": list(negative_classes),
                "excluded_columns": list(excluded_columns),
                "numeric_columns": list(numeric_columns),
                # Audit phase reads these dynamically — default to the user's
                # own uploaded dataset so drift is measured against the
                # training distribution, not a hardcoded PIMA reference.
                "reference_dataset_path": str(Path(dataset_path)),
                "fairness_attribute": fairness_attribute or "",
            },
            "project": {
                "condition_name": (condition_name or "").strip(),
                "positive_label": (positive_label or "Positive case").strip()
                or "Positive case",
                "negative_label": (negative_label or "Negative case").strip()
                or "Negative case",
            },
            "federated_learning": {
                "num_clients": int(num_clients),
                "num_rounds": int(num_rounds),
                # Server blocks until all target clients are connected before
                # starting the first round — proper federated behaviour.
                "min_fit_clients": int(num_clients),
                "min_available_clients": int(num_clients),
            },
            "model": {
                "hyperparameters": {
                    "local_epochs": int(local_epochs),
                }
            },
            "artifacts": {
                "root": str(artifact_root),
                "models_dir": str(artifact_root / "models"),
                "metrics_dir": str(artifact_root / "metrics"),
                "xai_dir": str(artifact_root / "xai"),
                "audit_dir": str(artifact_root / "audit"),
                "reports_dir": str(artifact_root / "reports"),
            },
        },
    }


def persist_uploaded_dataset(
    *,
    file_name: str,
    content: bytes,
    mode: str,
    run_id: str,
) -> str:
    if mode not in {"principal", "temporaire"}:
        raise ValueError(f"Unsupported mode: {mode}")

    # Uploaded client datasets must stay in a writable runtime area.
    # In Docker, `/app/data` is mounted read-only, so uploads are persisted
    # alongside the user run artifacts regardless of the selected mode.
    target_dir = Path("artifacts") / "user_runs" / run_id / "input"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{run_id}-{Path(file_name).name}"
    target_path.write_bytes(content)
    return str(target_path)


def save_run_metadata(
    runtime_config: dict[str, Any], status: str, extra: dict[str, Any] | None = None
) -> Path:
    artifact_root = Path(runtime_config["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": runtime_config["run_id"],
        "run_mode": runtime_config["run_mode"],
        "dataset_name": runtime_config["dataset_name"],
        "dataset_path": runtime_config["dataset_path"],
        "timestamp_utc": runtime_config["timestamp_utc"],
        "artifact_root": runtime_config["artifact_root"],
        "status": status,
    }
    if extra:
        payload.update(extra)
    target_path = artifact_root / "run_manifest.json"
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target_path


def save_effective_config(runtime_config: dict[str, Any]) -> Path:
    target_root = Path(runtime_config["artifact_root"])
    target_root.mkdir(parents=True, exist_ok=True)
    base_config = load_project_config()
    apply_config_override(base_config, runtime_config["config_override"])
    config_path = target_root / "effective_config.json"
    config_path.write_text(json.dumps(base_config, indent=2), encoding="utf-8")
    return config_path


def run_phase2_for_mode(runtime_config: dict[str, Any]) -> dict[str, Any]:
    effective_config_path = save_effective_config(runtime_config)
    # Regenerate run-scoped metadata + splits from the uploaded CSV so the
    # portal run is genuinely dataset-agnostic (own schema, own transformers,
    # own train/val/test). Server and client will both load from this run dir.
    prepared = prepare_phase2(config_path=effective_config_path)

    manifest_path = save_run_metadata(
        runtime_config, "phase2_completed", {"stage": "phase2"}
    )
    metadata = prepared.metadata
    return {
        "manifest_path": str(manifest_path),
        "raw_shape": metadata["raw_shape"],
        "cleaned_shape": metadata["shape_after_missing_drop"],
        "train_shape": metadata["train_shape"],
        "validation_shape": metadata["validation_shape"],
        "test_shape": metadata["test_shape"],
        "dropped_columns": runtime_config["config_override"]["data"][
            "excluded_columns"
        ],
        "num_clients": int(
            runtime_config["config_override"]["federated_learning"]["num_clients"]
        ),
        "numeric_columns": metadata["numeric_columns"],
        "categorical_columns": metadata["categorical_columns"],
        "class_balance": metadata.get("class_balance", {}),
    }


def build_user_context(username: str, runtime_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "client_id": username.strip().lower(),
        "username": username.strip().lower(),
        "artifact_root": runtime_config["artifact_root"],
        "global_artifact_root": runtime_config.get("global_artifact_root", "artifacts"),
        "dataset_path": runtime_config["dataset_path"],
        "run_id": runtime_config["run_id"],
        "run_mode": runtime_config["run_mode"],
        "config_override": runtime_config["config_override"],
    }


def run_training_for_mode(
    runtime_config: dict[str, Any], user_context: dict[str, Any]
) -> dict[str, Any]:
    tracer = PipelineTracer(
        runtime_config["artifact_root"], run_id=runtime_config.get("run_id", "")
    )

    tracer.mark("load_config", "start")
    config = load_effective_config(runtime_config)
    tracer.mark("load_config", "end")

    deployment_mode = os.environ.get(
        "DEPLOYMENT_MODE",
        config.get("federated_learning", {}).get("deployment_mode", "subprocess"),
    )

    user_context["artifact_root"] = runtime_config["artifact_root"]
    if "global_artifact_root" not in user_context:
        user_context["global_artifact_root"] = runtime_config.get(
            "global_artifact_root", runtime_config["artifact_root"]
        )
    user_context["config_override"] = runtime_config["config_override"]

    tracer.mark("training_full", "start", detail=f"mode={deployment_mode}")
    if deployment_mode == "docker":
        payload = _run_docker_training(runtime_config, user_context)
    else:
        # subprocess and docker_per_user both go through run_local_training;
        # the docker_per_user dispatch happens inside that function via
        # the DEPLOYMENT_MODE env var.
        payload = run_local_training(user_context, config)
    tracer.mark("training_full", "end")

    save_run_metadata(runtime_config, "audit_completed", {"stage": "audit"})
    tracer.flush()
    _logger.info("run_training_for_mode timing: %s", tracer.summary())
    return payload


def load_effective_config(runtime_config: dict[str, Any]) -> dict[str, Any]:
    base_config = load_project_config()
    apply_config_override(base_config, runtime_config["config_override"])
    return base_config


def run_xai_for_mode(
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    artifact_root = runtime_config.get(
        "global_artifact_root", runtime_config["artifact_root"]
    )
    artifact_paths = build_artifact_paths(artifact_root)
    effective_config_path = Path(artifact_root) / "effective_config.json"
    if not effective_config_path.exists():
        effective_config_path = (
            Path(runtime_config["artifact_root"]) / "effective_config.json"
        )
    artifacts = run_phase4_explainability(
        config_path=str(effective_config_path),
        model_path=artifact_paths["model"],
        artifact_root=artifact_root,
    )
    save_run_metadata(runtime_config, "xai_completed", {"stage": "xai"})
    return artifacts.validation_summary


def run_audit_for_mode(runtime_config: dict[str, Any]) -> dict[str, Any]:
    artifact_root = runtime_config.get(
        "global_artifact_root", runtime_config["artifact_root"]
    )
    artifact_paths = build_artifact_paths(artifact_root)
    effective_config_path = Path(artifact_root) / "effective_config.json"
    if not effective_config_path.exists():
        effective_config_path = (
            Path(runtime_config["artifact_root"]) / "effective_config.json"
        )
    artifacts = run_phase5_audit(
        config_path=str(effective_config_path),
        model_path=artifact_paths["model"],
        artifact_root=artifact_root,
    )
    save_run_metadata(runtime_config, "audit_completed", {"stage": "audit"})
    return artifacts.validation_summary


def summarize_run_outputs(runtime_config: dict[str, Any]) -> dict[str, Any]:
    global_artifact_root = runtime_config.get(
        "global_artifact_root", runtime_config["artifact_root"]
    )
    local_paths = build_artifact_paths(runtime_config["artifact_root"])
    global_paths = build_artifact_paths(global_artifact_root)
    summary = {
        "run_id": runtime_config["run_id"],
        "run_mode": runtime_config["run_mode"],
        "dataset_name": runtime_config["dataset_name"],
        "artifact_root": runtime_config["artifact_root"],
        "global_artifact_root": global_artifact_root,
        "paths": {
            "local_metadata": str(local_paths["metadata"]),
            "model": str(global_paths["model"]),
            "metrics": str(global_paths["metrics"]),
            "xai_root": str(global_paths["xai_root"]),
            "audit_root": str(global_paths["audit_root"]),
        },
        "available": {},
    }
    for key, path in summary["paths"].items():
        summary["available"][key] = Path(path).exists()

    metrics_path = global_paths["metrics"]
    if metrics_path.exists():
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        latest_round = metrics_payload["rounds"][-1]
        summary["metrics"] = {
            "num_rounds": metrics_payload["num_rounds"],
            "num_clients": metrics_payload["num_clients"],
            "selected_threshold": metrics_payload["selected_threshold"],
            "global_accuracy": latest_round["global_test_metrics"]["accuracy"],
            "global_recall": latest_round["global_test_metrics"]["recall"],
            "global_f1": latest_round["global_test_metrics"]["f1"],
        }

    manifest_path = Path(runtime_config["artifact_root"]) / "run_manifest.json"
    if manifest_path.exists():
        summary["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return summary
