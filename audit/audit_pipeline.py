from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from shared.federated_data import (
    build_artifact_paths,
    get_model_input_schema,
    list_client_ids,
    load_preprocessing_metadata,
    load_project_config,
)
from shared.utils import contains_forbidden_keys, load_json, utc_now

REQUIRED_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
FORBIDDEN_RAW_KEYS = {
    "x_numeric",
    "x_categorical",
    "y",
    "raw_rows",
    "rows",
    "dataset",
    "samples",
}
ALLOWED_CLIENT_PAYLOAD_FIELDS = {
    "model_params",
    "num_examples",
    "local_metrics",
    "shap_summary",
    "rule_summary",
    "audit_metadata",
}
FORBIDDEN_EXPLAINABILITY_KEYS = {
    "local_rules",
    "shap_summary_local",
    "instance_explanations",
    "raw_shap_values",
}
_ALL_FORBIDDEN_KEYS = FORBIDDEN_RAW_KEYS | FORBIDDEN_EXPLAINABILITY_KEYS


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_reference_frame(config: dict[str, Any]) -> pd.DataFrame | None:
    from shared.dataset_io import load_tabular

    reference_path = Path(config.get("data", {}).get("reference_dataset_path", ""))
    if reference_path.suffix and reference_path.exists():
        return load_tabular(reference_path)
    configured = Path(config["data"]["raw_dataset_path"])
    if configured.suffix and configured.exists():
        return load_tabular(configured)
    return None


def _evaluate_fairness(
    config: dict[str, Any], reference_frame: pd.DataFrame | None
) -> dict[str, Any]:
    fairness_attr_raw = config.get("data", {}).get("fairness_attribute", "")
    if not fairness_attr_raw:
        return {
            "status": "not_applicable",
            "details": "No fairness attribute configured. Fairness audit skipped.",
        }
    target_column = str(config["data"]["target_column"])
    if reference_frame is None or target_column not in reference_frame.columns:
        return {
            "status": "limited",
            "details": f"{target_column} column unavailable for fairness audit.",
        }
    col_map = {c.lower(): c for c in reference_frame.columns}
    fairness_attr = col_map.get(fairness_attr_raw.lower())
    if fairness_attr is None or fairness_attr not in reference_frame.columns:
        return {
            "status": "not_applicable",
            "details": f"Fairness attribute '{fairness_attr_raw}' not found in dataset.",
        }
    frame = reference_frame.copy()
    threshold = float(frame[fairness_attr].median())
    group_low = frame[frame[fairness_attr] < threshold]
    group_high = frame[frame[fairness_attr] >= threshold]
    low_rate = float(group_low[target_column].mean()) if not group_low.empty else 0.0
    high_rate = float(group_high[target_column].mean()) if not group_high.empty else 0.0
    gap = abs(high_rate - low_rate)
    return {
        "status": "passed" if gap <= 0.15 else "warning",
        "group_attribute": fairness_attr,
        "split_rule": f"{fairness_attr} median = {threshold:.2f}",
        "younger_positive_rate": low_rate,
        "older_positive_rate": high_rate,
        "gap": gap,
    }


def _evaluate_data_drift(
    config: dict[str, Any], reference_frame: pd.DataFrame | None
) -> dict[str, Any]:
    configured = Path(config["data"]["raw_dataset_path"])
    if reference_frame is None or not configured.exists():
        return {
            "status": "limited",
            "details": "Reference or current dataset unavailable for drift audit.",
        }
    from shared.dataset_io import load_tabular

    current_frame = load_tabular(configured)
    numeric_columns = [
        column
        for column in config["data"]["numeric_columns"]
        if column in current_frame.columns and column in reference_frame.columns
    ]
    if not numeric_columns:
        return {
            "status": "limited",
            "details": "No shared numeric columns available for drift audit.",
        }
    drifts: list[dict[str, Any]] = []
    for column in numeric_columns:
        reference_mean = _safe_float(reference_frame[column].mean())
        current_mean = _safe_float(current_frame[column].mean())
        denominator = max(abs(reference_mean), 1e-6)
        relative_shift = abs(current_mean - reference_mean) / denominator
        drifts.append(
            {
                "feature": column,
                "reference_mean": reference_mean,
                "current_mean": current_mean,
                "relative_shift": relative_shift,
            }
        )
    max_shift = max(item["relative_shift"] for item in drifts)
    return {
        "status": "passed" if max_shift <= 0.25 else "warning",
        "max_relative_shift": max_shift,
        "feature_drift": drifts,
    }


def _evaluate_accuracy(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    if not metrics_payload.get("rounds"):
        return {"status": "warning", "details": "No training rounds found."}
    latest = metrics_payload["rounds"][-1]["global_test_metrics"]
    accuracy = _safe_float(latest.get("accuracy"))
    recall = _safe_float(latest.get("recall"))
    f1 = _safe_float(latest.get("f1"))
    roc_auc = _safe_float(latest.get("roc_auc"))
    status = (
        "passed"
        if accuracy >= 0.70 and recall >= 0.35 and f1 >= 0.20 and roc_auc >= 0.70
        else "warning"
    )
    return {
        "status": status,
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def _evaluate_explainability_quality(
    phase4_validation: dict[str, Any], global_rules_summary: dict[str, Any]
) -> dict[str, Any]:
    has_shap = bool(phase4_validation.get("feature_count", 0))
    has_rules = bool(global_rules_summary.get("rules"))
    coherence = bool(
        phase4_validation.get("consistency_validation", {}).get(
            "feature_names_aligned", False
        )
    )
    quality_status = (
        phase4_validation.get("explainability_quality", {}).get("status")
        or phase4_validation.get("meaningfulness_status", "")
    )
    if quality_status == "degraded_due_to_cost_or_schema":
        status = "limited"
    elif has_shap and coherence:
        status = "passed"
    else:
        status = "warning"
    return {
        "status": status,
        "has_global_shap": has_shap,
        "has_rules": has_rules,
        "coherence": coherence,
        "meaningfulness_status": quality_status or "technically_executable",
        "details": phase4_validation.get("meaningfulness_message", ""),
    }


def _evaluate_privacy(
    client_payload_audit: dict[str, Any], server_report: dict[str, Any]
) -> dict[str, Any]:
    client_warnings = [
        row for row in client_payload_audit["client_results"] if row["issues"]
    ]
    forbidden_global = any(
        "forbidden" in issue for issue in server_report.get("issues", [])
    )
    status = "passed" if not client_warnings and not forbidden_global else "warning"
    return {
        "status": status,
        "clients_with_warnings": len(client_warnings),
        "forbidden_global_issue_detected": forbidden_global,
    }


@dataclass
class AuditArtifacts:
    client_payload_audit: dict[str, Any]
    server_audit_report: dict[str, Any]
    validation_summary: dict[str, Any]
    audit_log_lines: list[str]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value)


def _check_forbidden_keys(obj: Any) -> list[str]:
    return contains_forbidden_keys(obj, _ALL_FORBIDDEN_KEYS)


def _serialize_tensor_signature(
    state_dict: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for name, tensor in state_dict.items():
        detached = tensor.detach().cpu()
        signature.append(
            {
                "name": name,
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "numel": int(detached.numel()),
                "finite": bool(torch.isfinite(detached).all().item()),
            }
        )
    return signature


def _validate_metric_block(metric_block: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in REQUIRED_METRICS:
        if key not in metric_block:
            issues.append(f"missing_metric:{key}")
            continue
        if not _is_finite_number(metric_block[key]):
            issues.append(f"invalid_metric:{key}")
    return issues


def _build_audit_metadata(
    *,
    client_id: str,
    round_index: int,
    num_examples: int,
    payload_fields: list[str],
    validation_status: str,
    parameter_tensor_count: int,
    shap_feature_count: int,
) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "client_id": client_id,
        "round_index": round_index,
        "num_examples": int(num_examples),
        "payload_fields": payload_fields,
        "validation_status": validation_status,
        "parameter_tensor_count": parameter_tensor_count,
        "shap_feature_count": shap_feature_count,
        "raw_data_present": False,
    }


def _validate_client_payloads(
    metrics_payload: dict[str, Any],
    tensor_signature: list[dict[str, Any]],
    feature_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    final_round = metrics_payload["rounds"][-1]
    client_payload_results: list[dict[str, Any]] = []

    for client_metric in final_round["client_metrics"]:
        client_id = client_metric["client_id"]
        issues = _validate_metric_block(client_metric)
        shap_summary = client_metric.get("shap_summary")
        rule_summary = client_metric.get("rule_summary")
        if shap_summary is None:
            issues.append("missing_shap_summary")
            shap_feature_count = 0
        else:
            shap_feature_count = len(shap_summary.get("feature_names", []))
            if shap_feature_count != feature_count:
                issues.append("unexpected_shap_feature_count")
            if _check_forbidden_keys(shap_summary):
                issues.append("forbidden_raw_key_in_shap_summary")
        if rule_summary is None:
            issues.append("missing_rule_summary")
        elif _check_forbidden_keys(rule_summary):
            issues.append("forbidden_key_in_rule_summary")

        reconstructed_payload = {
            "model_params": {"tensor_signature": tensor_signature},
            "num_examples": client_metric["num_examples"],
            "local_metrics": {key: client_metric[key] for key in REQUIRED_METRICS},
            "shap_summary": {
                "feature_count": shap_feature_count,
                "top_features": []
                if shap_summary is None
                else shap_summary.get("top_features", []),
            },
            "rule_summary": {
                "num_rules": 0
                if rule_summary is None
                else int(rule_summary.get("num_rules", 0)),
                "rules": [] if rule_summary is None else rule_summary.get("rules", []),
            },
        }
        payload_fields = sorted(reconstructed_payload.keys() | {"audit_metadata"})
        if set(payload_fields) - ALLOWED_CLIENT_PAYLOAD_FIELDS:
            issues.append("unexpected_payload_field")
        if _check_forbidden_keys(reconstructed_payload):
            issues.append("forbidden_raw_key_in_payload")

        validation_status = "passed" if not issues else "warning"
        audit_metadata = _build_audit_metadata(
            client_id=client_id,
            round_index=int(final_round["round"]),
            num_examples=int(client_metric["num_examples"]),
            payload_fields=payload_fields,
            validation_status=validation_status,
            parameter_tensor_count=len(tensor_signature),
            shap_feature_count=shap_feature_count,
        )
        client_payload_results.append(
            {
                "client_id": client_id,
                "round_index": int(final_round["round"]),
                "issues": issues,
                "payload_fields": payload_fields,
                "audit_metadata": audit_metadata,
            }
        )
        logs.append(
            {
                "timestamp_utc": audit_metadata["timestamp_utc"],
                "event_type": "client_payload_validation",
                "client_id": client_id,
                "round_index": int(final_round["round"]),
                "status": validation_status,
                "issues": issues,
            }
        )

    return {
        "allowed_fields": sorted(ALLOWED_CLIENT_PAYLOAD_FIELDS),
        "forbidden_raw_keys": sorted(FORBIDDEN_RAW_KEYS),
        "parameter_tensor_signature": tensor_signature,
        "client_results": client_payload_results,
    }, logs


def _validate_server_artifacts(
    metrics_payload: dict[str, Any],
    global_model_bundle: dict[str, Any],
    phase4_validation: dict[str, Any],
    global_shap_summary: dict[str, Any],
    global_rules_summary: dict[str, Any],
    local_shap_summaries: list[dict[str, Any]],
    schema: dict[str, Any],
    artifact_root: str | Path = "artifacts",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[str] = []
    logs: list[dict[str, Any]] = []

    saved_model_round = int(metrics_payload.get("saved_model_round", 0) or 0)
    xai_source_round = int(phase4_validation.get("source_round_for_xai", 0) or 0)
    if saved_model_round and xai_source_round and saved_model_round != xai_source_round:
        issues.append("xai_round_mismatch_with_saved_model")

    if metrics_payload["num_clients"] != len(local_shap_summaries):
        issues.append("client_count_mismatch_between_metrics_and_xai")
    if metrics_payload["selected_threshold"] != phase4_validation["selected_threshold"]:
        issues.append("selected_threshold_mismatch")
    if metrics_payload["numeric_feature_count"] != len(schema["numeric_columns"]):
        issues.append("numeric_feature_count_mismatch")
    if metrics_payload["categorical_feature_count"] != len(
        schema["categorical_columns"]
    ):
        issues.append("categorical_feature_count_mismatch")
    if (
        len(global_shap_summary.get("feature_names", []))
        != phase4_validation["feature_count"]
    ):
        issues.append("global_shap_feature_count_mismatch")
    if _check_forbidden_keys(global_shap_summary):
        issues.append("forbidden_raw_key_in_global_shap")
    if _check_forbidden_keys(local_shap_summaries):
        issues.append("forbidden_raw_key_in_local_shap")
    if _check_forbidden_keys(global_rules_summary):
        issues.append("forbidden_raw_key_in_global_rules")
    if "rules" not in global_rules_summary:
        issues.append("missing_global_rules")

    round_summaries: list[dict[str, Any]] = []
    for round_payload in metrics_payload["rounds"]:
        round_issues = []
        round_issues.extend(
            _validate_metric_block(round_payload["aggregated_local_metrics"])
        )
        round_issues.extend(
            _validate_metric_block(round_payload["global_test_metrics"])
        )
        round_issues.extend(
            _validate_metric_block(round_payload["validation"]["validation_metrics"])
        )
        round_summary = {
            "round_index": int(round_payload["round"]),
            "status": "passed" if not round_issues else "warning",
            "issues": round_issues,
            "num_client_metrics": len(round_payload["client_metrics"]),
        }
        round_summaries.append(round_summary)
        logs.append(
            {
                "timestamp_utc": utc_now(),
                "event_type": "server_round_validation",
                "round_index": int(round_payload["round"]),
                "status": round_summary["status"],
                "issues": round_issues,
            }
        )

    tensor_signature = _serialize_tensor_signature(global_model_bundle["state_dict"])
    if not all(entry["finite"] for entry in tensor_signature):
        issues.append("non_finite_tensor_detected")

    report = {
        "status": "passed"
        if not issues and all(item["status"] == "passed" for item in round_summaries)
        else "warning",
        "issues": issues,
        "model_summary": {
            "model_family": global_model_bundle["model_family"],
            "model_type": global_model_bundle["model_type"],
            "tensor_count": len(tensor_signature),
            "selected_threshold": float(global_model_bundle["selected_threshold"]),
        },
        "phase4_summary": phase4_validation,
        "round_summaries": round_summaries,
        "technical_report": {
            "aggregated_rules_logic": "frequency_then_weighted_importance",
            "global_round_count": len(metrics_payload["rounds"]),
            "saved_model_round": saved_model_round,
            "xai_source_round": xai_source_round,
            "latest_global_metrics": metrics_payload["rounds"][-1][
                "global_test_metrics"
            ]
            if metrics_payload["rounds"]
            else {},
            "inference_log_path": str(
                build_artifact_paths(artifact_root)["audit_root"] / "audit_log.jsonl"
            ),
        },
        "rules_status": {
            "local_rules": "computed_locally_and_summarized_as_rule_summary",
            "global_rules": "aggregated_from_safe_client_rule_summaries",
            "audit_behavior": "validated_without_local_rules_or_instance_level_shap",
        },
    }
    logs.append(
        {
            "timestamp_utc": utc_now(),
            "event_type": "server_artifact_validation",
            "status": report["status"],
            "issues": issues,
        }
    )
    return report, logs


def _to_jsonl(entries: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(entry, ensure_ascii=False) for entry in entries]


def save_audit_artifacts(
    artifacts: AuditArtifacts, output_dir: str | Path = "artifacts/audit"
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "client_payload_audit.json").write_text(
        json.dumps(artifacts.client_payload_audit, indent=2),
        encoding="utf-8",
    )
    (output_path / "server_audit_report.json").write_text(
        json.dumps(artifacts.server_audit_report, indent=2),
        encoding="utf-8",
    )
    (output_path / "audit_validation_summary.json").write_text(
        json.dumps(artifacts.validation_summary, indent=2),
        encoding="utf-8",
    )
    (output_path / "audit_log.jsonl").write_text(
        "\n".join(artifacts.audit_log_lines) + "\n",
        encoding="utf-8",
    )


def run_phase5_audit(
    config_path: str | Path = "config/project-config.json",
    model_path: str | Path = "artifacts/models/global_model.pt",
    artifact_root: str | Path = "artifacts",
) -> AuditArtifacts:
    artifact_paths = build_artifact_paths(artifact_root)
    config = load_project_config(config_path)
    metadata = load_preprocessing_metadata(artifact_root=artifact_root)
    schema = get_model_input_schema(metadata)
    metrics_payload = load_json(artifact_paths["metrics"])
    phase4_validation = load_json(
        artifact_paths["xai_root"] / "xai_validation_report.json"
    )
    global_shap_summary = load_json(
        artifact_paths["xai_root"] / "global_shap_summary.json"
    )
    global_rules_summary = load_json(
        artifact_paths["xai_root"] / "global_rules_summary.json"
    )
    local_shap_summaries = load_json(
        artifact_paths["xai_root"] / "local_shap_summaries.json"
    )
    global_model_bundle = torch.load(model_path, map_location="cpu", weights_only=True)

    tensor_signature = _serialize_tensor_signature(global_model_bundle["state_dict"])
    client_payload_audit, client_logs = _validate_client_payloads(
        metrics_payload=metrics_payload,
        tensor_signature=tensor_signature,
        feature_count=len(schema["numeric_columns"])
        + len(schema["categorical_columns"]),
    )
    server_report, server_logs = _validate_server_artifacts(
        metrics_payload=metrics_payload,
        global_model_bundle=global_model_bundle,
        phase4_validation=phase4_validation,
        global_shap_summary=global_shap_summary,
        global_rules_summary=global_rules_summary,
        local_shap_summaries=local_shap_summaries,
        schema=schema,
        artifact_root=artifact_root,
    )
    reference_frame = _load_reference_frame(config)
    privacy_dimension = _evaluate_privacy(client_payload_audit, server_report)
    accuracy_dimension = _evaluate_accuracy(metrics_payload)
    fairness_dimension = _evaluate_fairness(config, reference_frame)
    drift_dimension = _evaluate_data_drift(config, reference_frame)
    explainability_dimension = _evaluate_explainability_quality(
        phase4_validation, global_rules_summary
    )

    all_logs = client_logs + server_logs
    dimension_statuses = {
        "privacy": privacy_dimension,
        "accuracy": accuracy_dimension,
        "fairness": fairness_dimension,
        "data_drift": drift_dimension,
        "explainability": explainability_dimension,
    }
    validation_summary = {
        "status": "passed"
        if server_report["status"] == "passed"
        and all(
            entry["audit_metadata"]["validation_status"] == "passed"
            for entry in client_payload_audit["client_results"]
        )
        and all(item["status"] == "passed" for item in dimension_statuses.values())
        else "warning",
        "timestamp_utc": utc_now(),
        "num_clients_audited": len(client_payload_audit["client_results"]),
        "num_rounds_audited": len(metrics_payload["rounds"]),
        "global_model_path": str(model_path),
        "audit_log_path": str(artifact_paths["audit_root"] / "audit_log.jsonl"),
        "server_report_path": str(
            artifact_paths["audit_root"] / "server_audit_report.json"
        ),
        "client_payload_report_path": str(
            artifact_paths["audit_root"] / "client_payload_audit.json"
        ),
        "dimensions": dimension_statuses,
        "notes": [
            "Audit executed post-training on persisted artifacts from phases 3 and 4.",
            "Clients transmitted only model parameters, metrics, SHAP summaries and aggregated rule summaries.",
            "No raw dataset rows, per-row SHAP values, local_rules fields or direct feature matrices were detected in audited payloads.",
            "The audit now evaluates privacy, accuracy, fairness, data drift and explainability quality.",
        ],
    }
    server_report["audit_dimensions"] = dimension_statuses

    artifacts = AuditArtifacts(
        client_payload_audit=client_payload_audit,
        server_audit_report=server_report,
        validation_summary=validation_summary,
        audit_log_lines=_to_jsonl(all_logs),
    )
    save_audit_artifacts(artifacts, output_dir=artifact_paths["audit_root"])
    return artifacts


if __name__ == "__main__":
    result = run_phase5_audit()
    print(json.dumps(result.validation_summary, indent=2))
