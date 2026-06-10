from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from audit.audit_pipeline import run_phase5_audit
from explainability.shap_explainer import run_phase4_explainability
from server.federated_trainer import run_local_distributed_training
from shared.data_preparation import prepare_phase2
from shared.federated_data import load_project_config
from shared.utils import find_free_port
from shared.utils import apply_config_override


_DIABETES_OVERRIDE = {
    "data": {
        "raw_dataset_path": "data/diabetes.csv",
        "target_column": "Outcome",
        "positive_class": 1,
        "negative_classes": [0],
        "numeric_columns": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        "fairness_attribute": "Age",
        "reference_dataset_path": "data/diabetes.csv",
    },
    "project": {
        "dataset_name": "pima-diabetes",
        "condition_name": "diabetes",
        "positive_label": "Diabetic patient",
        "negative_label": "Non-diabetic patient",
    },
}
_DIABETES_PRESET_PATH = "config/presets/diabetes.json"


def ensure_phase2_artifacts() -> None:
    required = [
        Path("artifacts/data/metadata/preprocessing_metadata.json"),
        Path("artifacts/data/splits/train.npz"),
        Path("artifacts/data/splits/validation.npz"),
        Path("artifacts/data/splits/test.npz"),
    ]
    if all(path.exists() for path in required):
        metadata = json.loads(required[0].read_text(encoding="utf-8"))
        if (
            metadata.get("target_column") == "Outcome"
            and len(metadata.get("numeric_columns", [])) == 8
        ):
            return
    prepare_phase2(config_override=_DIABETES_OVERRIDE)


def ensure_training_artifacts() -> None:
    ensure_phase2_artifacts()
    required = [
        Path("artifacts/models/global_model.pt"),
        Path("artifacts/metrics/federated_training_metrics.json"),
    ]
    if all(path.exists() for path in required):
        metrics_payload = json.loads(required[1].read_text(encoding="utf-8"))
        if metrics_payload.get("rounds"):
            final_round = metrics_payload["rounds"][-1]
            if (
                metrics_payload.get("numeric_feature_count") == 8
                and final_round["client_metrics"]
                and "shap_summary" in final_round["client_metrics"][0]
                and "rule_summary" in final_round["client_metrics"][0]
            ):
                return
    run_local_distributed_training(config_override=_DIABETES_OVERRIDE)


def ensure_xai_artifacts() -> None:
    ensure_training_artifacts()
    required = [
        Path("artifacts/xai/global_shap_summary.json"),
        Path("artifacts/xai/global_rules_summary.json"),
        Path("artifacts/xai/local_shap_summaries.json"),
        Path("artifacts/xai/instance_explanations.json"),
        Path("artifacts/xai/xai_validation_report.json"),
    ]
    if all(path.exists() for path in required):
        validation = json.loads(required[-1].read_text(encoding="utf-8"))
        if validation.get("feature_count") == 8:
            return
    run_phase4_explainability(
        config_path=_DIABETES_PRESET_PATH, artifact_root="artifacts"
    )


def ensure_audit_artifacts() -> None:
    ensure_xai_artifacts()
    required = [
        Path("artifacts/audit/client_payload_audit.json"),
        Path("artifacts/audit/server_audit_report.json"),
        Path("artifacts/audit/audit_log.jsonl"),
        Path("artifacts/audit/audit_validation_summary.json"),
    ]
    if all(path.exists() for path in required):
        summary = json.loads(required[-1].read_text(encoding="utf-8"))
        if "dimensions" in summary:
            return
    run_phase5_audit(
        config_path=_DIABETES_PRESET_PATH, artifact_root="artifacts"
    )


def launch_fl_server(config_override: dict) -> tuple[subprocess.Popen[str], str]:
    prepare_phase2(config_override=config_override)
    artifact_root = Path(config_override.get("artifacts", {}).get("root", "artifacts"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    effective_config = load_project_config(_DIABETES_PRESET_PATH)
    apply_config_override(effective_config, config_override)
    (artifact_root / "effective_config.json").write_text(
        json.dumps(effective_config, indent=2),
        encoding="utf-8",
    )
    address = f"127.0.0.1:{find_free_port()}"
    env = {
        **dict(os.environ),
        "FL_SERVER_ADDRESS": address,
        "CONFIG_OVERRIDE_JSON": json.dumps(config_override),
        "FL_FORCE_EXIT_ON_COMPLETE": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "server.run_training"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    return process, address


def wait_for_port(
    host: str, port: int, timeout: float = 30.0, interval: float = 0.5
) -> None:
    """Wait until a TCP port is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=2):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(interval)
    raise TimeoutError(f"Serveur sur {host}:{port} pas pret apres {timeout}s")
