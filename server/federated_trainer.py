from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, start_server
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from shared.aggregation import (
    aggregate_rules as aggregate_rule_summaries,
    aggregate_shap_summaries,
    extract_rule_entries,
    extract_shap_summaries,
)
from shared.federated_data import (
    build_artifact_paths,
    get_model_input_schema,
    load_preprocessing_metadata,
    load_project_config,
    load_test_dataset,
    load_train_dataset,
    load_validation_dataset,
    partition_dataset_for_clients,
)
from shared.modeling import (
    create_model,
    evaluate_predictions,
    get_model_parameters,
    predict_probabilities,
    select_decision_threshold,
    set_model_parameters,
)
from shared.utils import (
    apply_config_override,
    contains_forbidden_keys,
    find_free_port,
    PipelineTracer,
)

_logger = logging.getLogger("server.federated_trainer")

FORBIDDEN_CLIENT_METRIC_KEYS = {
    "local_rules",
    "shap_summary_local",
    "instance_explanations",
    "raw_shap_values",
    "x_numeric",
    "x_categorical",
    "y",
}
NUMERIC_METRIC_KEYS = (
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
)


def aggregate_fit_metrics(
    metrics: list[tuple[int, dict[str, Any]]],
) -> dict[str, float]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}
    aggregated: dict[str, float] = {}
    for key in NUMERIC_METRIC_KEYS:
        weighted_sum = 0.0
        for num_examples, metric_dict in metrics:
            weighted_sum += float(metric_dict[key]) * num_examples
        aggregated[key] = weighted_sum / total_examples
    return aggregated


def _decode_json_metric(name: str, raw_value: Any) -> dict[str, Any]:
    if not isinstance(raw_value, str):
        raise ValueError(
            f"Expected serialized JSON string for {name}, got {type(raw_value).__name__}"
        )
    decoded = json.loads(raw_value)
    if not isinstance(decoded, dict):
        raise ValueError(f"Objet JSON decode attendu pour {name}")
    forbidden = contains_forbidden_keys(decoded, FORBIDDEN_CLIENT_METRIC_KEYS)
    if forbidden:
        raise ValueError(f"Cles interdites detectees dans {name} : {forbidden}")
    return decoded


def _validate_client_metric_keys(metric_dict: Metrics) -> None:
    forbidden = sorted(set(metric_dict).intersection(FORBIDDEN_CLIENT_METRIC_KEYS))
    if forbidden:
        raise ValueError(
            f"Champs interdits dans les metriques client recus par le serveur : {forbidden}"
        )
    required = {
        "client_id",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "threshold",
        "shap_summary",
        "rule_summary",
    }
    missing = sorted(required.difference(metric_dict))
    if missing:
        raise ValueError(
            f"Champs obligatoires manquants dans les metriques client : {missing}"
        )


def parse_client_fit_metrics(num_examples: int, metric_dict: Metrics) -> dict[str, Any]:
    _validate_client_metric_keys(metric_dict)
    parsed = {
        "client_id": str(metric_dict["client_id"]),
        "num_examples": int(num_examples),
        "loss": float(metric_dict["loss"]),
        "accuracy": float(metric_dict["accuracy"]),
        "precision": float(metric_dict["precision"]),
        "recall": float(metric_dict["recall"]),
        "f1": float(metric_dict["f1"]),
        "roc_auc": float(metric_dict["roc_auc"]),
        "pr_auc": float(metric_dict["pr_auc"]),
        "threshold": float(metric_dict["threshold"]),
        "shap_summary": _decode_json_metric(
            "shap_summary", metric_dict["shap_summary"]
        ),
        "rule_summary": _decode_json_metric(
            "rule_summary", metric_dict["rule_summary"]
        ),
    }
    raw_aug = metric_dict.get("augmentation_summary")
    if raw_aug:
        parsed["augmentation_summary"] = _decode_json_metric(
            "augmentation_summary", raw_aug
        )
    return parsed


class DistributedFedAvgStrategy(FedAvg):
    def __init__(
        self,
        config: dict[str, Any],
        artifact_root: str | Path,
        force_exit_on_complete: bool = False,
    ) -> None:
        self.config = config
        self.artifact_root = str(artifact_root)
        self.artifact_paths = build_artifact_paths(self.artifact_root)
        self.metadata = load_preprocessing_metadata(artifact_root=self.artifact_root)
        self.schema = get_model_input_schema(self.metadata)
        self.configured_client_target = int(
            self.config["federated_learning"]["num_clients"]
        )
        self.min_fit_clients = max(
            1,
            int(
                self.config["federated_learning"].get(
                    "min_fit_clients", self.configured_client_target
                )
            ),
        )
        self.min_available_clients = max(
            self.min_fit_clients,
            int(
                self.config["federated_learning"].get(
                    "min_available_clients", self.min_fit_clients
                )
            ),
        )
        self.server_model = create_model(self.config, schema=self.schema)
        self.global_parameters_ndarrays = get_model_parameters(self.server_model)
        self.validation_dataset = load_validation_dataset(
            artifact_root=self.artifact_root
        )
        self.test_dataset = load_test_dataset(artifact_root=self.artifact_root)
        # Train set is kept for threshold-selection fallback when the
        # validation split is single-class (severely imbalanced datasets).
        try:
            self.train_dataset = load_train_dataset(artifact_root=self.artifact_root)
        except FileNotFoundError:
            self.train_dataset = None
        self.selected_threshold = float(self.config["training"]["default_threshold"])
        self.history: list[dict[str, Any]] = []
        self.force_exit_on_complete = force_exit_on_complete
        self._artifacts_saved = False
        # Best-model tracking: persist the round with the highest validation accuracy
        # so the saved model is not always the last round (which may have overfit).
        self._best_val_accuracy: float = -1.0
        self._best_parameters: list[np.ndarray] | None = None
        self._best_threshold: float = self.selected_threshold
        self._best_val_accuracy_round: int = 0

        super().__init__(
            fraction_fit=1.0,
            min_fit_clients=self.min_fit_clients,
            min_available_clients=self.min_available_clients,
            min_evaluate_clients=0,
            fraction_evaluate=0.0,
            initial_parameters=ndarrays_to_parameters(self.global_parameters_ndarrays),
            on_fit_config_fn=self._fit_config,
            accept_failures=False,
        )

    def _fit_config(self, server_round: int) -> dict[str, int]:
        return {
            "server_round": int(server_round),
            "local_epochs": int(
                self.config["model"]["hyperparameters"]["local_epochs"]
            ),
        }

    def _evaluate_global_model(
        self, parameters: list[np.ndarray], threshold: float
    ) -> dict[str, float]:
        set_model_parameters(self.server_model, parameters)
        probabilities = predict_probabilities(
            self.server_model,
            self.test_dataset.x_numeric,
            self.test_dataset.x_categorical,
        )
        evaluation = evaluate_predictions(
            self.test_dataset.y, probabilities, threshold=threshold
        )
        return {"loss": evaluation.loss, **evaluation.metrics}

    def _select_threshold(
        self, parameters: list[np.ndarray]
    ) -> tuple[float, dict[str, Any]]:
        set_model_parameters(self.server_model, parameters)
        probabilities = predict_probabilities(
            self.server_model,
            self.validation_dataset.x_numeric,
            self.validation_dataset.x_categorical,
        )
        fallback_y, fallback_proba = None, None
        if self.train_dataset is not None:
            fallback_y = self.train_dataset.y
            fallback_proba = predict_probabilities(
                self.server_model,
                self.train_dataset.x_numeric,
                self.train_dataset.x_categorical,
            )
        best_threshold, threshold_metrics = select_decision_threshold(
            self.validation_dataset.y,
            probabilities,
            candidate_thresholds=self.config["training"].get("threshold_candidates"),
            selection_strategy=str(
                self.config["training"].get("threshold_selection_strategy", "f1")
            ),
            min_recall=float(
                self.config["training"].get("threshold_selection_min_recall", 0.0)
            ),
            fallback_y_true=fallback_y,
            fallback_probabilities=fallback_proba,
        )
        validation_evaluation = evaluate_predictions(
            self.validation_dataset.y, probabilities, threshold=best_threshold
        )
        return best_threshold, {
            "selected_threshold": best_threshold,
            "selection_metrics": threshold_metrics,
            "validation_metrics": {
                "loss": validation_evaluation.loss,
                **validation_evaluation.metrics,
            },
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, Any]],
        failures: list[Any],
    ) -> tuple[Any, dict[str, Any]]:
        aggregated_parameters, _ = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is None:
            return None, {}

        self.global_parameters_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        client_round_metrics = [
            parse_client_fit_metrics(fit_res.num_examples, fit_res.metrics)
            for _, fit_res in results
        ]
        aggregated_local_metrics = aggregate_fit_metrics(
            [(entry["num_examples"], entry) for entry in client_round_metrics]
        )
        selected_threshold, validation_payload = self._select_threshold(
            self.global_parameters_ndarrays
        )
        self.selected_threshold = selected_threshold
        val_accuracy = validation_payload["validation_metrics"]["accuracy"]
        if val_accuracy > self._best_val_accuracy:
            self._best_val_accuracy = val_accuracy
            self._best_parameters = [p.copy() for p in self.global_parameters_ndarrays]
            self._best_threshold = selected_threshold
            self._best_val_accuracy_round = int(server_round)
        global_metrics = self._evaluate_global_model(
            self.global_parameters_ndarrays, threshold=selected_threshold
        )
        aggregated_shap_summary = aggregate_shap_summaries(
            extract_shap_summaries(client_round_metrics)
        )
        aggregated_rule_summary = aggregate_rule_summaries(
            extract_rule_entries(client_round_metrics)
        )

        self.history.append(
            {
                "round": int(server_round),
                "num_clients_participated": len(client_round_metrics),
                "participating_client_ids": [
                    str(entry["client_id"]) for entry in client_round_metrics
                ],
                "aggregated_local_metrics": aggregated_local_metrics,
                "validation": validation_payload,
                "global_test_metrics": global_metrics,
                "aggregated_shap_summary": aggregated_shap_summary,
                "aggregated_rule_summary": aggregated_rule_summary,
                "client_metrics": client_round_metrics,
            }
        )
        if server_round >= int(self.config["federated_learning"]["num_rounds"]):
            self.save_artifacts()
            if self.force_exit_on_complete:
                threading.Thread(
                    target=self._exit_process_after_delay, daemon=True
                ).start()
        return aggregated_parameters, {"selected_threshold": float(selected_threshold)}

    def build_metrics_payload(
        self, selected_threshold: float | None = None
    ) -> dict[str, Any]:
        observed_client_ids = sorted(
            {
                str(client_metric["client_id"])
                for round_payload in self.history
                for client_metric in round_payload.get("client_metrics", [])
            }
        )
        return {
            "model_family": self.config["model"]["family"],
            "model_type": self.config["model"]["type"],
            "numeric_feature_count": len(self.schema["numeric_columns"]),
            "categorical_feature_count": len(self.schema["categorical_columns"]),
            "categorical_cardinalities": self.schema["categorical_cardinalities"],
            "embedding_dimensions": self.schema["embedding_dimensions"],
            "num_clients": len(observed_client_ids)
            if observed_client_ids
            else self.min_fit_clients,
            "configured_client_target": self.configured_client_target,
            "min_fit_clients": self.min_fit_clients,
            "min_available_clients": self.min_available_clients,
            "num_rounds": int(self.config["federated_learning"]["num_rounds"]),
            "saved_model_round": int(self._best_val_accuracy_round),
            "selected_threshold": float(
                self.selected_threshold
                if selected_threshold is None
                else selected_threshold
            ),
            "final_round_selected_threshold": float(self.selected_threshold),
            "observed_client_ids": observed_client_ids,
            "rounds": self.history,
        }

    def save_artifacts(self) -> dict[str, Any]:
        tracer = PipelineTracer(self.artifact_root)

        tracer.mark("save_model_and_metrics", "start")
        models_dir = Path(self.config["artifacts"]["models_dir"])
        metrics_dir = Path(self.config["artifacts"]["metrics_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Use best-round parameters when available; fall back to final round.
        save_parameters = (
            self._best_parameters
            if self._best_parameters is not None
            else self.global_parameters_ndarrays
        )
        save_threshold = (
            self._best_threshold
            if self._best_parameters is not None
            else self.selected_threshold
        )
        set_model_parameters(self.server_model, save_parameters)
        payload = self.build_metrics_payload(selected_threshold=save_threshold)
        torch.save(
            {
                "state_dict": self.server_model.state_dict(),
                "schema": self.schema,
                "model_family": self.config["model"]["family"],
                "model_type": self.config["model"]["type"],
                "target_name": "target",
                "selected_threshold": save_threshold,
                "hyperparameters": self.config["model"]["hyperparameters"],
            },
            models_dir / "global_model.pt",
        )
        with (metrics_dir / "federated_training_metrics.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(payload, fh, indent=2)
        tracer.mark("save_model_and_metrics", "end")

        from audit.audit_pipeline import run_phase5_audit
        from explainability.shap_explainer import run_phase4_explainability

        model_path = models_dir / "global_model.pt"
        effective_config_path = Path(self.artifact_root) / "effective_config.json"
        config_path: str | Path = (
            effective_config_path
            if effective_config_path.exists()
            else "config/project-config.json"
        )

        tracer.mark("phase4_xai", "start")
        run_phase4_explainability(
            config_path=config_path,
            model_path=model_path,
            artifact_root=self.artifact_root,
        )
        tracer.mark("phase4_xai", "end")

        tracer.mark("phase5_audit", "start")
        run_phase5_audit(
            config_path=config_path,
            model_path=model_path,
            artifact_root=self.artifact_root,
        )
        tracer.mark("phase5_audit", "end")

        tracer.flush()
        _logger.info("save_artifacts timing: %s", tracer.summary())

        self._artifacts_saved = True
        return payload

    @staticmethod
    def _exit_process_after_delay() -> None:
        time.sleep(2.0)
        os._exit(0)


def run_distributed_training(
    config_path: str | Path = "config/project-config.json",
    config_override: dict[str, Any] | None = None,
    server_address: str = "0.0.0.0:8080",
    grpc_max_message_length: int = 2147483647,
    force_exit_on_complete: bool = False,
) -> dict[str, Any]:
    config = load_project_config(config_path)
    if config_override:
        apply_config_override(config, config_override)
    artifact_root = str(config["artifacts"]["root"])
    artifact_root_path = Path(artifact_root)
    artifact_root_path.mkdir(parents=True, exist_ok=True)
    (artifact_root_path / "effective_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    strategy = DistributedFedAvgStrategy(
        config=config,
        artifact_root=artifact_root,
        force_exit_on_complete=force_exit_on_complete,
    )
    round_timeout = float(
        config.get("federated_learning", {}).get("round_timeout_seconds", 300.0)
    )
    start_server(
        server_address=server_address,
        config=ServerConfig(
            num_rounds=int(config["federated_learning"]["num_rounds"]),
            round_timeout=round_timeout,
        ),
        strategy=strategy,
        grpc_max_message_length=grpc_max_message_length,
    )
    if strategy._artifacts_saved:
        return strategy.build_metrics_payload()
    return strategy.save_artifacts()


def run_local_distributed_training(
    config_override: dict[str, Any] | None = None,
    client_ids: list[str] | None = None,
) -> dict[str, Any]:
    config = load_project_config()
    if config_override:
        apply_config_override(config, config_override)
    if client_ids is None:
        client_ids = [
            f"client_user_{index + 1}"
            for index in range(int(config["federated_learning"]["num_clients"]))
        ]

    server_address = f"127.0.0.1:{find_free_port()}"
    server_env = {
        **dict(os.environ),
        "FL_SERVER_ADDRESS": server_address,
        "CONFIG_OVERRIDE_JSON": json.dumps(config_override or {}),
        "FL_FORCE_EXIT_ON_COMPLETE": "1",
    }
    server_process = subprocess.Popen(
        [sys.executable, "-m", "server.run_training"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=server_env,
    )
    processes: list[subprocess.Popen[str]] = []
    try:
        root_paths = build_artifact_paths(config["artifacts"]["root"])
        client_splits_dirs: list[Path] = []

        # ── Phase 1 : prepare client directories and shared splits ────────────
        # Build the per-client directory structure and copy the artifacts that
        # remain identical across clients (metadata, validation split, test split).
        # The partitioned train split is written separately below so that each
        # client genuinely trains on a distinct, non-overlapping data subset.
        for client_id in client_ids:
            client_root = (
                Path(config["artifacts"]["root"]) / "smoke_clients" / client_id
            )
            metadata_dir = client_root / "data" / "metadata"
            splits_dir = client_root / "data" / "splits"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            splits_dir.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "preprocessing_metadata.json").write_text(
                root_paths["metadata"].read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            # Validation and test remain global — the server evaluates on these.
            for split_name, source_path in (
                ("validation.npz", root_paths["validation"]),
                ("test.npz", root_paths["test"]),
            ):
                (splits_dir / split_name).write_bytes(source_path.read_bytes())
            client_splits_dirs.append(splits_dir)

        # ── Partition the training dataset  ───────────────────────────────────
        # Each client receives a disjoint, stratified subset of the global
        # train.npz.  When num_clients == 1 the full train set is used as-is.
        project_seed = int(config.get("project", {}).get("seed", 42))
        partition_dataset_for_clients(
            train_path=root_paths["train"],
            client_dirs=client_splits_dirs,
            seed=project_seed,
        )

        # ── Phase 2 : spawn one Flower client process per client ──────────────
        for client_id, client_splits_dir in zip(client_ids, client_splits_dirs):
            client_root = client_splits_dir.parents[
                1
            ]  # splits_dir -> data -> client_root
            client_env = {
                **dict(os.environ),
                "CONFIG_OVERRIDE_JSON": json.dumps(config_override or {}),
                "CLIENT_ARTIFACT_ROOT": str(client_root),
                "GLOBAL_ARTIFACT_ROOT": str(config["artifacts"]["root"]),
                "CLIENT_CONFIG_PATH": str(Path(client_root) / "effective_config.json"),
            }
            effective_config = json.dumps(config, indent=2)
            Path(client_env["CLIENT_CONFIG_PATH"]).write_text(
                effective_config, encoding="utf-8"
            )
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "client.runtime",
                        "--client-id",
                        client_id,
                        "--server-address",
                        server_address,
                        "--artifact-root",
                        str(client_root),
                        "--max-retries",
                        "25",
                        "--retry-delay",
                        "1",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=client_env,
                )
            )

        for process in processes:
            return_code = process.wait(timeout=180)
            if return_code != 0:
                raise RuntimeError(
                    f"Client process failed with exit code {return_code}."
                )

        server_return_code = server_process.wait(timeout=240)
        if server_return_code != 0:
            raise RuntimeError(
                f"Server process failed with exit code {server_return_code}."
            )
        metrics_path = (
            Path(config["artifacts"]["metrics_dir"]) / "federated_training_metrics.json"
        )
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        if server_process.poll() is None:
            server_process.kill()
