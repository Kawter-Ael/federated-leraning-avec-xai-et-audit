from __future__ import annotations

import json
from typing import Any

import numpy as np
from flwr.client import NumPyClient

from shared.modeling import create_model, get_model_parameters, set_model_parameters
from shared.user_client import (
    compute_metrics,
    generate_shap_summary,
    prepare_data,
    train_model,
)
from shared.utils import contains_forbidden_keys

FORBIDDEN_PAYLOAD_KEYS = {
    "local_rules",
    "shap_summary_local",
    "instance_explanations",
    "raw_shap_values",
    "instance_explanations",
    "private_instance_explanations",
    "x_numeric",
    "x_categorical",
    "y",
}


def build_safe_rule_summary(local_rules: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rules": local_rules,
        "num_rules": len(local_rules),
        "aggregation": "client_local_rule_summary",
    }


def encode_client_metrics_payload(
    *,
    client_id: str,
    metrics_payload: dict[str, Any],
    shap_summary: dict[str, Any],
    rule_summary: dict[str, Any],
) -> dict[str, float | int | str]:
    payload = {
        "client_id": client_id,
        "loss": float(metrics_payload["test_metrics"]["loss"]),
        "accuracy": float(metrics_payload["test_metrics"]["accuracy"]),
        "precision": float(metrics_payload["test_metrics"]["precision"]),
        "recall": float(metrics_payload["test_metrics"]["recall"]),
        "f1": float(metrics_payload["test_metrics"]["f1"]),
        "roc_auc": float(metrics_payload["test_metrics"]["roc_auc"]),
        "pr_auc": float(metrics_payload["test_metrics"]["pr_auc"]),
        "threshold": float(metrics_payload["selected_threshold"]),
        "shap_summary": json.dumps(shap_summary),
        "rule_summary": json.dumps(rule_summary),
    }
    forbidden = contains_forbidden_keys(payload, FORBIDDEN_PAYLOAD_KEYS)
    if forbidden:
        raise ValueError(f"Forbidden keys detected in client payload: {forbidden}")
    return payload


class FederatedClient(NumPyClient):
    def __init__(
        self, client_id: str, config: dict[str, Any], user_context: dict[str, Any]
    ) -> None:
        self.client_id = client_id
        self.config = config
        self.user_context = user_context
        self.prepared_data = prepare_data(user_context)
        self.metadata = self.prepared_data["metadata"]
        self.schema = self.prepared_data["schema"]
        self.model = create_model(config, schema=self.schema)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        _ = config
        return get_model_parameters(self.model)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, float | int | str]]:
        set_model_parameters(self.model, parameters)
        local_epochs = int(
            config.get(
                "local_epochs", self.config["model"]["hyperparameters"]["local_epochs"]
            )
        )
        local_config = {
            **self.config,
            "model": {
                **self.config["model"],
                "hyperparameters": {
                    **self.config["model"]["hyperparameters"],
                    "local_epochs": local_epochs,
                },
            },
        }
        aug_cfg = dict(self.config.get("data_augmentation", {}))
        balance_flags = self.metadata.get("class_balance", {}) or {}
        if balance_flags.get("severe_imbalance", False) and not aug_cfg.get(
            "enabled", False
        ):
            aug_cfg = {
                **aug_cfg,
                "enabled": True,
                "method": aug_cfg.get("method") or "smote",
                "factor": max(float(aug_cfg.get("factor", 1.0)), 1.0),
                "auto_enabled_reason": "severe_class_imbalance",
            }
        real_train_size = int(len(self.prepared_data["train_dataset"].y))
        aug_summary: dict[str, Any] | None = None
        if aug_cfg.get("enabled", False):
            from shared.data_augmentation import apply_augmentation

            train_ds = self.prepared_data["train_dataset"]
            x_num, x_cat, y_aug = apply_augmentation(
                train_ds.x_numeric,
                train_ds.x_categorical,
                train_ds.y,
                aug_cfg,
            )
            aug_dataset = type(train_ds)(x_numeric=x_num, x_categorical=x_cat, y=y_aug)
            aug_prepared = {**self.prepared_data, "train_dataset": aug_dataset}
            self.model = train_model(self.model, aug_prepared, local_config)
            augmented_size = int(len(y_aug))
            aug_summary = {
                "method": aug_cfg.get("method", "unknown"),
                "factor": float(aug_cfg.get("factor", 0)),
                "original_size": real_train_size,
                "augmented_size": augmented_size,
                "auto_enabled_reason": aug_cfg.get("auto_enabled_reason"),
            }
        else:
            self.model = train_model(self.model, self.prepared_data, local_config)
        metrics_payload = compute_metrics(self.model, self.prepared_data, local_config)
        shap_summary, local_rules, _private_instances = generate_shap_summary(
            self.model,
            self.prepared_data,
            local_config,
        )
        rule_summary = build_safe_rule_summary(local_rules)
        metrics = encode_client_metrics_payload(
            client_id=self.client_id,
            metrics_payload=metrics_payload,
            shap_summary=shap_summary,
            rule_summary=rule_summary,
        )
        if aug_summary is not None:
            metrics["augmentation_summary"] = json.dumps(aug_summary)
        return (
            get_model_parameters(self.model),
            real_train_size,
            metrics,
        )

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, float | int | str]]:
        _ = config
        set_model_parameters(self.model, parameters)
        metrics_payload = compute_metrics(self.model, self.prepared_data, self.config)
        return (
            float(metrics_payload["test_metrics"]["loss"]),
            int(len(self.prepared_data["train_dataset"].y)),
            {
                "client_id": self.client_id,
                "accuracy": float(metrics_payload["test_metrics"]["accuracy"]),
                "precision": float(metrics_payload["test_metrics"]["precision"]),
                "recall": float(metrics_payload["test_metrics"]["recall"]),
                "f1": float(metrics_payload["test_metrics"]["f1"]),
                "roc_auc": float(metrics_payload["test_metrics"]["roc_auc"]),
                "pr_auc": float(metrics_payload["test_metrics"]["pr_auc"]),
                "threshold": float(metrics_payload["selected_threshold"]),
            },
        )
