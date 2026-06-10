from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch
from sklearn.exceptions import ConvergenceWarning

from shared.aggregation import (
    aggregate_rules,
    aggregate_shap_summaries,
    extract_rule_entries,
    extract_shap_summaries,
)
from shared.business_interpretation import build_rule_sentence
from shared.federated_data import (
    StructuredDataset,
    build_artifact_paths,
    get_model_input_schema,
    load_preprocessing_metadata,
    load_project_config,
)
from shared.modeling import create_model, predict_probabilities


@dataclass
class ExplainabilityArtifacts:
    global_summary: dict[str, Any]
    local_summaries: list[dict[str, Any]]
    global_rules_summary: dict[str, Any]
    instance_explanations: list[dict[str, Any]]
    validation_summary: dict[str, Any]


class StructuredExplainabilityEngine:
    def __init__(
        self, model: Any, metadata: dict[str, Any], schema: dict[str, Any]
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.schema = schema
        self.feature_names = (
            self.schema["numeric_columns"] + self.schema["categorical_columns"]
        )
        self.inverse_vocabularies = self._build_inverse_vocabularies(
            self.metadata["categorical_vocabularies"]
        )

    @staticmethod
    def _build_inverse_vocabularies(
        vocabularies: dict[str, dict[str, int]],
    ) -> dict[str, dict[int, str]]:
        inverse: dict[str, dict[int, str]] = {}
        for column, mapping in vocabularies.items():
            inverse[column] = {
                int(index): str(category) for category, index in mapping.items()
            }
        return inverse

    def combine_inputs(self, dataset: StructuredDataset) -> np.ndarray:
        return np.concatenate(
            [
                dataset.x_numeric.astype(np.float32),
                dataset.x_categorical.astype(np.float32),
            ],
            axis=1,
        )

    def split_combined_inputs(
        self, combined: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        numeric_count = len(self.schema["numeric_columns"])
        x_numeric = combined[:, :numeric_count].astype(np.float32)
        x_categorical = combined[:, numeric_count:].astype(np.int64)
        for index, column in enumerate(self.schema["categorical_columns"]):
            max_index = int(self.schema["categorical_cardinalities"][column]) - 1
            x_categorical[:, index] = np.clip(
                np.rint(x_categorical[:, index]), 0, max_index
            ).astype(np.int64)
        return x_numeric, x_categorical

    def predict_from_combined(self, combined: np.ndarray) -> np.ndarray:
        x_numeric, x_categorical = self.split_combined_inputs(np.asarray(combined))
        probabilities = predict_probabilities(self.model, x_numeric, x_categorical)
        return probabilities.reshape(-1, 1)

    @staticmethod
    def sample_rows(combined: np.ndarray, max_rows: int) -> np.ndarray:
        if len(combined) <= max_rows:
            return combined
        indices = np.linspace(0, len(combined) - 1, num=max_rows, dtype=int)
        return combined[indices]

    def create_kernel_explainer(self, background: np.ndarray) -> shap.KernelExplainer:
        return shap.KernelExplainer(self.predict_from_combined, background)

    @staticmethod
    def normalize_shap_output(shap_values: Any) -> np.ndarray:
        values = np.asarray(shap_values)
        if values.ndim == 2:
            return values
        if values.ndim == 3:
            if values.shape[-1] == 1:
                return values[:, :, 0]
            if values.shape[0] == 1:
                return values[0]
        raise ValueError(f"Forme SHAP non supportee : {values.shape}")

    def decode_feature_value(self, feature_name: str, value: float) -> Any:
        if feature_name in self.schema["numeric_columns"]:
            return float(value)
        inverse_mapping = self.inverse_vocabularies[feature_name]
        return inverse_mapping.get(int(round(value)), "__unknown__")

    def summarize_shap_values(
        self,
        shap_values: np.ndarray,
        data: np.ndarray,
        top_k: int = 10,
    ) -> dict[str, Any]:
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        top_features = []
        for index in order[:top_k]:
            feature_name = self.feature_names[index]
            top_features.append(
                {
                    "feature": feature_name,
                    "mean_abs_shap": float(mean_abs[index]),
                    "sample_value": self.decode_feature_value(
                        feature_name, float(data[0, index])
                    ),
                }
            )
        return {
            "feature_names": self.feature_names,
            "mean_abs_shap": {
                self.feature_names[i]: float(mean_abs[i])
                for i in range(len(self.feature_names))
            },
            "top_features": top_features,
        }

    def build_instance_explanations(
        self,
        shap_values: np.ndarray,
        data: np.ndarray,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        explanations: list[dict[str, Any]] = []
        for row_index in range(shap_values.shape[0]):
            row_values = shap_values[row_index]
            order = np.argsort(np.abs(row_values))[::-1]
            top_features = []
            for index in order[:top_k]:
                feature_name = self.feature_names[index]
                top_features.append(
                    {
                        "feature": feature_name,
                        "shap_value": float(row_values[index]),
                        "feature_value": self.decode_feature_value(
                            feature_name, float(data[row_index, index])
                        ),
                    }
                )
            explanations.append({"row_index": row_index, "top_features": top_features})
        return explanations

    def explain_local_dataset(
        self,
        dataset: StructuredDataset,
        *,
        background_size: int,
        sample_size: int,
        nsamples: int,
        top_k: int,
        instance_explanations: int = 0,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        combined = self.combine_inputs(dataset)
        background = self.sample_rows(combined, background_size)
        sample = self.sample_rows(combined, sample_size)
        explainer = self.create_kernel_explainer(background)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            shap_values = self.normalize_shap_output(
                explainer.shap_values(
                    sample, nsamples=nsamples, l1_reg="num_features(10)"
                )
            )
        summary = self.summarize_shap_values(shap_values, sample, top_k=top_k)
        instances: list[dict[str, Any]] = []
        if instance_explanations > 0:
            instances = self.build_instance_explanations(
                shap_values[:instance_explanations],
                sample[:instance_explanations],
                top_k=min(8, top_k),
            )
        return summary, instances

    def extract_local_rules(
        self,
        dataset: StructuredDataset,
        shap_summary: dict[str, Any],
        *,
        max_rules: int,
        min_support: float,
        min_category_count: int,
    ) -> list[dict[str, Any]]:
        rules: list[dict[str, Any]] = []
        y = dataset.y.astype(np.int8)
        if y.sum() == 0 or y.sum() == len(y):
            return rules

        numeric_count = len(self.schema["numeric_columns"])
        top_features = shap_summary.get("top_features", [])[:max_rules]

        for item in top_features:
            feature = str(item["feature"])
            importance = float(item["mean_abs_shap"])
            if feature in self.schema["numeric_columns"]:
                index = self.schema["numeric_columns"].index(feature)
                values = dataset.x_numeric[:, index]
                pos_values = values[y == 1]
                neg_values = values[y == 0]
                if len(pos_values) == 0 or len(neg_values) == 0:
                    continue
                pos_mean = float(pos_values.mean())
                neg_mean = float(neg_values.mean())
                threshold = float((pos_mean + neg_mean) / 2.0)
                operator = ">=" if pos_mean >= neg_mean else "<="
                if operator == ">=":
                    support = float((values >= threshold).mean())
                else:
                    support = float((values <= threshold).mean())
                if support < min_support:
                    continue
                rules.append(
                    {
                        "feature": feature,
                        "feature_type": "numeric",
                        "operator": operator,
                        "value": round(threshold, 4),
                        "support": round(support, 4),
                        "importance": round(importance, 6),
                        "signature": f"{feature}|{operator}|{round(threshold, 2)}",
                        "description": f"{feature} {operator} {threshold:.2f}",
                    }
                )
            else:
                index = self.schema["categorical_columns"].index(feature)
                values = dataset.x_categorical[:, index]
                unique_values, counts = np.unique(values, return_counts=True)
                valid_categories = [
                    int(category)
                    for category, count in zip(unique_values, counts)
                    if int(count) >= min_category_count
                ]
                if not valid_categories:
                    continue
                best_category = None
                best_score = -np.inf
                for category in valid_categories:
                    mask = values == category
                    support = float(mask.mean())
                    if support < min_support:
                        continue
                    positive_rate = float(y[mask].mean()) if mask.any() else 0.0
                    baseline = float(y.mean())
                    score = positive_rate - baseline
                    if score > best_score:
                        best_score = score
                        best_category = category
                if best_category is None:
                    continue
                mask = values == best_category
                support = float(mask.mean())
                category_label = self.decode_feature_value(
                    feature, float(best_category)
                )
                rules.append(
                    {
                        "feature": feature,
                        "feature_type": "categorical",
                        "operator": "==",
                        "value": category_label,
                        "support": round(support, 4),
                        "importance": round(importance, 6),
                        "signature": f"{feature}|==|{category_label}",
                        "description": f"{feature} == {category_label}",
                    }
                )

        rules.sort(
            key=lambda item: (-float(item["importance"]), -float(item["support"]))
        )
        return rules[:max_rules]


def compute_client_safe_explanations(
    model: Any,
    metadata: dict[str, Any],
    schema: dict[str, Any],
    dataset: StructuredDataset,
    *,
    background_size: int,
    sample_size: int,
    nsamples: int,
    top_features: int,
    instance_explanations: int,
    max_rules: int,
    min_support: float,
    min_category_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    engine = StructuredExplainabilityEngine(
        model=model, metadata=metadata, schema=schema
    )
    shap_summary, private_instance_explanations = engine.explain_local_dataset(
        dataset,
        background_size=background_size,
        sample_size=sample_size,
        nsamples=nsamples,
        top_k=top_features,
        instance_explanations=instance_explanations,
    )
    local_rules = engine.extract_local_rules(
        dataset,
        shap_summary,
        max_rules=max_rules,
        min_support=min_support,
        min_category_count=min_category_count,
    )
    return shap_summary, local_rules, private_instance_explanations


class EmbeddingShapExplainer:
    def __init__(
        self,
        config_path: str | Path = "config/project-config.json",
        model_path: str | Path = "artifacts/models/global_model.pt",
        artifact_root: str | Path = "artifacts",
    ) -> None:
        self.config = load_project_config(config_path)
        self.artifact_root = str(artifact_root)
        self.artifact_paths = build_artifact_paths(self.artifact_root)
        self.metadata = load_preprocessing_metadata(artifact_root=self.artifact_root)
        self.schema = get_model_input_schema(self.metadata)
        bundle = torch.load(model_path, map_location="cpu", weights_only=True)
        self.selected_threshold = float(bundle["selected_threshold"])
        self.model = create_model(self.config, schema=bundle["schema"])
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()
        self.engine = StructuredExplainabilityEngine(
            self.model, self.metadata, self.schema
        )

    def _combine_inputs(self, dataset: StructuredDataset) -> np.ndarray:
        return self.engine.combine_inputs(dataset)

    def _sample_rows(self, combined: np.ndarray, max_rows: int) -> np.ndarray:
        return self.engine.sample_rows(combined, max_rows)

    def explain_dataset(
        self,
        dataset: StructuredDataset,
        *,
        background: np.ndarray,
        sample_size: int,
        nsamples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample = self.engine.sample_rows(
            self.engine.combine_inputs(dataset), sample_size
        )
        explainer = self.engine.create_kernel_explainer(background)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            shap_values = self.engine.normalize_shap_output(
                explainer.shap_values(
                    sample, nsamples=nsamples, l1_reg="num_features(10)"
                )
            )
        return sample, shap_values

    def _build_instance_explanations(
        self,
        shap_values: np.ndarray,
        data: np.ndarray,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        return self.engine.build_instance_explanations(shap_values, data, top_k=top_k)

    def _select_source_round(self, metrics_payload: dict[str, Any]) -> dict[str, Any]:
        """Return the round payload aligned with the saved model artifact.

        The global model may be saved from the best validation round instead of
        the final round. SHAP and rules must therefore be aggregated from the
        same round to keep XAI, metrics and model weights consistent.
        """
        rounds = list(metrics_payload.get("rounds", []))
        if not rounds:
            return {}

        saved_model_round = int(metrics_payload.get("saved_model_round", 0) or 0)
        if saved_model_round > 0:
            for round_payload in rounds:
                if int(round_payload.get("round", 0)) == saved_model_round:
                    return round_payload
        return rounds[-1]

    def _collect_saved_round_summaries(
        self, metrics_payload: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Collect SHAP summaries and rule entries from the saved-model round."""
        round_payload = self._select_source_round(metrics_payload)
        if not round_payload:
            return [], [], 0

        local_summaries = extract_shap_summaries(round_payload["client_metrics"])
        local_rule_entries = extract_rule_entries(round_payload["client_metrics"])
        return (
            local_summaries,
            local_rule_entries,
            int(round_payload.get("round", 0)),
        )

    def run(self) -> ExplainabilityArtifacts:
        metrics_payload = json.loads(
            self.artifact_paths["metrics"].read_text(encoding="utf-8")
        )

        local_summaries, local_rule_entries, source_round = (
            self._collect_saved_round_summaries(metrics_payload)
        )

        global_summary = aggregate_shap_summaries(local_summaries)
        global_rules_summary = aggregate_rules(local_rule_entries)
        instance_explanations: list[dict[str, Any]] = []

        num_rounds = len(metrics_payload.get("rounds", []))
        feature_count = len(global_summary.get("feature_names", []))
        max_cardinality = max(
            self.schema.get("categorical_cardinalities", {}).values(), default=0
        )
        if feature_count >= 40 or max_cardinality >= 100:
            explainability_status = "degraded_due_to_cost_or_schema"
            explainability_message = (
                "Feature space is wide or categorical cardinality is high. SHAP is available, but interpretability is degraded by dataset cost/schema."
            )
        elif not global_summary.get("top_features"):
            explainability_status = "technically_executable"
            explainability_message = (
                "XAI pipeline executed, but the global summary is limited for this dataset."
            )
        else:
            explainability_status = "meaningful_for_this_dataset"
            explainability_message = (
                "Global SHAP summary and aggregated rules are interpretable for this dataset."
            )

        validation_summary = {
            "model_path": str(self.artifact_paths["model"]),
            "selected_threshold": self.selected_threshold,
            "saved_model_round": int(metrics_payload.get("saved_model_round", 0) or 0),
            "source_round_for_xai": source_round,
            "num_clients_explained": len(local_summaries),
            "num_rounds_aggregated": num_rounds,
            "feature_count": feature_count,
            "numeric_feature_count": len(self.schema["numeric_columns"]),
            "categorical_feature_count": len(self.schema["categorical_columns"]),
            "shap_backend": "client_local_kernel_shap_aggregated_on_server",
            "instance_explanations_count": 0,
            "meaningfulness_status": explainability_status,
            "meaningfulness_message": explainability_message,
            "model_info": {
                "family": self.config["model"]["family"],
                "type": self.config["model"]["type"],
            },
            "data_structure": {
                "input_shape": {
                    "numeric_features": len(self.schema["numeric_columns"]),
                    "categorical_features": len(self.schema["categorical_columns"]),
                },
                "output_shape": {"prediction_dim": 1},
            },
            "generated_rules": [
                {
                    "signature": rule.get("signature", ""),
                    "technical_rule": rule.get("description", ""),
                    "business_rule": rule.get("business_description")
                    or build_rule_sentence(rule, config=self.config),
                }
                for rule in global_rules_summary.get("rules", [])
            ],
            "consistency_validation": {
                "feature_names_aligned": bool(global_summary.get("feature_names")),
                "client_shap_summaries_count": len(local_summaries),
                "global_rules_available": bool(global_rules_summary.get("rules")),
            },
            "privacy_validation": {
                "raw_data_exposed": False,
                "instance_level_shap_exposed": False,
                "only_aggregated_client_summaries_used": True,
            },
            "explainability_quality": {
                "global_summary_available": bool(global_summary.get("top_features")),
                "rules_generated": len(global_rules_summary.get("rules", [])),
                "quality_level": "satisfactory"
                if global_summary.get("top_features")
                else "limited",
                "status": explainability_status,
                "message": explainability_message,
                "max_categorical_cardinality": int(max_cardinality),
            },
            "artifacts": {
                "global_summary": str(
                    self.artifact_paths["xai_root"] / "global_shap_summary.json"
                ),
                "local_summaries": str(
                    self.artifact_paths["xai_root"] / "local_shap_summaries.json"
                ),
                "global_rules": str(
                    self.artifact_paths["xai_root"] / "global_rules_summary.json"
                ),
                "validation": str(
                    self.artifact_paths["xai_root"] / "xai_validation_report.json"
                ),
            },
        }

        return ExplainabilityArtifacts(
            global_summary=global_summary,
            local_summaries=local_summaries,
            global_rules_summary=global_rules_summary,
            instance_explanations=instance_explanations,
            validation_summary=validation_summary,
        )


def save_explainability_artifacts(
    artifacts: ExplainabilityArtifacts,
    output_dir: str | Path = "artifacts/xai",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "global_shap_summary.json").write_text(
        json.dumps(artifacts.global_summary, indent=2),
        encoding="utf-8",
    )
    (output_path / "local_shap_summaries.json").write_text(
        json.dumps(artifacts.local_summaries, indent=2),
        encoding="utf-8",
    )
    (output_path / "global_rules_summary.json").write_text(
        json.dumps(artifacts.global_rules_summary, indent=2),
        encoding="utf-8",
    )
    (output_path / "instance_explanations.json").write_text(
        json.dumps(artifacts.instance_explanations, indent=2),
        encoding="utf-8",
    )
    (output_path / "xai_validation_report.json").write_text(
        json.dumps(artifacts.validation_summary, indent=2),
        encoding="utf-8",
    )


def run_phase4_explainability(
    config_path: str | Path = "config/project-config.json",
    model_path: str | Path = "artifacts/models/global_model.pt",
    artifact_root: str | Path = "artifacts",
) -> ExplainabilityArtifacts:
    explainer = EmbeddingShapExplainer(
        config_path=config_path, model_path=model_path, artifact_root=artifact_root
    )
    artifacts = explainer.run()
    save_explainability_artifacts(artifacts, output_dir=Path(artifact_root) / "xai")
    return artifacts


if __name__ == "__main__":
    result = run_phase4_explainability()
    print(json.dumps(result.validation_summary, indent=2))
