from __future__ import annotations

import json
import unittest

import torch

from audit.audit_pipeline import (
    _serialize_tensor_signature,
    _validate_client_payloads,
    _validate_server_artifacts,
    run_phase5_audit,
)
from explainability.shap_explainer import EmbeddingShapExplainer
from shared.federated_data import get_model_input_schema, load_preprocessing_metadata
from tests.support import ensure_audit_artifacts


class ExplainabilityAndAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_audit_artifacts()
        cls.metadata = load_preprocessing_metadata()
        cls.schema = get_model_input_schema(cls.metadata)
        with open(
            "artifacts/metrics/federated_training_metrics.json", "r", encoding="utf-8"
        ) as fh:
            cls.metrics_payload = json.load(fh)
        with open(
            "artifacts/xai/global_shap_summary.json", "r", encoding="utf-8"
        ) as fh:
            cls.global_shap_summary = json.load(fh)
        with open(
            "artifacts/xai/global_rules_summary.json", "r", encoding="utf-8"
        ) as fh:
            cls.global_rules_summary = json.load(fh)
        with open(
            "artifacts/xai/local_shap_summaries.json", "r", encoding="utf-8"
        ) as fh:
            cls.local_shap_summaries = json.load(fh)
        with open(
            "artifacts/xai/xai_validation_report.json", "r", encoding="utf-8"
        ) as fh:
            cls.xai_validation_report = json.load(fh)
        cls.model_bundle = torch.load(
            "artifacts/models/global_model.pt", map_location="cpu", weights_only=True
        )

    def test_xai_artifacts_are_schema_aligned(self) -> None:
        feature_count = len(self.schema["numeric_columns"]) + len(
            self.schema["categorical_columns"]
        )
        self.assertEqual(self.xai_validation_report["feature_count"], feature_count)
        self.assertEqual(len(self.global_shap_summary["feature_names"]), feature_count)
        self.assertIn("rules", self.global_rules_summary)
        self.assertEqual(
            len(self.local_shap_summaries), self.metrics_payload["num_clients"]
        )
        for summary in self.local_shap_summaries:
            self.assertEqual(len(summary["feature_names"]), feature_count)
        self.assertIn("privacy_validation", self.xai_validation_report)
        self.assertIn("explainability_quality", self.xai_validation_report)

    def test_small_explainer_run_returns_expected_structure(self) -> None:
        explainer = EmbeddingShapExplainer()
        artifacts = explainer.run()
        self.assertEqual(artifacts.validation_summary["feature_count"], 8)
        self.assertEqual(
            len(artifacts.local_summaries), self.metrics_payload["num_clients"]
        )
        self.assertEqual(len(artifacts.instance_explanations), 0)

    def test_audit_validators_accept_current_artifacts(self) -> None:
        tensor_signature = _serialize_tensor_signature(self.model_bundle["state_dict"])
        client_payload_audit, client_logs = _validate_client_payloads(
            metrics_payload=self.metrics_payload,
            tensor_signature=tensor_signature,
            feature_count=8,
        )
        server_report, server_logs = _validate_server_artifacts(
            metrics_payload=self.metrics_payload,
            global_model_bundle=self.model_bundle,
            phase4_validation=self.xai_validation_report,
            global_shap_summary=self.global_shap_summary,
            global_rules_summary=self.global_rules_summary,
            local_shap_summaries=self.local_shap_summaries,
            schema=self.schema,
        )

        self.assertIn(server_report["status"], ("passed", "warning"))
        self.assertEqual(
            len(client_payload_audit["client_results"]),
            self.metrics_payload["num_clients"],
        )
        self.assertEqual(len(client_logs), self.metrics_payload["num_clients"])
        self.assertEqual(len(server_logs), len(self.metrics_payload["rounds"]) + 1)
        self.assertTrue(
            all(
                item["audit_metadata"]["validation_status"] == "passed"
                for item in client_payload_audit["client_results"]
            )
        )

    def test_full_audit_summary_contains_five_dimensions(self) -> None:
        artifacts = run_phase5_audit()
        dimensions = artifacts.validation_summary["dimensions"]
        self.assertEqual(
            set(dimensions.keys()),
            {"privacy", "accuracy", "fairness", "data_drift", "explainability"},
        )


if __name__ == "__main__":
    unittest.main()
