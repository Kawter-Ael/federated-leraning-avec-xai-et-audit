from __future__ import annotations

import importlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from dashboard.app import (
    ensure_default_global_model_artifact,
    explain_prediction,
    filter_case_history_for_run,
    load_dashboard_payload,
    load_runtime_components,
    normalize_dataframe_for_display,
    prepare_single_instance,
)
from shared.modeling import predict_probabilities
from tests.support import ensure_audit_artifacts


class DashboardAndDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_audit_artifacts()

    def test_dashboard_loads_expected_artifacts(self) -> None:
        payload = load_dashboard_payload()
        expected = {
            "model",
            "metrics",
            "global_shap",
            "global_rules",
            "local_shap",
            "instance_explanations",
            "xai_validation",
            "client_audit",
            "server_audit",
            "audit_summary",
            "audit_log",
        }
        self.assertTrue(expected.issubset({key for key, exists in payload["available"].items() if exists}))

    def test_dashboard_can_bootstrap_default_global_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            metadata_dir = tmp_root / "artifacts" / "data" / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = metadata_dir / "preprocessing_metadata.json"
            metadata_path.write_text(
                Path("artifacts/data/metadata/preprocessing_metadata.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            model_path = tmp_root / "artifacts" / "models" / "global_model.pt"
            result = ensure_default_global_model_artifact(
                config_path="config/project-config.json",
                metadata_path=metadata_path,
                model_path=model_path,
            )
            self.assertEqual(result["status"], "created")
            self.assertTrue(model_path.exists())

    def test_dashboard_prediction_path_works(self) -> None:
        runtime = load_runtime_components()
        values: dict[str, object] = {}
        for column in runtime["metadata"]["numeric_columns"]:
            values[column] = 0.0
        for column, vocab in runtime["metadata"]["categorical_vocabularies"].items():
            options = [value for value in vocab.keys() if value != "__unknown__"]
            values[column] = "missing" if "missing" in options else options[0]

        _, x_numeric, x_categorical = prepare_single_instance(values, runtime)
        probability = float(predict_probabilities(runtime["model"], x_numeric, x_categorical)[0])
        explanation = explain_prediction(x_numeric, x_categorical)

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertEqual(len(explanation[0]["top_features"]), 8)

    def test_display_normalization_handles_audit_log_lists(self) -> None:
        payload = load_dashboard_payload()
        frame = normalize_dataframe_for_display(__import__("pandas").DataFrame(payload["data"]["audit_log"]))
        self.assertIn("issues", frame.columns)
        self.assertTrue(all(isinstance(value, str) for value in frame["issues"].tolist()))

    def test_dashboard_filters_server_case_history_to_selected_run(self) -> None:
        rows = [
            {
                "case_id": "principal-20260506204210-cd817e27-case-1",
                "model_version": "principal-20260506204210-cd817e27",
            },
            {
                "case_id": "other-run-case-2",
                "model_version": "other-run",
            },
            {
                "case_id": "principal-20260506204210-cd817e27-case-legacy",
            },
        ]
        filtered = filter_case_history_for_run(
            rows, "principal-20260506204210-cd817e27"
        )
        self.assertEqual(len(filtered), 2)
        self.assertTrue(
            all("principal-20260506204210-cd817e27" in row["case_id"] for row in filtered)
        )

    def test_client_app_module_imports_without_relative_import_errors(self) -> None:
        module = importlib.import_module("client_app.app")
        self.assertTrue(hasattr(module, "main"))

    def test_docker_compose_config_is_valid(self) -> None:
        result = subprocess.run(
            ["docker", "compose", "--env-file", ".env.example", "config"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("mongodb:", result.stdout)
        # server is profile-gated ("fl") — not present in default compose config
        self.assertIn("dashboard:", result.stdout)
        self.assertIn("client-app:", result.stdout)
        self.assertNotIn("client_1:", result.stdout)


if __name__ == "__main__":
    unittest.main()
