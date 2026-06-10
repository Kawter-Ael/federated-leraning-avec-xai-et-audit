from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shared.client_workflow import (
    autodetect_dataset_roles,
    build_runtime_config,
    run_audit_for_mode,
    run_phase2_for_mode,
    run_training_for_mode,
    run_xai_for_mode,
)
from shared.dataset_io import load_tabular


class GenericDatasetPipelineTests(unittest.TestCase):
    def test_tb_fixture_runs_end_to_end(self) -> None:
        fixture_path = Path("tests/fixtures/tb_small.csv")
        frame = load_tabular(fixture_path)
        detected = autodetect_dataset_roles(frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            runtime_config = build_runtime_config(
                dataset_path=fixture_path,
                mode="temporaire",
                target_column=detected["target_column"],
                positive_class="1",
                negative_classes=["0"],
                excluded_columns=detected["excluded_columns"],
                numeric_columns=detected["numeric_columns"],
                num_clients=1,
                num_rounds=1,
                local_epochs=1,
                original_dataset_name=fixture_path.name,
                fairness_attribute=None,
            )
            runtime_config["artifact_root"] = str(tmp_root / "user_runs" / runtime_config["run_id"])
            runtime_config["global_artifact_root"] = runtime_config["artifact_root"]
            runtime_config["config_override"]["artifacts"] = {
                "root": runtime_config["artifact_root"],
                "models_dir": str(Path(runtime_config["artifact_root"]) / "models"),
                "metrics_dir": str(Path(runtime_config["artifact_root"]) / "metrics"),
                "xai_dir": str(Path(runtime_config["artifact_root"]) / "xai"),
                "audit_dir": str(Path(runtime_config["artifact_root"]) / "audit"),
                "reports_dir": str(Path(runtime_config["artifact_root"]) / "reports"),
            }
            runtime_config["config_override"]["federated_learning"].update(
                {
                    "num_clients": 1,
                    "num_rounds": 1,
                    "min_fit_clients": 1,
                    "min_available_clients": 1,
                }
            )

            phase2 = run_phase2_for_mode(runtime_config)
            self.assertIn("sex", phase2["categorical_columns"])
            self.assertNotIn("Glucose", phase2["numeric_columns"])

            payload = run_training_for_mode(
                runtime_config,
                {
                    "client_id": "tb_client",
                    "username": "tb_client",
                    "artifact_root": runtime_config["artifact_root"],
                    "dataset_path": str(fixture_path),
                    "config_override": runtime_config["config_override"],
                },
            )
            self.assertEqual(payload["num_rounds"], 1)

            xai = run_xai_for_mode(runtime_config)
            audit = run_audit_for_mode(runtime_config)
            self.assertGreaterEqual(xai["feature_count"], 1)
            self.assertEqual(audit["dimensions"]["fairness"]["status"], "not_applicable")

            metrics_payload = json.loads(
                (Path(runtime_config["artifact_root"]) / "metrics" / "federated_training_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics_payload["categorical_feature_count"], 1)
            self.assertIn("name", detected["excluded_columns"])


if __name__ == "__main__":
    unittest.main()
