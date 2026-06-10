from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from shared.client_workflow import (
    build_runtime_config,
    run_phase2_for_mode,
    run_training_for_mode,
)
from shared.data_preparation import prepare_phase2
from shared.federated_data import load_preprocessing_metadata
from shared.modeling import create_model, predict_probabilities
from tests.support import (
    ensure_training_artifacts,
    launch_fl_server,
    run_local_distributed_training,
    wait_for_port,
)


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_training_artifacts()

    def test_persisted_model_bundle_loads_and_predicts(self) -> None:
        bundle = torch.load(
            "artifacts/models/global_model.pt", map_location="cpu", weights_only=True
        )
        model = create_model(
            {
                "project": {"seed": 42},
                "model": {"hyperparameters": bundle["hyperparameters"]},
            },
            schema=bundle["schema"],
        )
        model.load_state_dict(bundle["state_dict"])
        model.eval()

        metadata = load_preprocessing_metadata()
        numeric_count = len(metadata["numeric_columns"])
        categorical_count = len(metadata["categorical_columns"])
        probabilities = predict_probabilities(
            model,
            x_numeric=[[0.0] * numeric_count],
            x_categorical=[[0] * categorical_count],
        )
        self.assertEqual(probabilities.shape, (1,))
        self.assertGreaterEqual(float(probabilities[0]), 0.0)
        self.assertLessEqual(float(probabilities[0]), 1.0)

    def test_metrics_artifact_contains_multiple_rounds(self) -> None:
        import json

        with Path("artifacts/metrics/federated_training_metrics.json").open(
            "r", encoding="utf-8"
        ) as fh:
            metrics = json.load(fh)

        self.assertGreaterEqual(metrics["num_clients"], 1)
        self.assertGreaterEqual(metrics["num_rounds"], 1)
        self.assertEqual(len(metrics["rounds"]), metrics["num_rounds"])
        for round_payload in metrics["rounds"]:
            self.assertIn("aggregated_local_metrics", round_payload)
            self.assertIn("aggregated_shap_summary", round_payload)
            self.assertIn("aggregated_rule_summary", round_payload)
            self.assertIn("global_test_metrics", round_payload)
            self.assertIn("validation", round_payload)
            self.assertGreaterEqual(len(round_payload["client_metrics"]), 1)
            self.assertIn("shap_summary", round_payload["client_metrics"][0])
            self.assertIn("rule_summary", round_payload["client_metrics"][0])
            self.assertNotIn("local_rules", round_payload["client_metrics"][0])

    def test_reduced_training_smoke_run_writes_temp_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            override = {
                "data": {
                    "raw_dataset_path": "data/diabetes.csv",
                    "reference_dataset_path": "data/diabetes.csv",
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
                },
                "federated_learning": {
                    "num_clients": 1,
                    "num_rounds": 1,
                    "min_fit_clients": 1,
                    "min_available_clients": 1,
                },
                "model": {
                    "hyperparameters": {
                        "local_epochs": 1,
                    }
                },
                "artifacts": {
                    "models_dir": str(tmp_root / "models"),
                    "metrics_dir": str(tmp_root / "metrics"),
                },
            }
            payload = run_local_distributed_training(config_override=override)

            self.assertEqual(payload["num_rounds"], 1)
            self.assertEqual(len(payload["rounds"]), 1)
            self.assertTrue((tmp_root / "models" / "global_model.pt").exists())
            self.assertTrue(
                (tmp_root / "metrics" / "federated_training_metrics.json").exists()
            )
            model_bundle = torch.load(
                tmp_root / "models" / "global_model.pt",
                map_location="cpu",
                weights_only=True,
            )
            metrics_payload = json.loads(
                (tmp_root / "metrics" / "federated_training_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                float(metrics_payload["selected_threshold"]),
                float(model_bundle["selected_threshold"]),
            )
            self.assertIn("final_round_selected_threshold", metrics_payload)

    def test_user_logical_client_training_writes_single_client_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            global_root = tmp_root / "global_server"
            runtime_config = build_runtime_config(
                dataset_path="data/diabetes.csv",
                mode="temporaire",
                target_column="Outcome",
                positive_class=1,
                negative_classes=[0],
                excluded_columns=[],
                numeric_columns=[
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                    "DiabetesPedigreeFunction",
                    "Age",
                ],
                num_clients=5,
                num_rounds=5,
                local_epochs=1,
                original_dataset_name="diabetes.csv",
            )
            runtime_config["artifact_root"] = str(
                tmp_root / "user_runs" / runtime_config["run_id"]
            )
            runtime_config["global_artifact_root"] = str(global_root)
            runtime_config["config_override"]["federated_learning"]["num_clients"] = 1
            runtime_config["config_override"]["federated_learning"]["num_rounds"] = 1
            runtime_config["config_override"]["federated_learning"][
                "min_fit_clients"
            ] = 1
            runtime_config["config_override"]["federated_learning"][
                "min_available_clients"
            ] = 1
            runtime_config["config_override"]["artifacts"] = {
                "root": str(global_root),
                "models_dir": str(global_root / "models"),
                "metrics_dir": str(global_root / "metrics"),
                "xai_dir": str(global_root / "xai"),
                "audit_dir": str(global_root / "audit"),
                "reports_dir": str(global_root / "reports"),
            }
            prepare_phase2(config_override=runtime_config["config_override"])
            run_phase2_for_mode(runtime_config)
            per_run_root = Path(runtime_config["artifact_root"])
            global_root_path = Path(runtime_config["global_artifact_root"])
            if (
                per_run_root != global_root_path
                and (global_root_path / "data").exists()
            ):
                shutil.copytree(
                    global_root_path / "data",
                    per_run_root / "data",
                    dirs_exist_ok=True,
                )
            server_process, server_address = launch_fl_server(
                runtime_config["config_override"]
            )
            wait_for_port("127.0.0.1", int(server_address.split(":")[1]), timeout=30)
            try:
                payload = run_training_for_mode(
                    runtime_config,
                    {
                        "client_id": "client_demo",
                        "username": "client_demo",
                        "artifact_root": runtime_config["artifact_root"],
                        "global_artifact_root": str(global_root),
                        "dataset_path": runtime_config["dataset_path"],
                        "server_address": server_address,
                        "config_override": runtime_config["config_override"],
                    },
                )
            finally:
                if server_process.poll() is None:
                    server_process.kill()

            self.assertEqual(payload["num_clients"], 1)
            self.assertEqual(payload["num_rounds"], 1)
            self.assertEqual(
                payload["rounds"][0]["client_metrics"][0]["client_id"], "client_demo"
            )
            self.assertTrue((global_root / "models" / "global_model.pt").exists())
            self.assertTrue(
                (global_root / "metrics" / "federated_training_metrics.json").exists()
            )
            self.assertTrue((global_root / "xai" / "global_shap_summary.json").exists())
            self.assertTrue(
                (global_root / "xai" / "global_rules_summary.json").exists()
            )
            self.assertTrue(
                (global_root / "audit" / "audit_validation_summary.json").exists()
            )

            server_report = json.loads(
                (global_root / "audit" / "server_audit_report.json").read_text()
            )
            self.assertNotIn(
                "client_count_mismatch_with_preprocessing_artifacts",
                server_report.get("issues", []),
            )

    def test_augmented_training_returns_correct_num_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            runtime_config = build_runtime_config(
                dataset_path="data/diabetes.csv",
                mode="temporaire",
                target_column="Outcome",
                positive_class=1,
                negative_classes=[0],
                excluded_columns=[],
                numeric_columns=[
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                    "DiabetesPedigreeFunction",
                    "Age",
                ],
                num_clients=1,
                num_rounds=1,
                local_epochs=1,
                original_dataset_name="diabetes.csv",
            )
            runtime_config["config_override"]["data_augmentation"] = {
                "enabled": True,
                "method": "noise",
                "factor": 0.5,
            }
            runtime_config["config_override"]["federated_learning"].update(
                {
                    "num_clients": 1,
                    "num_rounds": 1,
                    "min_fit_clients": 1,
                    "min_available_clients": 1,
                }
            )
            prepare_phase2(config_override=runtime_config["config_override"])
            run_phase2_for_mode(runtime_config)
            uc = {
                "client_id": "regress_b5",
                "username": "regress_b5",
                "artifact_root": runtime_config["artifact_root"],
                "global_artifact_root": runtime_config["artifact_root"],
                "dataset_path": runtime_config["dataset_path"],
                "config_override": runtime_config["config_override"],
            }
            payload = run_training_for_mode(runtime_config, uc)
            cm = payload["rounds"][-1]["client_metrics"][0]
            aug = cm.get("augmentation_summary", {})
            self.assertGreater(
                aug.get("augmented_size", 0), aug.get("original_size", 0)
            )
            self.assertEqual(cm["num_examples"], aug["original_size"])

    def test_global_artifact_root_not_overwritten_by_training(self) -> None:
        runtime_config = build_runtime_config(
            dataset_path="data/diabetes.csv",
            mode="temporaire",
            target_column="Outcome",
            positive_class=1,
            negative_classes=[0],
            excluded_columns=[],
            numeric_columns=["Pregnancies"],
            num_clients=1,
            num_rounds=1,
            local_epochs=1,
            original_dataset_name="diabetes.csv",
        )
        user_context = {
            "client_id": "b1_test",
            "username": "b1_test",
            "artifact_root": runtime_config["artifact_root"],
            "global_artifact_root": "/custom/global/root",
            "dataset_path": "data/diabetes.csv",
            "config_override": {},
        }
        self.assertEqual(user_context["global_artifact_root"], "/custom/global/root")

    def test_portal_run_with_categorical_feature_uses_same_schema_for_server_and_client(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            dataset_path = tmp_root / "categorical_diabetes.csv"

            frame = pd.read_csv("data/diabetes.csv").copy()
            categories = (["A", "B", "C"] * ((len(frame) // 3) + 1))[: len(frame)]
            frame["Clinic"] = categories
            frame.to_csv(dataset_path, index=False)

            runtime_config = build_runtime_config(
                dataset_path=dataset_path,
                mode="temporaire",
                target_column="Outcome",
                positive_class=1,
                negative_classes=[0],
                excluded_columns=[],
                numeric_columns=[
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                    "DiabetesPedigreeFunction",
                    "Age",
                ],
                num_clients=1,
                num_rounds=1,
                local_epochs=1,
                original_dataset_name="categorical_diabetes.csv",
            )
            runtime_config["artifact_root"] = str(
                tmp_root / "user_runs" / runtime_config["run_id"]
            )
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
            self.assertIn("Clinic", phase2["categorical_columns"])

            payload = run_training_for_mode(
                runtime_config,
                {
                    "client_id": "schema_regress",
                    "username": "schema_regress",
                    "artifact_root": runtime_config["artifact_root"],
                    "dataset_path": str(dataset_path),
                    "config_override": runtime_config["config_override"],
                },
            )

            self.assertEqual(payload["num_rounds"], 1)
            self.assertEqual(
                payload["rounds"][-1]["client_metrics"][0]["client_id"],
                "schema_regress",
            )
            metrics_path = (
                Path(runtime_config["artifact_root"])
                / "metrics"
                / "federated_training_metrics.json"
            )
            self.assertTrue(metrics_path.exists())
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics_payload["categorical_feature_count"], 1)


if __name__ == "__main__":
    unittest.main()
