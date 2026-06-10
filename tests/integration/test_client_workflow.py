from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from shared.client_workflow import (
    build_user_context,
    build_runtime_config,
    persist_uploaded_dataset,
    run_audit_for_mode,
    run_phase2_for_mode,
    run_training_for_mode,
    run_xai_for_mode,
    summarize_run_outputs,
    validate_uploaded_dataset,
)
from shared.data_preparation import prepare_phase2
from tests.support import ensure_training_artifacts, launch_fl_server, wait_for_port


class ClientWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_training_artifacts()
        cls.raw_df = pd.read_csv("data/diabetes.csv")

    def _build_small_dataset(self, output_path: Path) -> Path:
        sample = (
            self.raw_df.groupby("Outcome", group_keys=False)
            .head(120)
            .reset_index(drop=True)
        )
        sample.to_csv(output_path, index=False)
        return output_path

    def test_validate_uploaded_dataset_rejects_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "invalid.csv"
            self.raw_df.drop(columns=["Outcome"]).head(100).to_csv(
                dataset_path, index=False
            )
            validation = validate_uploaded_dataset(
                dataset_path,
                target_column="Outcome",
                positive_class=1,
                excluded_columns=[],
                numeric_columns=["Pregnancies", "Glucose", "BloodPressure"],
            )
            self.assertFalse(validation["valid"])
            self.assertTrue(
                any(
                    item["code"] == "missing_target_column"
                    for item in validation["blocking"]
                )
            )

    def test_validate_uploaded_dataset_rejects_multiclass_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "multiclass.csv"
            frame = self.raw_df.head(120).copy()
            frame["Outcome"] = [0, 1, 2] * 40
            frame.to_csv(dataset_path, index=False)
            validation = validate_uploaded_dataset(
                dataset_path,
                target_column="Outcome",
                positive_class=1,
                excluded_columns=[],
                numeric_columns=["Pregnancies", "Glucose"],
            )
            self.assertFalse(validation["valid"])
            self.assertTrue(
                any(item["code"] == "non_binary_target" for item in validation["blocking"])
            )

    def test_validate_uploaded_dataset_warns_high_cardinality_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "warning.csv"
            frame = self.raw_df.head(120).copy()
            frame["free_text_id"] = [f"patient_{index}" for index in range(len(frame))]
            frame.to_csv(dataset_path, index=False)
            validation = validate_uploaded_dataset(
                dataset_path,
                target_column="Outcome",
                positive_class=1,
                excluded_columns=[],
                numeric_columns=["Pregnancies", "Glucose"],
            )
            self.assertTrue(
                any(
                    item["code"] == "quasi_unique_columns"
                    for item in validation["warning"]
                )
            )

    def test_principal_upload_is_persisted_under_artifacts_runtime_area(self) -> None:
        dataset_path = persist_uploaded_dataset(
            file_name="client_dataset.csv",
            content=b"col1,col2\n1,2\n",
            mode="principal",
            run_id="principal-demo",
        )
        resolved = Path(dataset_path)
        try:
            self.assertIn(
                str(Path("artifacts") / "user_runs" / "principal-demo" / "input"),
                str(resolved),
            )
            self.assertNotIn(str(Path("data") / "client_uploads"), str(resolved))
            self.assertTrue(resolved.exists())
        finally:
            if resolved.exists():
                resolved.unlink()
            input_dir = resolved.parent
            run_root = input_dir.parent
            if input_dir.exists() and not any(input_dir.iterdir()):
                input_dir.rmdir()
            if run_root.exists() and not any(run_root.iterdir()):
                run_root.rmdir()

    def test_runtime_config_keeps_federated_rounds_non_blocking_for_single_user_runs(
        self,
    ) -> None:
        runtime_config = build_runtime_config(
            dataset_path="data/diabetes.csv",
            mode="temporaire",
            target_column="Outcome",
            positive_class=1,
            negative_classes=[0],
            excluded_columns=[],
            numeric_columns=["Pregnancies", "Glucose", "BloodPressure"],
            num_clients=7,
            num_rounds=3,
            local_epochs=1,
            original_dataset_name="diabetes.csv",
        )
        federated_config = runtime_config["config_override"]["federated_learning"]
        self.assertEqual(federated_config["num_clients"], 7)
        # Professional FL: server waits for all configured clients before starting.
        self.assertEqual(federated_config["min_fit_clients"], 7)
        self.assertEqual(federated_config["min_available_clients"], 7)

    def test_temporary_run_pipeline_writes_isolated_artifacts(self) -> None:
        principal_metrics = Path("artifacts/metrics/federated_training_metrics.json")
        principal_mtime = (
            principal_metrics.stat().st_mtime if principal_metrics.exists() else None
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = self._build_small_dataset(
                Path(tmpdir) / "client_dataset.csv"
            )
            global_root = Path(tmpdir) / "global_server"
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
                num_clients=2,
                num_rounds=1,
                local_epochs=1,
                original_dataset_name="client_dataset.csv",
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

            phase2_summary = run_phase2_for_mode(runtime_config)
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
            user_context = build_user_context("client_demo", runtime_config)
            user_context["server_address"] = server_address
            try:
                training_summary = run_training_for_mode(runtime_config, user_context)
            finally:
                if server_process.poll() is None:
                    server_process.kill()
            xai_summary = run_xai_for_mode(runtime_config)
            audit_summary = run_audit_for_mode(runtime_config)
            final_summary = summarize_run_outputs(runtime_config)

            temp_root = Path(runtime_config["artifact_root"])
            self.assertIn("user_runs", str(temp_root))
            self.assertTrue(
                (
                    temp_root / "data" / "metadata" / "preprocessing_metadata.json"
                ).exists()
            )
            self.assertTrue((global_root / "models" / "global_model.pt").exists())
            self.assertTrue(
                (global_root / "metrics" / "federated_training_metrics.json").exists()
            )
            self.assertTrue(
                (global_root / "xai" / "xai_validation_report.json").exists()
            )
            self.assertTrue(
                (global_root / "audit" / "server_audit_report.json").exists()
            )
            self.assertEqual(phase2_summary["num_clients"], 1)
            self.assertEqual(training_summary["num_rounds"], 1)
            self.assertEqual(xai_summary["num_clients_explained"], 1)
            self.assertIn(audit_summary["status"], {"passed", "warning"})
            self.assertEqual(final_summary["run_mode"], "temporaire")

            self.assertNotIn(
                "client_count_mismatch_with_preprocessing_artifacts",
                str(
                    Path(global_root / "audit" / "server_audit_report.json").read_text()
                ),
            )

        if principal_mtime is not None:
            self.assertEqual(principal_metrics.stat().st_mtime, principal_mtime)


if __name__ == "__main__":
    unittest.main()
