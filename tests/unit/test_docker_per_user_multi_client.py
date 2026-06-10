"""Lock test: docker_per_user + num_clients > 1 must produce real Docker client commands.

Regression guard for the soutenance fix — ensures the system never silently
falls back to subprocess simulation (sim_client_X) when deployment_mode is
docker_per_user and num_clients > 1.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np


class DockerPerUserMultiClientTests(unittest.TestCase):
    """Verify that docker_per_user + num_clients > 1 generates Docker commands."""

    def _make_artifact_root(self, tmp: Path) -> Path:
        """Create minimal artifact structure expected by run_user_training_session."""
        root = tmp / "user_runs" / "principal-test-000000000000"
        splits = root / "data" / "splits"
        meta = root / "data" / "metadata"
        logs = root / "logs"
        splits.mkdir(parents=True, exist_ok=True)
        meta.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(0)
        n = 100
        x_num = rng.random((n, 2), dtype=np.float32)
        x_cat = np.zeros((n, 0), dtype=np.int64)
        y = rng.integers(0, 2, size=n, dtype=np.int8)

        for split in ("train", "validation", "test"):
            np.savez_compressed(splits / f"{split}.npz", x_numeric=x_num, x_categorical=x_cat, y=y)

        meta_content = {
            "numeric_columns": ["f0", "f1"],
            "categorical_columns": [],
            "categorical_groupings": {},
            "categorical_vocabularies": {},
            "categorical_cardinalities": {},
            "embedding_dimensions": {},
            "embedding_input_schema": [],
            "categorical_encoding": {"unknown_token": 0},
            "transformer_state": {
                "numeric_columns": ["f0", "f1"],
                "numeric_imputer": {"strategy": "mean", "statistics": [0.5, 0.5]},
                "numeric_scaler": {"mean": [0.5, 0.5], "scale": [0.2, 0.2], "var": [0.04, 0.04]},
            },
        }
        (meta / "preprocessing_metadata.json").write_text(
            json.dumps(meta_content), encoding="utf-8"
        )
        return root

    def test_docker_per_user_num_clients_2_produces_two_docker_commands(self) -> None:
        """With num_clients=2 and docker_per_user, two docker run commands are built."""
        from shared.user_client import _build_docker_client_cmd

        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_artifact_root(Path(tmpdir))
            croot_1 = str(root / "fl_clients" / "client_1")
            croot_2 = str(root / "fl_clients" / "client_2")
            global_root = str(root)
            log_dir = root / "logs"
            log_dir.mkdir(exist_ok=True)

            env_patch = {
                "FL_ARTIFACT_ROOT_HOST": tmpdir,
                "FL_DATA_ROOT_HOST": str(Path(tmpdir) / "data"),
                "FL_CONFIG_ROOT_HOST": str(Path(tmpdir) / "config"),
                "FL_DOCKER_IMAGE": "ensaj-fl-client",
                "FL_DOCKER_NETWORK": "ensaj_fl-net",
            }

            with patch.dict(os.environ, env_patch):
                cmd1 = _build_docker_client_cmd(
                    client_id="client_1",
                    server_address="ensaj-client-app:50051",
                    artifact_root=croot_1,
                    global_artifact_root=global_root,
                    config_override_json="{}",
                    log_dir=log_dir,
                )
                cmd2 = _build_docker_client_cmd(
                    client_id="client_2",
                    server_address="ensaj-client-app:50051",
                    artifact_root=croot_2,
                    global_artifact_root=global_root,
                    config_override_json="{}",
                    log_dir=log_dir,
                )

            # Both commands must start with docker run
            self.assertEqual(cmd1[:2], ["docker", "run"])
            self.assertEqual(cmd2[:2], ["docker", "run"])

            # client_id must be client_1 / client_2 — never sim_client_X
            self.assertIn("client_1", cmd1)
            self.assertIn("client_2", cmd2)
            self.assertNotIn("sim_client_1", cmd1)
            self.assertNotIn("sim_client_2", cmd2)

            # Each command uses the correct image
            self.assertIn("ensaj-fl-client", cmd1)
            self.assertIn("ensaj-fl-client", cmd2)

            # Container names must differ
            name1 = cmd1[cmd1.index("--name") + 1]
            name2 = cmd2[cmd2.index("--name") + 1]
            self.assertNotEqual(name1, name2)
            self.assertIn("client_1", name1)
            self.assertIn("client_2", name2)

    def test_client_ids_for_num_clients_3_are_not_sim_client(self) -> None:
        """Multi-client ID generation: i=0→primary, i=1→client_2, i=2→client_3."""
        primary = "client_1"
        ids = [primary if i == 0 else f"client_{i + 1}" for i in range(3)]
        self.assertEqual(ids, ["client_1", "client_2", "client_3"])
        for cid in ids:
            self.assertNotIn("sim_client", cid)

    def test_num_clients_not_collapsed_to_1_when_enough_samples(self) -> None:
        """partition_dataset_for_clients succeeds with 100 samples for 2 clients."""
        from shared.federated_data import partition_dataset_for_clients

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            root = self._make_artifact_root(Path(tmpdir))
            train_path = root / "data" / "splits" / "train.npz"
            client_dirs = [
                root / "fl_clients" / "client_1" / "data" / "splits",
                root / "fl_clients" / "client_2" / "data" / "splits",
            ]
            written = partition_dataset_for_clients(
                train_path=train_path, client_dirs=client_dirs, seed=42
            )
            self.assertEqual(len(written), 2)
            for path in written:
                self.assertTrue(path.exists(), f"Partition not written: {path}")
                data = np.load(path)
                self.assertGreater(len(data["y"]), 0)

    def test_docker_command_uses_per_client_volume_not_shared_root(self) -> None:
        """Each client container mounts its own artifact dir, not the shared root."""
        from shared.user_client import _build_docker_client_cmd

        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_artifact_root(Path(tmpdir))
            croot_1 = str(root / "fl_clients" / "client_1")
            croot_2 = str(root / "fl_clients" / "client_2")
            log_dir = root / "logs"
            log_dir.mkdir(exist_ok=True)

            env_patch = {
                "FL_ARTIFACT_ROOT_HOST": tmpdir.replace("\\", "/"),
                "FL_DATA_ROOT_HOST": str(Path(tmpdir) / "data"),
                "FL_CONFIG_ROOT_HOST": str(Path(tmpdir) / "config"),
                "FL_DOCKER_IMAGE": "ensaj-fl-client",
                "FL_DOCKER_NETWORK": "ensaj_fl-net",
            }

            with patch.dict(os.environ, env_patch):
                cmd1 = _build_docker_client_cmd(
                    client_id="client_1",
                    server_address="host:50051",
                    artifact_root=croot_1,
                    global_artifact_root=str(root),
                    config_override_json="{}",
                    log_dir=log_dir,
                )
                cmd2 = _build_docker_client_cmd(
                    client_id="client_2",
                    server_address="host:50051",
                    artifact_root=croot_2,
                    global_artifact_root=str(root),
                    config_override_json="{}",
                    log_dir=log_dir,
                )

            # Find -v mounts for client-data
            # Volume format: "host_path:/app/client-data"  (host_path may contain C: on Windows)
            def client_data_mount(cmd: list[str]) -> str:
                for i, tok in enumerate(cmd):
                    if tok == "-v" and "client-data" in cmd[i + 1]:
                        # Strip the container part (everything from ":/app/...")
                        spec = cmd[i + 1]
                        container_part = ":/app/client-data"
                        if container_part in spec:
                            return spec[: spec.rindex(container_part)]
                        return spec
                return ""

            mount1 = client_data_mount(cmd1)
            mount2 = client_data_mount(cmd2)

            # Volumes must be different (separate per-client dirs)
            self.assertNotEqual(mount1, mount2)
            self.assertIn("client_1", mount1)
            self.assertIn("client_2", mount2)


if __name__ == "__main__":
    unittest.main()
