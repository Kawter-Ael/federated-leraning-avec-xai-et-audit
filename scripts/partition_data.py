"""Partition preprocessed training data into N client directories for Docker FL."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from shared.data_preparation import partition_training_indices
from shared.federated_data import load_project_config


def partition_for_docker(num_clients: int, artifact_root: str = "artifacts") -> None:
    root = Path(artifact_root)
    train_path = root / "data" / "splits" / "train.npz"
    validation_path = root / "data" / "splits" / "validation.npz"
    test_path = root / "data" / "splits" / "test.npz"
    metadata_path = root / "data" / "metadata" / "preprocessing_metadata.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")

    with np.load(train_path) as train_data:
        x_numeric = train_data["x_numeric"]
        x_categorical = train_data["x_categorical"]
        y = train_data["y"]

    root_effective_config = root / "effective_config.json"
    config = (
        json.loads(root_effective_config.read_text(encoding="utf-8"))
        if root_effective_config.exists()
        else load_project_config()
    )
    seed = int(config.get("project", {}).get("seed", 42))
    client_ids = [f"client_{i}" for i in range(1, num_clients + 1)]

    partitions = partition_training_indices(pd.Series(y), num_clients, seed=seed)

    clients_root = root / "clients"
    for client_id in client_ids:
        client_dir = clients_root / client_id / "data" / "splits"
        client_dir.mkdir(parents=True, exist_ok=True)

        indices = partitions[client_id]
        np.savez_compressed(
            client_dir / "train.npz",
            x_numeric=x_numeric[indices].astype(np.float32),
            x_categorical=x_categorical[indices].astype(np.int64),
            y=y[indices].astype(np.int8),
        )

        if validation_path.exists():
            shutil.copy2(validation_path, client_dir / "validation.npz")
        if test_path.exists():
            shutil.copy2(test_path, client_dir / "test.npz")

        if metadata_path.exists():
            meta_dir = clients_root / client_id / "data" / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metadata_path, meta_dir / "preprocessing_metadata.json")

        effective_config_path = clients_root / client_id / "effective_config.json"
        effective_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Partitioned data for {num_clients} clients into {clients_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition training data for Docker FL clients."
    )
    parser.add_argument(
        "--num-clients", type=int, default=3, help="Number of FL client partitions."
    )
    parser.add_argument(
        "--artifact-root", default="artifacts", help="Artifact root directory."
    )
    args = parser.parse_args()
    partition_for_docker(args.num_clients, args.artifact_root)


if __name__ == "__main__":
    main()
