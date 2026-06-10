from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from shared.utils import load_json


@dataclass
class StructuredDataset:
    x_numeric: np.ndarray
    x_categorical: np.ndarray
    y: np.ndarray


def load_project_config(path: str | Path = "config/project-config.json") -> dict[str, Any]:
    return load_json(path)


def build_artifact_paths(artifact_root: str | Path = "artifacts") -> dict[str, Path]:
    root = Path(artifact_root)
    return {
        "root": root,
        "data_root": root / "data",
        "metadata": root / "data" / "metadata" / "preprocessing_metadata.json",
        "train": root / "data" / "splits" / "train.npz",
        "validation": root / "data" / "splits" / "validation.npz",
        "test": root / "data" / "splits" / "test.npz",
        "clients": root / "data" / "clients",
        "model": root / "models" / "global_model.pt",
        "metrics": root / "metrics" / "federated_training_metrics.json",
        "xai_root": root / "xai",
        "audit_root": root / "audit",
        "audit_summary": root / "audit" / "audit_validation_summary.json",
    }


def load_preprocessing_metadata(
    path: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
) -> dict[str, Any]:
    metadata_path = Path(path) if path is not None else build_artifact_paths(artifact_root)["metadata"]
    return load_json(metadata_path)


def _rebuild_preprocessing_bundle_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    transformer_state = metadata.get("transformer_state")
    if not transformer_state:
        raise ValueError("transformer_state absent des metadonnees de preprocessing.")

    numeric_columns = transformer_state["numeric_columns"]
    feature_count = len(numeric_columns)
    dummy_numeric = np.zeros((2, feature_count), dtype=np.float64)

    numeric_imputer = SimpleImputer(strategy=transformer_state["numeric_imputer"]["strategy"])
    numeric_imputer.fit(dummy_numeric)
    numeric_imputer.statistics_ = np.asarray(transformer_state["numeric_imputer"]["statistics"], dtype=np.float64)
    numeric_imputer.n_features_in_ = feature_count
    numeric_imputer.feature_names_in_ = np.asarray(numeric_columns, dtype=object)

    numeric_scaler = StandardScaler()
    numeric_scaler.fit(dummy_numeric)
    numeric_scaler.mean_ = np.asarray(transformer_state["numeric_scaler"]["mean"], dtype=np.float64)
    numeric_scaler.scale_ = np.asarray(transformer_state["numeric_scaler"]["scale"], dtype=np.float64)
    numeric_scaler.var_ = np.asarray(transformer_state["numeric_scaler"]["var"], dtype=np.float64)
    numeric_scaler.n_features_in_ = feature_count

    return {
        "numeric_imputer": numeric_imputer,
        "numeric_scaler": numeric_scaler,
        "numeric_columns": metadata["numeric_columns"],
        "categorical_columns": metadata["categorical_columns"],
        "categorical_groupings": metadata["categorical_groupings"],
        "categorical_vocabularies": metadata["categorical_vocabularies"],
        "unknown_token": metadata["categorical_encoding"]["unknown_token"],
    }


def load_preprocessing_bundle(artifact_root: str | Path = "artifacts") -> dict[str, Any]:
    metadata = load_preprocessing_metadata(artifact_root=artifact_root)
    return _rebuild_preprocessing_bundle_from_metadata(metadata)


def load_structured_dataset(path: str | Path) -> StructuredDataset:
    with np.load(path) as data:
        return StructuredDataset(
            x_numeric=data["x_numeric"].astype(np.float32),
            x_categorical=data["x_categorical"].astype(np.int64),
            y=data["y"].astype(np.int8),
        )


def load_validation_dataset(
    path: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
) -> StructuredDataset:
    dataset_path = Path(path) if path is not None else build_artifact_paths(artifact_root)["validation"]
    return load_structured_dataset(dataset_path)


def load_test_dataset(
    path: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
) -> StructuredDataset:
    dataset_path = Path(path) if path is not None else build_artifact_paths(artifact_root)["test"]
    return load_structured_dataset(dataset_path)


def load_train_dataset(
    path: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
) -> StructuredDataset:
    dataset_path = Path(path) if path is not None else build_artifact_paths(artifact_root)["train"]
    return load_structured_dataset(dataset_path)


def list_client_ids(base_dir: str | Path | None = None, artifact_root: str | Path = "artifacts") -> list[str]:
    clients_dir = Path(base_dir) if base_dir is not None else build_artifact_paths(artifact_root)["clients"]
    return sorted(path.stem for path in clients_dir.glob("client_*.npz"))


def get_model_input_schema(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric_columns": metadata["numeric_columns"],
        "categorical_columns": metadata["categorical_columns"],
        "categorical_cardinalities": metadata["categorical_cardinalities"],
        "embedding_dimensions": metadata["embedding_dimensions"],
        "embedding_input_schema": metadata["embedding_input_schema"],
    }


def partition_dataset_for_clients(
    train_path: str | Path,
    client_dirs: list[Path],
    *,
    seed: int = 42,
) -> list[Path]:
    """Partition the global training dataset into N disjoint stratified client splits.

    Each simulated client receives a non-overlapping subset of the training data.
    Partitioning is IID-stratified: class proportions are approximately preserved
    in every partition so that no client is degenerate (all-positive or all-negative).

    Args:
        train_path: Path to the canonical global ``train.npz`` artifact.
        client_dirs: List of per-client ``splits/`` directories.  Each directory
            will receive a ``train.npz`` containing only that client's partition.
        seed: Random seed for reproducible shuffling.

    Returns:
        List of paths to the written per-client ``train.npz`` files, in the same
        order as *client_dirs*.

    Raises:
        ValueError: If there are fewer training samples than clients, which would
            produce empty partitions.
    """
    dataset = load_structured_dataset(train_path)
    n_clients = len(client_dirs)
    n_samples = len(dataset.y)

    if n_samples < n_clients:
        raise ValueError(
            f"Le dataset d'entrainement ({n_samples} exemples) est trop petit "
            f"pour etre partitionne entre {n_clients} clients."
        )

    rng = np.random.default_rng(seed)

    # Stratified shuffle: gather indices per class then interleave round-robin
    # across client buckets so class balance is maintained in every partition.
    classes = np.unique(dataset.y)
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]

    for cls in classes:
        cls_indices = np.where(dataset.y == cls)[0]
        rng.shuffle(cls_indices)
        # Distribute class indices round-robin across clients
        for position, idx in enumerate(cls_indices):
            client_indices[position % n_clients].append(int(idx))

    # Shuffle each client's index list to avoid ordered class blocks
    written_paths: list[Path] = []
    for client_index, (splits_dir, indices) in enumerate(zip(client_dirs, client_indices)):
        indices_array = np.array(sorted(indices), dtype=np.int64)
        rng.shuffle(indices_array)

        splits_dir.mkdir(parents=True, exist_ok=True)
        partition_path = splits_dir / "train.npz"
        np.savez_compressed(
            partition_path,
            x_numeric=dataset.x_numeric[indices_array],
            x_categorical=dataset.x_categorical[indices_array],
            y=dataset.y[indices_array],
        )
        written_paths.append(partition_path)

    return written_paths
