from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared.utils import apply_config_override


@dataclass
class StructuredSplit:
    x_numeric: np.ndarray
    x_categorical: np.ndarray
    y: np.ndarray


@dataclass
class PreparedData:
    cleaned_df: pd.DataFrame
    train_split: StructuredSplit
    validation_split: StructuredSplit
    test_split: StructuredSplit
    client_splits: dict[str, StructuredSplit]
    metadata: dict[str, Any]


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_raw_dataset(config: dict[str, Any]) -> pd.DataFrame:
    dataset_path = Path(config["data"]["raw_dataset_path"])
    from shared.dataset_io import load_tabular

    if not str(dataset_path).strip() or str(dataset_path) == ".":
        raise ValueError(
            "No dataset path configured. Load a dataset from the client portal or use a preset/override before running Phase 2."
        )
    return load_tabular(dataset_path)


def replace_question_marks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    object_columns = out.select_dtypes(include=["object", "string"]).columns
    if len(object_columns) == 0:
        return out
    out.loc[:, object_columns] = out.loc[:, object_columns].where(
        out.loc[:, object_columns] != "?", np.nan
    )
    return out


def normalize_diagnosis_code(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip().upper()
    if not text:
        return "missing"
    if text.startswith(("E", "V")):
        return text.split(".")[0]
    try:
        return str(int(float(text)))
    except Exception:
        return text.split(".")[0]


def group_diagnosis_category(value: Any) -> str:
    code = normalize_diagnosis_code(value)
    if code == "missing":
        return "other"
    if code.startswith("V"):
        return "external_causes"
    if code.startswith("E"):
        return "injury"

    try:
        numeric_code = int(code)
    except ValueError:
        return "other"

    if numeric_code == 250:
        return "diabetes_related"
    if 390 <= numeric_code <= 459 or numeric_code == 785:
        return "circulatory"
    if 460 <= numeric_code <= 519 or numeric_code == 786:
        return "respiratory"
    if 520 <= numeric_code <= 579 or numeric_code == 787:
        return "digestive"
    if 800 <= numeric_code <= 999:
        return "injury"
    if numeric_code in {780, 781, 782, 784, 788, 789}:
        return "other"
    return "other"


def normalize_and_group_diagnosis_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in out.columns:
            out[col] = out[col].map(group_diagnosis_category)
    return out


def _normalize_label_value(value: Any) -> Any:
    if pd.isna(value):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return stripped
        try:
            numeric = float(stripped)
        except ValueError:
            return stripped
        if numeric.is_integer():
            return int(numeric)
        return numeric
    return value


def build_target(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    target_col = config["data"]["target_column"]
    positive_class = _normalize_label_value(config["data"]["positive_class"])
    out = df.copy()
    normalized_target = out[target_col].map(_normalize_label_value)
    out["target"] = (normalized_target == positive_class).astype(np.int8)
    return out.drop(columns=[target_col])


def drop_configured_columns(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    excluded = list(config["data"]["excluded_columns"])
    dropped = [col for col in excluded if col in df.columns]
    return df.drop(columns=dropped), dropped


def drop_high_missing_columns(
    df: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    missing_ratio = df.isna().mean().to_dict()
    dropped = [
        col
        for col, ratio in missing_ratio.items()
        if col != "target" and ratio > threshold
    ]
    return df.drop(columns=dropped), dropped, missing_ratio


def infer_feature_types(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[list[str], list[str]]:
    x = df.drop(columns=["target"])
    configured_numeric = config["data"].get("numeric_columns", [])
    numeric_cols = [col for col in configured_numeric if col in x.columns]
    if not numeric_cols:
        numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in x.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def stratified_split(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    validation_size_within_train: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x = df.drop(columns=["target"])
    y = df["target"]
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_full,
        y_train_full,
        test_size=validation_size_within_train,
        random_state=seed,
        stratify=y_train_full,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def fit_numeric_transformer(
    x_train: pd.DataFrame, numeric_cols: list[str]
) -> tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    numeric_values = imputer.fit_transform(x_train[numeric_cols])
    scaler.fit(numeric_values)
    return imputer, scaler


def transform_numeric(
    x: pd.DataFrame,
    numeric_cols: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    numeric_values = imputer.transform(x[numeric_cols])
    return scaler.transform(numeric_values).astype(np.float32)


def prepare_categorical_frame(
    x: pd.DataFrame,
    categorical_cols: list[str],
    rare_threshold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = x[categorical_cols].copy()
    frame = frame.fillna("missing").astype(str)
    frame = frame.apply(lambda col: col.str.strip().replace({"": "missing"}))

    mappings: dict[str, Any] = {}
    prepared = pd.DataFrame(index=frame.index)

    for col in categorical_cols:
        value_counts = frame[col].value_counts()
        rare_values = value_counts[value_counts < rare_threshold].index
        column = frame[col].where(~frame[col].isin(rare_values), "other")
        prepared[col] = column
        mappings[col] = {
            "rare_values": sorted(str(value) for value in rare_values.tolist()),
            "value_counts": {str(k): int(v) for k, v in value_counts.to_dict().items()},
        }

    return prepared, mappings


def build_category_vocabularies(
    categorical_frame: pd.DataFrame,
    categorical_cols: list[str],
    unknown_token: str,
) -> dict[str, dict[str, int]]:
    vocabularies: dict[str, dict[str, int]] = {}
    for col in categorical_cols:
        categories = sorted(set(categorical_frame[col].astype(str).tolist()))
        ordered_categories = [unknown_token]
        for category in categories:
            if category != unknown_token:
                ordered_categories.append(category)
        vocabularies[col] = {
            category: idx for idx, category in enumerate(ordered_categories)
        }
    return vocabularies


def transform_categorical(
    x: pd.DataFrame,
    categorical_cols: list[str],
    category_groupings: dict[str, Any],
    vocabularies: dict[str, dict[str, int]],
    unknown_token: str,
) -> np.ndarray:
    frame = x[categorical_cols].copy()
    frame = frame.fillna("missing").astype(str)
    frame = frame.apply(lambda col: col.str.strip().replace({"": "missing"}))

    encoded_columns: list[np.ndarray] = []
    for col in categorical_cols:
        rare_values = set(category_groupings[col]["rare_values"])
        column = frame[col].where(~frame[col].isin(rare_values), "other")
        mapping = vocabularies[col]
        encoded = column.map(
            lambda value: mapping.get(str(value), mapping[unknown_token])
        ).to_numpy(dtype=np.int64)
        encoded_columns.append(encoded)

    if not encoded_columns:
        return np.empty((len(x), 0), dtype=np.int64)
    return np.stack(encoded_columns, axis=1)


def make_structured_split(
    x: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_imputer: SimpleImputer,
    numeric_scaler: StandardScaler,
    category_groupings: dict[str, Any],
    vocabularies: dict[str, dict[str, int]],
    unknown_token: str,
) -> StructuredSplit:
    x_numeric = transform_numeric(x, numeric_cols, numeric_imputer, numeric_scaler)
    x_categorical = transform_categorical(
        x, categorical_cols, category_groupings, vocabularies, unknown_token
    )
    return StructuredSplit(
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        y=y.to_numpy(dtype=np.int8),
    )


def partition_training_indices(
    y_train: pd.Series, num_clients: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    client_ids = [f"client_{i + 1}" for i in range(num_clients)]
    client_indices: dict[str, list[int]] = {cid: [] for cid in client_ids}

    for label in sorted(y_train.unique()):
        label_idx = y_train[y_train == label].index.to_numpy(copy=True)
        rng.shuffle(label_idx)
        proportions = rng.dirichlet(np.full(num_clients, 0.8))
        split_points = (np.cumsum(proportions)[:-1] * len(label_idx)).astype(int)
        chunks = np.split(label_idx, split_points)
        for cid, chunk in zip(client_ids, chunks):
            client_indices[cid].extend(chunk.tolist())

    min_rows = max(8, int(len(y_train) * 0.03))
    for cid in client_ids:
        if len(client_indices[cid]) >= min_rows:
            continue
        donor = max(client_ids, key=lambda key: len(client_indices[key]))
        if donor == cid or len(client_indices[donor]) <= min_rows:
            continue
        needed = min_rows - len(client_indices[cid])
        moved = client_indices[donor][:needed]
        client_indices[donor] = client_indices[donor][needed:]
        client_indices[cid].extend(moved)

    return {
        cid: np.asarray(indices, dtype=np.int64)
        for cid, indices in client_indices.items()
    }


def derive_embedding_dimensions(cardinalities: dict[str, int]) -> dict[str, int]:
    return {
        col: int(min(32, max(4, round(np.sqrt(size) * 2))))
        for col, size in cardinalities.items()
    }


def save_structured_npz(path: Path, split: StructuredSplit) -> None:
    np.savez_compressed(
        path,
        x_numeric=split.x_numeric.astype(np.float32),
        x_categorical=split.x_categorical.astype(np.int64),
        y=split.y.astype(np.int8),
    )


def save_phase2_artifacts(prepared: PreparedData, config: dict[str, Any]) -> None:
    data_root = Path(config["artifacts"]["root"]) / "data"
    splits_dir = data_root / "splits"
    clients_dir = data_root / "clients"
    metadata_dir = data_root / "metadata"
    for path in [splits_dir, clients_dir, metadata_dir]:
        path.mkdir(parents=True, exist_ok=True)

    save_structured_npz(splits_dir / "train.npz", prepared.train_split)
    save_structured_npz(splits_dir / "validation.npz", prepared.validation_split)
    save_structured_npz(splits_dir / "test.npz", prepared.test_split)

    for client_id, split in prepared.client_splits.items():
        save_structured_npz(clients_dir / f"{client_id}.npz", split)

    metadata_payload = dict(prepared.metadata)
    with (metadata_dir / "preprocessing_metadata.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(metadata_payload, fh, indent=2)


def serialize_transformer_state(
    numeric_imputer: SimpleImputer,
    numeric_scaler: StandardScaler,
    numeric_columns: list[str],
) -> dict[str, Any]:
    return {
        "numeric_columns": list(numeric_columns),
        "numeric_imputer": {
            "strategy": str(numeric_imputer.strategy),
            "statistics": [
                float(value) for value in numeric_imputer.statistics_.tolist()
            ],
        },
        "numeric_scaler": {
            "mean": [float(value) for value in numeric_scaler.mean_.tolist()],
            "scale": [float(value) for value in numeric_scaler.scale_.tolist()],
            "var": [float(value) for value in numeric_scaler.var_.tolist()],
        },
    }


def prepare_phase2(
    config_path: str | Path = "config/project-config.json",
    config_override: dict[str, Any] | None = None,
) -> PreparedData:
    config = load_config(config_path)
    if config_override:
        apply_config_override(config, config_override)
    seed = int(config["project"]["seed"])
    unknown_token = str(config["data"]["categorical_encoding"]["unknown_token"])
    rare_threshold = int(
        config["data"]["categorical_encoding"]["rare_category_threshold"]
    )
    high_missing_threshold = float(config["data"]["high_missing_threshold"])

    raw_df = load_raw_dataset(config)
    replaced_df = replace_question_marks(raw_df)
    if config.get("data", {}).get("apply_icd9_grouping", False):
        grouped_df = normalize_and_group_diagnosis_columns(replaced_df)
    else:
        grouped_df = replaced_df
    targeted_df = build_target(grouped_df, config)
    trimmed_df, config_dropped = drop_configured_columns(targeted_df, config)
    cleaned_df, high_missing_dropped, missing_ratio = drop_high_missing_columns(
        trimmed_df, threshold=high_missing_threshold
    )

    numeric_cols, categorical_cols = infer_feature_types(cleaned_df, config)
    x_train, x_validation, x_test, y_train, y_validation, y_test = stratified_split(
        cleaned_df,
        seed=seed,
        test_size=float(config["data"].get("test_size", 0.2)),
        validation_size_within_train=float(
            config["data"].get("validation_size_within_train", 0.125)
        ),
    )

    numeric_imputer, numeric_scaler = fit_numeric_transformer(x_train, numeric_cols)
    train_categorical_frame, category_groupings = prepare_categorical_frame(
        x_train, categorical_cols, rare_threshold
    )
    vocabularies = build_category_vocabularies(
        train_categorical_frame, categorical_cols, unknown_token
    )
    categorical_cardinalities = {
        col: len(vocabularies[col]) for col in categorical_cols
    }
    embedding_dimensions = derive_embedding_dimensions(categorical_cardinalities)

    train_split = make_structured_split(
        x_train,
        y_train,
        numeric_cols,
        categorical_cols,
        numeric_imputer,
        numeric_scaler,
        category_groupings,
        vocabularies,
        unknown_token,
    )
    validation_split = make_structured_split(
        x_validation,
        y_validation,
        numeric_cols,
        categorical_cols,
        numeric_imputer,
        numeric_scaler,
        category_groupings,
        vocabularies,
        unknown_token,
    )
    test_split = make_structured_split(
        x_test,
        y_test,
        numeric_cols,
        categorical_cols,
        numeric_imputer,
        numeric_scaler,
        category_groupings,
        vocabularies,
        unknown_token,
    )

    client_index_map = partition_training_indices(
        y_train, int(config["federated_learning"]["num_clients"]), seed
    )
    client_splits: dict[str, StructuredSplit] = {}
    client_summary: list[dict[str, Any]] = []
    train_index_positions = {index: pos for pos, index in enumerate(x_train.index)}
    for client_id, row_indices in client_index_map.items():
        positions = np.asarray(
            [train_index_positions[index] for index in row_indices], dtype=np.int64
        )
        split = StructuredSplit(
            x_numeric=train_split.x_numeric[positions],
            x_categorical=train_split.x_categorical[positions],
            y=train_split.y[positions],
        )
        client_splits[client_id] = split
        unique, counts = np.unique(split.y, return_counts=True)
        distribution = {
            str(int(label)): int(count) for label, count in zip(unique, counts)
        }
        client_summary.append(
            {
                "client_id": client_id,
                "num_examples": int(len(split.y)),
                "target_distribution": distribution,
            }
        )

    # Class-imbalance fingerprint: used downstream to auto-enable minority
    # oversampling and to surface a warning in the portal when splits risk
    # producing a single-class validation/test (severe-imbalance regime).
    train_positives = int((y_train == 1).sum())
    validation_positives = int((y_validation == 1).sum())
    test_positives = int((y_test == 1).sum())
    positive_rate = float(cleaned_df["target"].mean()) if len(cleaned_df) else 0.0
    min_split_positives = min(train_positives, validation_positives, test_positives)
    class_balance = {
        "positive_rate": positive_rate,
        "train_positives": train_positives,
        "validation_positives": validation_positives,
        "test_positives": test_positives,
        "min_split_positives": int(min_split_positives),
        "severe_imbalance": bool(positive_rate < 0.1 or min_split_positives < 5),
    }

    metadata = {
        "raw_shape": list(raw_df.shape),
        "shape_after_target": list(targeted_df.shape),
        "shape_after_column_drop": list(trimmed_df.shape),
        "shape_after_missing_drop": list(cleaned_df.shape),
        "class_balance": class_balance,
        "dropped_columns_config": config_dropped,
        "dropped_columns_high_missing": high_missing_dropped,
        "missing_ratio": missing_ratio,
        "target_distribution": {
            str(k): int(v)
            for k, v in cleaned_df["target"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "target_column": str(config["data"]["target_column"]),
        "target_semantics": {
            "negative": {
                "value": 0,
                "label": config.get("project", {}).get("negative_label", "Negative"),
            },
            "positive": {
                "value": 1,
                "label": config.get("project", {}).get("positive_label", "Positive"),
            },
            "risk_levels": {
                "low": {"min": 0.0, "max_exclusive": 0.33},
                "medium": {"min": 0.33, "max_exclusive": 0.66},
                "high": {"min": 0.66, "max_inclusive": 1.0},
            },
        },
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "categorical_groupings": category_groupings,
        "categorical_vocabularies": vocabularies,
        "categorical_cardinalities": categorical_cardinalities,
        "embedding_dimensions": embedding_dimensions,
        "embedding_input_schema": [
            {
                "name": col,
                "cardinality": categorical_cardinalities[col],
                "embedding_dim": embedding_dimensions[col],
                "unknown_index": vocabularies[col][unknown_token],
            }
            for col in categorical_cols
        ],
        "categorical_encoding": {
            "rare_category_threshold": rare_threshold,
            "unknown_token": unknown_token,
            "missing_token": "missing",
            "rare_replacement": "other",
        },
        "transformer_state": serialize_transformer_state(
            numeric_imputer=numeric_imputer,
            numeric_scaler=numeric_scaler,
            numeric_columns=numeric_cols,
        ),
        "train_shape": {
            "x_numeric": list(train_split.x_numeric.shape),
            "x_categorical": list(train_split.x_categorical.shape),
            "y": [int(train_split.y.shape[0])],
        },
        "validation_shape": {
            "x_numeric": list(validation_split.x_numeric.shape),
            "x_categorical": list(validation_split.x_categorical.shape),
            "y": [int(validation_split.y.shape[0])],
        },
        "test_shape": {
            "x_numeric": list(test_split.x_numeric.shape),
            "x_categorical": list(test_split.x_categorical.shape),
            "y": [int(test_split.y.shape[0])],
        },
        "train_target_distribution": {
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().to_dict().items()
        },
        "validation_target_distribution": {
            str(k): int(v)
            for k, v in y_validation.value_counts().sort_index().to_dict().items()
        },
        "test_target_distribution": {
            str(k): int(v)
            for k, v in y_test.value_counts().sort_index().to_dict().items()
        },
        "client_summary": client_summary,
    }

    prepared = PreparedData(
        cleaned_df=cleaned_df,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        client_splits=client_splits,
        metadata=metadata,
    )
    save_phase2_artifacts(prepared, config)
    return prepared
