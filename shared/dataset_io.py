from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd


class UnsupportedDatasetFormat(Exception):
    pass


def _coerce_numeric_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype != object:
            continue
        series = normalized[column]
        converted = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        converted_non_null = int(converted.notna().sum())
        if non_null > 0 and converted_non_null == non_null:
            normalized[column] = converted
    return normalized


def _detect_suffix(source: Any) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower()
    if hasattr(source, "name"):
        return Path(source.name).suffix.lower()
    return ".csv"


def load_tabular(source: str | Path | io.IOBase, **kwargs: Any) -> pd.DataFrame:
    suffix = _detect_suffix(source)
    if suffix == ".tsv":
        frame = pd.read_csv(source, sep="\t", **kwargs)
    elif suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(source, engine="openpyxl", **kwargs)
    elif suffix == ".parquet":
        frame = pd.read_parquet(source, **kwargs)
    elif suffix == ".csv":
        frame = pd.read_csv(source, **kwargs)
    else:
        raise UnsupportedDatasetFormat(
            f"Unsupported file format '{suffix}'. Supported: .csv, .tsv, .xlsx, .xls, .parquet"
        )
    frame.columns = [c.strip() for c in frame.columns]
    for col in frame.columns:
        if frame[col].dtype == object:
            frame[col] = frame[col].apply(
                lambda v: v.strip() if isinstance(v, str) else v
            )
    return _coerce_numeric_object_columns(frame)
