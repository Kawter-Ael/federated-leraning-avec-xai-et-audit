"""Shared Streamlit UI components used by both dashboards."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from explainability.shap_explainer import EmbeddingShapExplainer
from shared.business_interpretation import feature_to_label
from shared.data_preparation import transform_categorical, transform_numeric
from shared.federated_data import StructuredDataset


def default_numeric_value(column: str, metadata: dict[str, Any] | None = None) -> float:
    if metadata:
        transformer_state = metadata.get("transformer_state", {})
        imputer_stats = transformer_state.get("numeric_imputer", {}).get("statistics")
        numeric_columns = metadata.get("numeric_columns", [])
        if imputer_stats and column in numeric_columns:
            idx = numeric_columns.index(column)
            if idx < len(imputer_stats):
                return float(imputer_stats[idx])
    return 0.0


def default_categorical_value(column: str, metadata: dict[str, Any]) -> str:
    vocabularies = metadata["categorical_vocabularies"]
    options = [value for value in vocabularies[column].keys() if value != "__unknown__"]
    if not options:
        return "__unknown__"
    if "missing" in options:
        return "missing"
    value_counts = (
        metadata.get("categorical_groupings", {})
        .get(column, {})
        .get("value_counts", {})
    )
    if value_counts:
        sorted_values = sorted(
            value_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
        )
        for value, _count in sorted_values:
            if value in options:
                return str(value)
    return str(options[0])


def build_prediction_inputs(
    metadata: dict[str, Any],
    *,
    form_name: str = "prediction-form",
    key_prefix: str = "",
    caption: str = "The form rebuilds `x_numeric` and `x_categorical` with the same schema as Phase 2.",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with st.form(form_name, clear_on_submit=False):
        st.markdown("### New prediction")
        st.caption(caption)

        values: dict[str, Any] = {}
        numeric_columns = metadata["numeric_columns"]
        categorical_columns = metadata["categorical_columns"]

        numeric_cols = st.columns(2)
        for index, column in enumerate(numeric_columns):
            with numeric_cols[index % 2]:
                kwargs: dict[str, Any] = {
                    "label": feature_to_label(column, config=config).capitalize(),
                    "value": default_numeric_value(column, metadata=metadata),
                    "step": 1.0,
                    "format": "%.2f",
                }
                if key_prefix:
                    kwargs["key"] = f"{key_prefix}num-{column}"
                values[column] = st.number_input(**kwargs)

        categorical_cols = st.columns(2)
        vocabularies = metadata["categorical_vocabularies"]
        for index, column in enumerate(categorical_columns):
            options = [
                value for value in vocabularies[column].keys() if value != "__unknown__"
            ]
            preferred = default_categorical_value(column, metadata)
            with categorical_cols[index % 2]:
                kwargs = {
                    "label": feature_to_label(column, config=config).capitalize(),
                    "options": options,
                    "index": options.index(preferred) if preferred in options else 0,
                }
                if key_prefix:
                    kwargs["key"] = f"{key_prefix}cat-{column}"
                values[column] = st.selectbox(**kwargs)

        submitted = st.form_submit_button(
            "Predict and explain", use_container_width=True
        )
    return {"submitted": submitted, "values": values}


def prepare_single_instance(
    values: dict[str, Any], runtime: dict[str, Any]
) -> tuple[pd.DataFrame, Any, Any]:
    """Transform raw form values into preprocessed numeric and categorical arrays."""
    bundle = runtime["bundle"]
    metadata = runtime["metadata"]
    frame = pd.DataFrame(
        [values], columns=metadata["numeric_columns"] + metadata["categorical_columns"]
    )
    x_numeric = transform_numeric(
        frame,
        metadata["numeric_columns"],
        bundle["numeric_imputer"],
        bundle["numeric_scaler"],
    )
    x_categorical = transform_categorical(
        frame,
        metadata["categorical_columns"],
        bundle["categorical_groupings"],
        bundle["categorical_vocabularies"],
        bundle["unknown_token"],
    )
    return frame, x_numeric, x_categorical


def run_instance_explanation(
    explainer: EmbeddingShapExplainer,
    train_dataset: StructuredDataset,
    x_numeric: Any,
    x_categorical: Any,
    top_k: int = 8,
    background_max_rows: int = 20,
    nsamples: int = 80,
) -> list[dict[str, Any]]:
    """Run SHAP instance explanation using the provided explainer and background dataset."""
    feature_count = len(explainer.engine.feature_names)
    if feature_count >= 50:
        return []
    if feature_count >= 25:
        nsamples = min(nsamples, 40)
        top_k = min(top_k, 6)
    background = explainer._sample_rows(
        explainer._combine_inputs(train_dataset), max_rows=background_max_rows
    )
    sample_dataset = type(
        "SingleDataset",
        (),
        {"x_numeric": x_numeric, "x_categorical": x_categorical, "y": None},
    )()
    sample, shap_values = explainer.explain_dataset(
        sample_dataset,
        background=background,
        sample_size=1,
        nsamples=nsamples,
    )
    return explainer._build_instance_explanations(shap_values, sample, top_k=top_k)


def build_vertical_shap_figure(
    explainer: EmbeddingShapExplainer,
    train_dataset: StructuredDataset,
    x_numeric: Any,
    x_categorical: Any,
    *,
    background_max_rows: int = 20,
    nsamples: int = 80,
) -> Any | None:
    """Build a vertical SHAP waterfall figure for a single instance."""
    try:
        feature_count = len(explainer.engine.feature_names)
        if feature_count >= 50:
            return None
        if feature_count >= 25:
            nsamples = min(nsamples, 40)
        background = explainer._sample_rows(
            explainer._combine_inputs(train_dataset), max_rows=background_max_rows
        )
        combined = np.concatenate(
            [
                np.asarray(x_numeric, dtype=np.float32),
                np.asarray(x_categorical, dtype=np.float32),
            ],
            axis=1,
        )
        kernel_explainer = explainer.engine.create_kernel_explainer(background)
        shap_values = explainer.engine.normalize_shap_output(
            kernel_explainer.shap_values(
                combined, nsamples=nsamples, l1_reg="num_features(10)"
            )
        )
        expected_value = kernel_explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            base_value = float(np.asarray(expected_value).reshape(-1)[0])
        else:
            base_value = float(expected_value)
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=combined[0],
            feature_names=explainer.engine.feature_names,
        )
        figure = plt.figure(figsize=(10, 5.6))
        shap.plots.waterfall(
            explanation,
            max_display=min(10, len(explainer.engine.feature_names)),
            show=False,
        )
        figure.tight_layout()
        return figure
    except Exception:
        return None
    return None


def build_vertical_bar_figure(
    frame: pd.DataFrame,
    *,
    category_column: str,
    value_column: str,
    title: str,
    color: str = "#1f77b4",
) -> Any | None:
    if (
        frame.empty
        or category_column not in frame.columns
        or value_column not in frame.columns
    ):
        return None
    plot_frame = frame[[category_column, value_column]].copy().head(12)
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.bar(
        plot_frame[category_column], plot_frame[value_column], color=color, width=0.68
    )
    axis.set_title(title, fontsize=15, fontweight="bold")
    axis.set_xlabel(category_column.replace("_", " ").title())
    axis.set_ylabel(value_column.replace("_", " ").title())
    axis.tick_params(axis="x", rotation=80)
    axis.grid(axis="y", linestyle="--", alpha=0.24)
    figure.tight_layout()
    return figure
