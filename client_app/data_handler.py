"""Data loading and explanation utilities for the client portal."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import torch

from explainability.shap_explainer import EmbeddingShapExplainer
from shared.federated_data import (
    build_artifact_paths,
    load_preprocessing_bundle,
    load_preprocessing_metadata,
    load_project_config,
    load_train_dataset,
)
from shared.modeling import create_model
from shared.ui_components import build_vertical_shap_figure, run_instance_explanation


def _resolve_run_config_path(artifact_root: str) -> Path:
    run_config_path = Path(artifact_root) / "effective_config.json"
    return run_config_path if run_config_path.exists() else Path("config/project-config.json")


@st.cache_resource(show_spinner=False)
def load_run_runtime(artifact_root: str) -> dict[str, Any]:
    artifact_paths = build_artifact_paths(artifact_root)
    config = load_project_config(_resolve_run_config_path(artifact_root))
    metadata = load_preprocessing_metadata(artifact_root=artifact_root)
    bundle = load_preprocessing_bundle(artifact_root=artifact_root)
    model_bundle = torch.load(artifact_paths["model"], map_location="cpu", weights_only=True)
    model = create_model(config, schema=model_bundle["schema"])
    model.load_state_dict(model_bundle["state_dict"])
    model.eval()
    return {
        "artifact_paths": artifact_paths,
        "config": config,
        "metadata": metadata,
        "bundle": bundle,
        "model_bundle": model_bundle,
        "model": model,
    }


@st.cache_resource(show_spinner=False)
def load_run_explainer(artifact_root: str) -> EmbeddingShapExplainer:
    # Must pass the run-specific model path explicitly — the default in
    # EmbeddingShapExplainer points to the global production model, which has
    # a different schema from per-run models built on custom datasets.
    artifact_paths = build_artifact_paths(artifact_root)
    return EmbeddingShapExplainer(
        config_path=_resolve_run_config_path(artifact_root),
        model_path=artifact_paths["model"],
        artifact_root=artifact_root,
    )


def explain_prediction(runtime: dict[str, Any], x_numeric: Any, x_categorical: Any) -> list[dict[str, Any]]:
    explainer = load_run_explainer(str(runtime["artifact_paths"]["root"]))
    train_dataset = load_train_dataset(artifact_root=str(runtime["artifact_paths"]["root"]))
    return run_instance_explanation(explainer, train_dataset, x_numeric, x_categorical)


def build_prediction_shap_figure(runtime: dict[str, Any], x_numeric: Any, x_categorical: Any) -> Any | None:
    explainer = load_run_explainer(str(runtime["artifact_paths"]["root"]))
    train_dataset = load_train_dataset(artifact_root=str(runtime["artifact_paths"]["root"]))
    return build_vertical_shap_figure(explainer, train_dataset, x_numeric, x_categorical)
