from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainability.shap_explainer import EmbeddingShapExplainer
from shared.auth_store import list_all_case_history
from shared.business_interpretation import (
    build_client_result_payload,
    build_rule_sentence,
    build_server_case_table,
    feature_to_label,
)
from shared.report_export import build_prediction_report_pdf
from shared.federated_data import (
    get_model_input_schema,
    load_preprocessing_bundle,
    load_preprocessing_metadata,
    load_project_config,
    load_train_dataset,
)
from shared.modeling import create_model, predict_probabilities
from shared.ui_components import (
    build_vertical_shap_figure,
    build_prediction_inputs,
    build_vertical_bar_figure,
    prepare_single_instance,
    run_instance_explanation,
)
from shared.utils import load_json


def _artifact_paths(run_root: Path | None = None) -> dict[str, Path]:
    root = run_root or Path("artifacts")
    return {
        "model": root / "models" / "global_model.pt",
        "metrics": root / "metrics" / "federated_training_metrics.json",
        "global_shap": root / "xai" / "global_shap_summary.json",
        "global_rules": root / "xai" / "global_rules_summary.json",
        "local_shap": root / "xai" / "local_shap_summaries.json",
        "instance_explanations": root / "xai" / "instance_explanations.json",
        "xai_validation": root / "xai" / "xai_validation_report.json",
        "client_audit": root / "audit" / "client_payload_audit.json",
        "server_audit": root / "audit" / "server_audit_report.json",
        "audit_summary": root / "audit" / "audit_validation_summary.json",
        "audit_log": root / "audit" / "audit_log.jsonl",
    }


# Legacy constant kept for backward compatibility (empty-artifacts fallback).
ARTIFACT_PATHS = _artifact_paths()


def discover_completed_runs() -> list[dict[str, Any]]:
    """Return all runs that have model artifacts, newest first."""
    user_runs = Path("artifacts/user_runs")
    runs: list[dict[str, Any]] = []
    if not user_runs.exists():
        return runs
    for manifest_path in sorted(
        user_runs.glob("*/run_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_root = manifest_path.parent
            # Only include runs where training has produced a model
            if not (run_root / "models" / "global_model.pt").exists():
                continue
            runs.append(
                {
                    "run_id": manifest.get("run_id", run_root.name),
                    "dataset_name": manifest.get("dataset_name", ""),
                    "timestamp_utc": manifest.get("timestamp_utc", ""),
                    "status": manifest.get("status", ""),
                    "artifact_root": run_root,
                }
            )
        except Exception:
            continue
    return runs


def build_default_model_bundle(
    config: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    schema = get_model_input_schema(metadata)
    model = create_model(config, schema=schema)
    return {
        "state_dict": model.state_dict(),
        "schema": schema,
        "model_family": config["model"]["family"],
        "model_type": config["model"]["type"],
        "target_name": "target",
        "selected_threshold": float(config["training"]["default_threshold"]),
        "hyperparameters": config["model"]["hyperparameters"],
        "training_state": "not_trained_yet",
    }


def ensure_default_global_model_artifact(
    config_path: str | Path = "config/project-config.json",
    metadata_path: str | Path | None = None,
    model_path: str | Path | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    paths = _artifact_paths(run_root)
    resolved_model_path = Path(model_path) if model_path is not None else paths["model"]
    _run_root = run_root or Path("artifacts")
    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else _run_root / "data" / "metadata" / "preprocessing_metadata.json"
    )
    if resolved_model_path.exists():
        return {"status": "available", "path": str(resolved_model_path)}

    if not resolved_metadata_path.exists():
        return {"status": "metadata_missing", "path": str(resolved_model_path)}

    config = load_project_config(config_path)
    metadata = load_preprocessing_metadata(path=resolved_metadata_path)
    model_bundle = build_default_model_bundle(config, metadata)
    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_bundle, resolved_model_path)
    return {"status": "created", "path": str(resolved_model_path)}


def build_default_metrics_payload(
    config: dict[str, Any], selected_threshold: float
) -> dict[str, Any]:
    metadata = None
    try:
        metadata = load_preprocessing_metadata()
    except Exception:
        metadata = None
    return {
        "model_family": config["model"]["family"],
        "model_type": config["model"]["type"],
        "numeric_feature_count": len(metadata["numeric_columns"]) if metadata else 0,
        "categorical_feature_count": len(metadata["categorical_columns"])
        if metadata
        else 0,
        "categorical_cardinalities": metadata["categorical_cardinalities"]
        if metadata
        else {},
        "embedding_dimensions": metadata["embedding_dimensions"] if metadata else {},
        "num_clients": int(config["federated_learning"]["num_clients"]),
        "num_rounds": 0,
        "saved_model_round": 0,
        "selected_threshold": float(selected_threshold),
        "observed_client_ids": [],
        "rounds": [],
        "training_state": "not_trained_yet",
    }


def _dashboard_artifact_signature(
    run_root: Path | None = None,
) -> tuple[tuple[str, bool, float], ...]:
    paths = _artifact_paths(run_root)
    signature: list[tuple[str, bool, float]] = []
    for key, path in paths.items():
        exists = path.exists()
        signature.append((key, exists, path.stat().st_mtime if exists else 0.0))
    return tuple(signature)


@st.cache_data(show_spinner=False)
def _load_dashboard_payload_cached(
    signature: tuple[tuple[str, bool, float], ...],
    run_root_str: str | None,
) -> dict[str, Any]:
    run_root = Path(run_root_str) if run_root_str else None
    paths = _artifact_paths(run_root)
    payload: dict[str, Any] = {"available": {}, "data": {}}
    for key, path in paths.items():
        exists = path.exists()
        payload["available"][key] = exists
        if exists and path.suffix == ".json":
            payload["data"][key] = load_json(path)
        elif exists and path.name.endswith(".jsonl"):
            with path.open("r", encoding="utf-8") as fh:
                payload["data"][key] = [json.loads(line) for line in fh if line.strip()]
    return payload


def load_dashboard_payload(run_root: Path | None = None) -> dict[str, Any]:
    return _load_dashboard_payload_cached(
        _dashboard_artifact_signature(run_root),
        str(run_root) if run_root else None,
    )


def _runtime_component_signature(
    run_root: Path | None = None,
) -> tuple[float, float, float]:
    paths = _artifact_paths(run_root)
    model_exists = paths["model"].exists()
    _root = run_root or Path("artifacts")
    metadata_path = _root / "data" / "metadata" / "preprocessing_metadata.json"
    effective_config_path = _root / "effective_config.json"
    return (
        paths["model"].stat().st_mtime if model_exists else 0.0,
        metadata_path.stat().st_mtime if metadata_path.exists() else 0.0,
        effective_config_path.stat().st_mtime
        if effective_config_path.exists()
        else 0.0,
    )


@st.cache_resource(show_spinner=False)
def _load_runtime_components_cached(
    signature: tuple[float, float, float],
    run_root_str: str | None,
) -> dict[str, Any]:
    run_root = Path(run_root_str) if run_root_str else None
    paths = _artifact_paths(run_root)
    ensure_default_global_model_artifact(run_root=run_root)
    _root = run_root or Path("artifacts")
    effective_config_path = _root / "effective_config.json"
    config = load_project_config(
        effective_config_path
        if effective_config_path.exists()
        else "config/project-config.json"
    )
    metadata_path = _root / "data" / "metadata" / "preprocessing_metadata.json"
    metadata = load_preprocessing_metadata(
        path=metadata_path if metadata_path.exists() else None
    )
    bundle = load_preprocessing_bundle(
        artifact_root=str(_root) if metadata_path.exists() else "artifacts"
    )
    model_bundle = torch.load(paths["model"], map_location="cpu", weights_only=True)
    model = create_model(config, schema=model_bundle["schema"])
    model.load_state_dict(model_bundle["state_dict"])
    model.eval()
    return {
        "config": config,
        "metadata": metadata,
        "bundle": bundle,
        "model_bundle": model_bundle,
        "model": model,
    }


def load_runtime_components(run_root: Path | None = None) -> dict[str, Any]:
    return _load_runtime_components_cached(
        _runtime_component_signature(run_root),
        str(run_root) if run_root else None,
    )


@st.cache_resource(show_spinner=False)
def load_instance_explainer(run_root_str: str | None = None) -> EmbeddingShapExplainer:
    _root = Path(run_root_str) if run_root_str else Path("artifacts")
    paths = _artifact_paths(_root)
    effective_config_path = _root / "effective_config.json"
    return EmbeddingShapExplainer(
        config_path=effective_config_path
        if effective_config_path.exists()
        else "config/project-config.json",
        model_path=paths["model"],
        artifact_root=str(_root),
    )


def build_round_metrics_frame(metrics_payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for round_payload in metrics_payload.get("rounds", []):
        rows.append(
            {
                "round": int(round_payload["round"]),
                "global_accuracy": round_payload["global_test_metrics"]["accuracy"],
                "global_precision": round_payload["global_test_metrics"]["precision"],
                "global_recall": round_payload["global_test_metrics"]["recall"],
                "global_f1": round_payload["global_test_metrics"]["f1"],
                "global_roc_auc": round_payload["global_test_metrics"]["roc_auc"],
                "global_pr_auc": round_payload["global_test_metrics"]["pr_auc"],
                "num_clients_participated": round_payload.get(
                    "num_clients_participated",
                    len(round_payload.get("client_metrics", [])),
                ),
                "validation_f1": round_payload["validation"]["validation_metrics"][
                    "f1"
                ],
                "validation_recall": round_payload["validation"]["validation_metrics"][
                    "recall"
                ],
                "selected_threshold": round_payload["validation"]["selected_threshold"],
            }
        )
    return pd.DataFrame(rows)


def build_shap_frame(
    summary: dict[str, Any], config: dict[str, Any] | None = None
) -> pd.DataFrame:
    mean_abs = summary.get("mean_abs_shap", {})
    frame = pd.DataFrame(
        [
            {
                "feature": feature_to_label(feature, config=config),
                "mean_abs_shap": value,
            }
            for feature, value in mean_abs.items()
        ]
    )
    if frame.empty:
        return frame
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def to_display_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=False)
    if hasattr(value, "tolist"):
        return json.dumps(value.tolist(), ensure_ascii=False)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def normalize_dataframe_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    display_frame = frame.copy()
    for column in display_frame.columns:
        inferred = pd.api.types.infer_dtype(display_frame[column], skipna=True)
        if pd.api.types.is_object_dtype(display_frame[column]) or inferred.startswith(
            "mixed"
        ):
            display_frame[column] = display_frame[column].map(to_display_value)
    return display_frame


def render_key_value_table(
    mapping: dict[str, Any], order: list[str] | None = None
) -> None:
    rows: list[dict[str, Any]] = []
    for key in order or list(mapping.keys()):
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, dict):
            value = ", ".join(
                f"{sub_key}: {sub_value}" for sub_key, sub_value in value.items()
            )
        elif isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        rows.append({"Field": key, "Value": str(to_display_value(value))})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def filter_case_history_for_run(
    case_history: list[dict[str, Any]], run_id: str | None
) -> list[dict[str, Any]]:
    normalized_run_id = str(run_id or "").strip()
    if not normalized_run_id:
        return case_history
    return [
        record
        for record in case_history
        if str(record.get("model_version", "")).strip() == normalized_run_id
        or str(record.get("case_id", "")).strip().startswith(f"{normalized_run_id}-")
    ]


def explain_prediction(
    x_numeric: Any, x_categorical: Any, run_root: Path | None = None
) -> list[dict[str, Any]]:
    run_root_str = str(run_root) if run_root else None
    explainer = load_instance_explainer(run_root_str)
    _root = run_root or Path("artifacts")
    train_dataset = load_train_dataset(artifact_root=str(_root))
    return run_instance_explanation(explainer, train_dataset, x_numeric, x_categorical)


def build_dashboard_shap_figure(
    x_numeric: Any, x_categorical: Any, run_root: Path | None = None
) -> Any | None:
    run_root_str = str(run_root) if run_root else None
    explainer = load_instance_explainer(run_root_str)
    _root = run_root or Path("artifacts")
    train_dataset = load_train_dataset(artifact_root=str(_root))
    return build_vertical_shap_figure(
        explainer, train_dataset, x_numeric, x_categorical
    )


def render_header(metrics_payload: dict[str, Any], selected_threshold: float) -> None:
    st.markdown(
        """
        <style>
        :root {
            --ensaj-primary: #1f4e79;
            --ensaj-primary-deep: #183d60;
            --ensaj-secondary: #1b7a73;
            --ensaj-bg: #f4f7f9;
            --ensaj-card: rgba(255,255,255,.96);
            --ensaj-text: #1d2b3a;
            --ensaj-muted: #5b697a;
        }
        .stApp {
            background: linear-gradient(180deg, var(--ensaj-bg) 0%, #edf2f5 34%, #f9fbfc 100%);
            color: var(--ensaj-text);
        }
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3 {
            color: var(--ensaj-text);
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        [data-testid="stExpanderToggleIcon"] {
            color: var(--ensaj-text) !important;
            fill: var(--ensaj-text) !important;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(31,78,121,0.96), rgba(27,122,115,0.92));
            color: #fff8ef;
            border: 1px solid rgba(255,255,255,0.18);
            box-shadow: 0 18px 50px rgba(31,78,121,0.14);
            margin-bottom: 1rem;
        }
        .hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: 0.02em;}
        .hero p {margin: 0.4rem 0 0; max-width: 54rem; color: rgba(255,248,239,0.92);}
        .hero h1, .hero p, .hero strong {color: #fff8ef !important;}
        div[data-testid="stMetric"] {
            background: var(--ensaj-card);
            border: 1px solid rgba(31,78,121,0.10);
            padding: 0.7rem 0.9rem;
            border-radius: 16px;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] p,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricDelta"] {
            color: #1d2b3a !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
            background: #fffdf9 !important;
            color: var(--ensaj-text) !important;
        }
        div[data-baseweb="select"] input {
            color: var(--ensaj-text) !important;
        }
        div[role="listbox"] ul,
        div[role="option"] {
            background: #fffdf9 !important;
            color: var(--ensaj-text) !important;
        }
        button[kind="secondary"],
        button[kind="primary"] {
            color: #1d2b3a;
        }
        .stForm [data-testid="stFormSubmitButton"] button,
        .stButton button {
            background: linear-gradient(135deg, #1f4e79, #1b7a73) !important;
            color: #fff8ef !important;
            border: 1px solid rgba(31,78,121,0.18) !important;
            font-weight: 600;
        }
        .stForm [data-testid="stFormSubmitButton"] button:hover,
        .stButton button:hover {
            background: linear-gradient(135deg, #183d60, #15665e) !important;
            color: #fff8ef !important;
        }
        .stForm [data-testid="stFormSubmitButton"] button p,
        .stButton button p,
        .stForm [data-testid="stFormSubmitButton"] button span,
        .stButton button span {
            color: #fff8ef !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            color: var(--ensaj-muted) !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--ensaj-primary) !important;
        }
        .stAlert, .stInfo, .stWarning, .stSuccess, .stError {
            color: var(--ensaj-text) !important;
            border-radius: 16px;
        }
        [data-testid="stJson"] {
            background: #11161e !important;
            border-radius: 14px;
            border: 1px solid rgba(31,78,121,0.10);
        }
        [data-testid="stJson"] *, [data-testid="stJson"] code, [data-testid="stJson"] pre {
            color: #f4f1ea !important;
        }
        [data-testid="stDataFrame"] *,
        [data-testid="stExpander"] * {
            color: var(--ensaj-text);
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a2e42 0%, #1f3d55 100%);
        }
        div[data-testid="stSidebar"] * {
            color: #f7f1e7 !important;
        }
        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] span,
        div[data-testid="stSidebar"] p,
        div[data-testid="stSidebar"] div {
            color: #f7f1e7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    latest_round = (
        metrics_payload["rounds"][-1] if metrics_payload.get("rounds") else None
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1>FL Global Dashboard</h1>
            <p>Visualization of global decisions, aggregated SHAP explanations and business rules derived from clients. The active decision threshold is <strong>{selected_threshold:.2f}</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    num_clients = metrics_payload.get("num_clients", 0)
    if latest_round:
        cols[0].metric(
            "Global accuracy", f"{latest_round['global_test_metrics']['accuracy']:.3f}"
        )
        cols[1].metric(
            "Global recall", f"{latest_round['global_test_metrics']['recall']:.3f}"
        )
        cols[2].metric(
            "Global ROC AUC", f"{latest_round['global_test_metrics']['roc_auc']:.3f}"
        )
    else:
        cols[0].metric("Global accuracy", "Not trained")
        cols[1].metric("Global recall", "Not trained")
        cols[2].metric("Global ROC AUC", "Not trained")
    cols[3].metric("Selected threshold", f"{selected_threshold:.2f}")
    cols[4].metric("Aggregated clients", str(num_clients))


def render_missing_artifacts(availability: dict[str, bool]) -> None:
    missing = [key for key, exists in availability.items() if not exists]
    if missing:
        st.warning(f"Missing artifacts: {', '.join(missing)}")


def render_metrics_tab(metrics_payload: dict[str, Any]) -> None:
    st.subheader("Global metrics")
    num_clients = metrics_payload.get("num_clients", 0)
    num_rounds = metrics_payload.get("num_rounds", 0)
    saved_model_round = metrics_payload.get("saved_model_round", 0)
    configured_target = metrics_payload.get("configured_client_target", num_clients)
    observed_client_ids = metrics_payload.get("observed_client_ids", [])
    st.caption(
        f"Aggregated data from {num_clients} client(s) over {num_rounds} round(s)"
    )

    info_cols = st.columns(3)
    info_cols[0].metric("Configured target", configured_target)
    info_cols[1].metric("Observed clients", num_clients)
    info_cols[2].metric(
        "Saved round", saved_model_round if saved_model_round else "Last round"
    )

    if observed_client_ids:
        st.caption(f"Observed client IDs: {', '.join(observed_client_ids)}")

    if num_clients == 1 and num_rounds > 0:
        st.warning(
            "Single-client aggregation detected: displayed metrics come "
            "from a single simulated client. For truly federated aggregation, run "
            "training with Target number of clients >= 2 from the client interface."
        )

    if saved_model_round > 0 and num_rounds > 0:
        st.info(
            f"Saved model is from round {saved_model_round} "
            f"(best validation accuracy). Total rounds: {num_rounds}."
        )

    round_frame = build_round_metrics_frame(metrics_payload)

    if not round_frame.empty:
        latest = round_frame.iloc[-1]
        decision_text = (
            "Global model in production"
            if float(latest["global_accuracy"]) >= 0.70
            else "Global model needs monitoring"
        )
        st.info(f"Business interpretation: {decision_text}.")
        st.line_chart(
            round_frame.set_index("round")[
                ["global_accuracy", "global_recall", "global_f1", "global_roc_auc"]
            ]
        )
        with st.expander("Technical metrics view"):
            st.dataframe(
                normalize_dataframe_for_display(round_frame),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No global metrics available.")


def render_xai_tab(
    payload: dict[str, Any], config: dict[str, Any] | None = None
) -> None:
    st.subheader("Global explainability")
    global_summary = payload["data"].get("global_shap", {})
    validation_report = payload["data"].get("xai_validation", {})

    shap_num_clients = global_summary.get("num_clients", 0)
    shap_total_examples = global_summary.get("total_examples", 0)
    if shap_num_clients:
        st.caption(
            f"SHAP Global from {shap_num_clients} client(s) - {shap_total_examples} total examples"
        )

    if shap_num_clients == 1:
        st.warning(
            "Single-client SHAP: the chart below represents feature importance "
            "from a single client. Restart training with multiple clients "
            "to obtain truly aggregated SHAP."
        )
    xai_status = validation_report.get("explainability_quality", {}).get(
        "status"
    ) or validation_report.get("meaningfulness_status", "")
    xai_message = validation_report.get("explainability_quality", {}).get(
        "message"
    ) or validation_report.get("meaningfulness_message", "")
    if xai_status:
        st.info(f"XAI status: {xai_status}")
    if xai_message:
        st.caption(xai_message)

    global_frame = build_shap_frame(global_summary, config=config)
    if not global_frame.empty:
        st.markdown("### Most influential business factors")
        shap_chart = build_vertical_bar_figure(
            global_frame,
            category_column="feature",
            value_column="mean_abs_shap",
            title="SHAP Global",
            color="#1f4e79",
        )
        if shap_chart is not None:
            st.pyplot(shap_chart, clear_figure=True, use_container_width=True)
        st.dataframe(
            normalize_dataframe_for_display(
                global_frame.head(20).rename(
                    columns={
                        "feature": "Business factor",
                        "mean_abs_shap": "Importance",
                    }
                )
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Global SHAP summary missing.")

    if validation_report:
        with st.expander("XAI validation report"):
            cols = st.columns(4)
            cols[0].metric(
                "Explained clients", validation_report.get("num_clients_explained", 0)
            )
            cols[1].metric("Features", validation_report.get("feature_count", 0))
            cols[2].metric("Numeric", validation_report.get("numeric_feature_count", 0))
            cols[3].metric(
                "Categorical", validation_report.get("categorical_feature_count", 0)
            )
            render_key_value_table(
                validation_report,
                [
                    "model_path",
                    "selected_threshold",
                    "saved_model_round",
                    "source_round_for_xai",
                    "shap_backend",
                    "instance_explanations_count",
                ],
            )
            artifacts = validation_report.get("artifacts", {})
            if artifacts:
                st.markdown("#### XAI artifacts")
                render_key_value_table(artifacts)
            generated_rules = validation_report.get("generated_rules", [])
            if generated_rules:
                st.markdown("#### IF-THEN rules")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Business rule": item.get("business_rule", ""),
                                "Technical rule": item.get("technical_rule", ""),
                            }
                            for item in generated_rules
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            if validation_report.get("privacy_validation"):
                st.markdown("#### Quality and privacy")
                render_key_value_table(validation_report["privacy_validation"])
                render_key_value_table(
                    validation_report.get("explainability_quality", {})
                )


def condition_label(config: dict[str, Any] | None) -> str:
    if not config:
        return ""
    return str(config.get("project", {}).get("condition_name", "")).strip()


def condition_risk_phrase(config: dict[str, Any] | None) -> str:
    label = condition_label(config)
    return f"{label} risk assessment" if label else "risk assessment"


def report_title(prefix: str, config: dict[str, Any] | None) -> str:
    label = condition_label(config)
    return f"{prefix} {label} report" if label else f"{prefix} prediction report"


def render_rules_tab(
    payload: dict[str, Any],
    config: dict[str, Any] | None = None,
    run_root: Path | None = None,
) -> None:
    st.subheader("Aggregated rules")
    global_rules = payload["data"].get("global_rules", {})
    rules = global_rules.get("rules", [])
    num_unique_rules = global_rules.get("num_unique_rules", len(rules))
    rules_num_clients = global_rules.get("num_clients", 0)
    if rules:
        st.caption(
            f"{num_unique_rules} unique aggregated rules from {rules_num_clients} client(s)"
        )
        frame = pd.DataFrame(
            [
                {
                    "Applied business rule": rule.get("business_description")
                    or build_rule_sentence(rule, config=config),
                    "Simple explanation": rule.get("description", ""),
                    "Decision": f"Contributes to {condition_risk_phrase(config)}",
                }
                for rule in rules
            ]
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)
        with st.expander("Aggregated rules technical report"):
            st.dataframe(
                normalize_dataframe_for_display(pd.DataFrame(rules)),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No aggregated rules available.")

    try:
        case_history = list_all_case_history(limit=100)
    except Exception:
        case_history = []
    case_history = filter_case_history_for_run(
        case_history, str(run_root.name).strip() if run_root is not None else None
    )
    if case_history:
        st.markdown("### Server case table")
        st.dataframe(
            build_server_case_table(case_history),
            use_container_width=True,
            hide_index=True,
        )


def get_audit_status_icon(status: str) -> str:
    if status == "passed":
        return "[OK]"
    elif status == "not_applicable":
        return "[N/A]"
    elif status == "limited":
        return "[LIM]"
    elif status == "warning":
        return "[WARN]"
    return "[FAIL]"


def get_audit_status_label(status: str) -> str:
    if status == "passed":
        return "PASSED"
    elif status == "not_applicable":
        return "NOT APPLICABLE"
    elif status == "limited":
        return "LIMITED"
    elif status == "warning":
        return "WARNING"
    return "FAILED"


def build_audit_dimension_frame(dimensions: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dim_name, dim_data in dimensions.items():
        notes = dim_data.get("notes", [])
        detail = notes[0] if notes else ""
        rows.append(
            {
                "Dimension": dim_name.replace("_", " ").title(),
                "Status": get_audit_status_label(dim_data.get("status", "unknown")),
                "Detail": detail,
            }
        )
    return pd.DataFrame(rows)


DIMENSION_DESCRIPTIONS = {
    "privacy": {
        "title": "Privacy",
        "description": "No raw data, no forbidden keys, no leaks in client payloads.",
        "thresholds": {
            "clients_with_warnings": ("Clients with warnings", 0, "max"),
            "forbidden_global_issue_detected": (
                "Global issue detected",
                False,
                "exact",
            ),
        },
    },
    "accuracy": {
        "title": "Accuracy",
        "description": "Model performance against minimum thresholds.",
        "thresholds": {
            "accuracy": ("Accuracy", 0.70, "min"),
            "recall": ("Recall", 0.35, "min"),
            "f1": ("F1", 0.20, "min"),
            "roc_auc": ("ROC AUC", 0.70, "min"),
        },
    },
    "fairness": {
        "title": "Fairness",
        "description": "Prediction equity across demographic groups.",
        "thresholds": {
            "gap": ("Gap between groups", 0.15, "max"),
        },
    },
    "data_drift": {
        "title": "Data Drift",
        "description": "Feature distribution stability between reference and current run.",
        "thresholds": {
            "max_relative_shift": ("Max relative shift", 0.25, "max"),
        },
    },
    "explainability": {
        "title": "Explainability",
        "description": "Presence and coherence of SHAP explanations and IF-THEN rules.",
        "thresholds": {
            "has_global_shap": ("Global SHAP available", True, "exact"),
            "has_rules": ("Rules generated", True, "exact"),
            "coherence": ("Rule/SHAP coherence", True, "exact"),
        },
    },
}


def render_audit_tab(
    payload: dict[str, Any], config: dict[str, Any] | None = None
) -> None:
    st.subheader("Audit - 5-Dimension Validation")

    server_audit = payload["data"].get("server_audit", {})
    audit_summary = payload["data"].get("audit_summary", {})

    if not server_audit and not audit_summary:
        st.info(
            "No audit report available. Run a full training cycle to generate audit outputs."
        )
        return

    dimensions = audit_summary.get("dimensions", {}) if audit_summary else {}
    overall_status = audit_summary.get("status", server_audit.get("status", "unknown"))

    if not dimensions:
        st.warning("Audit dimensions are not available in the report.")
        with st.expander("Raw server report"):
            render_key_value_table(server_audit)
        return

    passed_count = sum(1 for d in dimensions.values() if d.get("status") == "passed")
    warning_count = sum(1 for d in dimensions.values() if d.get("status") == "warning")
    failed_count = sum(1 for d in dimensions.values() if d.get("status") == "failed")

    cols = st.columns(5)
    for idx, (dim_key, dim_data) in enumerate(dimensions.items()):
        status = dim_data.get("status", "unknown")
        desc = DIMENSION_DESCRIPTIONS.get(dim_key, {})
        title = desc.get("title", dim_key.replace("_", " ").title())
        icon = get_audit_status_icon(status)
        cols[idx].metric(f"{icon} {title}", get_audit_status_label(status))

    if overall_status == "passed" and warning_count == 0:
        st.success(f"Overall status: PASSED ({passed_count}/5 dimensions validated)")
    elif overall_status == "warning" or warning_count > 0:
        st.warning(
            f"Overall status: WARNING ({passed_count}/5 passed, {warning_count} warning)"
        )
    else:
        st.error(
            f"Overall status: FAILED ({passed_count}/5 passed, {failed_count} failed)"
        )

    st.caption(
        f"Audited clients: {audit_summary.get('num_clients_audited', '-')} | "
        f"Audited rounds: {audit_summary.get('num_rounds_audited', '-')} | "
        f"Timestamp: {audit_summary.get('timestamp_utc', '-')}"
    )

    st.markdown("---")
    st.markdown("### Details per dimension")

    for dim_key, dim_data in dimensions.items():
        desc = DIMENSION_DESCRIPTIONS.get(dim_key, {})
        title = desc.get("title", dim_key.replace("_", " ").title())
        description = desc.get("description", "")
        status = dim_data.get("status", "unknown")
        icon = get_audit_status_icon(status)

        with st.expander(f"{icon} {title} - {get_audit_status_label(status)}"):
            st.caption(description)
            threshold_defs = desc.get("thresholds", {})
            metric_rows: list[dict[str, str]] = []

            for metric_key, (
                metric_label,
                threshold_value,
                comparison,
            ) in threshold_defs.items():
                actual = dim_data.get(metric_key)
                if actual is None:
                    continue

                if isinstance(actual, bool):
                    actual_str = "Yes" if actual else "No"
                    threshold_str = "Yes" if threshold_value else "No"
                    meets = actual == threshold_value
                elif isinstance(actual, (int, float)):
                    if comparison == "min":
                        meets = actual >= threshold_value
                        threshold_str = f">= {threshold_value}"
                    else:
                        meets = actual <= threshold_value
                        threshold_str = f"<= {threshold_value}"
                    if abs(actual) < 0.001:
                        actual_str = f"{actual:.4f}"
                    elif abs(actual) < 1:
                        actual_str = f"{actual:.3f}"
                    else:
                        actual_str = f"{actual:.2f}"
                else:
                    actual_str = str(actual)
                    threshold_str = str(threshold_value)
                    meets = actual == threshold_value

                check = "OK" if meets else "FAIL"
                metric_rows.append(
                    {
                        "": check,
                        "Metric": metric_label,
                        "Value": actual_str,
                        "Threshold": threshold_str,
                    }
                )

            if metric_rows:
                st.dataframe(
                    pd.DataFrame(metric_rows), use_container_width=True, hide_index=True
                )

            if dim_data.get("status") in {"not_applicable", "limited"} and dim_data.get(
                "details"
            ):
                st.info(str(dim_data["details"]))

            if dim_key == "fairness" and dim_data.get("split_rule"):
                st.markdown(
                    f"**Segmentation rule**: {dim_data['split_rule']}  \n"
                    f"- Younger group: positive rate = {dim_data.get('younger_positive_rate', 0):.1%}  \n"
                    f"- Older group: positive rate = {dim_data.get('older_positive_rate', 0):.1%}  \n"
                    f"- Gap: {dim_data.get('gap', 0):.1%}"
                )

            if dim_key == "data_drift":
                feature_drift = dim_data.get("feature_drift", [])
                if feature_drift:
                    drift_frame = pd.DataFrame(
                        [
                            {
                                "Feature": feature_to_label(
                                    d["feature"], config=config
                                ),
                                "Reference (mean)": f"{d['reference_mean']:.3f}",
                                "Current (mean)": f"{d['current_mean']:.3f}",
                                "Relative shift": f"{d['relative_shift']:.4f}",
                            }
                            for d in feature_drift
                        ]
                    )
                    st.dataframe(drift_frame, use_container_width=True, hide_index=True)

    with st.expander("Methodology notes"):
        for note in audit_summary.get("notes", []):
            st.markdown(f"- {note}")

    if server_audit:
        with st.expander("Full technical report"):
            model_summary = server_audit.get("model_summary", {})
            if model_summary:
                st.markdown("#### Model summary")
                render_key_value_table(
                    model_summary,
                    [
                        "model_family",
                        "model_type",
                        "tensor_count",
                        "selected_threshold",
                    ],
                )
            round_summaries = server_audit.get("round_summaries", [])
            if round_summaries:
                st.markdown("#### Per-round validation")
                round_rows = [
                    {
                        "Round": r.get("round_index", "?"),
                        "Status": get_audit_status_label(r.get("status", "unknown")),
                        "Clients": r.get("num_client_metrics", "?"),
                    }
                    for r in round_summaries
                ]
                st.dataframe(
                    pd.DataFrame(round_rows), use_container_width=True, hide_index=True
                )

    audit_log = payload["data"].get("audit_log", [])
    if audit_log:
        with st.expander("Audit logs"):
            for entry in audit_log:
                ts = entry.get("timestamp_utc", "")
                event = entry.get("event", "")
                status = entry.get("status", "")
                st.markdown(f"**[{ts}]** `{event}` - {get_audit_status_label(status)}")


def render_prediction_tab(
    runtime: dict[str, Any],
    payload: dict[str, Any],
    config: dict[str, Any] | None = None,
    run_root: Path | None = None,
) -> None:
    st.subheader("User prediction")
    st.caption(
        "The global model reloads the PyTorch artifact and applies the same preprocessing as Phase 2."
    )
    st.caption(
        "Explanation quality depends on dataset quality, selected columns and the schema used to build the current run."
    )

    prediction_inputs = build_prediction_inputs(runtime["metadata"], config=config)
    if not prediction_inputs["submitted"]:
        if payload["data"].get("instance_explanations"):
            st.info("Submit new data to get a prediction and instance explanation.")
        return

    raw_frame, x_numeric, x_categorical = prepare_single_instance(
        prediction_inputs["values"], runtime
    )
    probability = float(
        predict_probabilities(runtime["model"], x_numeric, x_categorical)[0]
    )
    threshold = float(runtime["model_bundle"]["selected_threshold"])
    explanation = explain_prediction(x_numeric, x_categorical, run_root=run_root)[0][
        "top_features"
    ]
    decision_payload = build_client_result_payload(
        probability=probability,
        threshold=threshold,
        explanation_rows=explanation,
        config=config,
    )

    cols = st.columns(3)
    cols[0].metric("Diagnosis", decision_payload["diagnosis_label"])
    cols[1].metric("Risk level", decision_payload["risk_level"])
    cols[2].metric("Final decision", decision_payload["final_decision"])
    target_name = str(runtime["config"]["data"].get("target_column", "")).strip()
    if target_name:
        st.caption(f"Target column used for this decision: `{target_name}`")

    st.markdown("### Input values")
    st.dataframe(
        normalize_dataframe_for_display(raw_frame),
        use_container_width=True,
        hide_index=True,
    )

    shap_figure = build_dashboard_shap_figure(
        x_numeric, x_categorical, run_root=run_root
    )
    if shap_figure is not None:
        st.markdown("### SHAP visualization")
        st.pyplot(shap_figure, clear_figure=True, use_container_width=True)
    else:
        st.caption(
            "SHAP visualization was skipped or reduced because the current schema is too wide or costly for interactive local explanation."
        )

    st.markdown("### IF-THEN rules")
    for rule in decision_payload["if_then_rules"]:
        st.markdown(f"- {rule}")
    st.markdown("### Local explanation")
    explanation_frame = pd.DataFrame(explanation)
    if not explanation_frame.empty and "shap_value" in explanation_frame.columns:
        explanation_frame = explanation_frame.copy()
        explanation_frame["feature"] = explanation_frame["feature"].map(
            lambda f: feature_to_label(f, config=config)
        )
        explanation_chart = build_vertical_bar_figure(
            explanation_frame.rename(
                columns={"shap_value": "value", "feature": "Factor"}
            ),
            category_column="Factor",
            value_column="value",
            title="SHAP contribution per factor",
            color="#bf5700",
        )
        if explanation_chart is not None:
            st.pyplot(explanation_chart, clear_figure=True, use_container_width=True)
    st.dataframe(
        normalize_dataframe_for_display(explanation_frame),
        use_container_width=True,
        hide_index=True,
    )

    _metrics_data = payload["data"].get("metrics") or {}
    pdf_bytes = build_prediction_report_pdf(
        title=report_title("Admin", config),
        role="admin",
        actor="dashboard_admin",
        model_info={
            "Architecture": runtime.get("config", {})
            .get("model", {})
            .get("type", "neural_network"),
            "FL rounds": _metrics_data.get("num_rounds", "-"),
            "FL clients": _metrics_data.get("num_clients", "-"),
            "Decision threshold": f"{threshold:.3f}",
        },
        prediction_info={
            "Diagnosis": decision_payload["diagnosis_label"],
            "Risk level": decision_payload["risk_level"],
            "Final decision": decision_payload["final_decision"],
        },
        input_frame=raw_frame,
        explanation_frame=explanation_frame,
        rules_frame=pd.DataFrame(
            payload["data"].get("global_rules", {}).get("rules", [])
        ),
        shap_figure=build_dashboard_shap_figure(
            x_numeric, x_categorical, run_root=run_root
        ),
    )
    st.download_button(
        "Export prediction PDF",
        data=pdf_bytes,
        file_name="dashboard-prediction-report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="FL XAI Audit Dashboard",
        page_icon=":material/monitoring:",
        layout="wide",
    )

    # ── Run selector ─────────────────────────────────────────────────────────
    completed_runs = discover_completed_runs()
    run_root: Path | None = None

    with st.sidebar:
        st.markdown("### Training run")
        if completed_runs:
            run_labels = [
                f"{r['run_id'][:28]}  ({r['dataset_name'] or 'unknown'})"
                for r in completed_runs
            ]
            selected_idx = st.selectbox(
                "Select run",
                range(len(run_labels)),
                format_func=lambda i: run_labels[i],
                index=0,
            )
            run_root = completed_runs[selected_idx]["artifact_root"]
            st.caption(
                f"Status: {completed_runs[selected_idx]['status']}  \n"
                f"Time: {completed_runs[selected_idx]['timestamp_utc']}"
            )
        else:
            st.info("No completed training runs found yet.")
    # ─────────────────────────────────────────────────────────────────────────

    config = load_project_config()
    if run_root is not None:
        run_cfg_path = run_root / "effective_config.json"
        if run_cfg_path.exists():
            config = load_project_config(run_cfg_path)

    model_bootstrap = ensure_default_global_model_artifact(run_root=run_root)
    payload = load_dashboard_payload(run_root=run_root)
    runtime: dict[str, Any] | None = None
    selected_threshold = float(config["training"]["default_threshold"])
    if payload["available"]["model"]:
        try:
            runtime = load_runtime_components(run_root=run_root)
            selected_threshold = float(runtime["model_bundle"]["selected_threshold"])
        except Exception as exc:
            st.warning(f"Global model exists but full loading remains limited: {exc}")

    metrics_payload = payload["data"].get("metrics") or build_default_metrics_payload(
        config, selected_threshold
    )
    render_header(metrics_payload, selected_threshold)
    if model_bootstrap["status"] == "created":
        st.info(
            "Default global model initialized. Dashboard available before first federated training."
        )
    elif model_bootstrap["status"] == "metadata_missing":
        st.info(
            "Dashboard runs in limited mode until preprocessing metadata is available."
        )
    if not payload["available"]["metrics"]:
        st.info(
            "Federated metrics will appear after first training. The dashboard remains accessible with an initial untrained state."
        )
    render_missing_artifacts(payload["available"])

    tabs = st.tabs(["Metrics", "XAI", "Rules", "Prediction", "Audit"])
    with tabs[0]:
        render_metrics_tab(metrics_payload)
    with tabs[1]:
        render_xai_tab(payload, config=config)
    with tabs[2]:
        render_rules_tab(payload, config=config, run_root=run_root)
    with tabs[3]:
        if runtime is None:
            st.info(
                "User prediction available once the global model and preprocessing metadata are loaded."
            )
        else:
            render_prediction_tab(runtime, payload, config=config, run_root=run_root)
    with tabs[4]:
        render_audit_tab(payload, config=config)


if __name__ == "__main__":
    main()
