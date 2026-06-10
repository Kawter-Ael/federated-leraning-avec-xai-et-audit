"""Page renderers and shared UI helpers for the client portal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from shared.business_interpretation import (
    build_client_result_payload,
    feature_to_label,
    summarize_local_xai,
)
from shared.report_export import build_prediction_report_pdf
from shared.auth_store import (
    list_case_history,
    list_user_runs,
    save_case_history,
    upsert_user_run,
)
from shared.client_workflow import (
    autodetect_dataset_roles,
    build_runtime_config,
    build_user_context,
    persist_uploaded_dataset,
    run_audit_for_mode,
    run_phase2_for_mode,
    run_training_for_mode,
    run_xai_for_mode,
    summarize_run_outputs,
    validate_uploaded_dataset,
)
from shared.federated_data import build_artifact_paths, load_project_config
from shared.modeling import predict_probabilities
from shared.ui_components import (
    build_prediction_inputs,
    build_vertical_bar_figure,
    prepare_single_instance,
)
from shared.utils import apply_config_override

from client_app.auth import current_user
from client_app.data_handler import (
    build_prediction_shap_figure,
    explain_prediction,
    load_run_runtime,
)


def get_base_config() -> dict[str, Any]:
    return load_project_config()


def load_preset_config(name: str) -> dict[str, Any] | None:
    preset_path = Path("config") / "presets" / f"{name}.json"
    if not preset_path.exists():
        return None
    return json.loads(preset_path.read_text(encoding="utf-8"))


def get_runtime_display_config(runtime_config: dict[str, Any] | None) -> dict[str, Any]:
    config = get_base_config()
    if not runtime_config:
        return config
    effective_config_path = (
        Path(runtime_config["artifact_root"]) / "effective_config.json"
    )
    if effective_config_path.exists():
        return load_project_config(effective_config_path)
    apply_config_override(config, runtime_config.get("config_override", {}))
    return config


def latest_run_manifest() -> dict[str, Any] | None:
    manifests: list[Path] = []
    root_manifest = Path("artifacts/run_manifest.json")
    if root_manifest.exists():
        manifests.append(root_manifest)
    runs_root = Path("artifacts/user_runs")
    if runs_root.exists():
        manifests.extend(runs_root.glob("*/run_manifest.json"))
    if not manifests:
        return None
    latest = max(manifests, key=lambda item: item.stat().st_mtime)
    data = json.loads(latest.read_text(encoding="utf-8"))
    if "artifact_root" in data:
        data["artifact_root"] = data["artifact_root"].replace("\\", "/")
    return data


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ensaj-primary: #1f4e79;
            --ensaj-primary-deep: #183d60;
            --ensaj-secondary: #1b7a73;
            --ensaj-bg: #f4f7f9;
            --ensaj-bg-soft: #eef4f6;
            --ensaj-card: rgba(255,255,255,.96);
            --ensaj-text: #1d2b3a;
            --ensaj-muted: #5b697a;
            --ensaj-success: #2f855a;
            --ensaj-warning: #c97a10;
            --ensaj-error: #c0565b;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(31,78,121,.08), transparent 28%),
                radial-gradient(circle at top right, rgba(27,122,115,.08), transparent 22%),
                linear-gradient(180deg, var(--ensaj-bg) 0%, #edf2f5 100%);
            color: var(--ensaj-text);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1260px;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div, li {
            color: var(--ensaj-text) !important;
        }
        .hero {
            background: linear-gradient(135deg, var(--ensaj-primary) 0%, #2a638b 55%, var(--ensaj-secondary) 100%);
            border-radius: 26px;
            padding: 28px 30px;
            margin-bottom: 1.2rem;
            box-shadow: 0 24px 46px rgba(31, 78, 121, .18);
        }
        .hero * {
            color: #fff8ee !important;
        }
        .hero small {
            display: inline-block;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: .45rem;
            opacity: .82;
        }
        .section-title {
            margin: 1.1rem 0 .2rem 0;
            font-size: 1.42rem;
            font-weight: 800;
        }
        .section-subtitle {
            color: var(--ensaj-muted) !important;
            margin-bottom: .9rem;
            font-size: .95rem;
        }
        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: .6rem;
            margin: .2rem 0 .5rem 0;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: .58rem .9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: .9rem;
            border: 1px solid transparent;
        }
        .status-ok {
            background: var(--ensaj-success);
            color: #f3fffc !important;
            border-color: #276749;
        }
        .status-off {
            background: #dde4ea;
            color: var(--ensaj-text) !important;
            border-color: #c9d1d9;
        }
        div[data-testid="stMetric"] {
            background: var(--ensaj-card);
            border: 1px solid rgba(31, 78, 121, .10);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 16px 34px rgba(31, 78, 121, .06);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
            color: var(--ensaj-text) !important;
        }
        .stButton > button {
            min-height: 3rem;
            border-radius: 14px;
            border: 1px solid rgba(31, 78, 121, .12);
            background: linear-gradient(135deg, var(--ensaj-primary) 0%, var(--ensaj-secondary) 100%);
            color: #ffffff !important;
            font-weight: 700;
            box-shadow: 0 12px 28px rgba(31, 78, 121, .16);
        }
        /* Force white on all inner text nodes — global p/span rule would override otherwise */
        .stButton > button p,
        .stButton > button span,
        .stButton > button div,
        [data-testid="stFormSubmitButton"] button p,
        [data-testid="stFormSubmitButton"] button span,
        [data-testid="stFormSubmitButton"] button div {
            color: #ffffff !important;
        }
        .stButton > button:hover,
        .stButton > button:hover p,
        .stButton > button:hover span,
        .stButton > button:hover div {
            background: linear-gradient(135deg, var(--ensaj-primary-deep) 0%, #166962 100%);
            color: #ffffff !important;
        }
        /* Sidebar Logout button */
        div[data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(135deg, #1f4e79 0%, #1b7a73 100%) !important;
            color: #ffffff !important;
            border: none !important;
        }
        div[data-testid="stSidebar"] .stButton > button p,
        div[data-testid="stSidebar"] .stButton > button span,
        div[data-testid="stSidebar"] .stButton > button div {
            color: #ffffff !important;
        }
        div[data-testid="stSidebar"] .stButton > button:hover,
        div[data-testid="stSidebar"] .stButton > button:hover p,
        div[data-testid="stSidebar"] .stButton > button:hover span {
            color: #ffffff !important;
            opacity: .88;
        }
        .stRadio label,
        .stTextInput label,
        .stTextArea label,
        .stFileUploader label,
        .stNumberInput label {
            color: var(--ensaj-text) !important;
            font-weight: 700;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
            color: var(--ensaj-text) !important;
            background: rgba(255,255,255,.98) !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span {
            color: var(--ensaj-text) !important;
            background: rgba(255,255,255,.98) !important;
        }
        div[role="listbox"], div[role="option"] {
            background: #ffffff !important;
            color: var(--ensaj-text) !important;
        }
        div[data-testid="stDataFrame"] * {
            color: var(--ensaj-text) !important;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #172330 0%, #21374b 100%);
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
        div[data-testid="stSidebar"] [data-baseweb="radio"] label {
            color: #f7f1e7 !important;
            opacity: 1 !important;
        }
        div[data-testid="stSidebar"] [data-baseweb="radio"] label span {
            color: #f7f1e7 !important;
            opacity: 1 !important;
        }
        div[data-testid="stSidebar"] [data-baseweb="radio"] label div {
            color: #f7f1e7 !important;
            opacity: 1 !important;
        }
        div[data-testid="stSidebar"] [data-baseweb="radio"] label p {
            color: #f7f1e7 !important;
            opacity: 1 !important;
        }
        div[data-testid="stSidebar"] [role="radiogroup"] label,
        div[data-testid="stSidebar"] [role="radiogroup"] label *,
        div[data-testid="stSidebar"] [role="radiogroup"] div,
        div[data-testid="stSidebar"] [role="radiogroup"] span,
        div[data-testid="stSidebar"] [role="radiogroup"] p {
            color: #f7f1e7 !important;
            opacity: 1 !important;
        }
        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
            color: #f7f1e7 !important;
        }
        div[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        div[data-testid="stSidebar"] div[data-baseweb="select"] input,
        div[data-testid="stSidebar"] div[data-baseweb="select"] span {
            color: #f7f1e7 !important;
            background: rgba(255,255,255,.08) !important;
        }
        div[data-testid="stSidebar"] [role="listbox"],
        div[data-testid="stSidebar"] [role="option"] {
            background: #22374b !important;
            color: #f7f1e7 !important;
        }
        code {
            color: var(--ensaj-primary) !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 16px;
            border: 1px solid rgba(31, 78, 121, .08);
        }
        div[data-testid="stAlert"] * {
            color: var(--ensaj-text) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True
        )


def status_row(status_map: dict[str, bool]) -> None:
    html = ['<div class="status-row">']
    for label, state in status_map.items():
        css = "status-ok" if state else "status-off"
        marker = "Available" if state else "Missing"
        html.append(f"<div class='status-pill {css}'>{label} - {marker}</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def summary_table(mapping: dict[str, Any], order: list[str] | None = None) -> None:
    rows = []
    for key in order or list(mapping.keys()):
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, list):
            display = ", ".join(str(item) for item in value[:8])
            if len(value) > 8:
                display += f" ... (+{len(value) - 8})"
        elif isinstance(value, dict):
            display = f"{len(value)} fields"
        else:
            display = str(value) if not isinstance(value, str) else value
        rows.append({"Field": key, "Value": display})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def hero(kicker: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <small>{kicker}</small>
            <h1>{title}</h1>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_shap_feature_frame(summary: dict[str, Any] | None) -> pd.DataFrame:
    if not summary:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    top_features = summary.get("top_features") or []
    if top_features:
        frame = pd.DataFrame(top_features)
        if "mean_abs_shap" in frame.columns:
            return frame[["feature", "mean_abs_shap"]]
    mean_abs = summary.get("mean_abs_shap", {})
    if not isinstance(mean_abs, dict):
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    return (
        pd.DataFrame(
            [
                {"feature": key, "mean_abs_shap": value}
                for key, value in mean_abs.items()
            ]
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


def build_business_feature_frame(
    summary: dict[str, Any] | None, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    frame = build_shap_feature_frame(summary)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["feature"] = frame["feature"].map(
        lambda f: feature_to_label(f, config=config)
    )
    frame.rename(
        columns={"feature": "business_factor", "mean_abs_shap": "importance"},
        inplace=True,
    )
    return frame


def render_home() -> None:
    config = get_base_config()
    manifest = latest_run_manifest()
    user = current_user()
    hero(
        "Client Local Workspace",
        "Client local interface",
        "Load a dataset, participate in federated training and view readable business results without raw technical outputs.",
    )

    cols = st.columns(3)
    cols[0].metric("Active dataset", Path(config["data"]["raw_dataset_path"]).name)
    cols[1].metric("Target clients", str(config["federated_learning"]["num_clients"]))
    cols[2].metric("Reference rounds", str(config["federated_learning"]["num_rounds"]))

    if user:
        section("Client session", "Account currently connected to the client portal.")
        info_cols = st.columns(3)
        info_cols[0].metric("Username", user["username"])
        info_cols[1].metric("Role", user["role"])
        user_runs = list_user_runs(user["username"])
        info_cols[2].metric("Known runs", len(user_runs))

    section("Project status", "Global artifact availability currently detected.")
    # Resolve artifact root from latest run manifest (docker_per_user mode) or
    # fall back to legacy global paths so the status reflects the actual run state.
    _manifest_for_status = manifest
    if _manifest_for_status:
        _art_root = Path(_manifest_for_status.get("artifact_root", "artifacts"))
    else:
        _art_root = Path("artifacts")
    status_row(
        {
            "Metadata": (
                _art_root / "data" / "metadata" / "preprocessing_metadata.json"
            ).exists(),
            "Model": (_art_root / "models" / "global_model.pt").exists(),
            "Metrics": (
                _art_root / "metrics" / "federated_training_metrics.json"
            ).exists(),
            "XAI": (_art_root / "xai" / "xai_validation_report.json").exists(),
            "Audit": (_art_root / "audit" / "server_audit_report.json").exists(),
        }
    )

    section("Latest known run", "Latest client manifest is summarized here.")
    if manifest:
        cols = st.columns(4)
        cols[0].metric("Run ID", manifest.get("run_id", "-"))
        cols[1].metric("Mode", manifest.get("run_mode", "-"))
        cols[2].metric("Dataset", manifest.get("dataset_name", "-"))
        cols[3].metric("Status", manifest.get("status", "-"))
        summary_table(
            manifest, ["timestamp_utc", "artifact_root", "dataset_path", "stage"]
        )
    else:
        st.info("No client run detected yet.")

    if user:
        section("User history", "Latest runs linked to your client account.")
        user_runs = list_user_runs(user["username"])
        if user_runs:
            st.dataframe(
                pd.DataFrame(user_runs[:10]), use_container_width=True, hide_index=True
            )
        else:
            st.info("No runs linked to this account yet.")

        section("Case history", "Latest client decisions in business language.")
        cases = list_case_history(user["username"], limit=10)
        if cases:
            display = pd.DataFrame(cases)[
                ["timestamp", "final_decision", "risk_level", "diagnosis_label"]
            ]
            display.rename(
                columns={
                    "timestamp": "Date",
                    "final_decision": "Decision",
                    "risk_level": "Risk level",
                    "diagnosis_label": "Diagnosis",
                },
                inplace=True,
            )
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("No cases recorded for this account.")


def render_dataset() -> None:
    config = get_base_config()
    hero(
        "Dataset Intake",
        "Dataset configuration",
        "Load a CSV, choose a run mode and prepare a coherent run without permanently overwriting the source configuration.",
    )

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        section(
            "Data source",
            "Each user prepares their data locally in a dedicated run space under `artifacts/user_runs/<run_id>/`.",
        )
        preset_cols = st.columns(2)
        if preset_cols[0].button("Load PIMA preset", use_container_width=True):
            st.session_state["client_dataset_preset"] = load_preset_config("diabetes")
        if preset_cols[1].button("Clear preset", use_container_width=True):
            st.session_state.pop("client_dataset_preset", None)
        uploaded_file = st.file_uploader(
            "Load a dataset", type=["csv", "xlsx", "xls", "tsv", "parquet"]
        )
        mode = st.radio(
            "Run mode", options=["principal", "temporaire"], horizontal=True
        )
        st.caption(
            "Upload any binary-classification CSV. Column roles are auto-detected and can be adjusted below."
        )

    with right:
        section(
            "Exposed hyperparameters",
            "The Flower server no longer blocks on a fixed number of clients. The target is informative and multiple clients can participate if available.",
        )
        num_clients = st.number_input(
            "Target number of clients",
            min_value=1,
            max_value=50,
            value=int(config["federated_learning"]["num_clients"]),
        )
        num_rounds = st.number_input(
            "Number of rounds",
            min_value=1,
            max_value=20,
            value=int(config["federated_learning"]["num_rounds"]),
        )
        local_epochs = st.number_input(
            "Local epochs",
            min_value=1,
            max_value=20,
            value=int(config["model"]["hyperparameters"]["local_epochs"]),
        )

    if uploaded_file is None:
        st.info("Load a CSV to build a client run.")
        return

    # --- Dynamic column detection from uploaded CSV ---
    from shared.dataset_io import load_tabular

    try:
        _preview_frame = load_tabular(uploaded_file)
        _all_columns = _preview_frame.columns.tolist()
        _auto_numeric = _preview_frame.select_dtypes(
            include=["number"]
        ).columns.tolist()
    except Exception as _exc:
        st.error(f"Cannot parse dataset: {_exc}")
        return

    detected_roles = autodetect_dataset_roles(_preview_frame)
    preset = st.session_state.get("client_dataset_preset") or {}
    preset_data = preset.get("data", {})
    preset_project = preset.get("project", {})
    _auto_target = detected_roles["target_column"]
    _auto_numeric_filtered = detected_roles["numeric_columns"]
    _auto_pii = detected_roles["excluded_columns"]
    _auto_fairness = detected_roles["fairness_attribute"] or ""
    if preset_data.get("target_column") in _all_columns:
        _auto_target = str(preset_data["target_column"])
    preset_numeric = [
        column
        for column in preset_data.get("numeric_columns", [])
        if column in _all_columns
    ]
    if preset_numeric:
        _auto_numeric_filtered = [
            column for column in preset_numeric if column != _auto_target
        ]
    if str(preset_data.get("fairness_attribute", "")).strip() in _all_columns:
        _auto_fairness = str(preset_data.get("fairness_attribute", "")).strip()

    section(
        "Column configuration",
        "Map dataset columns to pipeline roles. Values are auto-detected — adjust as needed.",
    )
    _cfg_left, _cfg_right = st.columns(2)

    with _cfg_left:
        target_column = st.selectbox(
            "Target column (label to predict)",
            _all_columns,
            index=_all_columns.index(_auto_target)
            if _auto_target in _all_columns
            else 0,
        )
    _candidate_cols = [c for c in _all_columns if c != target_column]
    _target_values = sorted(
        [str(v) for v in _preview_frame[target_column].dropna().unique()]
    )
    with _cfg_right:
        positive_class = st.selectbox(
            "Positive class value (outcome = 1)",
            _target_values,
            index=(
                _target_values.index(str(preset_data.get("positive_class")))
                if str(preset_data.get("positive_class")) in _target_values
                else 0
            ),
        )
    negative_values = [v for v in _target_values if v != str(positive_class)]
    _default_numeric = [
        c for c in _auto_numeric_filtered if c != target_column and c in _candidate_cols
    ]
    numeric_columns = st.multiselect(
        "Numeric columns", _candidate_cols, default=_default_numeric
    )
    _default_excluded = [c for c in _auto_pii if c in _candidate_cols]
    excluded_columns = st.multiselect(
        "Excluded columns (drop before training)",
        _candidate_cols,
        default=_default_excluded,
    )
    _unexcluded_pii = [
        c for c in _auto_pii if c not in excluded_columns and c != target_column
    ]
    if _unexcluded_pii:
        st.warning(
            f"Columns {_unexcluded_pii} look like identifiers or free text. "
            "Consider excluding them to avoid privacy leaks and garbage features."
        )
    _fairness_options = ["(none)"] + [
        c for c in numeric_columns if c in _candidate_cols
    ]
    _fairness_index = (
        _fairness_options.index(_auto_fairness)
        if _auto_fairness in _fairness_options
        else 0
    )
    fairness_attribute_choice = st.selectbox(
        "Fairness attribute (audit split dimension)",
        _fairness_options,
        index=_fairness_index,
        help="Numeric attribute used by the audit phase to split groups (e.g. Age). Choose '(none)' to skip.",
    )
    fairness_attribute = (
        None if fairness_attribute_choice == "(none)" else fairness_attribute_choice
    )
    section(
        "Business labels",
        "These labels are persisted in the run configuration and reused by prediction cards and PDF exports.",
    )
    biz_cols = st.columns(3)
    condition_name = biz_cols[0].text_input(
        "Condition name",
        value=str(preset_project.get("condition_name", "")),
    )
    positive_label = biz_cols[1].text_input(
        "Positive label",
        value=str(preset_project.get("positive_label", "Positive case")),
    )
    negative_label = biz_cols[2].text_input(
        "Negative label",
        value=str(preset_project.get("negative_label", "Negative case")),
    )
    st.info(
        "Supported pipeline contract: binary classification only, one target column only, tabular rows only, no free-text modeling, no automatic multiclass support."
    )

    if st.button("Validate and prepare run", use_container_width=True):
        preview_id = f"preview-{mode}"
        dataset_path = persist_uploaded_dataset(
            file_name=uploaded_file.name,
            content=uploaded_file.getvalue(),
            mode=mode,
            run_id=preview_id,
        )
        validation = validate_uploaded_dataset(
            dataset_path,
            target_column=target_column,
            positive_class=positive_class,
            excluded_columns=excluded_columns,
            numeric_columns=numeric_columns,
        )
        st.session_state["client_validation"] = validation
        if validation["valid"]:
            runtime_config = build_runtime_config(
                dataset_path=dataset_path,
                mode=mode,
                target_column=target_column,
                positive_class=positive_class,
                negative_classes=negative_values,
                excluded_columns=excluded_columns,
                numeric_columns=numeric_columns,
                num_clients=int(num_clients),
                num_rounds=int(num_rounds),
                local_epochs=int(local_epochs),
                original_dataset_name=uploaded_file.name,
                fairness_attribute=fairness_attribute,
                condition_name=condition_name,
                positive_label=positive_label,
                negative_label=negative_label,
            )
            final_dataset_path = persist_uploaded_dataset(
                file_name=uploaded_file.name,
                content=uploaded_file.getvalue(),
                mode=mode,
                run_id=runtime_config["run_id"],
            )
            if mode == "principal":
                runtime_config = build_runtime_config(
                    dataset_path=final_dataset_path,
                    mode=mode,
                    target_column=target_column,
                    positive_class=positive_class,
                    negative_classes=negative_values,
                    excluded_columns=excluded_columns,
                    numeric_columns=numeric_columns,
                    num_clients=int(num_clients),
                    num_rounds=int(num_rounds),
                    local_epochs=int(local_epochs),
                    original_dataset_name=uploaded_file.name,
                    fairness_attribute=fairness_attribute,
                    condition_name=condition_name,
                    positive_label=positive_label,
                    negative_label=negative_label,
                )
            else:
                runtime_config["dataset_path"] = final_dataset_path
                runtime_config["config_override"]["data"]["raw_dataset_path"] = (
                    final_dataset_path
                )
                runtime_config["config_override"]["data"]["reference_dataset_path"] = (
                    final_dataset_path
                )
            st.session_state["client_runtime_config"] = runtime_config
            user = current_user()
            if user:
                upsert_user_run(
                    user["username"], runtime_config, "configured", "dataset"
                )
            st.success("Dataset validated and run configuration ready.")
        else:
            st.error("Invalid dataset for the current pipeline.")

    validation = st.session_state.get("client_validation")
    if validation:
        section("Dataset validation", "Structure summary before pipeline execution.")
        cols = st.columns(5)
        cols[0].metric("Status", "Valid" if validation["valid"] else "Invalid")
        cols[1].metric("Rows", validation["num_rows"])
        cols[2].metric("Columns", validation["num_columns"])
        cols[3].metric("Numeric", len(validation.get("detected_numeric_columns", [])))
        cols[4].metric(
            "Categorical", len(validation.get("detected_categorical_columns", []))
        )
        severity = validation.get("severity_counts", {})
        sev_cols = st.columns(3)
        sev_cols[0].metric("Blocking", severity.get("blocking", 0))
        sev_cols[1].metric("Warnings", severity.get("warning", 0))
        sev_cols[2].metric("Info", severity.get("info", 0))
        for item in validation.get("blocking", []):
            st.error(item["message"])
        for item in validation.get("warning", []):
            st.warning(item["message"])
        for item in validation.get("info", []):
            st.info(item["message"])
        st.dataframe(
            pd.DataFrame({"Detected columns": validation.get("columns", [])}),
            use_container_width=True,
            hide_index=True,
            height=260,
        )
        if isinstance(validation.get("preview"), pd.DataFrame):
            st.caption("Preview of first 10 rows")
            st.dataframe(
                validation["preview"].head(10),
                use_container_width=True,
                hide_index=True,
            )

    runtime_config = st.session_state.get("client_runtime_config")
    if runtime_config:
        section("Run configuration", "Compact view of effective overrides.")
        cols = st.columns(4)
        cols[0].metric("Run ID", runtime_config["run_id"])
        cols[1].metric("Mode", runtime_config["run_mode"])
        cols[2].metric("Dataset", runtime_config["dataset_name"])
        cols[3].metric("Artifacts", runtime_config["artifact_root"])
        summary_table(
            {
                "target_column": runtime_config["config_override"]["data"][
                    "target_column"
                ],
                "positive_class": runtime_config["config_override"]["data"][
                    "positive_class"
                ],
                "negative_classes": runtime_config["config_override"]["data"][
                    "negative_classes"
                ],
                "excluded_columns": runtime_config["config_override"]["data"][
                    "excluded_columns"
                ],
                "numeric_columns": runtime_config["config_override"]["data"][
                    "numeric_columns"
                ],
                "configured_client_target": runtime_config["config_override"][
                    "federated_learning"
                ]["num_clients"],
                "num_rounds": runtime_config["config_override"]["federated_learning"][
                    "num_rounds"
                ],
                "min_fit_clients": runtime_config["config_override"][
                    "federated_learning"
                ]["min_fit_clients"],
                "min_available_clients": runtime_config["config_override"][
                    "federated_learning"
                ]["min_available_clients"],
                "local_epochs": runtime_config["config_override"]["model"][
                    "hyperparameters"
                ]["local_epochs"],
            }
        )
        if isinstance(validation, dict):
            complexity = validation.get("complexity", {})
            complexity_rows = [
                {
                    "Field": "Detected numeric",
                    "Value": len(validation.get("detected_numeric_columns", [])),
                },
                {
                    "Field": "Detected categorical",
                    "Value": len(validation.get("detected_categorical_columns", [])),
                },
                {
                    "Field": "Target cardinality",
                    "Value": complexity.get("target_cardinality", 0),
                },
                {
                    "Field": "Class balance",
                    "Value": complexity.get("class_balance", {}),
                },
                {
                    "Field": "Max categorical cardinality",
                    "Value": complexity.get("max_categorical_cardinality", 0),
                },
                {
                    "Field": "Estimated SHAP cost",
                    "Value": complexity.get("shap_cost_level", "-"),
                },
                {
                    "Field": "Estimated split risk",
                    "Value": complexity.get("split_risk", "-"),
                },
            ]
            st.caption(
                "Dataset complexity affects training time and explanation quality. High-cardinality or highly imbalanced datasets can slow down XAI and audit."
            )
            st.dataframe(
                pd.DataFrame(complexity_rows), use_container_width=True, hide_index=True
            )


def render_preprocessing() -> None:
    hero(
        "Phase 2",
        "Preprocessing",
        "Local dataset preparation, cleaning, split creation and common schema generation reused throughout the pipeline.",
    )
    runtime_config = st.session_state.get("client_runtime_config")
    if not runtime_config:
        st.info("First validate a dataset in the Dataset tab.")
        return

    validation = st.session_state.get("client_validation") or {}
    complexity = validation.get("complexity", {})
    pretraining_warnings: list[str] = []
    if complexity.get("max_categorical_cardinality", 0) >= 30:
        pretraining_warnings.append(
            "High-cardinality categorical features detected. Training and XAI may be slower."
        )
    if complexity.get("split_risk") == "warning":
        pretraining_warnings.append(
            "Class distribution is small for at least one class. Validation/test metrics may be unstable."
        )
    if complexity.get("split_risk") == "blocking":
        pretraining_warnings.append(
            "Class distribution is too small for a safe split. Fix the dataset before continuing."
        )
    if complexity.get("shap_cost_level") == "high":
        pretraining_warnings.append(
            "High SHAP cost expected. Explainability may be degraded or deliberately reduced."
        )
    for message in pretraining_warnings:
        st.warning(message)

    section(
        "Data augmentation",
        "Optional: augment training data to improve model robustness.",
    )
    # Widget state is managed exclusively through Streamlit keys stored in
    # session_state. Never derive `value=` from runtime_config on re-renders —
    # mixing key-managed state with value= overrides causes React #185.
    _METHODS = {"SMOTE": "smote", "Noise injection": "noise", "Mixup": "mixup"}
    aug_enabled = st.checkbox("Enable data augmentation", key="aug_enabled_check")
    aug_method_label = st.selectbox(
        "Augmentation method",
        options=list(_METHODS.keys()),
        key="aug_method_sel",
        disabled=not aug_enabled,
    )
    aug_factor = st.slider(
        "Augmentation factor",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="aug_factor_slide",
        disabled=not aug_enabled,
    )
    aug_method = _METHODS[aug_method_label]

    section("Execution")
    if st.button("Run Phase 2", use_container_width=True):
        with st.spinner("Preparing data..."):
            # Apply augmentation config only inside the button handler to avoid
            # mutating session_state on every render (causes React re-render loop).
            if aug_enabled:
                runtime_config["config_override"]["data_augmentation"] = {
                    "enabled": True,
                    "method": aug_method,
                    "factor": aug_factor,
                }
            else:
                runtime_config["config_override"].pop("data_augmentation", None)
            st.session_state["client_runtime_config"] = runtime_config
            user = current_user()
            runtime_config["pending_client_id"] = user["username"]
            st.session_state["client_phase2"] = run_phase2_for_mode(runtime_config)
            if user:
                upsert_user_run(
                    user["username"], runtime_config, "phase2_completed", "phase2"
                )
        st.success("Phase 2 completed.")

    result = st.session_state.get("client_phase2")
    if result:
        balance = result.get("class_balance") or {}
        if balance.get("severe_imbalance"):
            positive_rate = balance.get("positive_rate", 0.0)
            min_pos = balance.get("min_split_positives", 0)
            st.warning(
                f"Severe class imbalance detected: positive rate {positive_rate:.2%}, "
                f"minimum positives across splits = {min_pos}. Minority oversampling "
                "will be auto-enabled during training to avoid a trivial all-negative "
                "model. For best results, upload more positive examples or enable "
                "SMOTE/Mixup explicitly above."
            )
        section("Preprocessing summary", "Shapes, final schema and processed columns.")
        cols = st.columns(5)
        cols[0].metric("Raw rows", result["raw_shape"][0])
        cols[1].metric("Raw cols", result["raw_shape"][1])
        cols[2].metric("Clean rows", result["cleaned_shape"][0])
        cols[3].metric("Clean cols", result["cleaned_shape"][1])
        cols[4].metric("Clients", result["num_clients"])
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Split": "Train x_numeric",
                        "Shape": str(result["train_shape"]["x_numeric"]),
                    },
                    {
                        "Split": "Train x_categorical",
                        "Shape": str(result["train_shape"]["x_categorical"]),
                    },
                    {
                        "Split": "Validation x_numeric",
                        "Shape": str(result["validation_shape"]["x_numeric"]),
                    },
                    {
                        "Split": "Validation x_categorical",
                        "Shape": str(result["validation_shape"]["x_categorical"]),
                    },
                    {
                        "Split": "Test x_numeric",
                        "Shape": str(result["test_shape"]["x_numeric"]),
                    },
                    {
                        "Split": "Test x_categorical",
                        "Shape": str(result["test_shape"]["x_categorical"]),
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        twin = st.columns(2)
        twin[0].dataframe(
            pd.DataFrame({"Numeric columns": result["numeric_columns"]}),
            use_container_width=True,
            hide_index=True,
        )
        twin[1].dataframe(
            pd.DataFrame({"Categorical columns": result["categorical_columns"]}),
            use_container_width=True,
            hide_index=True,
        )
        # Per-run schema is built from the uploaded CSV itself, so there is
        # no "global schema" to diverge from. Clients train on their own
        # numeric + categorical columns as detected in Phase 2.


def render_training() -> None:
    hero(
        "Phase 3",
        "User training",
        "The portal triggers a real Flower client identified by the authenticated user, participating in federated training.",
    )
    runtime_config = st.session_state.get("client_runtime_config")
    if not runtime_config:
        st.info("First validate a dataset in the Dataset tab.")
        return
    config = get_runtime_display_config(runtime_config)

    artifact_root = runtime_config["artifact_root"]
    metadata_path = (
        Path(artifact_root) / "data" / "metadata" / "preprocessing_metadata.json"
    )
    phase2_done = metadata_path.exists()
    phase2_result = st.session_state.get("client_phase2")

    if not phase2_done and not phase2_result:
        st.warning(
            "Phase 2 (preprocessing) must be completed before training. "
            "Go to the **Preprocessing** tab and click **Run Phase 2** first."
        )
        return

    section("Run configuration")
    _cfg_cols = st.columns(4)
    _cfg_cols[0].metric("Run ID", runtime_config["run_id"])
    _cfg_cols[1].metric("Mode", runtime_config["run_mode"])
    _cfg_cols[2].metric("Dataset", runtime_config["dataset_name"])
    _num_rounds = int(
        runtime_config["config_override"]["federated_learning"]["num_rounds"]
    )
    _local_epochs = int(
        runtime_config["config_override"]["model"]["hyperparameters"]["local_epochs"]
    )
    _cfg_cols[3].metric("Rounds x Epochs", f"{_num_rounds} x {_local_epochs}")

    section("Execution")
    if st.button("Start training", use_container_width=True):
        _progress = st.progress(0, text="Preparing data...")
        progress_map = {
            "prepare_data": 5,
            "server_start": 15,
            "training": 45,
            "xai": 78,
            "audit": 90,
            "finalize": 96,
        }

        def _progress_callback(stage: str, text: str) -> None:
            _progress.progress(progress_map.get(stage, 20), text=text)

        user = current_user()
        user_context = build_user_context(user["username"], runtime_config)
        user_context["progress_callback"] = _progress_callback
        try:
            st.session_state["client_training"] = run_training_for_mode(
                runtime_config, user_context
            )
        except RuntimeError as exc:
            _progress.empty()
            log_dir = Path(artifact_root) / "logs"
            with st.expander("Training failed — click to view details", expanded=True):
                st.error(str(exc)[:2000])
                for log_name in ("server_stderr.log", "client_stderr.log"):
                    log_path = log_dir / log_name
                    if log_path.exists():
                        tail = log_path.read_text(encoding="utf-8")[-3000:]
                        st.subheader(log_name)
                        st.code(tail, language="log")
                error_json = Path(artifact_root) / "client_prepare_error.json"
                if error_json.exists():
                    st.json(json.loads(error_json.read_text(encoding="utf-8")))
            return

        global_artifact_root = runtime_config.get(
            "global_artifact_root", runtime_config["artifact_root"]
        )
        global_paths = build_artifact_paths(global_artifact_root)
        xai_report_path = global_paths["xai_root"] / "xai_validation_report.json"
        global_shap_path = global_paths["xai_root"] / "global_shap_summary.json"
        audit_report_path = global_paths["audit_summary"]
        st.session_state["client_xai"] = (
            json.loads(xai_report_path.read_text(encoding="utf-8"))
            if xai_report_path.exists()
            else None
        )
        st.session_state["client_global_shap"] = (
            json.loads(global_shap_path.read_text(encoding="utf-8"))
            if global_shap_path.exists()
            else None
        )
        st.session_state["client_audit"] = (
            json.loads(audit_report_path.read_text(encoding="utf-8"))
            if audit_report_path.exists()
            else None
        )
        user = current_user()
        if user:
            upsert_user_run(
                user["username"], runtime_config, "audit_completed", "audit"
            )
        _progress.progress(100, text="Done!")
        st.success(
            "Training complete. Phases: server start -> FL rounds -> XAI -> audit."
        )

    payload = st.session_state.get("client_training")
    if payload:
        latest_round = payload["rounds"][-1]
        global_metrics = latest_round["global_test_metrics"]
        section("Global indicators", "Quick performance overview of the last round.")
        if payload.get("num_clients", 1) == 1:
            st.warning(
                "Single-client aggregation: displayed metrics come from a single "
                "simulated client. For truly federated aggregation, increase the "
                "target number of clients from the Dataset tab."
            )
        saved_model_round = payload.get("saved_model_round", 0)
        num_rounds = payload.get("num_rounds", 0)
        if saved_model_round > 0 and num_rounds > 0 and saved_model_round != num_rounds:
            st.info(
                f"Saved model uses round {saved_model_round} (best validation accuracy) "
                f"out of {num_rounds} total rounds."
            )
        cols = st.columns(4)
        cols[0].metric("Global performance", f"{global_metrics['accuracy']:.3f}")
        cols[1].metric("Positive case detection", f"{global_metrics['recall']:.3f}")
        cols[2].metric("Precision/recall balance", f"{global_metrics['f1']:.3f}")
        cols[3].metric("Decision threshold", f"{payload['selected_threshold']:.2f}")

        _aug_summaries = [
            cm.get("augmentation_summary")
            for cm in latest_round.get("client_metrics", [])
            if cm.get("augmentation_summary")
        ]
        if _aug_summaries:
            _aug = _aug_summaries[0]
            section(
                "Augmentation summary",
                "Data augmentation was applied during training. It modifies training rows only; columns remain unchanged.",
            )
            _aug_cols = st.columns(5)
            _aug_cols[0].metric("Method", str(_aug.get("method", "-")))
            _aug_cols[1].metric("Factor", f"{_aug.get('factor', 0):.1f}")
            _aug_cols[2].metric("Original rows", _aug.get("original_size", "-"))
            _aug_cols[3].metric("Augmented rows", _aug.get("augmented_size", "-"))
            _reason = _aug.get("auto_enabled_reason")
            _aug_cols[4].metric(
                "Trigger",
                "Auto (imbalance)" if _reason else "Manual",
            )
            st.caption(
                "Data augmentation does not add or remove columns. It only increases "
                "the number of training rows to improve model robustness."
            )
        else:
            _aug_cfg = runtime_config.get("config_override", {}).get(
                "data_augmentation", {}
            )
            if _aug_cfg.get("enabled"):
                st.caption(
                    "Augmentation was configured but no summary was returned by the client."
                )

        section(
            "My model contribution",
            "Local metrics and top features contributed by this client in the last federated round.",
        )
        user = current_user()
        my_client_id = user["username"].strip().lower() if user else None
        my_metric = next(
            (
                cm
                for cm in latest_round["client_metrics"]
                if cm.get("client_id", "").strip().lower() == my_client_id
            ),
            latest_round["client_metrics"][0]
            if latest_round["client_metrics"]
            else None,
        )
        if my_metric:
            _local_acc = float(my_metric.get("accuracy", 0))
            _local_rec = float(my_metric.get("recall", 0))
            _local_f1 = float(my_metric.get("f1", 0))
            _zero_metrics = _local_acc == 0.0 and _local_rec == 0.0 and _local_f1 == 0.0
            if _zero_metrics:
                st.warning(
                    "⚠️ Local test metrics are all 0. This usually means the local test partition "
                    "contains only one class (no positive or no negative samples). "
                    "See global indicators above for federated performance."
                )
            my_cols = st.columns(4)
            my_cols[0].metric("Local accuracy", f"{_local_acc:.3f}")
            my_cols[1].metric("Local recall", f"{_local_rec:.3f}")
            my_cols[2].metric("Local F1", f"{_local_f1:.3f}")
            my_cols[3].metric("Examples used", my_metric.get("num_examples", 0))
            top_features = my_metric.get("shap_summary", {}).get("top_features", [])
            if top_features:
                st.markdown("**Top contributing factors (local SHAP)**")
                shap_rows = [
                    {
                        "Factor": feature_to_label(f.get("feature", ""), config=config),
                        "Mean |SHAP|": round(f.get("mean_abs_shap", 0), 4),
                        "Direction": f.get("direction", ""),
                    }
                    for f in top_features[:8]
                ]
                st.dataframe(
                    pd.DataFrame(shap_rows), use_container_width=True, hide_index=True
                )
            num_rules = int(my_metric.get("rule_summary", {}).get("num_rules", 0))
            if num_rules:
                st.caption(
                    f"{num_rules} local IF-THEN rule(s) contributed to the federated aggregation."
                )
        else:
            st.info("No local metrics found for this client in the last round.")


def render_xai_audit() -> None:
    hero(
        "Phase 4 + Phase 5",
        "XAI and Audit",
        "XAI and audit outputs are produced server-side from federated artifacts. The client portal reviews them without executing server logic.",
    )
    runtime_config = st.session_state.get("client_runtime_config")
    if not runtime_config:
        st.info("First validate a dataset in the Dataset tab.")
        return
    config = get_runtime_display_config(runtime_config)

    section("Review")
    st.info(
        "XAI and audit outputs are generated from server global artifacts. They can be regenerated here after training."
    )
    global_artifact_root = runtime_config.get(
        "global_artifact_root", runtime_config["artifact_root"]
    )
    global_paths = build_artifact_paths(global_artifact_root)
    xai_report = global_paths["xai_root"] / "xai_validation_report.json"
    global_shap_path = global_paths["xai_root"] / "global_shap_summary.json"
    audit_report = global_paths["audit_summary"]
    action_cols = st.columns(2)
    if action_cols[0].button("Generate XAI", use_container_width=True):
        with st.spinner("Generating XAI..."):
            st.session_state["client_xai"] = run_xai_for_mode(runtime_config)
        st.success("XAI artifacts generated.")
    if action_cols[1].button("Generate audit", use_container_width=True):
        with st.spinner("Generating audit..."):
            st.session_state["client_audit"] = run_audit_for_mode(runtime_config)
        st.success("Audit artifacts generated.")

    st.session_state["client_xai"] = (
        json.loads(xai_report.read_text(encoding="utf-8"))
        if xai_report.exists()
        else None
    )
    st.session_state["client_global_shap"] = (
        json.loads(global_shap_path.read_text(encoding="utf-8"))
        if global_shap_path.exists()
        else None
    )
    st.session_state["client_audit"] = (
        json.loads(audit_report.read_text(encoding="utf-8"))
        if audit_report.exists()
        else None
    )

    xai_payload = st.session_state.get("client_xai")
    if xai_payload:
        section("XAI summary", "Explainability output summary from the server.")
        cols = st.columns(4)
        cols[0].metric("Explained clients", xai_payload.get("num_clients_explained", 0))
        cols[1].metric("Features", xai_payload.get("feature_count", 0))
        cols[2].metric("Numeric", xai_payload.get("numeric_feature_count", 0))
        cols[3].metric("Categorical", xai_payload.get("categorical_feature_count", 0))
        st.caption(
            "Global explanations are available and consistent with the current model."
        )
        xai_status = xai_payload.get("explainability_quality", {}).get(
            "status"
        ) or xai_payload.get("meaningfulness_status", "")
        xai_message = xai_payload.get("explainability_quality", {}).get(
            "message"
        ) or xai_payload.get("meaningfulness_message", "")
        if xai_status:
            st.info(f"XAI status: {xai_status}")
        if xai_message:
            st.caption(xai_message)
        global_shap_payload = st.session_state.get("client_global_shap")
        shap_frame = build_business_feature_frame(global_shap_payload, config=config)
        if not shap_frame.empty:
            st.markdown("#### SHAP LOCAL")
            shap_chart = build_vertical_bar_figure(
                shap_frame,
                category_column="business_factor",
                value_column="importance",
                title="SHAP LOCAL",
                color="#bf5a2a",
            )
            if shap_chart is not None:
                st.pyplot(shap_chart, clear_figure=True, use_container_width=True)
        generated_rules = xai_payload.get("generated_rules", [])
        if generated_rules:
            st.markdown("#### Generated IF-THEN rules")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Business rule": row.get("business_rule", ""),
                            "Technical rule": row.get("technical_rule", ""),
                        }
                        for row in generated_rules
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    audit_payload = st.session_state.get("client_audit")
    if audit_payload:
        section("Audit summary", "Privacy and coherence constraint validation.")
        cols = st.columns(3)
        cols[0].metric("Status", str(audit_payload.get("status", "-")).upper())
        cols[1].metric("Audited clients", audit_payload.get("num_clients_audited", 0))
        cols[2].metric("Audited rounds", audit_payload.get("num_rounds_audited", 0))
        for note in audit_payload.get("notes", []):
            st.markdown(f"- {note}")
        dimensions = audit_payload.get("dimensions", {})
        if dimensions:
            st.markdown("#### AI Audit - 5 dimensions")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Dimension": key.replace("_", " ").title(),
                            "Status": value.get("status", "-"),
                            "Summary": str(
                                value.get(
                                    "details",
                                    value.get("gap", value.get("accuracy", "")),
                                )
                            ),
                        }
                        for key, value in dimensions.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.caption("Full technical detail is available server-side for admin analysis.")
    if not xai_payload and not audit_payload:
        st.caption("No XAI or audit global artifacts found yet.")


def render_results() -> None:
    hero(
        "Results",
        "Final run summary",
        "Compact artifact view and final metrics for the current run.",
    )
    runtime_config = st.session_state.get("client_runtime_config")
    if not runtime_config:
        st.info("First validate a dataset in the Dataset tab.")
        return
    config = get_runtime_display_config(runtime_config)
    condition_name = str(config.get("project", {}).get("condition_name", "")).strip()
    report_title = (
        f"Client {condition_name} report"
        if condition_name
        else "Client prediction report"
    )

    summary = summarize_run_outputs(runtime_config)
    section("Run status", "Artifact availability at end of pipeline.")
    cols = st.columns(4)
    cols[0].metric("Run ID", summary["run_id"])
    cols[1].metric("Mode", summary["run_mode"])
    cols[2].metric("Dataset", summary["dataset_name"])
    cols[3].metric("Artifacts", summary["artifact_root"])
    status_row(
        {
            "Local metadata": summary["available"].get("local_metadata", False),
            "Model": summary["available"].get("model", False),
            "Metrics": summary["available"].get("metrics", False),
            "XAI": summary["available"].get("xai_root", False),
            "Audit": summary["available"].get("audit_root", False),
        }
    )

    metrics = summary.get("metrics")
    if metrics:
        section("Final metrics", "Quick global run status overview.")
        cols = st.columns(6)
        cols[0].metric("Rounds", metrics["num_rounds"])
        cols[1].metric("Clients", metrics["num_clients"])
        cols[2].metric("Threshold", f"{metrics['selected_threshold']:.2f}")
        cols[3].metric("Performance", f"{metrics['global_accuracy']:.3f}")
        cols[4].metric("Detection", f"{metrics['global_recall']:.3f}")
        cols[5].metric("Balance", f"{metrics['global_f1']:.3f}")

    runtime = None
    if summary["available"].get("local_metadata") and summary["available"].get("model"):
        runtime = load_run_runtime(runtime_config["artifact_root"])

    section(
        "User prediction",
        "Current run model applies the same preprocessing as Phase 2 and returns an instance explanation.",
    )
    if runtime is None:
        st.info(
            "Prediction available once model and metadata are generated for this run."
        )
    else:
        prediction_inputs = build_prediction_inputs(
            runtime["metadata"],
            form_name="client-prediction-form",
            key_prefix="predict-",
            caption="The form rebuilds the dataset variables with the same schema as Phase 2 of the current run.",
            config=config,
        )
        st.caption(
            "Explanation quality depends on dataset quality, selected columns and the amount of signal available in the uploaded data."
        )
        if prediction_inputs["submitted"]:
            raw_frame, x_numeric, x_categorical = prepare_single_instance(
                prediction_inputs["values"], runtime
            )
            probability = float(
                predict_probabilities(runtime["model"], x_numeric, x_categorical)[0]
            )
            threshold = float(runtime["model_bundle"]["selected_threshold"])
            explanation = explain_prediction(runtime, x_numeric, x_categorical)[0][
                "top_features"
            ]
            decision_payload = build_client_result_payload(
                probability=probability,
                threshold=threshold,
                explanation_rows=explanation,
                config=config,
            )
            local_xai_summary = summarize_local_xai(explanation)

            cols = st.columns(3)
            cols[0].metric("Diagnosis", decision_payload["diagnosis_label"])
            cols[1].metric("Risk level", decision_payload["risk_level"])
            cols[2].metric("Final decision", decision_payload["final_decision"])
            target_name = str(
                runtime["config"]["data"].get("target_column", "")
            ).strip()
            if target_name:
                st.caption(f"Target column used for this decision: `{target_name}`")

            st.markdown("#### Case context")
            st.dataframe(raw_frame, use_container_width=True, hide_index=True)
            shap_figure = build_prediction_shap_figure(
                runtime, x_numeric, x_categorical
            )
            if shap_figure is not None:
                st.markdown("#### SHAP visualization")
                st.pyplot(shap_figure, clear_figure=True, use_container_width=True)
            else:
                st.caption(
                    "SHAP visualization was skipped or reduced because the dataset schema is too wide or costly for an interactive local explanation."
                )
            st.markdown("#### Simple rule-based explanation")
            for rule in decision_payload["if_then_rules"]:
                st.markdown(f"- {rule}")
            st.markdown("#### Local interpretation (simplified XAI)")
            if local_xai_summary:
                st.dataframe(
                    pd.DataFrame(local_xai_summary),
                    use_container_width=True,
                    hide_index=True,
                )
            explanation_frame = pd.DataFrame(explanation)
            if (
                not explanation_frame.empty
                and "shap_value" in explanation_frame.columns
            ):
                business_frame = explanation_frame.copy()
                business_frame["feature"] = business_frame["feature"].map(
                    lambda f: feature_to_label(f, config=config)
                )
                explanation_chart = build_vertical_bar_figure(
                    business_frame.rename(
                        columns={"shap_value": "value", "feature": "business_factor"}
                    ),
                    category_column="business_factor",
                    value_column="value",
                    title="Local factor contribution",
                    color="#305068",
                )
                if explanation_chart is not None:
                    st.pyplot(
                        explanation_chart, clear_figure=True, use_container_width=True
                    )

            user = current_user()
            case_id = f"{runtime_config['run_id']}-case-{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S%f')}"
            if user:
                save_case_history(
                    user["username"],
                    {
                        "case_id": case_id,
                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                        "input_context": raw_frame.to_dict(orient="records")[0],
                        "final_decision": decision_payload["final_decision"],
                        "diagnosis_label": decision_payload["diagnosis_label"],
                        "risk_level": decision_payload["risk_level"],
                        "if_then_rules": decision_payload["if_then_rules"],
                        "rule_explanations": decision_payload["if_then_rules"],
                        "local_xai_summary": local_xai_summary,
                        "model_version": runtime_config["run_id"],
                    },
                )
            pdf_bytes = build_prediction_report_pdf(
                title=report_title,
                role=str(user["role"]) if user else "client",
                actor=str(user["username"]) if user else "client",
                model_info={
                    "Architecture": runtime.get("config", {})
                    .get("model", {})
                    .get("type", "neural_network"),
                    "Training mode": runtime_config["run_mode"],
                    "Training dataset": runtime_config["dataset_name"],
                },
                prediction_info={
                    "Diagnosis": decision_payload["diagnosis_label"],
                    "Risk level": decision_payload["risk_level"],
                    "Final decision": decision_payload["final_decision"],
                },
                input_frame=raw_frame,
                explanation_frame=pd.DataFrame(local_xai_summary),
                rules_frame=pd.DataFrame(
                    [
                        {"Applied local rule": rule, "Scope": "Local"}
                        for rule in decision_payload["if_then_rules"]
                    ]
                ),
                rules_title="Rule-based local",
                shap_figure=build_prediction_shap_figure(
                    runtime, x_numeric, x_categorical
                ),
            )
            st.download_button(
                "Export client PDF report",
                data=pdf_bytes,
                file_name=f"{runtime_config['run_id']}-prediction-report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    section(
        "Case history", "Full traceability of processed cases for the connected client."
    )
    user = current_user()
    if user:
        case_history = list_case_history(user["username"], limit=25)
        if case_history:
            case_frame = pd.DataFrame(case_history)
            if "input_context" in case_frame.columns:
                case_frame = case_frame.drop(columns=["input_context"])
            st.dataframe(case_frame, use_container_width=True, hide_index=True)
        else:
            st.info("No cases processed yet.")
