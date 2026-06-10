from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages


_BLANK_TEXT_VALUES = {"", "-", "--", "n/a", "none", "null", "nan"}


def _is_blank_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in _BLANK_TEXT_VALUES
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0 or all(_is_blank_value(item) for item in value)
    if isinstance(value, dict):
        return len(value) == 0 or all(_is_blank_value(item) for item in value.values())
    return False


def _clean_info_items(items: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in items.items() if not _is_blank_value(value)
    }


def _clean_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    cleaned = frame.copy()
    cleaned = cleaned.dropna(axis=1, how="all")
    if cleaned.empty:
        return pd.DataFrame()
    keep_rows = []
    for _, row in cleaned.iterrows():
        keep_rows.append(any(not _is_blank_value(value) for value in row.tolist()))
    cleaned = cleaned.loc[keep_rows].reset_index(drop=True)
    if cleaned.empty:
        return pd.DataFrame()
    keep_columns = []
    for column in cleaned.columns:
        keep_columns.append(any(not _is_blank_value(value) for value in cleaned[column]))
    cleaned = cleaned.loc[:, keep_columns]
    return cleaned.reset_index(drop=True)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return ", ".join(f"{key}={val}" for key, val in list(value.items())[:8])
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value[:8])
    return str(value)


def _split_sections(items: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    meta_keys = {"Role", "Actor"}
    prediction_keys = {
        "Predicted probability",
        "Predicted class",
        "Run mode",
        "Dataset",
        "Global rounds",
        "Global clients",
        "Diagnosis",
        "Risk level",
        "Final decision",
        "Diagnostic",
        "Niveau de risque",
        "Decision finale",
        "Threshold",
    }
    meta = {key: value for key, value in items.items() if key in meta_keys}
    prediction = {key: value for key, value in items.items() if key in prediction_keys}
    model = {key: value for key, value in items.items() if key not in meta_keys and key not in prediction_keys}
    return meta, model, prediction


def _draw_info_card(axis: Any, title: str, items: dict[str, Any], x: float, y: float, width: float, height: float) -> None:
    if not items:
        return
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#d6dce5",
        facecolor="#f8fafc",
        transform=axis.transAxes,
    )
    axis.add_patch(card)
    axis.text(x + 0.03, y + height - 0.06, title, fontsize=13, fontweight="bold", color="#1f3550", transform=axis.transAxes)
    current_y = y + height - 0.11
    for key, value in items.items():
        axis.text(x + 0.03, current_y, key, fontsize=10, fontweight="bold", color="#213547", transform=axis.transAxes)
        axis.text(
            x + 0.56 * width,
            current_y,
            _format_value(value),
            fontsize=10,
            color="#25384c",
            transform=axis.transAxes,
            ha="left",
        )
        current_y -= 0.06
        if current_y < y + 0.05:
            break


def _add_text_page(pdf: PdfPages, title: str, items: dict[str, Any]) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.patch.set_facecolor("#f4f6f8")
    axis = figure.add_subplot(111)
    axis.axis("off")
    hero = FancyBboxPatch(
        (0.03, 0.82),
        0.94,
        0.13,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=0,
        facecolor="#20364b",
        transform=axis.transAxes,
    )
    axis.add_patch(hero)
    axis.text(0.06, 0.915, title, fontsize=22, fontweight="bold", color="#fff8ef", va="center", transform=axis.transAxes)
    axis.text(
        0.06,
        0.865,
        "Prediction report with model metadata, decision summary and explainability context.",
        fontsize=11,
        color="#e8eef4",
        va="center",
        transform=axis.transAxes,
    )

    meta, model, prediction = (
        _clean_info_items(section_items) for section_items in _split_sections(items)
    )
    _draw_info_card(axis, "Identity", meta, 0.05, 0.58, 0.42, 0.18)
    _draw_info_card(axis, "Model", model, 0.53, 0.48, 0.42, 0.28)
    _draw_info_card(axis, "Prediction", prediction, 0.05, 0.30, 0.90, 0.20)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _add_table_page(pdf: PdfPages, title: str, frame: pd.DataFrame) -> None:
    frame = _clean_frame(frame)
    if frame.empty:
        return
    if frame.shape[0] == 1 and frame.shape[1] > 8:
        flattened = pd.DataFrame(
            [{"Field": column, "Value": _format_value(frame.iloc[0][column])} for column in frame.columns]
        )
        _add_table_page(pdf, title, flattened)
        return
    detailed_rule_columns = {"description", "feature", "operator", "value"}
    if title == "Rule-based summary" and detailed_rule_columns.issubset(set(frame.columns)):
        _add_rule_summary_page(pdf, title, frame)
        return
    if frame.shape[1] > 6:
        flattened_rows: list[dict[str, Any]] = []
        for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
            flattened_rows.append({"Field": f"Item {row_index}", "Value": ""})
            for column in frame.columns:
                flattened_rows.append(
                    {
                        "Field": column,
                        "Value": _format_value(row[column]),
                    }
                )
        _add_table_page(pdf, title, pd.DataFrame(flattened_rows))
        return
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.patch.set_facecolor("#ffffff")
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(0.02, 0.965, title, fontsize=17, fontweight="bold", color="#1f3550", transform=axis.transAxes)
    preview = frame.copy().head(18).astype(str)
    if frame.shape[0] > len(preview):
        axis.text(
            0.02,
            0.93,
            f"Showing first {len(preview)} rows out of {len(frame)}.",
            fontsize=9,
            color="#5c6f82",
            transform=axis.transAxes,
        )
    table = axis.table(
        cellText=preview.values,
        colLabels=preview.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.05, 0.96, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.38)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#20364b")
            cell.set_text_props(color="#fff8ef", weight="bold")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 == 0 else "#eef2f6")
            cell.set_edgecolor("#d6dce5")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _add_rule_summary_page(pdf: PdfPages, title: str, frame: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.patch.set_facecolor("#ffffff")
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(0.02, 0.965, title, fontsize=17, fontweight="bold", color="#1f3550", transform=axis.transAxes)

    current_y = 0.88
    for row_index, (_, row) in enumerate(frame.head(5).iterrows(), start=1):
        card_height = 0.145
        if current_y - card_height < 0.05:
            break
        card = FancyBboxPatch(
            (0.02, current_y - card_height),
            0.96,
            card_height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#d6dce5",
            facecolor="#f8fafc",
            transform=axis.transAxes,
        )
        axis.add_patch(card)

        axis.text(
            0.04,
            current_y - 0.03,
            f"Rule {row_index}: {row.get('description', row.get('signature', '-'))}",
            fontsize=12,
            fontweight="bold",
            color="#1f3550",
            transform=axis.transAxes,
        )

        left_items = [
            ("Feature", row.get("feature", "-")),
            ("Operator", row.get("operator", "-")),
            ("Value", row.get("value", "-")),
            ("Type", row.get("feature_type", "-")),
        ]
        right_items = [
            ("Frequency", row.get("frequency", "-")),
            ("Importance", row.get("importance", "-")),
            ("Support", row.get("support", "-")),
            ("Clients", row.get("client_ids", "-")),
        ]

        left_y = current_y - 0.07
        for label, value in left_items:
            axis.text(0.05, left_y, f"{label}:", fontsize=9.5, fontweight="bold", color="#213547", transform=axis.transAxes)
            axis.text(0.18, left_y, _format_value(value), fontsize=9.5, color="#25384c", transform=axis.transAxes)
            left_y -= 0.023

        right_y = current_y - 0.07
        for label, value in right_items:
            axis.text(0.53, right_y, f"{label}:", fontsize=9.5, fontweight="bold", color="#213547", transform=axis.transAxes)
            axis.text(0.67, right_y, _format_value(value), fontsize=9.5, color="#25384c", transform=axis.transAxes)
            right_y -= 0.023

        current_y -= 0.17

    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _add_local_rule_text_page(pdf: PdfPages, title: str, frame: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.patch.set_facecolor("#ffffff")
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(
        0.04,
        0.96,
        title,
        fontsize=18,
        fontweight="bold",
        color="#1f3550",
        transform=axis.transAxes,
    )

    current_y = 0.89
    rule_column = (
        "Applied local rule"
        if "Applied local rule" in frame.columns
        else frame.columns[0]
    )
    for row_index, rule_text in enumerate(frame[rule_column].head(12), start=1):
        text = str(rule_text).strip()
        if not text:
            continue
        card_height = 0.06
        if current_y - card_height < 0.06:
            break
        card = FancyBboxPatch(
            (0.04, current_y - card_height),
            0.92,
            card_height,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=0.8,
            edgecolor="#d6dce5",
            facecolor="#f8fafc",
            transform=axis.transAxes,
        )
        axis.add_patch(card)
        axis.text(
            0.06,
            current_y - 0.031,
            f"{row_index}. {text}",
            fontsize=10.5,
            color="#25384c",
            transform=axis.transAxes,
            va="center",
            wrap=True,
        )
        current_y -= 0.078

    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def build_prediction_report_pdf(
    *,
    title: str,
    role: str,
    actor: str,
    model_info: dict[str, Any],
    prediction_info: dict[str, Any],
    input_frame: pd.DataFrame,
    explanation_frame: pd.DataFrame,
    rules_frame: pd.DataFrame | None = None,
    rules_title: str = "Rule-based summary",
    shap_figure: Any | None = None,
) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        model_info = _clean_info_items(model_info)
        prediction_info = _clean_info_items(prediction_info)
        input_frame = _clean_frame(input_frame)
        explanation_frame = _clean_frame(explanation_frame)
        rules_frame = _clean_frame(rules_frame)
        _add_text_page(
            pdf,
            title,
            {
                "Role": role,
                "Actor": actor,
                **model_info,
                **prediction_info,
            },
        )
        if not input_frame.empty:
            _add_table_page(pdf, "Prediction inputs", input_frame)
        if not explanation_frame.empty:
            _add_table_page(pdf, "SHAP explanation", explanation_frame)
        if not rules_frame.empty:
            if rules_title == "Rule-based local":
                _add_local_rule_text_page(pdf, rules_title, rules_frame)
            else:
                _add_table_page(pdf, rules_title, rules_frame)
        if shap_figure is not None:
            pdf.savefig(shap_figure, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()
