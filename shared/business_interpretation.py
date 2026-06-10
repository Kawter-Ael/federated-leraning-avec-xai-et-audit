from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


RISK_LEVELS = (
    {"label": "Low risk", "min": 0.0, "max": 0.33},
    {"label": "Medium risk", "min": 0.33, "max": 0.66},
    {"label": "High risk", "min": 0.66, "max": 1.01},
)

FEATURE_LABELS = {
    "Pregnancies": "number of pregnancies",
    "Glucose": "glucose level",
    "BloodPressure": "blood pressure",
    "SkinThickness": "skin fold thickness",
    "Insulin": "insulin level",
    "BMI": "body mass index",
    "DiabetesPedigreeFunction": "diabetes pedigree function",
    "Age": "age",
}


@dataclass
class BusinessDecision:
    diagnosis_label: str
    diagnosis_code: int
    risk_level: str
    final_decision: str
    probability: float
    threshold: float


def load_risk_levels(
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    if config and "risk_levels" in config:
        return tuple(config["risk_levels"])
    return RISK_LEVELS


def prediction_to_diagnosis(
    prediction: int, config: dict[str, Any] | None = None
) -> str:
    if config and "project" in config:
        positive = config["project"].get("positive_label", "Positive case")
        negative = config["project"].get("negative_label", "Negative case")
        positive = str(positive).strip() or "Positive case"
        negative = str(negative).strip() or "Negative case"
        return positive if int(prediction) == 1 else negative
    return "Positive outcome" if int(prediction) == 1 else "Negative outcome"


def prediction_to_risk_level(
    probability: float, config: dict[str, Any] | None = None
) -> str:
    value = float(probability)
    for level in load_risk_levels(config):
        if level["min"] <= value < level["max"]:
            return str(level["label"])
    return "High risk"


def feature_to_label(feature_name: str, config: dict[str, Any] | None = None) -> str:
    """Return a human-readable feature label.

    `FEATURE_LABELS` is only a seed fallback. Per-run labels from
    `config["data"]["feature_labels"]` take priority when available.
    """
    if config and "data" in config and config["data"].get("feature_labels"):
        custom = config["data"]["feature_labels"]
        if feature_name in custom:
            return str(custom[feature_name])
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").lower())


def format_feature_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return "unknown"
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_if_then_explanation(
    explanation_rows: list[dict[str, Any]] | None,
    *,
    max_rules: int = 3,
    config: dict[str, Any] | None = None,
) -> list[str]:
    if not explanation_rows:
        return []
    condition_name = "the condition"
    if config and "project" in config:
        condition_name = (
            str(config["project"].get("condition_name", condition_name)).strip()
            or condition_name
        )
    rules: list[str] = []
    for row in explanation_rows[:max_rules]:
        feature = feature_to_label(str(row.get("feature", "factor")), config=config)
        feature_value = format_feature_value(
            row.get("feature_value", row.get("sample_value", ""))
        )
        shap_value = float(row.get("shap_value", row.get("mean_abs_shap", 0.0)))
        if shap_value >= 0:
            impact = "increases"
        else:
            impact = "decreases"
        rules.append(
            f"If {feature} = {feature_value}, then this {impact} {condition_name} risk."
        )
    return rules


def build_rule_sentence(
    rule: dict[str, Any], config: dict[str, Any] | None = None
) -> str:
    condition_name = "the condition"
    if config and "project" in config:
        condition_name = (
            str(config["project"].get("condition_name", condition_name)).strip()
            or condition_name
        )
    feature = feature_to_label(str(rule.get("feature", "factor")), config=config)
    operator = str(rule.get("operator", "=="))
    value = format_feature_value(rule.get("value", ""))
    return f"If {feature} {operator} {value}, then {condition_name} risk is influenced by this condition."


def summarize_local_xai(
    explanation_rows: list[dict[str, Any]] | None, *, max_items: int = 3
) -> list[dict[str, Any]]:
    if not explanation_rows:
        return []
    output: list[dict[str, Any]] = []
    for row in explanation_rows[:max_items]:
        output.append(
            {
                "feature": feature_to_label(str(row.get("feature", "factor"))),
                "impact": "increased risk"
                if float(row.get("shap_value", row.get("mean_abs_shap", 0.0))) >= 0
                else "decreased risk",
                "value": format_feature_value(
                    row.get("feature_value", row.get("sample_value", ""))
                ),
            }
        )
    return output


def format_client_decision(
    probability: float, threshold: float, config: dict[str, Any] | None = None
) -> BusinessDecision:
    prediction = int(float(probability) >= float(threshold))
    diagnosis = prediction_to_diagnosis(prediction, config=config)
    risk_level = prediction_to_risk_level(probability, config=config)
    pct = float(probability) * 100.0
    diagnosis_pct = f"{pct:.1f}% {diagnosis}"
    risk_pct = f"{risk_level} ({pct:.1f}%)"
    final_decision = f"{pct:.1f}% - {diagnosis} - {risk_level}"
    return BusinessDecision(
        diagnosis_label=diagnosis_pct,
        diagnosis_code=prediction,
        risk_level=risk_pct,
        final_decision=final_decision,
        probability=float(probability),
        threshold=float(threshold),
    )


def build_client_result_payload(
    *,
    probability: float,
    threshold: float,
    explanation_rows: list[dict[str, Any]] | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = format_client_decision(probability, threshold, config=config)
    rules = build_if_then_explanation(explanation_rows, config=config)
    return {
        "diagnosis_label": decision.diagnosis_label,
        "diagnosis_code": decision.diagnosis_code,
        "risk_level": decision.risk_level,
        "final_decision": decision.final_decision,
        "probability": decision.probability,
        "threshold": decision.threshold,
        "if_then_rules": rules,
    }


def build_server_case_table(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=["Case", "Applied business rule", "Simple explanation", "Decision"]
        )
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rules = record.get("if_then_rules") or record.get("rule_explanations") or []
        rows.append(
            {
                "Case": record.get("case_id", f"case-{index}"),
                "Applied business rule": rules[0]
                if rules
                else "No readable rule available",
                "Simple explanation": " | ".join(rules[:2])
                if rules
                else record.get("risk_level", "-"),
                "Decision": record.get("final_decision", "-"),
            }
        )
    return pd.DataFrame(rows)
