"""Shared aggregation functions for SHAP summaries and rules."""

from __future__ import annotations

from typing import Any

from shared.business_interpretation import build_rule_sentence


def extract_shap_summaries(
    client_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract SHAP summaries from client metric dicts into flat format.

    Each output entry contains ``feature_names``, ``mean_abs_shap``,
    ``client_id`` and ``num_examples`` at the top level.
    """
    normalized: list[dict[str, Any]] = []
    for entry in client_metrics:
        summary = dict(entry["shap_summary"])
        summary["client_id"] = entry["client_id"]
        summary["num_examples"] = int(entry["num_examples"])
        normalized.append(summary)
    return normalized


def extract_rule_entries(client_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract rule entries from client metric dicts into flat format.

    Each output entry contains ``client_id``, ``num_examples`` and
    ``rules`` at the top level.
    """
    return [
        {
            "client_id": str(entry["client_id"]),
            "num_examples": int(entry["num_examples"]),
            "rules": entry.get("rule_summary", {}).get("rules", []),
        }
        for entry in client_metrics
    ]


def aggregate_shap_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate SHAP summaries using weighted averaging by num_examples.

    Each entry in *summaries* must contain ``feature_names``,
    ``mean_abs_shap`` and ``num_examples``.
    """
    if not summaries:
        return {"feature_names": [], "mean_abs_shap": {}, "top_features": []}

    feature_names = list(summaries[0]["feature_names"])
    for entry in summaries[1:]:
        if list(entry["feature_names"]) != feature_names:
            raise ValueError("SHAP feature_names mismatch between clients.")

    total_examples = sum(int(entry["num_examples"]) for entry in summaries)
    aggregated = {feature: 0.0 for feature in feature_names}
    for entry in summaries:
        weight = float(entry["num_examples"]) / float(total_examples)
        for feature in feature_names:
            aggregated[feature] += (
                float(entry["mean_abs_shap"].get(feature, 0.0)) * weight
            )

    top_features = sorted(
        [{"feature": f, "mean_abs_shap": float(v)} for f, v in aggregated.items()],
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )[:10]

    return {
        "feature_names": feature_names,
        "mean_abs_shap": aggregated,
        "top_features": top_features,
        "aggregation": "weighted_mean_by_num_examples",
        "num_clients": len(summaries),
        "total_examples": int(total_examples),
    }


def aggregate_rules(rule_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rules from multiple clients using frequency and weighted importance.

    Each entry in *rule_entries* must contain ``client_id``,
    ``num_examples`` and ``rules`` (list of dicts with ``signature``,
    ``feature``, ``operator``, ``value``, etc.).
    """
    aggregated: dict[str, dict[str, Any]] = {}
    for entry in rule_entries:
        client_id = str(entry["client_id"])
        num_examples = int(entry["num_examples"])
        for rule in entry.get("rules", []):
            signature = str(rule["signature"])
            current = aggregated.setdefault(
                signature,
                {
                    "feature": rule["feature"],
                    "operator": rule["operator"],
                    "value": rule["value"],
                    "description": rule["description"],
                    "business_description": build_rule_sentence(rule),
                    "feature_type": rule.get("feature_type", "unknown"),
                    "frequency": 0,
                    "weighted_importance_sum": 0.0,
                    "weighted_support_sum": 0.0,
                    "weight_total": 0,
                    "client_ids": [],
                },
            )
            current["frequency"] += 1
            current["weighted_importance_sum"] += (
                float(rule["importance"]) * num_examples
            )
            current["weighted_support_sum"] += float(rule["support"]) * num_examples
            current["weight_total"] += num_examples
            current["client_ids"].append(client_id)

    ranked_rules = []
    for signature, item in aggregated.items():
        ranked_rules.append(
            {
                "signature": signature,
                "feature": item["feature"],
                "operator": item["operator"],
                "value": item["value"],
                "description": item["description"],
                "business_description": item["business_description"],
                "feature_type": item["feature_type"],
                "frequency": item["frequency"],
                "importance": round(
                    item["weighted_importance_sum"] / item["weight_total"], 6
                ),
                "support": round(
                    item["weighted_support_sum"] / item["weight_total"], 4
                ),
                "client_ids": sorted(item["client_ids"]),
            }
        )

    ranked_rules.sort(
        key=lambda item: (
            -int(item["frequency"]),
            -float(item["importance"]),
            -float(item["support"]),
        )
    )
    unique_clients = sorted({str(entry["client_id"]) for entry in rule_entries})
    return {
        "rules": ranked_rules[:10],
        "aggregation": "frequency_then_weighted_importance",
        "num_unique_rules": len(ranked_rules),
        "num_clients": len(unique_clients),
        "client_ids": unique_clients,
    }
