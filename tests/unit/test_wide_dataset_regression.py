"""Regression tests for wide-dataset robustness.

Tests that prediction form helpers, PDF export and XAI paths do not
break or produce unreadable output when the dataset has many columns
(e.g. 30+ features).  No Streamlit session or trained model required.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from shared.report_export import (
    _clean_frame,
    _clean_info_items,
    build_prediction_report_pdf,
)
from shared.ui_components import default_numeric_value


def _wide_metadata(n_numeric: int = 30, n_categorical: int = 5) -> dict:
    numeric_columns = [f"feat_{i:02d}" for i in range(n_numeric)]
    categorical_columns = [f"cat_{j}" for j in range(n_categorical)]
    imputer_stats = [float(i) * 1.1 for i in range(n_numeric)]
    vocabularies = {
        col: {"__unknown__": 0, "A": 1, "B": 2}
        for col in categorical_columns
    }
    groupings = {
        col: {"rare_values": [], "value_counts": {"A": 50, "B": 50}}
        for col in categorical_columns
    }
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_vocabularies": vocabularies,
        "categorical_groupings": groupings,
        "categorical_cardinalities": {col: 3 for col in categorical_columns},
        "transformer_state": {
            "numeric_imputer": {
                "strategy": "median",
                "statistics": imputer_stats,
            },
            "numeric_scaler": {
                "mean": imputer_stats,
                "scale": [1.0] * n_numeric,
                "var": [1.0] * n_numeric,
            },
            "numeric_columns": numeric_columns,
        },
    }


class WideDatasetDefaultValueTests(unittest.TestCase):
    """default_numeric_value returns imputer median for any column in wide metadata."""

    def test_returns_correct_median_for_known_column(self) -> None:
        metadata = _wide_metadata(n_numeric=30)
        # feat_05 has index 5 → statistics[5] = 5 * 1.1 = 5.5
        value = default_numeric_value("feat_05", metadata)
        self.assertAlmostEqual(value, 5.5, places=5)

    def test_returns_zero_for_unknown_column(self) -> None:
        metadata = _wide_metadata(n_numeric=30)
        value = default_numeric_value("nonexistent_column", metadata)
        self.assertEqual(value, 0.0)

    def test_returns_zero_when_no_metadata(self) -> None:
        value = default_numeric_value("any_column", None)
        self.assertEqual(value, 0.0)

    def test_all_columns_have_non_nan_defaults(self) -> None:
        metadata = _wide_metadata(n_numeric=40)
        for col in metadata["numeric_columns"]:
            value = default_numeric_value(col, metadata)
            self.assertFalse(
                np.isnan(value), f"default for {col!r} is NaN"
            )


class WideDatasetPdfExportTests(unittest.TestCase):
    """PDF export must not raise and must produce non-empty bytes for wide data."""

    def _make_wide_input_frame(self, n_cols: int = 35) -> pd.DataFrame:
        return pd.DataFrame(
            {f"feat_{i:02d}": [float(i) * 1.5] for i in range(n_cols)}
        )

    def _make_wide_explanation_frame(self, n_features: int = 30) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": [f"feat_{i:02d}" for i in range(n_features)],
                "shap_value": [round((i - n_features / 2) * 0.01, 4) for i in range(n_features)],
                "feature_value": [float(i) for i in range(n_features)],
            }
        )

    def test_pdf_export_does_not_raise_on_wide_input(self) -> None:
        pdf_bytes = build_prediction_report_pdf(
            title="Wide Dataset Prediction Report",
            role="client",
            actor="test_user",
            model_info={"Model type": "neural_network", "Threshold": "0.5"},
            prediction_info={"Diagnosis": "Positive", "Risk level": "High"},
            input_frame=self._make_wide_input_frame(35),
            explanation_frame=self._make_wide_explanation_frame(30),
            rules_frame=None,
            shap_figure=None,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 100)

    def test_pdf_export_with_many_shap_features(self) -> None:
        """50-feature explanation must not crash or produce empty output."""
        pdf_bytes = build_prediction_report_pdf(
            title="Stress Test Report",
            role="admin",
            actor="admin_user",
            model_info={"Dataset": "wide_test", "Rounds": 3},
            prediction_info={"Predicted class": 1, "Predicted probability": 0.72},
            input_frame=self._make_wide_input_frame(50),
            explanation_frame=self._make_wide_explanation_frame(50),
            rules_frame=pd.DataFrame(
                {
                    "Feature": ["feat_01", "feat_02"],
                    "Operator": [">=", "<="],
                    "Value": [1.5, 3.0],
                }
            ),
            shap_figure=None,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 100)

    def test_pdf_export_with_empty_explanation_frame(self) -> None:
        """Empty SHAP explanation (XAI degraded) must not crash."""
        pdf_bytes = build_prediction_report_pdf(
            title="Degraded XAI Report",
            role="client",
            actor="test_user",
            model_info={"Model type": "neural_network"},
            prediction_info={"Diagnosis": "Negative"},
            input_frame=self._make_wide_input_frame(10),
            explanation_frame=pd.DataFrame(
                columns=["feature", "shap_value", "feature_value"]
            ),
            rules_frame=None,
            shap_figure=None,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 100)

    def test_clean_frame_drops_placeholder_rule_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "description": ["-", "-", "-"],
                "feature": ["-", "-", "-"],
                "operator": ["", "", ""],
                "value": [None, None, None],
            }
        )
        self.assertTrue(_clean_frame(frame).empty)

    def test_clean_info_items_drops_blank_model_fields(self) -> None:
        cleaned = _clean_info_items(
            {
                "Architecture": "neural_network",
                "Training mode": "",
                "Training dataset": "-",
                "Decision threshold": 0.5,
            }
        )
        self.assertEqual(
            cleaned,
            {
                "Architecture": "neural_network",
                "Decision threshold": 0.5,
            },
        )

    def test_pdf_export_supports_simple_local_rule_table(self) -> None:
        pdf_bytes = build_prediction_report_pdf(
            title="Client Local Rules Report",
            role="client",
            actor="test_user",
            model_info={"Architecture": "neural_network"},
            prediction_info={"Diagnosis": "Positive"},
            input_frame=self._make_wide_input_frame(8),
            explanation_frame=self._make_wide_explanation_frame(5),
            rules_frame=pd.DataFrame(
                [
                    {
                        "Applied local rule": "If glucose level is high, then risk increases.",
                        "Scope": "Local",
                    }
                ]
            ),
            rules_title="Rule-based local",
            shap_figure=None,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 100)


class WideDatasetShapDegradationTests(unittest.TestCase):
    """SHAP summary helpers must handle wide feature lists without crashing."""

    def test_shap_aggregation_on_many_features(self) -> None:
        from shared.aggregation import aggregate_shap_summaries

        n = 60
        feature_names = [f"feat_{i:02d}" for i in range(n)]
        # Simulate two clients sending SHAP summaries with 60 features each
        summaries = [
            {
                "num_examples": 100,
                "feature_names": feature_names,
                "mean_abs_shap": {f: float(i) * 0.01 for i, f in enumerate(feature_names)},
                "top_features": feature_names[:10],
            },
            {
                "num_examples": 80,
                "feature_names": feature_names,
                "mean_abs_shap": {f: float(i) * 0.015 for i, f in enumerate(feature_names)},
                "top_features": feature_names[:10],
            },
        ]
        result = aggregate_shap_summaries(summaries)
        self.assertIn("feature_names", result)
        self.assertIn("mean_abs_shap", result)
        # Top features capped at 10 (config default)
        self.assertLessEqual(len(result["top_features"]), 10)

    def test_aggregate_shap_single_client(self) -> None:
        from shared.aggregation import aggregate_shap_summaries

        n = 40
        feature_names = [f"f_{i}" for i in range(n)]
        summaries = [
            {
                "num_examples": 200,
                "feature_names": feature_names,
                "mean_abs_shap": {f: 0.05 for f in feature_names},
                "top_features": feature_names[:5],
            }
        ]
        result = aggregate_shap_summaries(summaries)
        self.assertEqual(len(result["feature_names"]), n)


if __name__ == "__main__":
    unittest.main()
