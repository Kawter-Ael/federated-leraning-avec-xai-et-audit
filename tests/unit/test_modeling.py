from __future__ import annotations

import unittest
import warnings

import numpy as np

from shared.modeling import (
    compute_classification_metrics,
    compute_pos_weight,
    create_model,
    evaluate_predictions,
    get_model_parameters,
    predict_probabilities,
    select_decision_threshold,
    set_model_parameters,
)


class TestComputeClassificationMetrics(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.9, 0.95])
        metrics = compute_classification_metrics(y_true, probs, threshold=0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertGreater(metrics["roc_auc"], 0.99)

    def test_all_negative(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        with warnings.catch_warnings(record=True) as caught:
            metrics = compute_classification_metrics(y_true, probs, threshold=0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["pr_auc"], 0.0)
        self.assertEqual(len(caught), 0)

    def test_all_positive(self) -> None:
        y_true = np.array([1, 1, 1, 1])
        probs = np.array([0.6, 0.7, 0.8, 0.9])
        with warnings.catch_warnings(record=True) as caught:
            metrics = compute_classification_metrics(y_true, probs, threshold=0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(len(caught), 0)

    def test_realistic_imbalanced_distribution_produces_stable_metrics(self) -> None:
        y_true = np.array([0] * 95 + [1] * 5)
        probs = np.array([0.05] * 80 + [0.2] * 15 + [0.7] * 5, dtype=np.float64)
        metrics = compute_classification_metrics(y_true, probs, threshold=0.5)
        self.assertGreaterEqual(metrics["roc_auc"], 0.9)
        self.assertGreaterEqual(metrics["pr_auc"], 0.5)


class TestSelectDecisionThreshold(unittest.TestCase):
    def test_returns_threshold_with_best_f1(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold, metrics = select_decision_threshold(y_true, probs)
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0.1)
        self.assertLessEqual(threshold, 0.9)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)

    def test_respects_min_recall(self) -> None:
        y_true = np.array([0] * 90 + [1] * 10)
        probs = np.concatenate([np.full(90, 0.1), np.full(10, 0.5)])
        threshold, metrics = select_decision_threshold(y_true, probs, min_recall=0.9)
        self.assertGreaterEqual(metrics["recall"], 0.9)

    def test_custom_candidate_thresholds(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.3, 0.8, 0.9])
        threshold, _ = select_decision_threshold(y_true, probs, candidate_thresholds=[0.3, 0.5])
        self.assertIn(threshold, [0.3, 0.5])

    def test_single_class_target_falls_back_without_warning_prone_scorers(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        probs = np.array([0.1, 0.2, 0.2, 0.3])
        with warnings.catch_warnings(record=True) as caught:
            threshold, metrics = select_decision_threshold(y_true, probs)
        self.assertEqual(threshold, 0.5)
        self.assertIn("precision", metrics)
        self.assertEqual(len(caught), 0)


class TestComputePosWeight(unittest.TestCase):
    def test_balanced_dataset(self) -> None:
        y = np.array([0, 0, 1, 1])
        weight = compute_pos_weight(y, scale=1.0)
        self.assertIsInstance(weight, np.ndarray)
        self.assertAlmostEqual(float(weight[0]), 1.0, places=4)

    def test_imbalanced_dataset(self) -> None:
        y = np.array([0] * 9 + [1])
        weight = compute_pos_weight(y, scale=1.0)
        self.assertGreater(float(weight[0]), 1.0)

    def test_scale_parameter(self) -> None:
        y = np.array([0] * 9 + [1])
        weight_scaled = compute_pos_weight(y, scale=0.0)
        self.assertAlmostEqual(float(weight_scaled[0]), 1.0, places=4)


class TestEvaluatePredictions(unittest.TestCase):
    def test_returns_loss_and_metrics(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.9, 0.95])
        result = evaluate_predictions(y_true, probs, threshold=0.5)
        self.assertIsInstance(result.loss, float)
        self.assertGreater(result.loss, 0.0)
        self.assertIsInstance(result.metrics, dict)
        self.assertIn("accuracy", result.metrics)


class TestModelParametersRoundtrip(unittest.TestCase):
    def test_get_set_parameters_preserves_predictions(self) -> None:
        config = {
            "project": {"seed": 42},
            "model": {
                "family": "tabular_nn",
                "type": "embedded",
                "hyperparameters": {
                    "hidden_dims": [8, 4],
                    "dropout": 0.1,
                    "use_batch_norm": False,
                },
            },
        }
        schema = {
            "numeric_columns": ["num_a", "num_b"],
            "categorical_columns": ["cat_a"],
            "categorical_cardinalities": {"cat_a": 3},
            "embedding_dimensions": {"cat_a": 2},
        }
        model = create_model(config, schema=schema)
        x_numeric = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        x_categorical = np.array([[0], [1]], dtype=np.int64)

        original_probs = predict_probabilities(model, x_numeric, x_categorical)
        params = get_model_parameters(model)
        set_model_parameters(model, params)
        restored_probs = predict_probabilities(model, x_numeric, x_categorical)

        np.testing.assert_array_almost_equal(original_probs, restored_probs, decimal=5)

    def test_set_parameters_shape_mismatch_raises(self) -> None:
        config = {
            "project": {"seed": 42},
            "model": {
                "family": "tabular_nn",
                "type": "embedded",
                "hyperparameters": {"hidden_dims": [4], "dropout": 0.0, "use_batch_norm": False},
            },
        }
        schema = {
            "numeric_columns": ["a"],
            "categorical_columns": [],
            "categorical_cardinalities": {},
            "embedding_dimensions": {},
        }
        model = create_model(config, schema=schema)
        with self.assertRaises(ValueError):
            set_model_parameters(model, [np.array([1.0])])


if __name__ == "__main__":
    unittest.main()
