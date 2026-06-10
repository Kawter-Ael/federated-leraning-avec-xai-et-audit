"""Unit tests for dataset viability checks in validate_uploaded_dataset."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from shared.client_workflow import validate_uploaded_dataset


def _write_csv(tmpdir: str, frame: pd.DataFrame, name: str = "data.csv") -> Path:
    path = Path(tmpdir) / name
    frame.to_csv(path, index=False)
    return path


class BinaryTargetWithTextLabelsTests(unittest.TestCase):
    """Target column with string labels (e.g. 'Yes'/'No') should be accepted."""

    def test_text_binary_target_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = pd.DataFrame(
                {
                    "age": list(range(50)),
                    "score": [float(i) * 0.1 for i in range(50)],
                    "diagnosis": ["Yes"] * 30 + ["No"] * 20,
                }
            )
            path = _write_csv(tmpdir, frame)
            result = validate_uploaded_dataset(
                path,
                target_column="diagnosis",
                positive_class="Yes",
                excluded_columns=[],
                numeric_columns=["age", "score"],
            )
            self.assertTrue(result["valid"], result["blocking"])
            self.assertEqual(len(result["blocking"]), 0)

    def test_mixed_case_target_warns_normalization(self) -> None:
        """Labels differing only by case → warning, not blocking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = pd.DataFrame(
                {
                    "x": list(range(40)),
                    "label": ["Yes"] * 20 + ["yes"] * 20,
                }
            )
            path = _write_csv(tmpdir, frame)
            result = validate_uploaded_dataset(
                path,
                target_column="label",
                positive_class="yes",
                excluded_columns=[],
                numeric_columns=["x"],
            )
            codes = [item["code"] for item in result["warning"]]
            self.assertIn("target_values_normalize_to_same_token", codes)


class TooFewPositivesTests(unittest.TestCase):
    def test_too_few_examples_blocking(self) -> None:
        """< 3 examples in minority class → blocking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = pd.DataFrame(
                {
                    "x": list(range(12)),
                    "label": [0] * 10 + [1] * 2,
                }
            )
            path = _write_csv(tmpdir, frame)
            result = validate_uploaded_dataset(
                path,
                target_column="label",
                positive_class=1,
                excluded_columns=[],
                numeric_columns=["x"],
            )
            self.assertFalse(result["valid"])
            codes = [item["code"] for item in result["blocking"]]
            self.assertIn("too_few_examples_per_class", codes)

    def test_small_minority_class_warning(self) -> None:
        """< 10 but >= 3 examples in minority class → warning only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = pd.DataFrame(
                {
                    "x": list(range(50)),
                    "label": [0] * 45 + [1] * 5,
                }
            )
            path = _write_csv(tmpdir, frame)
            result = validate_uploaded_dataset(
                path,
                target_column="label",
                positive_class=1,
                excluded_columns=[],
                numeric_columns=["x"],
            )
            # valid (no blocking), but warning present
            self.assertTrue(result["valid"], result["blocking"])
            codes = [item["code"] for item in result["warning"]]
            self.assertIn("small_minority_class", codes)


class HighCardinalityCategoricalTests(unittest.TestCase):
    def test_high_cardinality_categorical_warns(self) -> None:
        """Categorical column with many unique values → warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 200
            frame = pd.DataFrame(
                {
                    "label": [0] * (n // 2) + [1] * (n // 2),
                    "city": [f"city_{i}" for i in range(n)],  # 200 unique
                    "age": list(range(n)),
                }
            )
            path = _write_csv(tmpdir, frame)
            result = validate_uploaded_dataset(
                path,
                target_column="label",
                positive_class=1,
                excluded_columns=[],
                numeric_columns=["age"],
            )
            codes = [item["code"] for item in result["warning"]]
            self.assertIn("high_cardinality_categorical_columns", codes)


class QuasiUniqueIdentifierTests(unittest.TestCase):
    def test_identifier_column_warns(self) -> None:
        """Column that is ~unique per row → quasi_unique_columns warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 100
            frame = pd.DataFrame(
                {
                    "patient_id": [f"ID-{i:04d}" for i in range(n)],
                    "score": [float(i) for i in range(n)],
                    "outcome": [0] * (n // 2) + [1] * (n // 2),
                }
            )
            path = _write_csv(tmpdir, frame)
            result = validate_uploaded_dataset(
                path,
                target_column="outcome",
                positive_class=1,
                excluded_columns=[],
                numeric_columns=["score"],
            )
            codes = [item["code"] for item in result["warning"]]
            self.assertIn("quasi_unique_columns", codes)

    def test_excluded_identifier_does_not_warn(self) -> None:
        """Quasi-unique column that is excluded should still be flagged in warnings —
        it is the user's responsibility to exclude, the validator reports what it sees."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 100
            frame = pd.DataFrame(
                {
                    "name": [f"Patient {i}" for i in range(n)],
                    "score": [float(i) for i in range(n)],
                    "outcome": [0] * (n // 2) + [1] * (n // 2),
                }
            )
            path = _write_csv(tmpdir, frame)
            # When user has already excluded it, validation still flags it for visibility
            result = validate_uploaded_dataset(
                path,
                target_column="outcome",
                positive_class=1,
                excluded_columns=["name"],
                numeric_columns=["score"],
            )
            # dataset is still valid — no blocking — excluded column only in warning
            self.assertTrue(result["valid"])

    def test_multiclass_target_blocking(self) -> None:
        """3-class target → non_binary_target blocking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = pd.DataFrame(
                {
                    "x": list(range(90)),
                    "label": [0, 1, 2] * 30,
                }
            )
            path = _write_csv(tmpdir, frame)
            result = validate_uploaded_dataset(
                path,
                target_column="label",
                positive_class=1,
                excluded_columns=[],
                numeric_columns=["x"],
            )
            self.assertFalse(result["valid"])
            codes = [item["code"] for item in result["blocking"]]
            self.assertIn("non_binary_target", codes)


if __name__ == "__main__":
    unittest.main()
