from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from shared.client_workflow import autodetect_dataset_roles


class Phase3AutodetectTests(unittest.TestCase):
    def test_detects_diabetes_shape(self) -> None:
        frame = pd.read_csv("data/diabetes.csv")
        detected = autodetect_dataset_roles(frame)
        self.assertEqual(detected["target_column"], "Outcome")
        self.assertIn("Glucose", detected["numeric_columns"])
        self.assertEqual(detected["excluded_columns"], [])
        self.assertEqual(detected["fairness_attribute"], "Age")

    def test_detects_tb_roles_and_pii(self) -> None:
        frame = pd.read_csv(Path("tests/fixtures/tb_small.csv"))
        detected = autodetect_dataset_roles(frame)
        self.assertEqual(detected["target_column"], "tb_label")
        self.assertIn("age", detected["numeric_columns"])
        self.assertIn("name", detected["excluded_columns"])
        self.assertEqual(detected["fairness_attribute"], "age")

    def test_prefers_binary_target_when_last_column_not_binary(self) -> None:
        frame = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "label": [0, 1, 0, 1],
                "free_text": ["a", "b", "c", "d"],
            }
        )
        detected = autodetect_dataset_roles(frame)
        self.assertEqual(detected["target_column"], "label")
        self.assertIn("id", detected["excluded_columns"])
        self.assertEqual(detected["numeric_columns"], [])
        self.assertIn("free_text", detected["excluded_columns"])


if __name__ == "__main__":
    unittest.main()
