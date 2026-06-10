from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from shared.dataset_io import UnsupportedDatasetFormat, load_tabular


class DatasetIoTests(unittest.TestCase):
    def test_load_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_text(" a , b \n 1 , 2 \n", encoding="utf-8")
            frame = load_tabular(path)
            self.assertEqual(frame.columns.tolist(), ["a", "b"])
            self.assertEqual(int(frame.iloc[0]["a"]), 1)

    def test_load_xlsx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.xlsx"
            pd.DataFrame({"target": [0, 1], "value": [1.2, 3.4]}).to_excel(
                path, index=False
            )
            frame = load_tabular(path)
            self.assertEqual(frame.shape, (2, 2))
            self.assertIn("target", frame.columns)

    def test_load_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.parquet"
            pd.DataFrame({"target": [0, 1], "value": [1.2, 3.4]}).to_parquet(path)
            frame = load_tabular(path)
            self.assertEqual(frame.shape, (2, 2))
            self.assertIn("value", frame.columns)

    def test_unsupported_format_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.json"
            path.write_text('{"a": 1}', encoding="utf-8")
            with self.assertRaises(UnsupportedDatasetFormat):
                load_tabular(path)


if __name__ == "__main__":
    unittest.main()
