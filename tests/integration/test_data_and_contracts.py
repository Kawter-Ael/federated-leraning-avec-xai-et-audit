from __future__ import annotations

import unittest

from shared.data_preparation import (
    build_target,
    drop_configured_columns,
    infer_feature_types,
    load_config,
    load_raw_dataset,
    replace_question_marks,
)
from shared.federated_data import load_preprocessing_metadata, load_train_dataset
from tests.support import ensure_phase2_artifacts


class DataAndContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_phase2_artifacts()
        cls.config = load_config("config/presets/diabetes.json")
        cls.raw_df = load_raw_dataset(cls.config)
        cls.metadata = load_preprocessing_metadata()

    def test_raw_dataset_loads(self) -> None:
        self.assertEqual(self.raw_df.shape, (768, 9))
        self.assertIn("Outcome", self.raw_df.columns)

    def test_target_definition_is_binary(self) -> None:
        replaced = replace_question_marks(self.raw_df)
        targeted = build_target(replaced, self.config)
        self.assertNotIn("Outcome", targeted.columns)
        self.assertIn("target", targeted.columns)
        self.assertEqual(set(targeted["target"].unique().tolist()), {0, 1})

    def test_identifier_columns_are_removed_by_contract(self) -> None:
        replaced = replace_question_marks(self.raw_df)
        targeted = build_target(replaced, self.config)
        trimmed, dropped = drop_configured_columns(targeted, self.config)
        self.assertEqual(dropped, [])
        self.assertEqual(trimmed.shape[1], 9)

    def test_metadata_feature_schema_is_consistent(self) -> None:
        numeric_columns = self.metadata["numeric_columns"]
        categorical_columns = self.metadata["categorical_columns"]
        self.assertEqual(len(numeric_columns), 8)
        self.assertEqual(len(categorical_columns), 0)
        self.assertEqual(len(set(numeric_columns + categorical_columns)), 8)
        self.assertEqual(self.metadata["categorical_cardinalities"], {})
        self.assertEqual(self.metadata["embedding_dimensions"], {})

    def test_feature_type_inference_matches_saved_metadata(self) -> None:
        replaced = replace_question_marks(self.raw_df)
        targeted = build_target(replaced, self.config)
        trimmed, _ = drop_configured_columns(targeted, self.config)
        trimmed = trimmed.drop(columns=self.metadata["dropped_columns_high_missing"])
        numeric_cols, categorical_cols = infer_feature_types(trimmed, self.config)
        self.assertEqual(numeric_cols, self.metadata["numeric_columns"])
        self.assertEqual(categorical_cols, self.metadata["categorical_columns"])

    def test_train_artifact_matches_metadata_shapes(self) -> None:
        train_dataset = load_train_dataset()
        self.assertEqual(tuple(train_dataset.x_numeric.shape), tuple(self.metadata["train_shape"]["x_numeric"]))
        self.assertEqual(tuple(train_dataset.x_categorical.shape), tuple(self.metadata["train_shape"]["x_categorical"]))
        self.assertEqual(train_dataset.y.shape[0], self.metadata["train_shape"]["y"][0])


if __name__ == "__main__":
    unittest.main()
