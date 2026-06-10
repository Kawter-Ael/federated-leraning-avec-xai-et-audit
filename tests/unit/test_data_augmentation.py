from __future__ import annotations

import unittest

import numpy as np

from shared.data_augmentation import (
    apply_augmentation,
    augment_mixup,
    augment_noise,
    augment_smote,
)


def _make_binary_data(n: int = 100, n_features: int = 4, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    x_numeric = rng.standard_normal((n, n_features)).astype(np.float32)
    x_categorical = np.zeros((n, 0), dtype=np.int64)
    y = rng.integers(0, 2, size=n).astype(np.int8)
    return x_numeric, x_categorical, y


class TestAugmentNoise(unittest.TestCase):
    def test_increases_sample_count(self) -> None:
        x_num, x_cat, y = _make_binary_data(50)
        x_out, _, y_out = augment_noise(x_num, x_cat, y, factor=0.5, seed=0)
        self.assertGreater(len(y_out), len(y))

    def test_preserves_original_data(self) -> None:
        x_num, x_cat, y = _make_binary_data(20)
        x_out, _, y_out = augment_noise(x_num, x_cat, y, factor=0.5, seed=1)
        np.testing.assert_array_equal(x_out[:20], x_num)
        np.testing.assert_array_equal(y_out[:20], y)

    def test_zero_factor_is_noop(self) -> None:
        x_num, x_cat, y = _make_binary_data(20)
        x_out, x_cat_out, y_out = augment_noise(x_num, x_cat, y, factor=0.0, seed=2)
        self.assertEqual(len(y_out), len(y))
        np.testing.assert_array_equal(x_out, x_num)
        np.testing.assert_array_equal(x_cat_out, x_cat)
        np.testing.assert_array_equal(y_out, y)


class TestAugmentMixup(unittest.TestCase):
    def test_increases_sample_count(self) -> None:
        x_num, x_cat, y = _make_binary_data(50)
        x_out, _, y_out = augment_mixup(x_num, x_cat, y, factor=0.5, seed=0)
        self.assertGreater(len(y_out), len(y))

    def test_single_class_returns_original(self) -> None:
        x_num = np.random.randn(20, 3).astype(np.float32)
        x_cat = np.zeros((20, 0), dtype=np.int64)
        y = np.zeros(20, dtype=np.int8)
        x_out, _, y_out = augment_mixup(x_num, x_cat, y, factor=0.5, seed=3)
        self.assertEqual(len(y_out), len(y))

    def test_zero_factor_is_noop(self) -> None:
        x_num, x_cat, y = _make_binary_data(30)
        x_out, x_cat_out, y_out = augment_mixup(x_num, x_cat, y, factor=0.0, seed=7)
        self.assertEqual(len(y_out), len(y))
        np.testing.assert_array_equal(x_out, x_num)
        np.testing.assert_array_equal(y_out, y)


class TestAugmentSMOTE(unittest.TestCase):
    def test_increases_sample_count(self) -> None:
        x_num, x_cat, y = _make_binary_data(100)
        x_out, _, y_out = augment_smote(x_num, x_cat, y, factor=1.0, seed=0)
        self.assertGreater(len(y_out), len(y))

    def test_fallback_on_tiny_data(self) -> None:
        x_num = np.random.randn(4, 3).astype(np.float32)
        x_cat = np.zeros((4, 0), dtype=np.int64)
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        x_out, _, y_out = augment_smote(x_num, x_cat, y, factor=1.0, seed=5)
        self.assertGreaterEqual(len(y_out), len(y))

    def test_fallback_on_single_class(self) -> None:
        x_num = np.random.randn(30, 3).astype(np.float32)
        x_cat = np.zeros((30, 0), dtype=np.int64)
        y = np.zeros(30, dtype=np.int8)
        x_out, _, y_out = augment_smote(x_num, x_cat, y, factor=1.0, seed=6)
        self.assertEqual(len(y_out), len(y))


class TestApplyAugmentation(unittest.TestCase):
    def test_disabled_returns_original(self) -> None:
        x_num, x_cat, y = _make_binary_data(30)
        x_out, _, y_out = apply_augmentation(x_num, x_cat, y, {"enabled": False})
        self.assertIs(x_out, x_num)
        self.assertIs(y_out, y)

    def test_dispatches_smote(self) -> None:
        x_num, x_cat, y = _make_binary_data(80)
        x_out, _, y_out = apply_augmentation(
            x_num, x_cat, y, {"enabled": True, "method": "smote", "factor": 1.0}
        )
        self.assertGreater(len(y_out), len(y))

    def test_dispatches_noise(self) -> None:
        x_num, x_cat, y = _make_binary_data(30)
        x_out, _, y_out = apply_augmentation(
            x_num,
            x_cat,
            y,
            {"enabled": True, "method": "noise", "factor": 0.5, "noise_std": 0.05},
        )
        self.assertGreater(len(y_out), len(y))

    def test_dispatches_mixup(self) -> None:
        x_num, x_cat, y = _make_binary_data(40)
        x_out, _, y_out = apply_augmentation(
            x_num,
            x_cat,
            y,
            {"enabled": True, "method": "mixup", "factor": 0.5, "mixup_alpha": 0.2},
        )
        self.assertGreater(len(y_out), len(y))

    def test_unknown_method_defaults_to_smote(self) -> None:
        x_num, x_cat, y = _make_binary_data(80)
        x_out, _, y_out = apply_augmentation(
            x_num, x_cat, y, {"enabled": True, "method": "unknown", "factor": 1.0}
        )
        self.assertGreater(len(y_out), len(y))


class TestAugmentationContracts(unittest.TestCase):
    def test_noise_preserves_column_count(self) -> None:
        x_num, x_cat, y = _make_binary_data(40, n_features=6)
        x_out, x_cat_out, y_out = augment_noise(x_num, x_cat, y, factor=0.5, seed=10)
        self.assertEqual(x_out.shape[1], x_num.shape[1])
        self.assertEqual(x_cat_out.shape[1], x_cat.shape[1])

    def test_mixup_preserves_column_count(self) -> None:
        x_num, x_cat, y = _make_binary_data(40, n_features=6)
        x_out, x_cat_out, y_out = augment_mixup(x_num, x_cat, y, factor=0.5, seed=11)
        self.assertEqual(x_out.shape[1], x_num.shape[1])
        self.assertEqual(x_cat_out.shape[1], x_cat.shape[1])

    def test_smote_preserves_column_count(self) -> None:
        x_num, x_cat, y = _make_binary_data(80, n_features=6)
        x_out, x_cat_out, y_out = augment_smote(x_num, x_cat, y, factor=1.0, seed=12)
        self.assertEqual(x_out.shape[1], x_num.shape[1])
        self.assertEqual(x_cat_out.shape[1], x_cat.shape[1])

    def test_augmentation_preserves_dtypes(self) -> None:
        x_num, x_cat, y = _make_binary_data(50, n_features=4)
        for method in ("noise", "mixup"):
            x_out, x_cat_out, y_out = apply_augmentation(
                x_num,
                x_cat,
                y,
                {"enabled": True, "method": method, "factor": 0.5},
            )
            self.assertEqual(x_out.dtype, np.float32, f"{method} x_numeric dtype")
            self.assertIn(y_out.dtype, (np.int8, np.int64), f"{method} y dtype")

    def test_noise_factor_zero_returns_identical_arrays(self) -> None:
        x_num, x_cat, y = _make_binary_data(25)
        x_out, x_cat_out, y_out = apply_augmentation(
            x_num,
            x_cat,
            y,
            {"enabled": True, "method": "noise", "factor": 0.0},
        )
        self.assertIs(x_out, x_num)
        self.assertIs(y_out, y)

    def test_mixup_factor_zero_returns_identical_arrays(self) -> None:
        x_num, x_cat, y = _make_binary_data(25)
        x_out, x_cat_out, y_out = apply_augmentation(
            x_num,
            x_cat,
            y,
            {"enabled": True, "method": "mixup", "factor": 0.0},
        )
        self.assertIs(x_out, x_num)
        self.assertIs(y_out, y)


if __name__ == "__main__":
    unittest.main()
