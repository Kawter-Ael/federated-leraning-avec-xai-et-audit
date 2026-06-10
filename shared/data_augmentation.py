"""Tabular data augmentation strategies for federated learning clients."""

from __future__ import annotations

from typing import Any

import numpy as np


def augment_smote(
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    factor: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min())
    if min_class_count < 6 or len(unique) < 2:
        return x_numeric, x_categorical, y

    from imblearn.over_sampling import SMOTE

    n_target = int(len(y) * (1.0 + factor))
    sampling_strategy: dict[int, int] = {}
    for cls in unique:
        cls_count = int(counts[unique == cls][0])
        target = max(cls_count, n_target // len(unique))
        if target > cls_count:
            sampling_strategy[int(cls)] = target

    if not sampling_strategy:
        return x_numeric, x_categorical, y

    has_categorical = x_categorical.shape[1] > 0

    if has_categorical:
        combined = np.concatenate([x_numeric, x_categorical.astype(np.float32)], axis=1)
    else:
        combined = x_numeric.copy()

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=seed,
        k_neighbors=min(5, min_class_count - 1),
    )
    try:
        resampled, y_resampled = smote.fit_resample(combined, y)
    except ValueError:
        return x_numeric, x_categorical, y

    n_num = x_numeric.shape[1]
    x_num_out = resampled[:, :n_num].astype(np.float32)
    x_cat_out: np.ndarray
    if has_categorical:
        x_cat_out = np.round(resampled[:, n_num:]).astype(np.int64)
    else:
        x_cat_out = np.zeros((len(y_resampled), 0), dtype=np.int64)

    return x_num_out, x_cat_out, y_resampled.astype(np.int8)


def augment_noise(
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    noise_std: float = 0.05,
    factor: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_augment = int(len(y) * factor)
    if n_augment <= 0:
        return x_numeric, x_categorical, y
    indices = rng.choice(len(y), size=n_augment, replace=True)

    x_num_aug = x_numeric[indices].copy()
    if x_num_aug.shape[1] > 0:
        stds = np.std(x_numeric, axis=0, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        noise = rng.normal(loc=0.0, scale=noise_std, size=x_num_aug.shape) * stds
        x_num_aug = x_num_aug + noise.astype(np.float32)

    x_cat_aug = x_categorical[indices].copy()
    y_aug = y[indices].copy()

    x_numeric_out = np.concatenate([x_numeric, x_num_aug], axis=0)
    x_categorical_out = np.concatenate([x_categorical, x_cat_aug], axis=0)
    y_out = np.concatenate([y, y_aug], axis=0)

    return x_numeric_out, x_categorical_out, y_out


def augment_mixup(
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.2,
    factor: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return x_numeric, x_categorical, y

    factor_val = float(factor)
    if factor_val <= 0:
        return x_numeric, x_categorical, y

    x_num_list = [x_numeric]
    x_cat_list = [x_categorical]
    y_list = [y]

    for cls in unique_classes:
        cls_mask = y == cls
        cls_indices = np.where(cls_mask)[0]
        if len(cls_indices) < 2:
            continue
        n_cls = max(1, int(len(cls_indices) * factor_val))
        for _ in range(n_cls):
            i, j = rng.choice(cls_indices, size=2, replace=False)
            lam = float(rng.beta(alpha, alpha))
            x_num_mixed = (
                (lam * x_numeric[i] + (1 - lam) * x_numeric[j])
                .reshape(1, -1)
                .astype(np.float32)
            )
            x_num_list.append(x_num_mixed)

            if x_categorical.shape[1] > 0:
                chosen = (
                    x_categorical[i].reshape(1, -1)
                    if lam >= 0.5
                    else x_categorical[j].reshape(1, -1)
                )
                x_cat_list.append(chosen)
            else:
                x_cat_list.append(x_categorical[i : i + 1])

            y_list.append(np.array([cls], dtype=np.int8))

    if len(x_num_list) == 1:
        return x_numeric, x_categorical, y

    x_numeric_out = np.concatenate(x_num_list, axis=0)
    x_categorical_out = np.concatenate(x_cat_list, axis=0)
    y_out = np.concatenate(y_list, axis=0)

    return x_numeric_out, x_categorical_out, y_out


def augment_minority_balance(
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate-with-noise oversampling for the minority class.

    Used as the guaranteed fallback when SMOTE cannot run (e.g., fewer than
    6 minority samples). Balances the minority to match the majority count;
    numeric features get a light Gaussian perturbation to avoid creating
    exact duplicates that the model can memorize.
    """
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return x_numeric, x_categorical, y
    majority_count = int(counts.max())
    rng = np.random.default_rng(seed)
    x_num_list = [x_numeric]
    x_cat_list = [x_categorical]
    y_list = [y]
    for cls, cls_count in zip(unique, counts):
        deficit = majority_count - int(cls_count)
        if deficit <= 0:
            continue
        cls_indices = np.where(y == cls)[0]
        picks = rng.choice(cls_indices, size=deficit, replace=True)
        x_num_aug = x_numeric[picks].astype(np.float32).copy()
        if x_num_aug.shape[1] > 0:
            stds = np.std(x_numeric, axis=0, keepdims=True)
            stds = np.where(stds < 1e-8, 1.0, stds)
            noise = rng.normal(0.0, noise_std, size=x_num_aug.shape) * stds
            x_num_aug = (x_num_aug + noise).astype(np.float32)
        x_cat_aug = x_categorical[picks].copy()
        y_aug = np.full(deficit, int(cls), dtype=np.int8)
        x_num_list.append(x_num_aug)
        x_cat_list.append(x_cat_aug)
        y_list.append(y_aug)
    return (
        np.concatenate(x_num_list, axis=0),
        np.concatenate(x_cat_list, axis=0),
        np.concatenate(y_list, axis=0),
    )


def _smote_with_fallback(
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    factor: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SMOTE when feasible, minority-duplicate otherwise.

    SMOTE needs at least `k_neighbors + 1` minority samples (k_neighbors>=1),
    i.e. at least 2 minority rows. Below that threshold it silently returns
    the unchanged arrays, which for severely imbalanced data means the model
    keeps collapsing to all-negative. Routing to duplicate-with-noise
    oversampling guarantees the minority gets represented.
    """
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or int(counts.min()) < 2:
        return augment_minority_balance(x_numeric, x_categorical, y, seed=seed)
    if int(counts.min()) < 6:
        smote_out = augment_smote(x_numeric, x_categorical, y, factor=factor, seed=seed)
        # augment_smote returns the input unchanged when it cannot run; in that
        # case fall back to duplicate-with-noise balancing.
        if smote_out[2].shape == y.shape and np.array_equal(smote_out[2], y):
            return augment_minority_balance(x_numeric, x_categorical, y, seed=seed)
        return smote_out
    return augment_smote(x_numeric, x_categorical, y, factor=factor, seed=seed)


def apply_augmentation(
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not config.get("enabled", False):
        return x_numeric, x_categorical, y

    method = config.get("method", "smote").lower()
    seed = int(config.get("seed", 42))
    factor = float(config.get("factor", 1.0))

    if method == "smote":
        return _smote_with_fallback(
            x_numeric, x_categorical, y, factor=factor, seed=seed
        )
    elif method == "noise":
        return augment_noise(
            x_numeric,
            x_categorical,
            y,
            noise_std=float(config.get("noise_std", 0.05)),
            factor=factor,
            seed=seed,
        )
    elif method == "mixup":
        return augment_mixup(
            x_numeric,
            x_categorical,
            y,
            alpha=float(config.get("mixup_alpha", 0.2)),
            factor=factor,
            seed=seed,
        )
    elif method == "minority_balance":
        return augment_minority_balance(x_numeric, x_categorical, y, seed=seed)
    else:
        return _smote_with_fallback(
            x_numeric, x_categorical, y, factor=factor, seed=seed
        )
