from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class EvaluationResult:
    loss: float
    metrics: dict[str, float]


def _has_both_classes(y_true: np.ndarray) -> bool:
    labels = np.unique(np.asarray(y_true))
    return labels.size >= 2


class EmbeddedTabularNeuralNetwork(nn.Module):
    def __init__(
        self,
        numeric_input_dim: int,
        categorical_cardinalities: list[int],
        embedding_dimensions: list[int],
        hidden_dims: list[int],
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality, embedding_dim in zip(categorical_cardinalities, embedding_dimensions)]
        )
        mlp_input_dim = numeric_input_dim + int(sum(embedding_dimensions))

        layers: list[nn.Module] = []
        current_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        embedded_parts: list[torch.Tensor] = []
        if x_categorical.shape[1] > 0:
            for index, embedding_layer in enumerate(self.embeddings):
                embedded_parts.append(embedding_layer(x_categorical[:, index]))
        if embedded_parts:
            categorical_embedding = torch.cat(embedded_parts, dim=1)
            combined = torch.cat([x_numeric, categorical_embedding], dim=1)
        else:
            combined = x_numeric
        return self.network(combined).squeeze(-1)


def get_device() -> torch.device:
    return torch.device("cpu")


def create_model(config: dict[str, Any], schema: dict[str, Any]) -> EmbeddedTabularNeuralNetwork:
    params = config["model"]["hyperparameters"]
    seed = int(config["project"]["seed"])
    torch.manual_seed(seed)
    categorical_columns = schema["categorical_columns"]
    cardinalities = [int(schema["categorical_cardinalities"][col]) for col in categorical_columns]
    embedding_dimensions = [int(schema["embedding_dimensions"][col]) for col in categorical_columns]
    return EmbeddedTabularNeuralNetwork(
        numeric_input_dim=len(schema["numeric_columns"]),
        categorical_cardinalities=cardinalities,
        embedding_dimensions=embedding_dimensions,
        hidden_dims=[int(value) for value in params["hidden_dims"]],
        dropout=float(params.get("dropout", 0.0)),
        use_batch_norm=bool(params.get("use_batch_norm", False)),
    )


def create_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    params = config["model"]["hyperparameters"]
    optimizer_name = str(params.get("optimizer", "adam")).lower()
    learning_rate = float(params.get("learning_rate", 1e-3))
    weight_decay = float(params.get("weight_decay", 0.0))
    if optimizer_name != "adam":
        raise ValueError(f"Optimiseur non supporte : {optimizer_name}")
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy().astype(np.float32) for tensor in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(state_dict) != len(parameters):
        raise ValueError(f"Nombre de parametres incoherent : attendu {len(state_dict)}, recu {len(parameters)}")
    updated_state = {}
    for (name, current_tensor), parameter in zip(state_dict.items(), parameters):
        array = np.asarray(parameter)
        if tuple(current_tensor.shape) != tuple(array.shape):
            raise ValueError(f"Incoherence de forme pour {name} : attendu {tuple(current_tensor.shape)}, recu {tuple(array.shape)}")
        updated_state[name] = torch.tensor(array, dtype=current_tensor.dtype)
    model.load_state_dict(updated_state, strict=True)


def compute_classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int8)
    has_both_classes = _has_both_classes(y_true)
    if has_both_classes:
        roc_auc = float(roc_auc_score(y_true, probabilities))
        pr_auc = float(average_precision_score(y_true, probabilities))
    else:
        positive_present = bool(np.any(np.asarray(y_true) == 1))
        roc_auc = 1.0
        pr_auc = 1.0 if positive_present else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
    }


def predict_probabilities(model: nn.Module, x_numeric: np.ndarray, x_categorical: np.ndarray) -> np.ndarray:
    device = get_device()
    model = model.to(device)
    model.eval()
    x_numeric_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=device)
    x_categorical_tensor = torch.tensor(x_categorical, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x_numeric_tensor, x_categorical_tensor)
        probabilities = torch.sigmoid(logits)
    return probabilities.detach().cpu().numpy().astype(np.float64)


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> EvaluationResult:
    metrics = compute_classification_metrics(y_true, probabilities, threshold)
    loss = float(log_loss(y_true, probabilities, labels=[0, 1]))
    return EvaluationResult(loss=loss, metrics=metrics)


def evaluate_model(
    model: nn.Module,
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationResult:
    probabilities = predict_probabilities(model, x_numeric, x_categorical)
    return evaluate_predictions(y, probabilities, threshold=threshold)


def select_decision_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    candidate_thresholds: list[float] | None = None,
    selection_strategy: str = "f1",
    min_recall: float = 0.0,
    fallback_y_true: np.ndarray | None = None,
    fallback_probabilities: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    # When the primary target set is single-class we cannot compute a
    # meaningful precision/recall trade-off on it. A fallback set (typically
    # the training data, which is guaranteed to contain both classes after
    # stratified split + augmentation) gives a usable operating point.
    if not _has_both_classes(y_true):
        if (
            fallback_y_true is not None
            and fallback_probabilities is not None
            and _has_both_classes(fallback_y_true)
        ):
            return select_decision_threshold(
                fallback_y_true,
                fallback_probabilities,
                candidate_thresholds=candidate_thresholds,
                selection_strategy=selection_strategy,
                min_recall=min_recall,
            )
        predictions = (probabilities >= 0.5).astype(np.int8)
        precision = float(precision_score(y_true, predictions, zero_division=0))
        recall = float(recall_score(y_true, predictions, zero_division=0))
        f1 = float(f1_score(y_true, predictions, zero_division=0))
        f2 = float(fbeta_score(y_true, predictions, beta=2, zero_division=0))
        return 0.5, {"precision": precision, "recall": recall, "f1": f1, "f2": f2}

    if candidate_thresholds is None:
        candidate_thresholds = [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]

    evaluated_candidates: list[tuple[float, dict[str, float]]] = []
    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(np.int8)
        precision = float(precision_score(y_true, predictions, zero_division=0))
        recall = float(recall_score(y_true, predictions, zero_division=0))
        f1 = float(f1_score(y_true, predictions, zero_division=0))
        f2 = float(fbeta_score(y_true, predictions, beta=2, zero_division=0))
        if recall < min_recall:
            score = float("-inf")
        elif selection_strategy == "precision_with_min_recall":
            score = precision
        elif selection_strategy == "f2":
            score = f2
        else:
            score = f1
        evaluated_candidates.append(
            (
                float(threshold),
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "f2": f2,
                    "score": score,
                },
            )
        )

    valid_candidates = [item for item in evaluated_candidates if item[1]["recall"] >= min_recall]
    candidate_pool = valid_candidates if valid_candidates else evaluated_candidates
    if not candidate_pool:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0}

    best_threshold, best_metrics = candidate_pool[0]
    for threshold, metrics in candidate_pool[1:]:
        if metrics["score"] > best_metrics["score"]:
            best_threshold, best_metrics = threshold, metrics
            continue
        if metrics["score"] == best_metrics["score"]:
            if metrics["recall"] > best_metrics["recall"] or (
                metrics["recall"] == best_metrics["recall"] and metrics["precision"] > best_metrics["precision"]
            ):
                best_threshold, best_metrics = threshold, metrics

    return best_threshold, {key: value for key, value in best_metrics.items() if key != "score"}


def build_data_loader(
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.tensor(x_numeric, dtype=torch.float32),
        torch.tensor(x_categorical, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def compute_pos_weight(y: np.ndarray, scale: float = 1.0) -> np.ndarray:
    positive_count = max(int((y == 1).sum()), 1)
    negative_count = max(int((y == 0).sum()), 1)
    balanced_ratio = negative_count / positive_count
    scaled_ratio = 1.0 + (balanced_ratio - 1.0) * float(scale)
    return np.asarray([scaled_ratio], dtype=np.float32)


def train_local_model(
    model: nn.Module,
    x_numeric: np.ndarray,
    x_categorical: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
) -> None:
    device = get_device()
    params = config["model"]["hyperparameters"]
    batch_size = int(params.get("batch_size", 256))
    local_epochs = int(params.get("local_epochs", 1))
    seed = int(config["project"]["seed"])
    early_stopping_patience = int(params.get("early_stopping_patience", 0))

    model = model.to(device)
    model.train()
    optimizer = create_optimizer(model, config)
    pos_weight = torch.tensor(
        compute_pos_weight(y_train, scale=float(params.get("pos_weight_scale", 1.0))),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = build_data_loader(x_numeric, x_categorical, y_train, batch_size=batch_size, shuffle=True, seed=seed)

    best_epoch_loss: float | None = None
    epochs_without_improvement = 0

    for _ in range(local_epochs):
        running_loss = 0.0
        batch_count = 0
        for batch_numeric, batch_categorical, batch_y in loader:
            batch_numeric = batch_numeric.to(device)
            batch_categorical = batch_categorical.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_numeric, batch_categorical)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())
            batch_count += 1

        if batch_count == 0:
            continue
        mean_epoch_loss = running_loss / batch_count
        if best_epoch_loss is None or mean_epoch_loss < best_epoch_loss - 1e-4:
            best_epoch_loss = mean_epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            break
