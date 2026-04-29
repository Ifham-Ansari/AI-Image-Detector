from pathlib import Path
import sys

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import CLASS_NAMES


def _unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def calculate_metrics(model, dataloader, device: str) -> dict:
    model.eval()
    true_labels = []
    preds = []
    probas = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, _ = _unpack_batch(batch)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)[:, 1]
            predictions = outputs.argmax(dim=1)
            true_labels.extend(targets.cpu().tolist())
            preds.extend(predictions.cpu().tolist())
            probas.extend(probabilities.cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(true_labels, preds),
        "precision": precision_score(true_labels, preds, zero_division=0),
        "recall": recall_score(true_labels, preds, zero_division=0),
        "f1": f1_score(true_labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(true_labels, probas) if len(set(true_labels)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(true_labels, preds).tolist(),
        "per_class_accuracy": _compute_per_class_accuracy(true_labels, preds),
    }
    return metrics


def _compute_per_class_accuracy(true_labels, preds):
    per_class = {}
    cm = confusion_matrix(true_labels, preds)
    for idx, label in enumerate(CLASS_NAMES):
        support = cm[idx].sum()
        correct = cm[idx, idx]
        per_class[label] = float(correct / support) if support else 0.0
    return per_class


def plot_confusion_matrix(cm, labels, output_path: Path) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_curve(true_labels, probas, output_path: Path) -> None:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(true_labels, probas)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.savefig(output_path)
    plt.close(fig)
