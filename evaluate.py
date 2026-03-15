"""
evaluate.py  —  Metrics computation, confusion matrix, ROC curve, and full report.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, average_precision_score,
)


# ──────────────────────────────────────────────────────────────
# Core Metrics
# ──────────────────────────────────────────────────────────────
def compute_metrics_from_arrays(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute all classification metrics from raw logits and true labels."""
    preds = np.argmax(logits, axis=1)
    probs = _softmax(logits)[:, 1]   # P(real)

    metrics = {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "roc_auc":   float(roc_auc_score(labels, probs)),
        "avg_precision": float(average_precision_score(labels, probs)),
    }
    return metrics


def compute_metrics_from_logits(eval_pred) -> Dict[str, float]:
    """Adapter for HuggingFace Trainer's compute_metrics callback."""
    logits, labels = eval_pred
    return compute_metrics_from_arrays(logits, labels)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ──────────────────────────────────────────────────────────────
# Full Evaluation Report
# ──────────────────────────────────────────────────────────────
def full_evaluation_report(
    logits: np.ndarray,
    labels: np.ndarray,
    output_dir: Optional[str] = None,
    label_names: list = ["Fake", "Real"],
) -> Dict:
    """
    Print and optionally save a full evaluation report:
    - Classification metrics
    - Confusion matrix heatmap
    - ROC curve
    - Precision-Recall curve
    """
    preds = np.argmax(logits, axis=1)
    probs = _softmax(logits)[:, 1]

    metrics = compute_metrics_from_arrays(logits, labels)

    print("\n" + "═" * 50)
    print("          EVALUATION REPORT")
    print("═" * 50)
    print(classification_report(labels, preds, target_names=label_names))
    for k, v in metrics.items():
        print(f"  {k:<16}: {v:.4f}")
    print("═" * 50)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        _plot_confusion_matrix(labels, preds, label_names, out / "confusion_matrix.png")
        _plot_roc_curve(labels, probs, out / "roc_curve.png")
        _plot_pr_curve(labels, probs, out / "pr_curve.png")
        print(f"\n📊 Plots saved to {output_dir}/")

    return metrics


# ──────────────────────────────────────────────────────────────
# Plot Helpers
# ──────────────────────────────────────────────────────────────
def _plot_confusion_matrix(labels, preds, label_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_pr_curve(labels, probs, save_path):
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#16a34a", lw=2, label=f"PR Curve (AP = {ap:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Plot train/val loss and F1 over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_f1"], "g-o", label="Val F1")
    axes[1].plot(epochs, history["val_acc"], "m-o", label="Val Accuracy")
    axes[1].set_title("Validation Metrics", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved training plot to {save_path}")
    else:
        plt.show()
    plt.close()
