"""Evaluation metrics for the fraud detection binary classifier.

All metrics are adapted for extreme class imbalance (0.172% fraud rate).
Primary metric: PR-AUC (AUPRC) as recommended by the dataset authors.
"""

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full metric suite for logging to MLflow.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class (fraud).
        threshold: Decision threshold to convert probabilities to labels.

    Returns:
        Dictionary of metric names → values, ready for mlflow.log_metrics().
    """
    y_pred = (y_proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "pr_auc": pr_auc,  # PRIMARY metric (AUPRC)
        "roc_auc": roc_auc,
        "f1_fraud": f1,
        "precision_fraud": precision,
        "recall_fraud": recall,
        "threshold": threshold,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }

    logger.info(
        "Metrics — PR-AUC: %.4f | ROC-AUC: %.4f | F1: %.4f | "
        "Precision: %.4f | Recall: %.4f | threshold: %.3f",
        pr_auc,
        roc_auc,
        f1,
        precision,
        recall,
        threshold,
    )
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f1",
) -> float:
    """Find the decision threshold that optimises the chosen strategy.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted probabilities.
        strategy: 'f1'         — maximise F1-score (default)
                  'recall_at_precision_90' — highest recall for precision >= 0.9

    Returns:
        Optimal threshold as a float.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    if strategy == "f1":
        # Avoid division by zero
        denom = precisions + recalls
        f1_scores = np.where(denom > 0, 2 * precisions * recalls / denom, 0.0)
        best_idx = np.argmax(f1_scores[:-1])  # thresholds has len-1 vs precisions
        optimal = float(thresholds[best_idx])
        logger.info(
            "Optimal threshold (F1 strategy): %.4f — F1=%.4f, P=%.4f, R=%.4f",
            optimal,
            f1_scores[best_idx],
            precisions[best_idx],
            recalls[best_idx],
        )

    elif strategy == "recall_at_precision_90":
        mask = precisions[:-1] >= 0.90
        if not mask.any():
            logger.warning("No threshold achieves precision >= 0.90; falling back to F1.")
            return find_optimal_threshold(y_true, y_proba, strategy="f1")
        best_idx = np.argmax(recalls[:-1][mask])
        optimal = float(thresholds[mask][best_idx])
        logger.info("Optimal threshold (Recall@P≥0.90 strategy): %.4f", optimal)

    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")

    return optimal


def classification_report_str(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Return a formatted classification report with class names."""
    return classification_report(
        y_true,
        y_pred,
        target_names=["Normal (0)", "Fraud (1)"],
        digits=4,
    )
