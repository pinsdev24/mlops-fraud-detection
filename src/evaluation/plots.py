"""Plotting utilities for MLflow artefacts.

All functions return the figure path (str) so callers can pass it to
mlflow.log_artifact().  Every plot uses a consistent dark-style theme.
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

matplotlib.use("Agg")  # Non-interactive backend for Docker / CI headless runs

logger = logging.getLogger(__name__)

STYLE = "seaborn-v0_8-darkgrid"
FIG_DPI = 150


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return str(path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path = "artifacts/confusion_matrix.png",
) -> str:
    """Plot and save the confusion matrix."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fraud"])
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    return _save(fig, Path(output_path))


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pr_auc: float,
    output_path: str | Path = "artifacts/pr_curve.png",
) -> str:
    """Plot and save the Precision-Recall curve."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        PrecisionRecallDisplay.from_predictions(
            y_true, y_proba, ax=ax, name=f"Model (PR-AUC={pr_auc:.4f})"
        )
        ax.set_title("Precision-Recall Curve (primary metric)", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        # Baseline (random classifier)
        baseline = y_true.mean()
        ax.axhline(
            baseline, color="grey", linestyle="--", alpha=0.7, label=f"Baseline ({baseline:.4f})"
        )
        ax.legend()
    return _save(fig, Path(output_path))


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    roc_auc: float,
    output_path: str | Path = "artifacts/roc_curve.png",
) -> str:
    """Plot and save the ROC curve."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=f"Model (AUC={roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random classifier")
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
    return _save(fig, Path(output_path))


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    output_path: str | Path = "artifacts/feature_importance.png",
    top_n: int = 20,
) -> str:
    """Bar chart of top-N feature importances (tree-based models)."""
    indices = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(top_n), top_vals[::-1], color="steelblue", edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
        ax.invert_xaxis()
    return _save(fig, Path(output_path))


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    output_path: str | Path = "artifacts/shap_summary.png",
) -> str:
    """SHAP beeswarm summary plot saved to file."""
    try:
        import shap

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(10, 7))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
                plot_size=None,
            )
            plt.title("SHAP Feature Impact — Fraud Detection", fontsize=14, fontweight="bold")
            plt.tight_layout()
        return _save(plt.gcf(), Path(output_path))
    except ImportError:
        logger.warning("shap not installed — skipping SHAP summary plot.")
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP plot failed: %s", exc)
        return ""


def plot_threshold_analysis(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    optimal_threshold: float,
    output_path: str | Path = "artifacts/threshold_analysis.png",
) -> str:
    """Plot precision and recall as a function of the decision threshold."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, precisions[:-1], label="Precision", color="dodgerblue")
        ax.plot(thresholds, recalls[:-1], label="Recall", color="tomato")
        ax.axvline(
            optimal_threshold,
            color="gold",
            linestyle="--",
            label=f"Optimal threshold ({optimal_threshold:.3f})",
        )
        ax.set_xlabel("Decision Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Precision / Recall vs. Decision Threshold", fontsize=14, fontweight="bold")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    return _save(fig, Path(output_path))
