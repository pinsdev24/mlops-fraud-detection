"""CI-5 — Automatic model comparison and promotion gate.

Compares a new MLflow run against the current champion run.
Promotes the new model only if it does not degrade performance
beyond a configurable tolerance threshold.

Rule (from CI-5 spec):
    "Le modèle n'est promu que si AUC > modèle précédent − 1%."

Usage:
    # From CLI (integrated in GitLab CI pipeline):
    python -m src.models.compare --experiment fraud-detection-v1 \\
                                  --new-run-id <RUN_ID> \\
                                  --metric pr_auc --tolerance 0.01
"""

import argparse
import logging
import sys
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def get_champion_run(
    experiment_name: str,
    metric: str = "pr_auc",
) -> Optional[mlflow.entities.Run]:
    """Find the best completed run in an experiment by a given metric.

    Returns None if no finished runs exist yet.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("Experiment '%s' not found — no champion.", experiment_name)
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        logger.info("No finished runs in '%s' — no champion yet.", experiment_name)
        return None

    champion = runs[0]
    champion_val = champion.data.metrics.get(metric, None)
    logger.info(
        "Current champion → run_id: %s | %s: %.4f",
        champion.info.run_id,
        metric,
        champion_val if champion_val is not None else float("nan"),
    )
    return champion


def should_promote(
    new_run_id: str,
    reference_run_id: str,
    metric: str = "pr_auc",
    tolerance: float = 0.01,
) -> bool:
    """CI-5: Return True if the new model should replace the champion.

    Promotion condition: new_metric >= reference_metric - tolerance

    Example from spec:
        "Le modèle n'est promu que si l'AUC > modèle précédent − 1%."

    Args:
        new_run_id: MLflow run_id of the candidate model.
        reference_run_id: MLflow run_id of the current champion.
        metric: The metric name to compare (default: 'pr_auc').
        tolerance: Allowed degradation (default: 0.01 = 1%).

    Returns:
        True if the new model should be promoted, False otherwise.
    """
    client = MlflowClient()

    new_run = client.get_run(new_run_id)
    ref_run = client.get_run(reference_run_id)

    new_val = new_run.data.metrics.get(metric)
    ref_val = ref_run.data.metrics.get(metric)

    if new_val is None:
        logger.error("Metric '%s' not found in new run %s", metric, new_run_id)
        return False
    if ref_val is None:
        logger.warning(
            "Metric '%s' not found in reference run %s — promoting new run by default.",
            metric, reference_run_id,
        )
        return True

    threshold = ref_val - tolerance
    promoted = new_val >= threshold

    logger.info(
        "CI-5 Comparison | metric=%s | new=%.4f | champion=%.4f | "
        "min_required=%.4f | promoted=%s",
        metric, new_val, ref_val, threshold, promoted,
    )

    # Log comparison result back to the new run
    client.set_tag(new_run_id, f"ci5_comparison_{metric}", f"{new_val:.4f}_vs_{ref_val:.4f}")
    client.set_tag(new_run_id, "ci5_reference_run_id", reference_run_id)

    return promoted


def compare_all_runs(
    experiment_name: str,
    metric: str = "pr_auc",
    max_results: int = 10,
) -> list[dict]:
    """Return a sorted leaderboard of the best N runs for the experiment.

    Useful for the rapport section 8.6 (tableau comparatif).
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=max_results,
    )

    leaderboard = []
    for run in runs:
        m = run.data.metrics
        p = run.data.params
        leaderboard.append({
            "run_id": run.info.run_id[:8],
            "run_name": run.info.run_name,
            "model": p.get("model_name", "?"),
            "strategy": p.get("imbalance_strategy", "?"),
            "pr_auc": m.get("pr_auc", None),
            "roc_auc": m.get("roc_auc", None),
            "f1_fraud": m.get("f1_fraud", None),
            "recall_fraud": m.get("recall_fraud", None),
            "precision_fraud": m.get("precision_fraud", None),
            "threshold": m.get("threshold", None),
        })

    return leaderboard


def print_leaderboard(experiment_name: str, metric: str = "pr_auc") -> None:
    """Pretty-print the run leaderboard to stdout."""
    rows = compare_all_runs(experiment_name, metric)
    if not rows:
        print("No finished runs found.")
        return

    header = (
        f"{'Run':8} | {'Model':20} | {'Strategy':14} | "
        f"{'PR-AUC':7} | {'ROC-AUC':7} | {'F1':7} | {'Recall':7}"
    )
    print("\n" + "=" * len(header))
    print(f"{'Leaderboard — ' + experiment_name:^{len(header)}}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['run_id']:8} | {r['model']:20} | {r['strategy']:14} | "
            f"{_fmt(r['pr_auc']):7} | {_fmt(r['roc_auc']):7} | "
            f"{_fmt(r['f1_fraud']):7} | {_fmt(r['recall_fraud']):7}"
        )
    print("=" * len(header) + "\n")


def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "  N/A "


# ---------------------------------------------------------------------------
# CLI (used by GitLab CI step CI-5)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CI-5 — Automatic model comparison and promotion gate"
    )
    parser.add_argument("--experiment", default="fraud-detection-v1")
    parser.add_argument("--new-run-id", help="Run ID of the candidate model")
    parser.add_argument("--metric", default="pr_auc")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--leaderboard", action="store_true", help="Print leaderboard and exit")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    if args.leaderboard:
        print_leaderboard(args.experiment, metric=args.metric)
        sys.exit(0)

    if not args.new_run_id:
        print("Error: --new-run-id is required when not using --leaderboard")
        sys.exit(1)

    champion = get_champion_run(args.experiment, args.metric)
    if champion is None or champion.info.run_id == args.new_run_id:
        print("No previous champion — new run is the reference. ✅")
        sys.exit(0)

    promoted = should_promote(
        args.new_run_id, champion.info.run_id, args.metric, args.tolerance
    )
    print(f"Promotion decision: {'✅ PROMOTED' if promoted else '❌ REJECTED'}")
    sys.exit(0 if promoted else 1)
