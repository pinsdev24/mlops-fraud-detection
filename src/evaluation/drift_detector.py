"""CI-9 — Data drift and concept drift detection.

Uses Evidently to compute data drift reports and log them as MLflow artefacts.
Also provides a simulation helper that injects synthetic drift as specified
in the project brief.

Simulation spec (from sujet):
    Data drift:
      - Amount × 1.3 on 20% of events after t0
      - Missing rate: 0% → 5% on one key variable
    Concept drift:
      - Before t0: fraud if amount > 800 AND international=1
      - After t0:  fraud if amount > 300 AND online=1

Usage:
    python -m src.evaluation.drift_detector \\
        --reference data/processed/reference.parquet \\
        --current   data/processed/current.parquet   \\
        --output    artifacts/drift_report.html
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.data.loader import AMOUNT_COL

logger = logging.getLogger(__name__)

# Drift alert thresholds
DRIFT_SHARE_ALERT = 0.20  # >20% drifted features triggers alert
DATASET_DRIFT_THRESHOLD = 0.05  # p-value threshold for each feature


# ---------------------------------------------------------------------------
# Core drift detection
# ---------------------------------------------------------------------------


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str | Path = "artifacts/drift_report.html",
    log_to_mlflow: bool = True,
) -> dict:
    """Run Evidently drift detection and optionally log to MLflow.

    Args:
        reference_df: Historical baseline DataFrame (training distribution).
        current_df: Recent/production data to compare against.
        output_path: Path to write the HTML report.
        log_to_mlflow: If True, attach the report and alert tag to the active run.

    Returns:
        Dict with 'drift_share' (0–1) and 'drifted_features' list.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        logger.error("evidently not installed — cannot run drift detection.")
        return {"drift_share": None, "drifted_features": []}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select numeric columns present in both DataFrames
    common_cols = [
        c
        for c in reference_df.columns
        if c in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[c])
    ]

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[common_cols],
        current_data=current_df[common_cols],
    )
    report.save_html(str(output_path))
    logger.info("Drift report saved: %s", output_path)

    # Parse results
    result = report.as_dict()
    drift_results = result["metrics"][0]["result"]
    drift_share = drift_results.get("share_of_drifted_columns", 0.0)
    drifted = [
        col
        for col, stats in drift_results.get("drift_by_columns", {}).items()
        if stats.get("drift_detected", False)
    ]

    logger.info(
        "Drift share: %.1f%% | Drifted features: %s",
        drift_share * 100,
        drifted if drifted else "none",
    )

    if log_to_mlflow:
        try:
            mlflow.log_metric("drift_share", drift_share)
            mlflow.log_metric("n_drifted_features", len(drifted))
            mlflow.log_artifact(str(output_path))

            if drift_share > DRIFT_SHARE_ALERT:
                alert_msg = (
                    f"DATA_DRIFT_DETECTED: {drift_share:.1%} of features drifted. "
                    f"Affected: {', '.join(drifted)}"
                )
                mlflow.set_tag("alert", alert_msg)
                logger.warning("🚨 %s", alert_msg)
            else:
                mlflow.set_tag("alert", "none")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not log drift results to MLflow: %s", exc)

    return {"drift_share": drift_share, "drifted_features": drifted}


# ---------------------------------------------------------------------------
# Drift simulation (from project spec)
# ---------------------------------------------------------------------------


def simulate_data_drift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Inject synthetic data drift as specified in the project brief.

    Applies the following changes to a *copy* of df:
      - Amount × 1.3 on a random 20% of rows
      - Missing rate 5% injected on V1 (one key PCA feature)

    Args:
        df: Original feature DataFrame.
        seed: Random seed for reproducibility.

    Returns:
        drifted_df: Copy of df with drift injected.
    """
    rng = np.random.default_rng(seed)
    drifted = df.copy()

    # 1. Amount drift: multiply by 1.3 on 20% of transactions
    n_affected = int(len(drifted) * 0.20)
    affected_idx = rng.choice(len(drifted), size=n_affected, replace=False)
    drifted.iloc[affected_idx, drifted.columns.get_loc(AMOUNT_COL)] *= 1.3
    logger.info("Amount drift applied to %d rows (×1.3)", n_affected)

    # 2. Missing rate drift on V1: inject 5% NaN
    if "V1" in drifted.columns:
        n_missing = int(len(drifted) * 0.05)
        missing_idx = rng.choice(len(drifted), size=n_missing, replace=False)
        drifted.loc[drifted.index[missing_idx], "V1"] = np.nan
        logger.info("Missing rate drift: %.0f NaN injected into V1", n_missing)

    return drifted


def simulate_concept_drift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate concept drift by flipping the fraud rule.

    Before t0: fraud if Amount > 800 (big transactions targeted)
    After  t0: fraud if Amount > 300 (fraudsters fractionate)

    This creates a new 'Class' column based on the new rule,
    to test if model performance degrades detectably.

    Args:
        df: Feature DataFrame (must contain Amount column).
        seed: Not used, kept for API consistency.

    Returns:
        df_drifted: Copy with 'Class' re-labeled under new rule.
    """
    drifted = df.copy()
    # Simplified concept drift: fraud is now triggered at lower amounts
    drifted["Class"] = ((drifted[AMOUNT_COL] > 300) & (drifted[AMOUNT_COL] < 700)).astype(int)
    new_rate = drifted["Class"].mean()
    logger.info(
        "Concept drift applied — new fraud rate: %.3f%% (was rule: Amount>800)",
        new_rate * 100,
    )
    return drifted


def run_drift_monitoring(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path = "artifacts/drift_report.html",
) -> dict:
    """Load reference and current DataFrames and run drift detection."""
    reference_path = Path(reference_path)
    current_path = Path(current_path)

    # Support .csv and .parquet
    def _load(p: Path) -> pd.DataFrame:
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)

    reference_df = _load(reference_path)
    current_df = _load(current_path)
    logger.info("Loaded reference=%d rows, current=%d rows", len(reference_df), len(current_df))
    return detect_drift(reference_df, current_df, output_path=output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI-9 Drift Detector")
    parser.add_argument("--reference", required=True, help="Path to reference dataset")
    parser.add_argument("--current", required=True, help="Path to current/production dataset")
    parser.add_argument("--output", default="artifacts/drift_report.html")
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=DRIFT_SHARE_ALERT,
        help="Fraction of drifted features that triggers a non-zero exit code",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    results = run_drift_monitoring(args.reference, args.current, args.output)
    drift_share = results.get("drift_share", 0.0) or 0.0
    if drift_share > args.alert_threshold:
        logger.warning("🚨 Drift alert: %.1f%% features drifted", drift_share * 100)
        sys.exit(2)  # Exit 2 = drift alert (not a code error)
    sys.exit(0)
