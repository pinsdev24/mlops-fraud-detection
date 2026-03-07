"""CI-3 — Data Quality Gate.

Blocks the training pipeline if the raw dataset violates any quality rule.
The gate can be run standalone:

    python -m src.data.quality_gate --data data/raw/creditcard.csv
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .loader import TARGET_COL, AMOUNT_COL, TIME_COL, PCA_FEATURES, load_raw_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality rules (all thresholds configurable)
# ---------------------------------------------------------------------------

CRITICAL_COLUMNS = [TARGET_COL, AMOUNT_COL] + PCA_FEATURES


@dataclass
class QualityConfig:
    """Thresholds for each quality rule."""

    max_missing_rate: float = 0.05  # CI-3 mandatory rule
    min_rows: int = 100_000  # sanity check on volume
    max_amount: float = 100_000.0  # outlier cap for Amount
    fraud_rate_min: float = 0.001  # expected class distribution
    fraud_rate_max: float = 0.005
    max_duplicate_rate: float = 0.01  # tolerate up to 1% exact duplicates


@dataclass
class QualityReport:
    """Collects all violations found during quality checks."""

    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def add_violation(self, msg: str) -> None:
        logger.error("[QUALITY GATE] VIOLATION: %s", msg)
        self.violations.append(msg)

    def add_warning(self, msg: str) -> None:
        logger.warning("[QUALITY GATE] WARNING: %s", msg)
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = ["=== Data Quality Report ==="]
        if self.passed:
            lines.append("✅ All quality checks passed.")
        else:
            lines.append(f"❌ {len(self.violations)} violation(s) found — pipeline BLOCKED:")
            for v in self.violations:
                lines.append(f"  • {v}")
        if self.warnings:
            lines.append(f"\n⚠️  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                lines.append(f"  • {w}")
        return "\n".join(lines)


class DataQualityGate:
    """Runs a series of quality checks on the raw DataFrame.

    Usage::

        gate = DataQualityGate()
        report = gate.run(df)
        if not report.passed:
            sys.exit(1)
    """

    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()

    def run(self, df: pd.DataFrame) -> QualityReport:
        report = QualityReport()
        self._check_volume(df, report)
        self._check_missing_values(df, report)
        self._check_duplicates(df, report)
        self._check_class_distribution(df, report)
        self._check_amount_outliers(df, report)
        self._check_negative_values(df, report)
        print(report.summary())
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_volume(self, df: pd.DataFrame, report: QualityReport) -> None:
        if len(df) < self.config.min_rows:
            report.add_violation(
                f"Dataset has only {len(df):,} rows — expected at least "
                f"{self.config.min_rows:,}."
            )
        else:
            logger.info("Volume check OK: %d rows", len(df))

    def _check_missing_values(self, df: pd.DataFrame, report: QualityReport) -> None:
        """CI-3: block if >5% missing values in any critical column."""
        for col in CRITICAL_COLUMNS:
            if col not in df.columns:
                continue
            missing_rate = df[col].isna().mean()
            if missing_rate > self.config.max_missing_rate:
                report.add_violation(
                    f"Column '{col}' has {missing_rate:.1%} missing values "
                    f"(threshold: {self.config.max_missing_rate:.0%})."
                )
            elif missing_rate > 0:
                report.add_warning(f"Column '{col}' has {missing_rate:.2%} missing values.")

    def _check_duplicates(self, df: pd.DataFrame, report: QualityReport) -> None:
        dup_rate = df.duplicated().mean()
        if dup_rate > self.config.max_duplicate_rate:
            report.add_violation(
                f"Duplicate row rate is {dup_rate:.1%} "
                f"(threshold: {self.config.max_duplicate_rate:.0%})."
            )
        elif dup_rate > 0:
            report.add_warning(f"Dataset contains {int(dup_rate * len(df))} duplicate rows.")

    def _check_class_distribution(self, df: pd.DataFrame, report: QualityReport) -> None:
        if TARGET_COL not in df.columns:
            return
        fraud_rate = df[TARGET_COL].mean()
        if not (self.config.fraud_rate_min <= fraud_rate <= self.config.fraud_rate_max):
            report.add_warning(
                f"Fraud rate {fraud_rate:.3%} is outside expected range "
                f"[{self.config.fraud_rate_min:.3%}, {self.config.fraud_rate_max:.3%}]. "
                "Possible concept drift."
            )
        else:
            logger.info("Class distribution OK: fraud rate = %.3f%%", fraud_rate * 100)

    def _check_amount_outliers(self, df: pd.DataFrame, report: QualityReport) -> None:
        if AMOUNT_COL not in df.columns:
            return
        max_val = df[AMOUNT_COL].max()
        if max_val > self.config.max_amount:
            report.add_warning(
                f"Amount column has extreme value {max_val:,.2f} "
                f"(cap: {self.config.max_amount:,.0f})."
            )

    def _check_negative_values(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Amount and Time must be non-negative."""
        for col in [AMOUNT_COL, TIME_COL]:
            if col not in df.columns:
                continue
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                report.add_violation(f"Column '{col}' contains {neg_count} negative value(s).")


def run_quality_gate(data_path: str | Path) -> QualityReport:
    """Convenience function: load data and run quality gate."""
    df = load_raw_data(data_path)
    gate = DataQualityGate()
    return gate.run(df)


# ---------------------------------------------------------------------------
# CLI entry point (called directly by GitLab CI step CI-3)
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI-3 Data Quality Gate")
    parser.add_argument(
        "--data",
        default="data/raw/creditcard.csv",
        help="Path to the raw CSV file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    report = run_quality_gate(args.data)
    sys.exit(0 if report.passed else 1)
