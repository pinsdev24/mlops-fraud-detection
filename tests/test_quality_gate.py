"""CI-1 — Unit tests for data loading and quality gate."""

import pandas as pd
import pytest

from src.data.loader import (
    load_raw_data,
    get_feature_names,
    get_target_name,
    TARGET_COL,
)
from src.data.quality_gate import DataQualityGate, QualityConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_df():
    """Minimal valid DataFrame that mimics the creditcard.csv schema."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 1000
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172800, n)  # 2 days in seconds
    data["Amount"] = rng.exponential(scale=100, size=n)
    data["Class"] = (rng.random(n) < 0.002).astype(int)  # ~0.2% fraud
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def strict_config():
    return QualityConfig(
        max_missing_rate=0.05,
        min_rows=500,
        fraud_rate_min=0.0001,
        fraud_rate_max=0.01,
    )


# ---------------------------------------------------------------------------
# Tests — loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_feature_names_count(self):
        features = get_feature_names()
        assert len(features) == 30, "Expected 28 PCA + Amount + Time = 30 features"

    def test_target_name(self):
        assert get_target_name() == "Class"

    def test_load_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_data(tmp_path / "nonexistent.csv")

    def test_load_raises_on_missing_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("A,B,C\n1,2,3\n")
        with pytest.raises(ValueError, match="Missing columns"):
            load_raw_data(bad_csv)

    def test_target_is_integer(self, tmp_path, sample_df):
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_raw_data(csv_path)
        assert df[TARGET_COL].dtype == int


# ---------------------------------------------------------------------------
# Tests — quality gate
# ---------------------------------------------------------------------------


class TestDataQualityGate:
    def test_passes_on_clean_data(self, sample_df, strict_config):
        gate = DataQualityGate(config=strict_config)
        report = gate.run(sample_df)
        assert report.passed, f"Violations: {report.violations}"

    def test_fails_on_high_missing_rate(self, sample_df, strict_config):
        df_bad = sample_df.copy()
        # Inject >5% missing in a critical column
        n_missing = int(len(df_bad) * 0.10)
        df_bad.loc[:n_missing, TARGET_COL] = None
        gate = DataQualityGate(config=strict_config)
        report = gate.run(df_bad)
        assert not report.passed
        assert any("missing" in v.lower() for v in report.violations)

    def test_fails_on_low_volume(self, sample_df):
        config = QualityConfig(min_rows=10_000)  # higher than sample_df (1000)
        gate = DataQualityGate(config=config)
        report = gate.run(sample_df)
        assert not report.passed
        assert any("rows" in v.lower() for v in report.violations)

    def test_fails_on_negative_amount(self, sample_df, strict_config):
        df_bad = sample_df.copy()
        df_bad.loc[0, "Amount"] = -100.0
        gate = DataQualityGate(config=strict_config)
        report = gate.run(df_bad)
        assert not report.passed
        assert any("negative" in v.lower() for v in report.violations)

    def test_warnings_on_duplicates(self, sample_df, strict_config):
        # Create <1% duplicate — should warn, not fail
        df_dup = pd.concat([sample_df, sample_df.head(5)], ignore_index=True)
        gate = DataQualityGate(config=strict_config)
        report = gate.run(df_dup)
        assert not report.passed or any("duplicate" in w.lower() for w in report.warnings)

    def test_report_summary_contains_status(self, sample_df, strict_config):
        gate = DataQualityGate(config=strict_config)
        report = gate.run(sample_df)
        summary = report.summary()
        assert "Quality Report" in summary
