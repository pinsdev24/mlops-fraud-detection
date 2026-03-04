"""CI-1 — Unit tests for model building, training and CI-5 compare.

These tests use lightweight synthetic data and a local MLflow file-based
tracking store to avoid any network dependency.
"""

import numpy as np
import pytest
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.train import build_model, load_config
from src.models.compare import should_promote, get_champion_run, compare_all_runs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_Xy():
    """Small synthetic dataset (1000 rows, 30 features, ~0.2% fraud)."""
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.standard_normal((n, 30))
    y = (rng.random(n) < 0.05).astype(int)   # 5% fraud to ensure enough positives
    return X, y


@pytest.fixture
def mlflow_tmp(tmp_path):
    """Temporary local MLflow tracking URI (no server required for tests)."""
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri
    mlflow.set_tracking_uri("http://localhost:5000")   # reset to default


# ---------------------------------------------------------------------------
# Tests — build_model
# ---------------------------------------------------------------------------

class TestBuildModel:
    @pytest.mark.parametrize("model_name,clf_class", [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        ("xgboost", XGBClassifier),
    ])
    def test_returns_correct_classifier_type(self, model_name, clf_class):
        clf, params = build_model(
            model_name=model_name,
            hyperparams={},
            imbalance_strategy="class_weight",
        )
        assert isinstance(clf, clf_class)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("unicorn", {}, "class_weight")

    def test_logistic_regression_has_balanced_weight(self):
        clf, _ = build_model("logistic_regression", {}, "class_weight")
        assert clf.class_weight == "balanced"

    def test_xgboost_scale_pos_weight_set(self):
        clf, _ = build_model("xgboost", {}, "class_weight", scale_pos_weight=100.0)
        assert clf.scale_pos_weight == 100.0

    def test_xgboost_no_scale_pos_weight_for_smote(self):
        clf, _ = build_model("xgboost", {}, "smote", scale_pos_weight=100.0)
        # with SMOTE strategy, scale_pos_weight should be 1.0
        assert clf.scale_pos_weight == 1.0

    def test_params_dict_returned(self):
        _, params = build_model("random_forest", {"n_estimators": 10}, "class_weight")
        assert "n_estimators" in params
        assert params["n_estimators"] == 10

    def test_random_state_is_set(self):
        clf, params = build_model(
            "logistic_regression", {}, "class_weight", random_state=99
        )
        assert clf.random_state == 99

    def test_model_fits_on_synthetic_data(self, sample_Xy):
        X, y = sample_Xy
        clf, _ = build_model("xgboost", {"n_estimators": 5, "max_depth": 2}, "class_weight")
        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1]
        assert proba.shape == (len(y),)
        assert 0.0 <= proba.min() <= proba.max() <= 1.0


# ---------------------------------------------------------------------------
# Tests — CI-5 comparison (compare.py)
# ---------------------------------------------------------------------------

class TestModelComparison:
    """Test should_promote using a local MLflow tracking store."""

    def _log_dummy_run(self, exp_name: str, pr_auc: float, mlflow_uri: str) -> str:
        """Log a minimal run returning its run_id."""
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(exp_name)
        with mlflow.start_run() as run:
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("roc_auc", pr_auc + 0.05)
            mlflow.log_param("model_name", "xgboost")
            mlflow.log_param("imbalance_strategy", "class_weight")
            return run.info.run_id

    def test_promote_when_new_is_better(self, mlflow_tmp):
        exp = "test-compare-better"
        ref_id = self._log_dummy_run(exp, 0.80, mlflow_tmp)
        new_id = self._log_dummy_run(exp, 0.85, mlflow_tmp)
        assert should_promote(new_id, ref_id, metric="pr_auc", tolerance=0.01)

    def test_promote_when_within_tolerance(self, mlflow_tmp):
        exp = "test-compare-within"
        ref_id = self._log_dummy_run(exp, 0.80, mlflow_tmp)
        # 0.795 is within 1% of 0.80
        new_id = self._log_dummy_run(exp, 0.795, mlflow_tmp)
        assert should_promote(new_id, ref_id, metric="pr_auc", tolerance=0.01)

    def test_reject_when_below_tolerance(self, mlflow_tmp):
        exp = "test-compare-reject"
        ref_id = self._log_dummy_run(exp, 0.80, mlflow_tmp)
        # 0.78 is 2.5% below 0.80 — exceeds tolerance of 1%
        new_id = self._log_dummy_run(exp, 0.78, mlflow_tmp)
        assert not should_promote(new_id, ref_id, metric="pr_auc", tolerance=0.01)

    def test_get_champion_returns_best_run(self, mlflow_tmp):
        exp = "test-champion"
        self._log_dummy_run(exp, 0.70, mlflow_tmp)
        self._log_dummy_run(exp, 0.85, mlflow_tmp)
        self._log_dummy_run(exp, 0.60, mlflow_tmp)
        champion = get_champion_run(exp, metric="pr_auc")
        assert champion is not None
        assert abs(champion.data.metrics["pr_auc"] - 0.85) < 1e-6

    def test_get_champion_returns_none_for_empty_experiment(self, mlflow_tmp):
        mlflow.set_experiment("totally-empty-experiment")
        champion = get_champion_run("totally-empty-experiment", "pr_auc")
        assert champion is None

    def test_leaderboard_returns_sorted_list(self, mlflow_tmp):
        exp = "test-leaderboard"
        self._log_dummy_run(exp, 0.75, mlflow_tmp)
        self._log_dummy_run(exp, 0.82, mlflow_tmp)
        self._log_dummy_run(exp, 0.68, mlflow_tmp)
        board = compare_all_runs(exp, metric="pr_auc")
        assert len(board) == 3
        # Should be sorted descending by pr_auc
        values = [r["pr_auc"] for r in board]
        assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# Tests — load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model:\n  name: xgboost\ndata:\n  random_seed: 42\n")
        cfg = load_config(cfg_file)
        assert cfg["model"]["name"] == "xgboost"
        assert cfg["data"]["random_seed"] == 42

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")
