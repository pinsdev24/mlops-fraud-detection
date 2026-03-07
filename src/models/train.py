"""CI-4 — Training pipeline with full MLflow tracking.

Usage:
    # From project root with activated venv:
    python -m src.models.train --config configs/train_config.yaml

    # Or inside Docker (MLflow server at http://mlflow:5000):
    MLFLOW_TRACKING_URI=http://mlflow:5000 python -m src.models.train

The script:
1. Loads and validates data (CI-3 quality gate)
2. Cleans and splits the dataset (stratified)
3. Builds a sklearn preprocessing + model pipeline
4. Handles class imbalance with the chosen strategy
5. Optimises the decision threshold on the validation set
6. Logs everything to MLflow: params, metrics, artefacts, model
7. Compares against the previous champion (CI-5)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.loader import get_feature_names
from src.data.quality_gate import run_quality_gate
from src.evaluation.metrics import (
    classification_report_str,
    compute_all_metrics,
    find_optimal_threshold,
)
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
    plot_shap_summary,
    plot_threshold_analysis,
)
from src.models.compare import should_promote, get_champion_run
from src.preprocessing.cleaner import (
    clean_data,
    split_features_target,
    train_val_test_split,
)
from src.preprocessing.features import (
    apply_imbalance_strategy,
    build_preprocessor,
    compute_class_weight_ratio,
)

from src.data.loader import load_raw_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", path)
    return cfg


def _deep_get(d: dict, *keys, default=None):
    """Safe nested dict access."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(
    model_name: str,
    hyperparams: dict,
    imbalance_strategy: str,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
):
    """Instantiate a classifier based on config.

    Args:
        model_name: 'logistic_regression' | 'random_forest' | 'xgboost'
        hyperparams: Dict of kwargs passed to the classifier constructor.
        imbalance_strategy: Used to set class_weight when relevant.
        scale_pos_weight: XGBoost-specific imbalance weight.
        random_state: Reproducibility seed.

    Returns:
        Fitted-ready sklearn/xgboost estimator.
    """
    class_weight = "balanced" if imbalance_strategy == "class_weight" else None

    if model_name == "logistic_regression":
        params = {
            "C": hyperparams.get("C", 1.0),
            "max_iter": hyperparams.get("max_iter", 1000),
            "solver": hyperparams.get("solver", "lbfgs"),
            "class_weight": class_weight,
            "random_state": random_state,
        }
        clf = LogisticRegression(**params)

    elif model_name == "random_forest":
        params = {
            "n_estimators": hyperparams.get("n_estimators", 200),
            "max_depth": hyperparams.get("max_depth", None),
            "min_samples_leaf": hyperparams.get("min_samples_leaf", 4),
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**params)

    elif model_name == "xgboost":
        # For XGBoost use scale_pos_weight instead of class_weight
        params = {
            "n_estimators": hyperparams.get("n_estimators", 300),
            "max_depth": hyperparams.get("max_depth", 6),
            "learning_rate": hyperparams.get("learning_rate", 0.05),
            "subsample": hyperparams.get("subsample", 0.8),
            "colsample_bytree": hyperparams.get("colsample_bytree", 0.8),
            "scale_pos_weight": (scale_pos_weight if imbalance_strategy == "class_weight" else 1.0),
            "eval_metric": "aucpr",
            "random_state": random_state,
            "n_jobs": -1,
        }
        clf = XGBClassifier(**params)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose: logistic_regression | random_forest | xgboost"
        )

    logger.info("Built model: %s | strategy: %s", model_name, imbalance_strategy)
    return clf, params


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------


def train_pipeline(config: dict) -> str:
    """Run the end-to-end training pipeline and return the MLflow run_id.

    Args:
        config: Parsed YAML config dict.

    Returns:
        MLflow run_id string.
    """
    # -----------------------------------------------------------------------
    # 0. Setup MLflow
    # -----------------------------------------------------------------------
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        _deep_get(config, "mlflow", "tracking_uri", default="http://localhost:5000"),
    )
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = _deep_get(config, "mlflow", "experiment_name", default="fraud-detection-v1")
    mlflow.set_experiment(exp_name)

    random_seed = _deep_get(config, "data", "random_seed", default=42)
    np.random.seed(random_seed)

    model_name = _deep_get(config, "model", "name", default="xgboost")
    imbalance_strategy = _deep_get(
        config, "preprocessing", "imbalance_strategy", default="class_weight"
    )
    threshold_strategy = _deep_get(config, "threshold", "optimization_strategy", default="f1")
    artifacts_dir = Path(_deep_get(config, "artifacts", "output_dir", default="artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load & validate data (CI-3)
    # -----------------------------------------------------------------------
    raw_path = _deep_get(config, "data", "raw_path", default="data/raw/creditcard.csv")
    logger.info("=== Step 1: Data Quality Gate (CI-3) ===")
    report = run_quality_gate(raw_path)
    if not report.passed:
        logger.error("Quality gate FAILED — aborting training. Violations:\n%s", report.violations)
        sys.exit(1)

    df = load_raw_data(raw_path)

    # -----------------------------------------------------------------------
    # 2. Clean & split
    # -----------------------------------------------------------------------
    logger.info("=== Step 2: Cleaning & Splitting ===")
    df = clean_data(df)
    X, y = split_features_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X,
        y,
        test_size=_deep_get(config, "data", "test_size", default=0.20),
        val_size=_deep_get(config, "data", "val_size", default=0.10),
        random_state=random_seed,
    )

    # -----------------------------------------------------------------------
    # 3. Preprocessing — fit ONLY on train, transform all splits
    # -----------------------------------------------------------------------
    logger.info("=== Step 3: Preprocessing (no data leakage) ===")
    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    scale_pos_weight = compute_class_weight_ratio(y_train)

    # Apply resampling only if strategy is smote/undersample
    X_train_final, y_train_final = apply_imbalance_strategy(
        X_train_scaled, y_train, strategy=imbalance_strategy, random_state=random_seed
    )

    # -----------------------------------------------------------------------
    # 4. Build model
    # -----------------------------------------------------------------------
    logger.info("=== Step 4: Building model — %s ===", model_name)
    hyperparams = _deep_get(config, "model", "hyperparams", default={}) or {}
    clf, clf_params = build_model(
        model_name,
        hyperparams,
        imbalance_strategy,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
    )

    # -----------------------------------------------------------------------
    # 5. Train & evaluate with full MLflow tracking
    # -----------------------------------------------------------------------
    logger.info("=== Step 5: Training & MLflow Tracking ===")
    # Build run name from config for readability in MLflow UI
    run_name = f"{model_name}_{imbalance_strategy}_seed{random_seed}"
    tags = _deep_get(config, "mlflow", "tags", default={}) or {}
    tags["model"] = model_name
    tags["imbalance_strategy"] = imbalance_strategy
    tags["threshold_strategy"] = threshold_strategy

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

        # --- Log all config params ---
        mlflow.log_params(
            {
                "model_name": model_name,
                "imbalance_strategy": imbalance_strategy,
                "threshold_strategy": threshold_strategy,
                "random_seed": random_seed,
                "test_size": _deep_get(config, "data", "test_size", default=0.20),
                "val_size": _deep_get(config, "data", "val_size", default=0.10),
                "train_samples": len(X_train_final),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "fraud_rate_train": float(y_train.mean()),
                "n_features": len(get_feature_names()),
            }
        )
        mlflow.log_params({f"clf_{k}": v for k, v in clf_params.items()})

        # --- Train ---
        logger.info("Fitting %s on %d samples ...", model_name, len(X_train_final))
        clf.fit(X_train_final, y_train_final)

        # --- Validate: find optimal threshold on val set ---
        y_val_proba = clf.predict_proba(X_val_scaled)[:, 1]
        optimal_threshold = find_optimal_threshold(
            y_val.values, y_val_proba, strategy=threshold_strategy
        )
        mlflow.log_param("optimal_threshold", optimal_threshold)

        # --- Final evaluation on HELD-OUT test set ---
        y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

        test_metrics = compute_all_metrics(y_test.values, y_test_proba, threshold=optimal_threshold)
        mlflow.log_metrics(test_metrics)

        # Also log val metrics to detect overfitting
        val_metrics = compute_all_metrics(y_val.values, y_val_proba, threshold=optimal_threshold)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # --- Save classification report as artefact ---
        report_txt = classification_report_str(y_test.values, y_test_pred)
        logger.info("\n%s", report_txt)
        report_path = artifacts_dir / "classification_report.txt"
        report_path.write_text(report_txt)
        mlflow.log_artifact(str(report_path))

        # --- Generate and log all figures ---
        logger.info("Generating evaluation figures ...")
        feature_names = get_feature_names()

        plot_paths = []

        cm_path = plot_confusion_matrix(
            y_test.values, y_test_pred, output_path=artifacts_dir / "confusion_matrix.png"
        )
        plot_paths.append(cm_path)

        pr_path = plot_pr_curve(
            y_test.values,
            y_test_proba,
            pr_auc=test_metrics["pr_auc"],
            output_path=artifacts_dir / "pr_curve.png",
        )
        plot_paths.append(pr_path)

        roc_path = plot_roc_curve(
            y_test.values,
            y_test_proba,
            roc_auc=test_metrics["roc_auc"],
            output_path=artifacts_dir / "roc_curve.png",
        )
        plot_paths.append(roc_path)

        # Threshold analysis
        precisions_arr, recalls_arr, thresholds_arr = precision_recall_curve(
            y_test.values, y_test_proba
        )
        thresh_path = plot_threshold_analysis(
            precisions_arr,
            recalls_arr,
            thresholds_arr,
            optimal_threshold,
            output_path=artifacts_dir / "threshold_analysis.png",
        )
        plot_paths.append(thresh_path)

        # Feature importance (tree-based models only)
        if hasattr(clf, "feature_importances_"):
            fi_path = plot_feature_importance(
                clf.feature_importances_,
                feature_names,
                output_path=artifacts_dir / "feature_importance.png",
            )
            plot_paths.append(fi_path)

        # SHAP summary (optional, controlled by config)
        save_shap = _deep_get(config, "artifacts", "save_shap", default=True)
        if save_shap:
            try:
                import shap

                shap_samples = _deep_get(config, "artifacts", "shap_max_samples", default=1000)
                X_shap = X_test_scaled[:shap_samples]
                if model_name == "xgboost":
                    explainer = shap.TreeExplainer(clf)
                else:
                    explainer = shap.Explainer(clf, X_train_scaled[:500])
                shap_vals = explainer.shap_values(X_shap)
                shap_path = plot_shap_summary(
                    shap_vals, X_shap, feature_names, output_path=artifacts_dir / "shap_summary.png"
                )
                if shap_path:
                    plot_paths.append(shap_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SHAP failed (skipping): %s", exc)

        # Upload all figures to MLflow
        for p in plot_paths:
            if p:
                mlflow.log_artifact(p)

        # --- Log the trained model (MLflow Model format) ---
        # Build a full sklearn Pipeline (preprocessor + classifier) for inference
        inference_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
        mlflow.sklearn.log_model(
            sk_model=inference_pipeline,
            artifact_path="model",
            registered_model_name=f"fraud-detector-{model_name}",
            input_example=X_test.head(5),
        )
        logger.info("Model registered: fraud-detector-%s | run_id: %s", model_name, run_id)

        # --- CI-5: Compare against champion ---
        promotion_cfg = _deep_get(config, "promotion", default={}) or {}
        promo_metric = promotion_cfg.get("metric", "pr_auc")
        promo_tol = promotion_cfg.get("tolerance", 0.01)

        champion_run = get_champion_run(exp_name, promo_metric)
        if champion_run and champion_run.info.run_id != run_id:
            promoted = should_promote(
                new_run_id=run_id,
                reference_run_id=champion_run.info.run_id,
                metric=promo_metric,
                tolerance=promo_tol,
            )
            mlflow.set_tag("promotion_status", "champion" if promoted else "rejected")
            if promoted:
                logger.info("✅ New model PROMOTED as champion (CI-5).")
            else:
                logger.warning(
                    "❌ Model NOT promoted — below champion by >%.1f%% on %s.",
                    promo_tol * 100,
                    promo_metric,
                )
        else:
            mlflow.set_tag("promotion_status", "first_run")
            logger.info("First run — automatically set as champion reference.")

        # Final summary
        logger.info(
            "=== Training complete — PR-AUC: %.4f | ROC-AUC: %.4f | F1: %.4f ===",
            test_metrics["pr_auc"],
            test_metrics["roc_auc"],
            test_metrics["f1_fraud"],
        )

    return run_id


# ---------------------------------------------------------------------------
# CLI entry point (CI-4 — triggered by GitLab pipeline)
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CI-4 — Automated model training with MLflow tracking"
    )
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to YAML training config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    run_id = train_pipeline(cfg)
    print(f"\nMLflow run_id: {run_id}")
