"""Feature engineering and class imbalance strategies.

Builds a scikit-learn Pipeline compatible preprocessor that:
- Scales Amount and Time (the only non-PCA features)
- Leaves V1–V28 untouched (already PCA-transformed)

Imbalance strategies (to be compared in experiments):
- 'class_weight' : pass class_weight='balanced' to classifiers (default)
- 'smote'        : SMOTE oversampling (train set only)
- 'undersample'  : RandomUnderSampler on majority class
"""

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.data.loader import AMOUNT_COL, PCA_FEATURES, TIME_COL

logger = logging.getLogger(__name__)

# Only Amount and Time need scaling; V1-V28 are already unit-variance from PCA
SCALE_COLS = [AMOUNT_COL, TIME_COL]
PASSTHROUGH_COLS = PCA_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that scales Amount & Time, passes V1-V28 through.

    IMPORTANT: This must be fit ONLY on the training data to prevent leakage.

    Returns:
        sklearn ColumnTransformer (unfitted).
    """
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), SCALE_COLS),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def apply_imbalance_strategy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "class_weight",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the chosen imbalance strategy to the training data.

    Args:
        X_train: Already-preprocessed training features (numpy or DataFrame).
        y_train: Training labels.
        strategy: One of 'class_weight', 'smote', 'undersample'.
            - 'class_weight': No resampling needed — returns data unchanged.
              The classifier itself handles weighting.
            - 'smote': SMOTE oversampling of the minority class.
            - 'undersample': Random undersampling of the majority class.
        random_state: Reproducibility seed.

    Returns:
        (X_resampled, y_resampled) as numpy arrays.
    """
    if isinstance(X_train, pd.DataFrame):
        X_arr = X_train.values
    else:
        X_arr = X_train
    y_arr = y_train.values if hasattr(y_train, "values") else y_train

    if strategy == "class_weight":
        logger.info("Imbalance strategy: class_weight (no resampling)")
        return X_arr, y_arr

    elif strategy == "smote":
        logger.info("Imbalance strategy: SMOTE — resampling training set...")
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_arr, y_arr)
        logger.info(
            "After SMOTE — shape: %s | fraud rate: %.2f%%",
            X_res.shape,
            y_res.mean() * 100,
        )
        return X_res, y_res

    elif strategy == "undersample":
        logger.info("Imbalance strategy: RandomUnderSampler")
        rus = RandomUnderSampler(sampling_strategy=0.1, random_state=random_state)
        X_res, y_res = rus.fit_resample(X_arr, y_arr)
        logger.info(
            "After undersampling — shape: %s | fraud rate: %.2f%%",
            X_res.shape,
            y_res.mean() * 100,
        )
        return X_res, y_res

    else:
        raise ValueError(
            f"Unknown imbalance strategy '{strategy}'. "
            "Choose from: 'class_weight', 'smote', 'undersample'."
        )


def compute_class_weight_ratio(y_train: pd.Series) -> float:
    """Return scale_pos_weight for XGBoost (ratio negative/positive samples)."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ratio = n_neg / n_pos
    logger.info("scale_pos_weight for XGBoost: %.1f (neg=%d, pos=%d)", ratio, n_neg, n_pos)
    return ratio
