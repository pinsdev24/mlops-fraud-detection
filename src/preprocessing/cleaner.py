"""Data cleaning module.

Handles:
- Type coercion
- Duplicate removal
- Outlier capping on Amount
- Train / validation / test stratified split
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import TARGET_COL, ALL_FEATURES

logger = logging.getLogger(__name__)

# Cap for Amount (99.9th percentile threshold applied during training)
AMOUNT_CAP_QUANTILE = 0.999


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Apply minimal cleaning steps on the raw DataFrame.

    Steps:
      1. Drop exact duplicate rows (optional).
      2. Ensure correct dtypes.
      3. Keep only expected columns.

    Note: We do NOT cap/scale here — that is done inside the sklearn Pipeline
    to avoid data leakage between train and test splits.

    Args:
        df: Raw DataFrame from loader.load_raw_data().
        drop_duplicates: Remove exact duplicate rows.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Starting data cleaning — initial shape: %s", df.shape)
    original_len = len(df)

    if drop_duplicates:
        df = df.drop_duplicates()
        dropped = original_len - len(df)
        if dropped:
            logger.info("Dropped %d duplicate rows", dropped)

    # Keep only the known columns
    cols_to_keep = ALL_FEATURES + [TARGET_COL]
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()

    # Coerce dtypes
    for col in ALL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Drop rows where any feature is NaN (after coercion)
    before = len(df)
    df = df.dropna(subset=ALL_FEATURES)
    after = len(df)
    if before != after:
        logger.warning("Dropped %d rows with NaN feature values", before - after)

    logger.info("Cleaning done — final shape: %s", df.shape)
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X from target vector y."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified three-way split: train / validation / test.

    The stratify parameter preserves the extreme class imbalance (0.172%)
    in every split, ensuring each set sees fraud examples.

    Args:
        X: Feature matrix.
        y: Binary target (0 = normal, 1 = fraud).
        test_size: Fraction of total data reserved for the held-out test set.
        val_size: Fraction of total data reserved for the validation set.
        random_state: Reproducibility seed.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: carve out the test set
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: carve val from the remaining (train + val) data
    # Adjust val_size relative to the remaining pool
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=relative_val, stratify=y_tmp, random_state=random_state
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d | fraud in train: %d (%.3f%%)",
        len(X_train),
        len(X_val),
        len(X_test),
        y_train.sum(),
        y_train.mean() * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
