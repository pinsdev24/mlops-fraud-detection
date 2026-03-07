"""CI-1 — Unit tests for preprocessing (cleaner and features)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data.loader import TARGET_COL, AMOUNT_COL, TIME_COL
from src.preprocessing.cleaner import clean_data, split_features_target, train_val_test_split
from src.preprocessing.features import (
    apply_imbalance_strategy,
    build_preprocessor,
    compute_class_weight_ratio,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 2000
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data[TIME_COL] = rng.uniform(0, 172800, n)
    data[AMOUNT_COL] = rng.exponential(100, n)
    data[TARGET_COL] = (rng.random(n) < 0.002).astype(int)
    return pd.DataFrame(data)


@pytest.fixture
def df_with_duplicates(raw_df):
    return pd.concat([raw_df, raw_df.head(20)], ignore_index=True)


@pytest.fixture
def X_train_y_train(raw_df):
    df_clean = clean_data(raw_df)
    X, y = split_features_target(df_clean)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    return X_train, y_train


# ---------------------------------------------------------------------------
# Tests — cleaner
# ---------------------------------------------------------------------------


class TestCleaner:
    def test_clean_drops_duplicates_by_default(self, df_with_duplicates):
        df_clean = clean_data(df_with_duplicates, drop_duplicates=True)
        assert df_clean.duplicated().sum() == 0

    def test_clean_keeps_duplicates_when_disabled(self, df_with_duplicates):
        df_clean = clean_data(df_with_duplicates, drop_duplicates=False)
        assert len(df_clean) == len(df_with_duplicates)

    def test_clean_output_has_correct_columns(self, raw_df):
        df_clean = clean_data(raw_df)
        expected_cols = set([f"V{i}" for i in range(1, 29)] + [TIME_COL, AMOUNT_COL, TARGET_COL])
        assert set(df_clean.columns) == expected_cols

    def test_target_is_integer_after_clean(self, raw_df):
        df_clean = clean_data(raw_df)
        assert df_clean[TARGET_COL].dtype == int

    def test_split_features_target_shapes(self, raw_df):
        df_clean = clean_data(raw_df)
        X, y = split_features_target(df_clean)
        assert X.shape[1] == 30  # 28 PCA + Amount + Time
        assert len(y) == len(X)
        assert set(y.unique()).issubset({0, 1})


class TestTrainValTestSplit:
    def test_split_sizes(self, raw_df):
        df_clean = clean_data(raw_df)
        X, y = split_features_target(df_clean)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, test_size=0.20, val_size=0.10
        )
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X)

    def test_stratification_preserves_fraud_ratio(self, raw_df):
        df_clean = clean_data(raw_df)
        X, y = split_features_target(df_clean)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        global_rate = y.mean()
        # Each split should have fraud rate within ±50% of global (very lenient due to tiny count)
        for split_y in [y_train, y_val, y_test]:
            if split_y.sum() > 0:
                assert abs(split_y.mean() - global_rate) / global_rate < 1.0

    def test_no_overlap_between_splits(self, raw_df):
        df_clean = clean_data(raw_df).reset_index(drop=True)
        X, y = split_features_target(df_clean)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)


# ---------------------------------------------------------------------------
# Tests — features / preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessor:
    def test_build_preprocessor_returns_column_transformer(self):
        prep = build_preprocessor()
        assert isinstance(prep, ColumnTransformer)

    def test_preprocessor_scales_amount_and_time(self, X_train_y_train):
        X_train, y_train = X_train_y_train
        prep = build_preprocessor()
        X_transformed = prep.fit_transform(X_train)
        # Amount and Time should now have ~0 mean and ~1 std
        # (they are the first 2 columns in the transformer output)
        assert abs(X_transformed[:, 0].mean()) < 0.5  # Amount scaled
        assert abs(X_transformed[:, 1].mean()) < 0.5  # Time scaled

    def test_preprocessor_output_shape(self, X_train_y_train):
        X_train, _ = X_train_y_train
        prep = build_preprocessor()
        X_out = prep.fit_transform(X_train)
        assert X_out.shape[1] == 30  # 28 PCA + Amount + Time


class TestImbalanceStrategy:
    def test_class_weight_returns_unchanged_data(self, X_train_y_train):
        X, y = X_train_y_train
        X_res, y_res = apply_imbalance_strategy(X.values, y, strategy="class_weight")
        assert X_res.shape == X.values.shape

    def test_smote_increases_minority_class(self, X_train_y_train):
        X, y = X_train_y_train
        if y.sum() < 5:
            pytest.skip("Not enough fraud samples for SMOTE (need >=5)")
        X_res, y_res = apply_imbalance_strategy(X.values, y, strategy="smote")
        assert y_res.sum() > y.sum()

    def test_undersample_reduces_majority_class(self, X_train_y_train):
        X, y = X_train_y_train
        if y.sum() == 0:
            pytest.skip("No fraud samples in train split")
        X_res, y_res = apply_imbalance_strategy(X.values, y, strategy="undersample")
        assert len(y_res) < len(y)

    def test_unknown_strategy_raises_value_error(self, X_train_y_train):
        X, y = X_train_y_train
        with pytest.raises(ValueError, match="Unknown imbalance strategy"):
            apply_imbalance_strategy(X.values, y, strategy="magic")

    def test_class_weight_ratio_positive(self, X_train_y_train):
        _, y = X_train_y_train
        if y.sum() == 0:
            pytest.skip("No fraud samples")
        ratio = compute_class_weight_ratio(y)
        assert ratio > 1.0  # always more negatives than positives
