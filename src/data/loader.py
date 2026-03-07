"""Data loader for the Credit Card Fraud Detection dataset."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Column definitions
TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES = PCA_FEATURES + [TIME_COL, AMOUNT_COL]

# Expected schema
EXPECTED_COLUMNS = ALL_FEATURES + [TARGET_COL]
EXPECTED_DTYPES = {col: float for col in ALL_FEATURES}
EXPECTED_DTYPES[TARGET_COL] = int


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw creditcard CSV into a DataFrame.

    Args:
        path: Path to creditcard.csv

    Returns:
        DataFrame with validated schema and correct dtypes.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the schema is invalid.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Dataset not found at %s. Attempting to download...", path)
        import subprocess

        project_root = Path(__file__).resolve().parent.parent.parent
        script_path = project_root / "download_data.sh"

        if script_path.exists():
            try:
                subprocess.run(["bash", str(script_path)], cwd=project_root, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Error downloading dataset: {e}")
        else:
            raise FileNotFoundError(f"Dataset not found and {script_path} is missing.")

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found even after running download_data.sh: {path}"
            )

    logger.info("Loading dataset from %s ...", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", len(df), df.shape[1])

    _validate_schema(df)

    # Ensure target is integer
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Check that all expected columns are present."""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        logger.warning("Unexpected extra columns (will be ignored): %s", extra_cols)


def get_feature_names() -> list[str]:
    """Return the list of feature column names used for modelling."""
    return ALL_FEATURES


def get_target_name() -> str:
    """Return the target column name."""
    return TARGET_COL


def get_pca_feature_names() -> list[str]:
    """Return only the PCA-transformed features V1–V28."""
    return PCA_FEATURES
