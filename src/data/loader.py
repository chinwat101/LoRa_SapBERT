"""
src/data/loader.py
Load and do initial validation of Lexitron CSV files.
"""
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# BOM prefix that Python csv adds to first column name when reading UTF-8-BOM files
_BOM_PREFIX = "\ufeff"


def _strip_bom(columns: list[str]) -> list[str]:
    """Remove BOM character from column names."""
    return [c.lstrip(_BOM_PREFIX) for c in columns]


def load_lexitron(
    csv_path: Path,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a Lexitron CSV file and return a cleaned DataFrame.

    Args:
        csv_path: Path to the CSV file.
        encoding: File encoding (default utf-8; handles utf-8-sig automatically).

    Returns:
        DataFrame with BOM-stripped column names.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If the file has no rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Loading: {csv_path}  (encoding={encoding})")

    # Use utf-8-sig to automatically strip BOM
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = _strip_bom(list(df.columns))

    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def validate_thai_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """
    Check which required columns are present in the DataFrame.

    Args:
        df: Source DataFrame.
        required: List of expected column names.

    Returns:
        List of found column names (missing columns are logged as warnings).
    """
    found = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]

    if missing:
        logger.warning(f"Missing columns: {missing}")
    logger.info(f"Using columns: {found}")

    return found
