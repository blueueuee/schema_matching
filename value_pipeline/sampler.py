"""
Stage 1 (Value half): Value sampling module.
Pulls representative non-null samples from column values or CSV sources.
"""

from typing import Any, List, Optional
import random
import numpy as np
import pandas as pd


def _is_null(val: Any) -> bool:
    """Check if a value is considered null or missing."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip().lower() in {"", "null", "none", "nan", "na", "<na>"}:
        return True
    return False


def sample_column(values: list, n: int = 500, random_state: Optional[int] = 42) -> list:
    """
    Given a list of raw column values (may contain None/NaN),
    return up to n non-null values sampled without replacement.
    If len(non_null) <= n, return all non-null values.
    """
    if values is None:
        return []

    non_null = [v for v in values if not _is_null(v)]
    if len(non_null) <= n:
        return list(non_null)

    rng = random.Random(random_state) if random_state is not None else random
    return rng.sample(non_null, n)


def load_csv_column(filepath: str, column_name: str, n: int = 500, random_state: Optional[int] = 42) -> list:
    """
    Load a column from a CSV file and return a sample of non-null values.
    """
    df = pd.read_csv(filepath, usecols=[column_name])
    raw_values = df[column_name].tolist()
    return sample_column(raw_values, n=n, random_state=random_state)
