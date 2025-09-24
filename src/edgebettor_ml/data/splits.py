from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def time_based_split(df: pd.DataFrame, test_season: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "season" not in df.columns:
        raise ValueError("DataFrame must contain 'season' column for time-based split")
    train = df[df["season"] <= test_season - 2].copy()
    val = df[df["season"] == test_season - 1].copy()
    test = df[df["season"] == test_season].copy()
    return train, val, test


