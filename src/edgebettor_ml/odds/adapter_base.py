from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


STANDARD_COLUMNS = [
    "game_id",
    "home_team",
    "away_team",
    "moneyline_home",
    "moneyline_away",
    "spread_home",
    "spread_price_home",
    "spread_price_away",
    "total_points",
    "total_over_price",
    "total_under_price",
]


class OddsAdapter(ABC):
    @abstractmethod
    def fetch_odds(self, season: int, week: int) -> pd.DataFrame:
        raise NotImplementedError


def ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required odds columns: {missing}")
    # Keep required first, preserve any extra informative columns (e.g., commence_time, bookmaker)
    ordered = STANDARD_COLUMNS + [c for c in df.columns if c not in STANDARD_COLUMNS]
    return df[ordered].copy()


