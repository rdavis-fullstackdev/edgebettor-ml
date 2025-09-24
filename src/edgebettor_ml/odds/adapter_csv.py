from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .adapter_base import OddsAdapter, ensure_standard_columns


class CsvOddsAdapter(OddsAdapter):
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def fetch_odds(self, season: int, week: int) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        df = pd.read_csv(self.csv_path)
        df_std = ensure_standard_columns(df)
        return df_std


