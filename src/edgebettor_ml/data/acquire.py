from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

try:
    import nfl_data_py as nfl
except Exception:  # pragma: no cover - optional during early scaffolding
    nfl = None


RAW_DIR = Path(".data/raw")


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def season_bounds() -> Tuple[int, int]:
    now = datetime.now()
    # Include current season; downstream code should handle partial weeks
    return 2009, now.year


def pull_core(season_start: int, season_end: int) -> None:
    if nfl is None:
        raise RuntimeError("nfl_data_py is not available")
    seasons = list(range(season_start, season_end + 1))
    schedules = nfl.import_schedules(seasons)
    schedules.to_csv(RAW_DIR / "schedules.csv", index=False)

    # Betting lines API can differ by version; try a few options, else skip.
    betting = None
    try:
        if hasattr(nfl, "import_betting_lines"):
            betting = nfl.import_betting_lines(seasons)
        elif hasattr(nfl, "import_lines"):
            betting = nfl.import_lines(seasons)
    except Exception:
        betting = None
    if betting is not None and isinstance(betting, pd.DataFrame) and not betting.empty:
        betting.to_csv(RAW_DIR / "betting_lines.csv", index=False)

    # Team weekly stats API compatibility
    weekly = None
    try:
        if hasattr(nfl, "import_team_weekly"):
            weekly = nfl.import_team_weekly(seasons)
        elif hasattr(nfl, "import_weekly_team_stats"):
            weekly = nfl.import_weekly_team_stats(seasons)
    except Exception:
        weekly = None
    if weekly is not None and isinstance(weekly, pd.DataFrame) and not weekly.empty:
        weekly.to_csv(RAW_DIR / "team_weekly.csv", index=False)

    # play-by-play-derived advanced (EPA etc.)
    try:
        pbp = nfl.import_pbp_data(seasons)
        pbp.to_csv(RAW_DIR / "pbp.csv", index=False)
    except Exception:
        pass


def main() -> None:
    ensure_dirs()
    s0, s1 = season_bounds()
    pull_core(s0, s1)
    print(f"Wrote raw CSVs under {RAW_DIR}")


if __name__ == "__main__":
    main()


