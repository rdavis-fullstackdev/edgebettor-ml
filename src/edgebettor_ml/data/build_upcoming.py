from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureBuildConfig, build_features


RAW_DIR = Path(".data/raw")


def build_upcoming_features(season: int, week: int) -> pd.DataFrame:
    """Build features for upcoming games in a given season/week.

    Uses historical team weekly stats strictly before the target week.
    """
    schedules = pd.read_csv(RAW_DIR / "schedules.csv")
    weekly_path = RAW_DIR / "team_weekly.csv"
    if weekly_path.exists():
        weekly = pd.read_csv(weekly_path)
    else:
        # Fallback: derive minimal weekly from schedules
        hist_sched = schedules[(schedules["season"] < season) | ((schedules["season"] == season) & (schedules["week"] < week))]
        home = hist_sched[["season", "week", "home_team", "home_score", "away_score"]].copy()
        home.rename(columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against"}, inplace=True)
        away = hist_sched[["season", "week", "away_team", "away_score", "home_score"]].copy()
        away.rename(columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against"}, inplace=True)
        weekly = pd.concat([home, away], axis=0, ignore_index=True)

    # Filter schedule to target week
    cols_needed = ["season", "week", "game_id", "home_team", "away_team"]
    for c in cols_needed:
        if c not in schedules.columns:
            schedules[c] = np.nan
    week_sched = schedules.loc[(schedules["season"] == season) & (schedules["week"] == week), cols_needed].copy()

    # Restrict team stats to weeks before target week in the same season
    stats_mask = (weekly["season"] < season) | ((weekly["season"] == season) & (weekly["week"] < week))
    weekly_hist = weekly.loc[stats_mask].copy()

    # Shift weekly stats forward one week for inference so each team's rolling
    # features reflect information up to week-1 when predicting week
    feats = build_features(
        week_sched,
        weekly_hist,
        FeatureBuildConfig(include_market_inputs=True),
        shift_weeks_for_merge=1,
    )
    return feats


