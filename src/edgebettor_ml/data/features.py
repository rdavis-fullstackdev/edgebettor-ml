from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROLLING_WINDOWS = (3, 5)


@dataclass
class FeatureBuildConfig:
    rolling_windows: Tuple[int, int] = ROLLING_WINDOWS
    include_market_inputs: bool = True


def _compute_team_rolling(stats: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    required = ["team", "season", "week"]
    for col in required:
        if col not in stats.columns:
            raise ValueError(f"Missing column in team stats: {col}")
    stats = stats.sort_values(["team", "season", "week"]).copy()
    numeric_cols = [c for c in stats.columns if c not in required]
    grouped = stats.groupby("team", as_index=False, group_keys=False)
    frames = [stats[["team", "season", "week"]].copy()]
    for w in windows:
        rolled = grouped[numeric_cols].rolling(window=w, min_periods=1).mean().reset_index(drop=True)
        rolled.columns = [f"{c}_roll{w}" for c in rolled.columns]
        frames.append(rolled)
    out = pd.concat(frames, axis=1)
    return out


def _merge_game_rows(
    schedule: pd.DataFrame,
    team_roll: pd.DataFrame,
) -> pd.DataFrame:
    req = ["season", "week", "game_id", "home_team", "away_team"]
    for col in req:
        if col not in schedule.columns:
            raise ValueError(f"Missing column in schedule: {col}
")
    # Merge home and away rolling stats
    home = schedule.merge(
        team_roll.add_prefix("home_"),
        left_on=["season", "week", "home_team"],
        right_on=["home_season", "home_week", "home_team"],
        how="left",
    )
    both = home.merge(
        team_roll.add_prefix("away_"),
        left_on=["season", "week", "away_team"],
        right_on=["away_season", "away_week", "away_team"],
        how="left",
    )
    # Build matchup diffs for rolled numeric features
    rolled_cols_home = [c for c in both.columns if c.startswith("home_") and any(s in c for s in ["_roll3", "_roll5"])]
    for col_h in rolled_cols_home:
        base = col_h[len("home_") :]
        col_a = f"away_{base}"
        if col_a in both.columns:
            both[f"{col_h}_minus_{col_a}"] = both[col_h] - both[col_a]
    return both


def build_features(
    schedule: pd.DataFrame,
    team_weekly_stats: pd.DataFrame,
    config: Optional[FeatureBuildConfig] = None,
) -> pd.DataFrame:
    """Return per-game feature rows including targets when available.

    Expects schedule to include outcome fields when available:
      - home_score, away_score, closing_spread, closing_total
    Targets generated:
      - y_home_win, y_home_cover_vs_closing, y_over_total_vs_closing
    """
    cfg = config or FeatureBuildConfig()
    team_roll = _compute_team_rolling(team_weekly_stats, cfg.rolling_windows)
    games = _merge_game_rows(schedule, team_roll)

    # Market inputs passthrough
    if cfg.include_market_inputs:
        for col in [
            "closing_spread",
            "closing_total",
            "moneyline_home",
            "moneyline_away",
        ]:
            if col not in games.columns:
                games[col] = np.nan

    # Simple context features
    for col in ["rest_days_home", "rest_days_away", "home_field", "dome", "travel_miles_home", "travel_miles_away"]:
        if col not in games.columns:
            games[col] = np.nan

    # Targets
    if {"home_score", "away_score"}.issubset(games.columns):
        games["y_home_win"] = (games["home_score"] > games["away_score"]).astype(float)
    else:
        games["y_home_win"] = np.nan

    if {"closing_spread", "home_score", "away_score"}.issubset(games.columns):
        margin = games["home_score"] - games["away_score"]
        games["y_home_cover_vs_closing"] = (margin + games["closing_spread"]) > 0
        games["y_home_cover_vs_closing"] = games["y_home_cover_vs_closing"].astype(float)
    else:
        games["y_home_cover_vs_closing"] = np.nan

    if {"closing_total", "home_score", "away_score"}.issubset(games.columns):
        total_pts = games["home_score"] + games["away_score"]
        games["y_over_total_vs_closing"] = (total_pts > games["closing_total"]).astype(float)
    else:
        games["y_over_total_vs_closing"] = np.nan

    return games


