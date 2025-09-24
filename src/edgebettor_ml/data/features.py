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
    candidate_cols = [c for c in stats.columns if c not in required]
    for c in candidate_cols:
        stats[c] = pd.to_numeric(stats[c], errors="coerce")
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(stats[c])]
    grouped = stats.groupby("team", as_index=False, group_keys=False)
    frames = [stats[["team", "season", "week"]].copy()]
    for w in windows:
        rolled = grouped[numeric_cols].rolling(window=w, min_periods=1).mean().reset_index(drop=True)
        rolled.columns = [f"{c}_roll{w}" for c in rolled.columns]
        frames.append(rolled)
    out = pd.concat(frames, axis=1)
    return out


def _ensure_margin_column(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy()
    if "margin" not in weekly.columns:
        if {"points_for", "points_against"}.issubset(weekly.columns):
            weekly["points_for"] = pd.to_numeric(weekly["points_for"], errors="coerce")
            weekly["points_against"] = pd.to_numeric(weekly["points_against"], errors="coerce")
            weekly["margin"] = weekly["points_for"] - weekly["points_against"]
        else:
            weekly["margin"] = np.nan
    return weekly


def _add_opponent_margin(schedule: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    sched_cols = ["season", "week", "home_team", "away_team"]
    for c in sched_cols:
        if c not in schedule.columns:
            schedule[c] = np.nan
    base = schedule[sched_cols].copy()
    home_side = base[["season", "week", "home_team", "away_team"]].copy()
    home_side.rename(columns={"home_team": "team", "away_team": "opp"}, inplace=True)
    away_side = base[["season", "week", "home_team", "away_team"]].copy()
    away_side.rename(columns={"away_team": "team", "home_team": "opp"}, inplace=True)
    pairs = pd.concat([home_side, away_side], axis=0, ignore_index=True)

    weekly2 = _ensure_margin_column(weekly)
    opp_margin = weekly2[["season", "week", "team", "margin"]].copy()
    opp_margin.rename(columns={"team": "opp", "margin": "opponent_margin"}, inplace=True)
    annotated = pairs.merge(opp_margin, on=["season", "week", "opp"], how="left")
    annotated = annotated[["season", "week", "team", "opponent_margin"]]
    out = weekly2.merge(annotated, on=["season", "week", "team"], how="left")
    return out


def _first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _augment_weekly_with_derived(schedule: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    # Start with opponent margin
    w = _add_opponent_margin(schedule, weekly).copy()

    # Third-down offense: converted / attempts
    td_conv_col = _first_existing(w, [
        "third_down_converted", "third_down_successes", "third_down_conversions",
    ])
    td_att_col = _first_existing(w, [
        "third_down_attempts", "third_down_tries", "third_down_opps",
    ])
    if td_conv_col and td_att_col:
        w["third_down_off"] = pd.to_numeric(w[td_conv_col], errors="coerce") / pd.to_numeric(w[td_att_col], errors="coerce")
    else:
        w["third_down_off"] = np.nan

    # Turnovers offense: giveaways/turnovers
    to_off_col = _first_existing(w, ["giveaways", "turnovers", "turnovers_offense"])  # positive for offense committing TOs
    if to_off_col:
        w["turnovers_off"] = pd.to_numeric(w[to_off_col], errors="coerce")
    else:
        w["turnovers_off"] = np.nan

    # Red zone efficiency offense: TDs / trips
    rz_td_col = _first_existing(w, ["red_zone_td", "red_zone_tds", "rzt_td"])  # touchdowns
    rz_trips_col = _first_existing(w, ["red_zone_trips", "rzt_trips", "red_zone_attempts"])
    if rz_td_col and rz_trips_col:
        w["redzone_off"] = pd.to_numeric(w[rz_td_col], errors="coerce") / pd.to_numeric(w[rz_trips_col], errors="coerce")
    else:
        w["redzone_off"] = np.nan

    # Build defense metrics from opponent's offensive metrics by pairing
    cols_needed = ["season", "week", "team", "third_down_off", "turnovers_off", "redzone_off"]
    for c in ["season", "week", "team"]:
        if c not in w.columns:
            w[c] = np.nan
    off_metrics = w[cols_needed].copy()
    off_metrics.rename(columns={
        "team": "opp",
        "third_down_off": "third_down_def",
        "turnovers_off": "turnovers_def",
        "redzone_off": "redzone_def",
    }, inplace=True)

    # Use schedule mapping to attach opponent defensive metrics to team rows
    sched_base = schedule[["season", "week", "home_team", "away_team"]].copy()
    m1 = sched_base.rename(columns={"home_team": "team", "away_team": "opp"})
    m2 = sched_base.rename(columns={"away_team": "team", "home_team": "opp"})
    pairs = pd.concat([m1, m2], axis=0, ignore_index=True)

    w = w.merge(pairs[["season", "week", "team", "opp"]], on=["season", "week", "team"], how="left")
    w = w.merge(off_metrics, on=["season", "week", "opp"], how="left")
    w.drop(columns=["opp"], inplace=True)

    # Simple SOS proxy: opponent margin (already present); add alias columns for clarity
    w["sos"] = pd.to_numeric(w["opponent_margin"], errors="coerce")

    return w


def _merge_game_rows(
    schedule: pd.DataFrame,
    team_roll: pd.DataFrame,
) -> pd.DataFrame:
    req = ["season", "week", "game_id", "home_team", "away_team"]
    for col in req:
        if col not in schedule.columns:
            raise ValueError(f"Missing column in schedule: {col}")
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
            # ensure numeric before subtraction
            both[col_h] = pd.to_numeric(both[col_h], errors="coerce")
            both[col_a] = pd.to_numeric(both[col_a], errors="coerce")
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
    weekly_aug = _augment_weekly_with_derived(schedule, team_weekly_stats)
    team_roll = _compute_team_rolling(weekly_aug, cfg.rolling_windows)
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


