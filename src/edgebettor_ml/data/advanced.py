from __future__ import annotations

import pandas as pd
import numpy as np


def _is_neutral(play: pd.Series) -> bool:
    try:
        sd = play.get("score_differential")
        qtr = play.get("qtr")
        return (pd.notna(sd) and -7 <= float(sd) <= 7) and (pd.notna(qtr) and int(qtr) <= 3)
    except Exception:
        return False


def compute_team_week_advanced(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-week advanced metrics from play-by-play.

    Returns columns:
      season, week, team,
      epa_off, epa_def,
      success_off, success_def,
      pass_rate_off,
      neutral_pace (plays per minute in neutral situations)
    """
    required = ["season", "week", "posteam", "defteam", "epa", "pass"]
    for c in required:
        if c not in pbp.columns:
            pbp[c] = np.nan

    # Offense aggregates
    off = pbp.groupby(["season", "week", "posteam"], dropna=False).agg(
        epa_off=("epa", "mean"),
        success_off=("epa", lambda s: float(np.mean(np.where(pd.to_numeric(s, errors="coerce") > 0, 1.0, 0.0)))) ,
        pass_rate_off=("pass", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce")))) ,
        plays_off=("epa", "count"),
    ).reset_index().rename(columns={"posteam": "team"})

    # Defense aggregates
    deff = pbp.groupby(["season", "week", "defteam"], dropna=False).agg(
        epa_def=("epa", "mean"),
        success_def=("epa", lambda s: float(np.mean(np.where(pd.to_numeric(s, errors="coerce") > 0, 1.0, 0.0)))) ,
        plays_def=("epa", "count"),
    ).reset_index().rename(columns={"defteam": "team"})

    # Neutral pace: estimate as plays per minute in neutral situations
    neutral_mask = pbp.apply(_is_neutral, axis=1)
    neutral = pbp.loc[neutral_mask]
    if "game_seconds_remaining" in neutral.columns:
        neutral = neutral.copy()
        # approximate elapsed time per play delta within game
        neutral["elapsed"] = neutral.groupby(["season", "week", "posteam"]).apply(
            lambda g: g["game_seconds_remaining"].shift(1) - g["game_seconds_remaining"]
        ).reset_index(level=[0,1,2], drop=True)
        neutral_agg = neutral.groupby(["season", "week", "posteam"]).agg(
            total_elapsed=("elapsed", lambda s: float(pd.to_numeric(s, errors="coerce").clip(lower=0).sum())),
            plays=("epa", "count"),
        ).reset_index()
        neutral_agg["neutral_pace"] = neutral_agg.apply(
            lambda r: (r["plays"] / (r["total_elapsed"] / 60.0)) if r["total_elapsed"] and r["total_elapsed"] > 0 else np.nan,
            axis=1,
        )
        neutral_agg = neutral_agg.rename(columns={"posteam": "team"})
        pace = neutral_agg[["season", "week", "team", "neutral_pace"]]
    else:
        pace = off[["season", "week", "team"]].copy()
        pace["neutral_pace"] = np.nan

    df = off.merge(deff, on=["season", "week", "team"], how="outer")
    df = df.merge(pace, on=["season", "week", "team"], how="left")

    # League z-scores by season-week for each metric
    for col in ["epa_off", "epa_def", "success_off", "success_def", "pass_rate_off", "neutral_pace"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        grp = df.groupby(["season", "week"])[col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        df[f"{col}_z"] = (df[col] - mean) / std

    return df


