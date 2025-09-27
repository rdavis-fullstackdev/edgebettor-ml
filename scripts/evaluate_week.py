from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from edgebettor_ml.modeling.evaluate import brier_score, log_loss


def evaluate_week(season: int, week: int, preds_json_path: Path) -> dict:
    preds = pd.read_json(preds_json_path)
    # Load realized outcomes from schedules
    sched_path = Path(".data/raw/schedules.csv")
    if not sched_path.exists():
        raise SystemExit("Missing schedules.csv; run data acquisition")
    sched = pd.read_csv(sched_path)
    cols = ["season", "week", "game_id", "home_team", "away_team", "home_score", "away_score"]
    for c in cols:
        if c not in sched.columns:
            sched[c] = np.nan
    week_sched = sched.loc[(sched["season"] == season) & (sched["week"] == week), cols].copy()

    df = preds.merge(week_sched, on=["game_id", "home_team", "away_team"], how="inner")
    # Drop rows without final scores
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        return {"error": "No completed games found for evaluation."}

    # Home win head
    y_true_win = (df["home_score"] > df["away_score"]).astype(int).to_numpy()
    p_win = df["p_home_win"].clip(0, 1).to_numpy()

    metrics = {
        "games": int(df.shape[0]),
        "home_win_brier": brier_score(y_true_win, p_win),
        "home_win_logloss": log_loss(y_true_win, p_win),
    }

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--preds-json", type=str, default=None)
    args = ap.parse_args()

    preds_json = Path(args.preds_json) if args.preds_json else Path("outputs") / f"preds_{args.season}_wk{args.week}.json"
    if not preds_json.exists():
        raise SystemExit(f"Predictions not found: {preds_json}")

    metrics = evaluate_week(args.season, args.week, preds_json)

    out_dir = Path("reports") / f"{args.season}_{args.week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


