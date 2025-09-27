from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from edgebettor_ml.odds.adapter_theoddsapi import TheOddsApiAdapter
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


DATE_CANDIDATE_COLS = [
    "gameday",
    "game_day",
    "gamedate",
    "game_date",
    "start_time",
    "start_time_utc",
    "game_time",
    "game_datetime",
    "datetime",
    "date",
]


def extract_unique_dates(df: pd.DataFrame) -> List[str]:
    for col in DATE_CANDIDATE_COLS:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            dates = sorted({d.date().isoformat() for d in s.dropna().to_list()})
            return dates
    # Fallback: no date column found
    return []


def filter_out_august(dates: List[str]) -> List[str]:
    out: List[str] = []
    for d in dates:
        try:
            dt = datetime.fromisoformat(d)
            if dt.month != 8:
                out.append(d)
        except Exception:
            # accept if parseable by pandas
            ts = pd.to_datetime(d, errors="coerce")
            if pd.notna(ts) and int(ts.month) != 8:
                out.append(ts.date().isoformat())
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--book", type=str, default="Caesars")
    args = ap.parse_args()

    raw_dir = Path(".data/raw")
    sched_path = raw_dir / "schedules.csv"
    if not sched_path.exists():
        raise SystemExit("Missing schedules.csv. Run data acquisition first.")
    schedules = pd.read_csv(sched_path)
    for c in ["season", "week", "home_team", "away_team"]:
        if c not in schedules.columns:
            schedules[c] = pd.NA
    sched_wk = schedules[(schedules["season"] == args.season) & (schedules["week"] == args.week)].copy()
    if sched_wk.empty:
        raise SystemExit(f"No schedule rows for {args.season} week {args.week}")

    dates = extract_unique_dates(sched_wk)
    dates = filter_out_august(dates)
    if not dates:
        # fallback: attempt Fri..Tue around the min existing gameday-like value
        # but if absent, bail
        raise SystemExit("No valid dates found for this week in schedules.csv")

    adapter = TheOddsApiAdapter(bookmaker_title=args.book)

    raw_out_dir = Path(f".data/odds_raw/{args.season}")
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for d in dates:
        df = adapter.fetch_odds_for_date(d)
        # write raw JSON snapshot
        try:
            # we don't have raw JSON from adapter; reconstruct from DataFrame for traceability
            (raw_out_dir / f"{d}.json").write_text(df.to_json(orient="records"), encoding="utf-8")
        except Exception:
            pass
        frames.append(df)

    if not frames:
        raise SystemExit("No odds data retrieved for the given dates.")
    all_odds = pd.concat(frames, axis=0, ignore_index=True)

    # Consolidate per schedule game by home/away abbreviations
    odds_cols = [
        "game_id","home_team","away_team","moneyline_home","moneyline_away",
        "spread_home","spread_price_home","spread_price_away","total_points",
        "total_over_price","total_under_price","commence_time","bookmaker",
    ]
    for c in odds_cols:
        if c not in all_odds.columns:
            all_odds[c] = pd.NA
    all_odds = all_odds[odds_cols].copy()

    # Pick the latest record per (home, away) pair by commence_time
    all_odds["commence_ts"] = pd.to_datetime(all_odds["commence_time"], errors="coerce", utc=True)
    all_odds.sort_values(["home_team","away_team","commence_ts"], inplace=True)
    latest = all_odds.groupby(["home_team","away_team"], as_index=False).tail(1)

    # Join onto schedule by teams
    join_cols = ["home_team","away_team"]
    consolidated = sched_wk.merge(latest, on=join_cols, how="left")

    out_dir = Path(f".data/odds/{args.season}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wk{args.week}.csv"
    consolidated[odds_cols].to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


