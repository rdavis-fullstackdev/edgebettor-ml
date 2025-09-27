from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from edgebettor_ml.odds.adapter_theoddsapi import TheOddsApiAdapter

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


WEEKDAY_NAMES = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}


def daterange_days(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-09-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat(), help="End date (YYYY-MM-DD)")
    ap.add_argument("--book", default="Caesars", help="Bookmaker title (e.g., Caesars)")
    ap.add_argument("--days", default="thu,sat,sun,mon", help="Comma list of weekdays to fetch (mon..sun)")
    ap.add_argument("--skip_august", action="store_true", help="Exclude August dates (preseason)")
    args = ap.parse_args()

    weekday_map = {s.strip().lower() for s in args.days.split(",") if s.strip()}
    valid = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    if not weekday_map.issubset(valid):
        raise SystemExit("Invalid --days; use mon,tue,wed,thu,fri,sat,sun")

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    adapter = TheOddsApiAdapter(bookmaker_title=args.book)

    out_root = Path(".data/odds_raw_daily")
    out_root.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    total_calls = 0
    for day in daterange_days(start_dt, end_dt):
        wd = day.strftime("%a").lower()
        if wd not in weekday_map:
            continue
        if args.skip_august and day.month == 8:
            continue
        # Use noon UTC snapshot per day
        date_param = day.strftime("%Y-%m-%dT12:00:00Z")
        try:
            df = adapter.fetch_odds_for_date(date_param)
        except Exception as e:
            # continue on HTTP/plan limits
            print(f"WARN: {date_param} fetch failed: {e}")
            continue
        if df.empty:
            continue
        # Save raw for traceability
        out_day_dir = out_root / day.strftime("%Y")
        out_day_dir.mkdir(parents=True, exist_ok=True)
        (out_day_dir / f"{day.date().isoformat()}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        frames.append(df)
        total_calls += 1

    if not frames:
        print("No daily odds fetched.")
        return

    all_df = pd.concat(frames, axis=0, ignore_index=True)
    print(f"Fetched {len(frames)} daily snapshots across {total_calls} API calls.")

    # Map to season/week using schedules
    sched_path = Path('.data/raw/schedules.csv')
    if not sched_path.exists():
        print("Missing schedules.csv; skipping weekly consolidation.")
        return
    schedules = pd.read_csv(sched_path)
    date_cols = [
        "gameday","game_day","gamedate","game_date","start_time_utc","start_time",
        "game_time","game_datetime","datetime","date"
    ]
    sched = schedules.copy()
    sched_dt = None
    for c in date_cols:
        if c in sched.columns:
            sched_dt = pd.to_datetime(sched[c], errors="coerce", utc=True)
            break
    if sched_dt is None:
        print("No parsable date column in schedules.csv; skipping weekly consolidation.")
        return
    sched["date_iso"] = sched_dt.dt.date.astype(str)
    for c in ["season","week","home_team","away_team"]:
        if c not in sched.columns:
            sched[c] = pd.NA
    sched_small = sched[["season","week","home_team","away_team","date_iso"]].dropna(subset=["home_team","away_team","date_iso"]).copy()

    all_df = all_df.copy()
    all_df["date_iso"] = pd.to_datetime(all_df["commence_time"], errors="coerce", utc=True).dt.date.astype(str)
    merged = all_df.merge(sched_small, on=["home_team","away_team","date_iso"], how="left")
    mapped = merged.dropna(subset=["season","week"]).copy()
    if mapped.empty:
        print("No rows mapped to season/week; daily raw saved only.")
        return

    cols = [
        "game_id","home_team","away_team","moneyline_home","moneyline_away",
        "spread_home","spread_price_home","spread_price_away","total_points",
        "total_over_price","total_under_price","commence_time","bookmaker",
    ]
    for (season, week), part in mapped.groupby(["season","week"]):
        season_i = int(season)
        week_i = int(week)
        out_dir = Path(f".data/odds/{season_i}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"wk{week_i}.csv"
        # within-week, take latest per matchup
        p2 = part.copy()
        p2["commence_ts"] = pd.to_datetime(p2["commence_time"], errors="coerce", utc=True)
        p2.sort_values(["home_team","away_team","commence_ts"], inplace=True)
        latest = p2.groupby(["home_team","away_team"], as_index=False).tail(1)
        latest = latest[cols]
        latest.to_csv(out_path, index=False)
    print("Consolidated weekly CSVs under .data/odds/<season>/wk<week>.csv")


if __name__ == "__main__":
    main()


