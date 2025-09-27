from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Rams": "LAR",
    "Los Angeles Chargers": "LAC",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def _abbr(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return TEAM_NAME_TO_ABBR.get(name, name)


def extract_bookmaker(game: Dict[str, Any], bookmaker_key: str) -> Optional[Dict[str, Any]]:
    for b in game.get("bookmakers", []) or []:
        if (b.get("key") or "").lower() == bookmaker_key.lower():
            return b
    return None


def parse_historical_json(payload: Dict[str, Any], bookmaker_key: str) -> pd.DataFrame:
    data = payload.get("data") or payload
    rows: List[Dict[str, Any]] = []
    for game in data:
        game_id = str(game.get("id"))
        commence_time = game.get("commence_time")
        home_team = _abbr(game.get("home_team"))
        away_team = _abbr(game.get("away_team"))
        bm = extract_bookmaker(game, bookmaker_key)
        if not bm:
            continue
        row = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "moneyline_home": None,
            "moneyline_away": None,
            "spread_home": None,
            "spread_price_home": None,
            "spread_price_away": None,
            "total_points": None,
            "total_over_price": None,
            "total_under_price": None,
            "commence_time": commence_time,
            "bookmaker": bm.get("title") or bookmaker_key,
        }
        inferred_away = None
        for m in bm.get("markets", []) or []:
            key = m.get("key")
            outcomes = m.get("outcomes", []) or []
            if key == "h2h" and len(outcomes) >= 2:
                for o in outcomes:
                    name = _abbr(o.get("name"))
                    if name == home_team:
                        row["moneyline_home"] = o.get("price")
                    else:
                        row["moneyline_away"] = o.get("price")
                        inferred_away = name or inferred_away
            elif key == "spreads" and outcomes:
                for o in outcomes:
                    name = _abbr(o.get("name"))
                    if name == home_team:
                        row["spread_home"] = o.get("point")
                        row["spread_price_home"] = o.get("price")
                    else:
                        row["spread_price_away"] = o.get("price")
                        inferred_away = name or inferred_away
            elif key == "totals" and len(outcomes) >= 2:
                for o in outcomes:
                    nm = (o.get("name") or "").lower()
                    if nm.startswith("over"):
                        row["total_points"] = o.get("point")
                        row["total_over_price"] = o.get("price")
                    elif nm.startswith("under"):
                        row["total_under_price"] = o.get("price")
        if not away_team and inferred_away:
            row["away_team"] = inferred_away
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to oddsapi historical JSON")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--book_key", default="williamhill_us")
    ap.add_argument("--bulk", action="store_true", help="Process entire file and split by season/week using schedules.csv")
    args = ap.parse_args()

    payload_path = Path(args.json)
    if not payload_path.exists():
        raise SystemExit(f"JSON not found: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    df = parse_historical_json(payload, args.book_key)
    if df.empty:
        raise SystemExit("No odds parsed from historical JSON")

    cols = [
        "game_id","home_team","away_team","moneyline_home","moneyline_away",
        "spread_home","spread_price_home","spread_price_away","total_points",
        "total_over_price","total_under_price","commence_time","bookmaker",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    if args.bulk or args.season is None or args.week is None:
        # bulk mode: map each game to season/week via schedules
        sched_path = Path('.data/raw/schedules.csv')
        if not sched_path.exists():
            raise SystemExit("Missing schedules.csv for bulk processing")
        schedules = pd.read_csv(sched_path)
        # derive a date column
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
            raise SystemExit("No parsable date column in schedules.csv")
        sched["date_iso"] = sched_dt.dt.date.astype(str)
        for c in ["season","week","home_team","away_team"]:
            if c not in sched.columns:
                sched[c] = pd.NA
        sched_small = sched[["season","week","home_team","away_team","date_iso"]].dropna(subset=["home_team","away_team","date_iso"]).copy()

        df = df.copy()
        df["date_iso"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True).dt.date.astype(str)
        merged = df.merge(sched_small, on=["home_team","away_team","date_iso"], how="left")
        mapped = merged.dropna(subset=["season","week"]).copy()
        if mapped.empty:
            raise SystemExit("No odds rows mapped to season/week using schedules")

        for (season, week), part in mapped.groupby(["season","week"]):
            season_i = int(season)
            week_i = int(week)
            raw_dir = Path(f".data/odds_raw/{season_i}")
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"wk{week_i}.json").write_text(part[cols].to_json(orient="records"), encoding="utf-8")
            out_dir = Path(f".data/odds/{season_i}")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"wk{week_i}.csv"
            part[cols].to_csv(out_path, index=False)
        print("Wrote consolidated weekly odds under .data/odds/<season>/wk<week>.csv")
    else:
        raw_dir = Path(f".data/odds_raw/{args.season}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"wk{args.week}.json").write_text(df.to_json(orient="records"), encoding="utf-8")

        out_dir = Path(f".data/odds/{args.season}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"wk{args.week}.csv"
        df[cols].to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


