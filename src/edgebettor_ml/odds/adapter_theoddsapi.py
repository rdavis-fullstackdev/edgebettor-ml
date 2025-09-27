from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .adapter_base import OddsAdapter, ensure_standard_columns


BASE_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
HISTORICAL_URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds"

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

def _abbr(team_name: Optional[str]) -> Optional[str]:
    if team_name is None:
        return None
    return TEAM_NAME_TO_ABBR.get(team_name, team_name)


BOOKMAKER_TITLE_TO_KEY = {
    "caesars": "caesars",
    "caesars sportsbook": "caesars",
    "draftkings": "draftkings",
}


class TheOddsApiAdapter(OddsAdapter):
    def __init__(self, api_key: Optional[str] = None, market: str = "american", region: str = "us", bookmaker_title: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("Missing ODDS_API_KEY for The Odds API")
        self.market = market
        self.region = region
        self.bookmaker_title = bookmaker_title  # e.g., "Caesars"
        self.bookmaker_key = None
        if bookmaker_title:
            self.bookmaker_key = BOOKMAKER_TITLE_TO_KEY.get(bookmaker_title.lower(), bookmaker_title.lower())

    def _build_params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": ",".join(["h2h", "spreads", "totals"]),
            "oddsFormat": self.market,
        }
        if self.bookmaker_key:
            params["bookmakers"] = self.bookmaker_key
        if extra:
            params.update(extra)
        return params

    def _select_bookmaker(self, bookmakers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        bm: Optional[Dict[str, Any]] = None
        if self.bookmaker_title:
            for b in bookmakers:
                title = (b.get("title") or "").lower()
                key = (b.get("key") or "").lower()
                if title == self.bookmaker_title.lower() or key == (self.bookmaker_key or "") or self.bookmaker_title.lower() in title:
                    bm = b
                    break
        if bm is None and bookmakers:
            bm = max(bookmakers, key=lambda b: len(b.get("markets", [])))
        return bm

    def _parse_odds_payload(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        # Allow both live list payloads and historical dict payloads with a "data" field
        if isinstance(data, dict) and "data" in data:
            data = data.get("data") or []
        rows: List[Dict[str, Any]] = []
        for game in data:
            game_id = str(game.get("id"))
            home_full = game.get("home_team")
            teams = game.get("teams") or []
            away_full = next((t for t in teams if t != home_full), None)
            home_team = _abbr(home_full)
            away_team = _abbr(away_full)
            commence_time = game.get("commence_time")  # ISO timestamp
            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue
            bm = self._select_bookmaker(bookmakers)
            if not bm:
                continue
            markets = bm.get("markets", [])
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
                "bookmaker": bm.get("title") if bm else None,
            }
            inferred_away_from_outcomes = None
            for m in markets:
                key = m.get("key")
                outcomes = m.get("outcomes", [])
                if key == "h2h" and len(outcomes) >= 2:
                    for o in outcomes:
                        if _abbr(o.get("name")) == home_team:
                            row["moneyline_home"] = o.get("price")
                        else:
                            row["moneyline_away"] = o.get("price")
                            inferred_away_from_outcomes = _abbr(o.get("name")) or inferred_away_from_outcomes
                elif key == "spreads" and outcomes:
                    for o in outcomes:
                        if _abbr(o.get("name")) == home_team:
                            row["spread_home"] = o.get("point")
                            row["spread_price_home"] = o.get("price")
                        else:
                            row["spread_price_away"] = o.get("price")
                            inferred_away_from_outcomes = _abbr(o.get("name")) or inferred_away_from_outcomes
                elif key == "totals" and len(outcomes) >= 2:
                    for o in outcomes:
                        if o.get("name", "").lower().startswith("over"):
                            row["total_points"] = o.get("point")
                            row["total_over_price"] = o.get("price")
                        elif o.get("name", "").lower().startswith("under"):
                            row["total_under_price"] = o.get("price")
            if away_team is None and inferred_away_from_outcomes is not None:
                row["away_team"] = inferred_away_from_outcomes
            rows.append(row)
        df = pd.DataFrame(rows)
        return ensure_standard_columns(df)

    def fetch_odds_for_date(self, date_iso: str) -> pd.DataFrame:
        # Historical endpoint requires explicit date. Example format: 2020-09-01T12:00:00Z
        params = self._build_params({"dateFormat": "iso", "date": date_iso})
        resp = requests.get(HISTORICAL_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_odds_payload(data)

    def fetch_odds(self, season: int, week: int) -> pd.DataFrame:
        # This is a simplified placeholder; The Odds API does not index strictly by NFL week.
        # In production, you would map week to dates and filter games accordingly.
        params = self._build_params()
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_odds_payload(data)


