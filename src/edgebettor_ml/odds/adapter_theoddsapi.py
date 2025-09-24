from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .adapter_base import OddsAdapter, ensure_standard_columns


BASE_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


class TheOddsApiAdapter(OddsAdapter):
    def __init__(self, api_key: Optional[str] = None, market: str = "american", region: str = "us"):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("Missing ODDS_API_KEY for The Odds API")
        self.market = market
        self.region = region

    def fetch_odds(self, season: int, week: int) -> pd.DataFrame:
        # This is a simplified placeholder; The Odds API does not index strictly by NFL week.
        # In production, you would map week to dates and filter games accordingly.
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": ",".join(["h2h", "spreads", "totals"]),
            "oddsFormat": self.market,
        }
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Parse minimal fields from first bookmaker lines per game
        rows: List[Dict[str, Any]] = []
        for game in data:
            game_id = str(game.get("id"))
            home_team = game.get("home_team")
            away_team = next((t for t in game.get("commence_time", ""),), None)  # placeholder
            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue
            bm = bookmakers[0]
            markets = bm.get("markets", [])
            row = {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": None,
                "moneyline_home": None,
                "moneyline_away": None,
                "spread_home": None,
                "spread_price_home": None,
                "spread_price_away": None,
                "total_points": None,
                "total_over_price": None,
                "total_under_price": None,
            }
            for m in markets:
                key = m.get("key")
                outcomes = m.get("outcomes", [])
                if key == "h2h" and len(outcomes) >= 2:
                    for o in outcomes:
                        if o.get("name") == home_team:
                            row["moneyline_home"] = o.get("price")
                        else:
                            row["moneyline_away"] = o.get("price")
                elif key == "spreads" and outcomes:
                    # The Odds API provides point and price for each team
                    for o in outcomes:
                        if o.get("name") == home_team:
                            row["spread_home"] = o.get("point")
                            row["spread_price_home"] = o.get("price")
                        else:
                            row["spread_price_away"] = o.get("price")
                elif key == "totals" and len(outcomes) >= 2:
                    # Outcomes typically named Over/Under
                    for o in outcomes:
                        if o.get("name", "").lower().startswith("over"):
                            row["total_points"] = o.get("point")
                            row["total_over_price"] = o.get("price")
                        elif o.get("name", "").lower().startswith("under"):
                            row["total_under_price"] = o.get("price")
            rows.append(row)

        df = pd.DataFrame(rows)
        df_std = ensure_standard_columns(df)
        return df_std


