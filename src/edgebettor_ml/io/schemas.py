from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PredictionRow(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    p_home_win: float
    p_away_win: float
    p_home_cover: float
    p_away_cover: float
    p_over: float
    p_under: float
    # extras
    home_ml_pct: float | None = None
    home_ml: int | None = None
    home_spread_pct: float | None = None
    home_spread: float | None = None
    away_ml_pct: float | None = None
    away_ml: int | None = None
    away_spread_pct: float | None = None
    away_spread: float | None = None


class EvRow(BaseModel):
    game_id: str
    market: str  # moneyline|spread|total
    side: str    # home|away|over|under
    price: int
    implied: float
    edge: float
    ev: float


