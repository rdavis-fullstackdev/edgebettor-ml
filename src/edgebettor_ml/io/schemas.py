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


class EvRow(BaseModel):
    game_id: str
    market: str  # moneyline|spread|total
    side: str    # home|away|over|under
    price: int
    implied: float
    edge: float
    ev: float


