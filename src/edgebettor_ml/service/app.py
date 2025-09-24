from __future__ import annotations

import os
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="NFL EV Service", version="0.1.0")


class PredictRequest(BaseModel):
    season: int
    week: int
    odds_source: Literal["csv", "theoddsapi"]
    odds_csv_path: Optional[str] = Field(default=None, description="Path to odds CSV if source=csv")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": {"TZ": os.environ.get("TZ", "")}}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    # Placeholder implementation; will be wired to pipeline.
    return {
        "season": req.season,
        "week": req.week,
        "odds_source": req.odds_source,
        "message": "Prediction pipeline not yet connected."
    }


