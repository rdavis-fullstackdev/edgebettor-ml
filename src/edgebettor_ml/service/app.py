from __future__ import annotations

import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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
    try:
        from pathlib import Path
        import pandas as pd
        from edgebettor_ml.odds.adapter_csv import CsvOddsAdapter
        from edgebettor_ml.odds.adapter_theoddsapi import TheOddsApiAdapter
        from edgebettor_ml.data.build_upcoming import build_upcoming_features
        from edgebettor_ml.modeling.infer import predict_proba
        from edgebettor_ml.odds.pricing import price_option
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))

    if req.odds_source == "csv":
        if not req.odds_csv_path:
            raise HTTPException(status_code=400, detail="odds_csv_path required for csv source")
        adapter = CsvOddsAdapter(req.odds_csv_path)
    else:
        adapter = TheOddsApiAdapter()

    odds_df = adapter.fetch_odds(req.season, req.week)
    feats = build_upcoming_features(req.season, req.week)

    art_root = Path("artifacts")
    run_dirs = sorted([p for p in art_root.iterdir() if p.is_dir()]) if art_root.exists() else []
    if not run_dirs:
        raise HTTPException(status_code=500, detail="No artifacts found; train a model first")
    artifacts_dir = run_dirs[-1]
    probs = predict_proba(feats, artifacts_dir)

    merged = feats.merge(odds_df, on=["game_id", "home_team", "away_team"], how="left")
    preds = []
    evs = []
    for idx, r in merged.iterrows():
        p_home_win = float(probs["p_home_win"][idx])
        p_home_cover = float(probs["p_home_cover"][idx])
        p_over = float(probs["p_over"][idx])
        preds.append({
            "game_id": str(r.game_id),
            "home_team": r.home_team,
            "away_team": r.away_team,
            "p_home_win": p_home_win,
            "p_away_win": 1 - p_home_win,
            "p_home_cover": p_home_cover,
            "p_away_cover": 1 - p_home_cover,
            "p_over": p_over,
            "p_under": 1 - p_over,
        })
        if pd.notna(r.get("moneyline_home")):
            pr = price_option(p_home_win, int(r["moneyline_home"]))
            evs.append({"game_id": str(r.game_id), "market": "moneyline", "side": "home", **{**pr, "price": int(r["moneyline_home"])}})
        if pd.notna(r.get("moneyline_away")):
            pr = price_option(1 - p_home_win, int(r["moneyline_away"]))
            evs.append({"game_id": str(r.game_id), "market": "moneyline", "side": "away", **{**pr, "price": int(r["moneyline_away"])}})
        if pd.notna(r.get("spread_price_home")):
            pr = price_option(p_home_cover, int(r["spread_price_home"]))
            evs.append({"game_id": str(r.game_id), "market": "spread", "side": "home", **{**pr, "price": int(r["spread_price_home"])}})
        if pd.notna(r.get("spread_price_away")):
            pr = price_option(1 - p_home_cover, int(r["spread_price_away"]))
            evs.append({"game_id": str(r.game_id), "market": "spread", "side": "away", **{**pr, "price": int(r["spread_price_away"])}})
        if pd.notna(r.get("total_over_price")):
            pr = price_option(p_over, int(r["total_over_price"]))
            evs.append({"game_id": str(r.game_id), "market": "total", "side": "over", **{**pr, "price": int(r["total_over_price"])}})
        if pd.notna(r.get("total_under_price")):
            pr = price_option(1 - p_over, int(r["total_under_price"]))
            evs.append({"game_id": str(r.game_id), "market": "total", "side": "under", **{**pr, "price": int(r["total_under_price"])}})

    return {"season": req.season, "week": req.week, "predictions": preds, "ev": evs}


