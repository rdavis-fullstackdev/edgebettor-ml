from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from edgebettor_ml.odds.adapter_csv import CsvOddsAdapter
from edgebettor_ml.odds.adapter_theoddsapi import TheOddsApiAdapter
from edgebettor_ml.odds.pricing import price_option
from edgebettor_ml.io.schemas import EvRow, PredictionRow
from edgebettor_ml.io.sinks import maybe_post_to_api, write_ev_csv, write_predictions_csv, write_predictions_json
from edgebettor_ml.data.build_upcoming import build_upcoming_features
from edgebettor_ml.modeling.infer import predict_proba
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--odds-source", choices=["csv", "theoddsapi"], required=True)
    p.add_argument("--odds-csv-path", default=None)
    args = p.parse_args()

    if args.odds_source == "csv":
        if not args.odds_csv_path:
            raise SystemExit("--odds-csv-path is required when --odds-source=csv")
        adapter = CsvOddsAdapter(args.odds_csv_path)
    else:
        adapter = TheOddsApiAdapter()

    odds_df = adapter.fetch_odds(args.season, args.week)
    feats = build_upcoming_features(args.season, args.week)
    # Load latest artifacts directory (simple heuristic: most recent dir)
    art_root = Path("artifacts")
    if not art_root.exists():
        raise SystemExit("No artifacts found. Train a model first with make train")
    season_prefix = f"run_{args.season}_"
    season_runs = [p for p in art_root.iterdir() if p.is_dir() and p.name.startswith(season_prefix) and (p / "model.pt").exists()]
    if season_runs:
        artifacts_dir = max(season_runs, key=lambda p: (p / "model.pt").stat().st_mtime)
    else:
        candidates = [p for p in art_root.iterdir() if p.is_dir() and (p / "model.pt").exists()]
        if not candidates:
            raise SystemExit("No valid artifact directories with model.pt present.")
        artifacts_dir = max(candidates, key=lambda p: (p / "model.pt").stat().st_mtime)
    probs = predict_proba(feats, artifacts_dir)

    preds = []
    ev_rows = []
    merged = feats.merge(odds_df, on=["game_id", "home_team", "away_team"], how="left", suffixes=("_feat", "_odds"))
    for idx, r in merged.iterrows():
        p_home_win = float(probs["p_home_win"][idx])
        p_home_cover = float(probs["p_home_cover"][idx])
        p_over = float(probs["p_over"][idx])
        preds.append(
            PredictionRow(
                game_id=str(r.game_id),
                home_team=r.home_team,
                away_team=r.away_team,
                p_home_win=p_home_win,
                p_away_win=1 - p_home_win,
                p_home_cover=p_home_cover,
                p_away_cover=1 - p_home_cover,
                p_over=p_over,
                p_under=1 - p_over,
            )
        )

        # Moneyline EV
        ml_home = r.get("moneyline_home_odds", r.get("moneyline_home"))
        ml_away = r.get("moneyline_away_odds", r.get("moneyline_away"))
        if pd.notna(ml_home):
            pr = price_option(p_home_win, int(ml_home))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="moneyline", side="home", price=int(ml_home), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))
        if pd.notna(ml_away):
            pr = price_option(1 - p_home_win, int(ml_away))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="moneyline", side="away", price=int(ml_away), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))

        # Spread EV (home spread/price)
        sp_home = r.get("spread_price_home_odds", r.get("spread_price_home"))
        sp_away = r.get("spread_price_away_odds", r.get("spread_price_away"))
        if pd.notna(sp_home):
            pr = price_option(p_home_cover, int(sp_home))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="spread", side="home", price=int(sp_home), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))
        if pd.notna(sp_away):
            pr = price_option(1 - p_home_cover, int(sp_away))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="spread", side="away", price=int(sp_away), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))

        # Total EV
        to_price = r.get("total_over_price_odds", r.get("total_over_price"))
        tu_price = r.get("total_under_price_odds", r.get("total_under_price"))
        if pd.notna(to_price):
            pr = price_option(p_over, int(to_price))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="total", side="over", price=int(to_price), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))
        if pd.notna(tu_price):
            pr = price_option(1 - p_over, int(tu_price))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="total", side="under", price=int(tu_price), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))

    out_csv = Path("outputs") / f"preds_{args.season}_wk{args.week}.csv"
    out_json = Path("outputs") / f"preds_{args.season}_wk{args.week}.json"
    write_predictions_csv(out_csv, preds)
    write_predictions_json(out_json, preds)
    maybe_post_to_api([p.model_dump() for p in preds])

    write_ev_csv(Path("outputs") / f"ev_{args.season}_wk{args.week}.csv", ev_rows)

    print(f"Wrote {out_csv} and {out_json}")


if __name__ == "__main__":
    main()


