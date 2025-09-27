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
    p.add_argument("--book", default=None, help="Specific bookmaker title (e.g., Caesars)")
    args = p.parse_args()

    if args.odds_source == "csv":
        if not args.odds_csv_path:
            raise SystemExit("--odds-csv-path is required when --odds-source=csv")
        adapter = CsvOddsAdapter(args.odds_csv_path)
    else:
        from edgebettor_ml.odds.adapter_theoddsapi import TheOddsApiAdapter
        book = args.book or "Caesars"
        adapter = TheOddsApiAdapter(bookmaker_title=book)

    odds_df = adapter.fetch_odds(args.season, args.week)
    feats = build_upcoming_features(args.season, args.week)
    # Merge odds into features BEFORE prediction to condition spread/total
    odds_slim = odds_df[[
        "game_id","home_team","away_team","spread_home","total_points"
    ]].copy()
    feats_for_pred = feats.merge(odds_slim, on=["game_id","home_team","away_team"], how="left")
    # Set model feature names consistently used in training
    if "spread_home" in feats_for_pred.columns:
        feats_for_pred["closing_spread"] = feats_for_pred["spread_home"]
    if "total_points" in feats_for_pred.columns:
        feats_for_pred["closing_total"] = feats_for_pred["total_points"]

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
    probs = predict_proba(feats_for_pred, artifacts_dir)

    preds = []
    ev_rows = []
    # Map probabilities by game_id to avoid row order issues
    # Use DataFrame indexes to ensure 1:1 mapping by merge order
    prob_df = feats_for_pred[["game_id"]].copy()
    prob_df["p_home_win"] = probs["p_home_win"]
    prob_df["p_home_cover"] = probs["p_home_cover"]
    prob_df["p_over"] = probs["p_over"]

    merged = feats.merge(prob_df, on=["game_id"], how="left").merge(
        odds_df, on=["game_id", "home_team", "away_team"], how="left", suffixes=("_feat", "_odds")
    )
    for _, r in merged.iterrows():
        p_home_win = float(r.p_home_win) if pd.notna(r.p_home_win) else 0.5
        p_home_cover = float(r.p_home_cover) if pd.notna(r.p_home_cover) else 0.5
        p_over = float(r.p_over) if pd.notna(r.p_over) else 0.5
        # Odds fields
        ml_home = r.get("moneyline_home_odds", r.get("moneyline_home"))
        ml_away = r.get("moneyline_away_odds", r.get("moneyline_away"))
        sp_home = r.get("spread_price_home_odds", r.get("spread_price_home"))
        sp_away = r.get("spread_price_away_odds", r.get("spread_price_away"))
        spread_home = r.get("spread_home_odds", r.get("spread_home"))
        total_points = r.get("total_points_odds", r.get("total_points"))
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
                home_ml_pct=p_home_win,
                home_ml=int(ml_home) if pd.notna(ml_home) else None,
                home_spread_pct=p_home_cover,
                home_spread=float(spread_home) if pd.notna(spread_home) else None,
                away_ml_pct=1 - p_home_win,
                away_ml=int(ml_away) if pd.notna(ml_away) else None,
                away_spread_pct=1 - p_home_cover,
                away_spread=float(-spread_home) if pd.notna(spread_home) else None,
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

    # Debug: write a compact CSV to inspect mapping
    try:
        dbg_rows = []
        for r in preds:
            dbg_rows.append({
                "game_id": r.game_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_spread": r.home_spread,
                "away_spread": r.away_spread,
                "p_home_win": r.p_home_win,
                "p_home_cover": r.p_home_cover,
                "p_away_cover": r.p_away_cover,
                "home_ml": r.home_ml,
                "away_ml": r.away_ml,
            })
        pd.DataFrame(dbg_rows).to_csv(Path("outputs") / f"preds_debug_{args.season}_wk{args.week}.csv", index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()


