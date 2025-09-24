from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from edgebettor_ml.odds.adapter_csv import CsvOddsAdapter
from edgebettor_ml.odds.adapter_theoddsapi import TheOddsApiAdapter
from edgebettor_ml.odds.pricing import price_option
from edgebettor_ml.io.schemas import EvRow, PredictionRow
from edgebettor_ml.io.sinks import maybe_post_to_api, write_ev_csv, write_predictions_csv, write_predictions_json


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

    # Placeholder: predictions would be produced by the model. For now, uniform priors.
    preds = []
    ev_rows = []
    for _, r in odds_df.iterrows():
        p_home_win = 0.5
        p_home_cover = 0.5
        p_over = 0.5
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
        if pd.notna(r.moneyline_home):
            pr = price_option(p_home_win, int(r.moneyline_home))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="moneyline", side="home", price=int(r.moneyline_home), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))
        if pd.notna(r.moneyline_away):
            pr = price_option(1 - p_home_win, int(r.moneyline_away))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="moneyline", side="away", price=int(r.moneyline_away), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))

        # Spread EV (home spread/price)
        if pd.notna(r.spread_price_home):
            pr = price_option(p_home_cover, int(r.spread_price_home))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="spread", side="home", price=int(r.spread_price_home), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))
        if pd.notna(r.spread_price_away):
            pr = price_option(1 - p_home_cover, int(r.spread_price_away))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="spread", side="away", price=int(r.spread_price_away), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))

        # Total EV
        if pd.notna(r.total_over_price):
            pr = price_option(p_over, int(r.total_over_price))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="total", side="over", price=int(r.total_over_price), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))
        if pd.notna(r.total_under_price):
            pr = price_option(1 - p_over, int(r.total_under_price))
            ev_rows.append(EvRow(game_id=str(r.game_id), market="total", side="under", price=int(r.total_under_price), implied=pr["implied"], edge=pr["edge"], ev=pr["ev"]))

    out_csv = Path("outputs") / f"preds_{args.season}_wk{args.week}.csv"
    out_json = Path("outputs") / f"preds_{args.season}_wk{args.week}.json"
    write_predictions_csv(out_csv, preds)
    write_predictions_json(out_json, preds)
    maybe_post_to_api([p.model_dump() for p in preds])

    write_ev_csv(Path("outputs") / f"ev_{args.season}_wk{args.week}.csv", ev_rows)

    print(f"Wrote {out_csv} and {out_json}")


if __name__ == "__main__":
    main()


