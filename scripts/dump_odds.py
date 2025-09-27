from __future__ import annotations

import argparse
from pathlib import Path

from edgebettor_ml.odds.adapter_theoddsapi import TheOddsApiAdapter
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--book", type=str, default="Caesars")
    ap.add_argument("--show", type=int, default=0, help="Print the first N rows to stdout")
    ap.add_argument("--debug", action="store_true", help="Write a JSON debug dump alongside CSV")
    args = ap.parse_args()

    adapter = TheOddsApiAdapter(bookmaker_title=args.book)
    df = adapter.fetch_odds(args.season, args.week)
    out = Path("outputs") / f"odds_{args.book.lower()}_{args.season}_wk{args.week}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")

    if args.show and args.show > 0:
        cols = [c for c in [
            "game_id","home_team","away_team","commence_time","bookmaker",
            "moneyline_home","moneyline_away","spread_home","spread_price_home","spread_price_away",
            "total_points","total_over_price","total_under_price"
        ] if c in df.columns]
        print(df[cols].head(args.show).to_string(index=False))

    if args.debug:
        debug_path = out.with_suffix(".debug.json")
        try:
            df.to_json(debug_path, orient="records", indent=2)
            print(f"Wrote {debug_path}")
        except Exception as e:
            print(f"Failed to write debug JSON: {e}")


if __name__ == "__main__":
    main()


