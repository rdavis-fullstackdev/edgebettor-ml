from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from edgebettor_ml.io.sinks import maybe_post_to_api


def make_idempotency_key(game_id: str) -> str:
    return hashlib.sha256(game_id.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()

    preds_path = Path("outputs") / f"preds_{args.season}_wk{args.week}.json"
    if not preds_path.exists():
        raise SystemExit(f"Missing predictions: {preds_path}")
    preds = json.loads(preds_path.read_text(encoding="utf-8"))

    # Create per-game payloads
    payloads = []
    for row in preds:
        payload = {
            "idempotency_key": make_idempotency_key(row["game_id"]),
            "game_id": row["game_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "probabilities": {
                "home_win": row["p_home_win"],
                "away_win": row["p_away_win"],
                "home_cover": row["p_home_cover"],
                "away_cover": row["p_away_cover"],
                "over": row["p_over"],
                "under": row["p_under"],
            },
        }
        payloads.append(payload)

    out_dir = Path("outputs") / f"export_{args.season}_wk{args.week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in payloads:
        (out_dir / f"{p['game_id']}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")

    maybe_post_to_api(payloads)
    print(f"Exported {len(payloads)} game payloads to {out_dir}")


if __name__ == "__main__":
    main()


