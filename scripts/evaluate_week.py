from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from edgebettor_ml.modeling.evaluate import brier_score, log_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--preds-csv", type=str, required=True)
    args = ap.parse_args()

    preds = pd.read_csv(args.preds_csv)
    # Expect realized outcomes available in a joined file; for now, placeholder
    # Users can extend to compute realized ROI given settled bets.
    print("Evaluation placeholder; integrate with realized outcomes dataset.")

    out_dir = Path("reports") / f"{args.season}_{args.week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text("Add realized outcomes to compute ROI and calibration.")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()


