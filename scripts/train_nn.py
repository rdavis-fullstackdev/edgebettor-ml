from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from edgebettor_ml.data.features import FeatureBuildConfig, build_features
from edgebettor_ml.data.splits import time_based_split
from edgebettor_ml.modeling.calibration import fit_isotonic_per_head, save_calibrators
from edgebettor_ml.modeling.datasets import (
    PreprocessArtifacts,
    build_feature_matrix,
    fit_preprocessor,
    save_artifacts,
)
from edgebettor_ml.modeling.evaluate import auc_score, brier_score, log_loss
from edgebettor_ml.modeling.model_nn import TrainConfig, train_model


TARGET_COLS = [
    "y_home_win",
    "y_home_cover_vs_closing",
    "y_over_total_vs_closing",
]


def load_configs(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_weekly_from_schedule(schedules: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "week", "home_team", "away_team", "home_score", "away_score"]
    for c in cols:
        if c not in schedules.columns:
            schedules[c] = np.nan
    played = schedules.dropna(subset=["home_score", "away_score"])  # completed games
    home_rows = played[["season", "week", "home_team", "home_score", "away_score"]].copy()
    home_rows.rename(columns={
        "home_team": "team",
        "home_score": "points_for",
        "away_score": "points_against",
    }, inplace=True)
    away_rows = played[["season", "week", "away_team", "away_score", "home_score"]].copy()
    away_rows.rename(columns={
        "away_team": "team",
        "away_score": "points_for",
        "home_score": "points_against",
    }, inplace=True)
    weekly = pd.concat([home_rows, away_rows], axis=0, ignore_index=True)
    weekly["margin"] = weekly["points_for"] - weekly["points_against"]
    return weekly


def assemble_training_frame(raw_dir: Path) -> pd.DataFrame:
    # Load raw components
    schedules = pd.read_csv(raw_dir / "schedules.csv")
    betting_path = raw_dir / "betting_lines.csv"
    betting = pd.read_csv(betting_path) if betting_path.exists() else pd.DataFrame()
    weekly_path = raw_dir / "team_weekly.csv"
    if weekly_path.exists():
        weekly = pd.read_csv(weekly_path)
    else:
        weekly = _build_weekly_from_schedule(schedules)

    # Normalize schedules minimal columns
    keep = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "stadium",
        "venue",
    ]
    for c in keep:
        if c not in schedules.columns:
            schedules[c] = np.nan
    sched = schedules[keep].copy()

    # Join closing lines if available (simplified merge)
    bl_keep = [
        "game_id",
        "spread_close",
        "total_close",
        "home_moneyline_close",
        "away_moneyline_close",
    ]
    if betting.empty:
        df = sched.copy()
        df["closing_spread"] = np.nan
        df["closing_total"] = np.nan
        df["moneyline_home"] = np.nan
        df["moneyline_away"] = np.nan
    else:
        for c in bl_keep:
            if c not in betting.columns:
                betting[c] = np.nan
        bl = betting[bl_keep].drop_duplicates("game_id")
        df = sched.merge(bl, on="game_id", how="left")
        df.rename(
            columns={
                "spread_close": "closing_spread",
                "total_close": "closing_total",
                "home_moneyline_close": "moneyline_home",
                "away_moneyline_close": "moneyline_away",
            },
            inplace=True,
        )

    feats = build_features(df, weekly, FeatureBuildConfig())
    # Keep rows with targets (completed games)
    feats = feats.dropna(subset=TARGET_COLS)
    return feats


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded_prefixes = ["y_"]
    excluded_exact = {
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
    }
    candidates = []
    for c in df.columns:
        if c in excluded_exact:
            continue
        if any(c.startswith(p) for p in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args()

    cfg = load_configs(args.config)
    artifacts_root = Path(cfg.get("artifacts_dir", "artifacts"))

    raw_dir = Path(".data/raw")
    if not raw_dir.exists():
        raise SystemExit("Raw data not found. Run: make data")

    df = assemble_training_frame(raw_dir)
    test_season = int(df["season"].max())
    train_df, val_df, test_df = time_based_split(df, test_season)

    feature_names = select_feature_columns(train_df)

    X_train = build_feature_matrix(train_df, feature_names)
    X_val = build_feature_matrix(val_df, feature_names)
    X_test = build_feature_matrix(test_df, feature_names)

    y_train = {
        "y_home_win": train_df["y_home_win"].to_numpy(float),
        "y_home_cover_vs_closing": train_df["y_home_cover_vs_closing"].to_numpy(float),
        "y_over_total_vs_closing": train_df["y_over_total_vs_closing"].to_numpy(float),
    }
    y_val = {
        "y_home_win": val_df["y_home_win"].to_numpy(float),
        "y_home_cover_vs_closing": val_df["y_home_cover_vs_closing"].to_numpy(float),
        "y_over_total_vs_closing": val_df["y_over_total_vs_closing"].to_numpy(float),
    }
    y_test = {
        "y_home_win": test_df["y_home_win"].to_numpy(float),
        "y_home_cover_vs_closing": test_df["y_home_cover_vs_closing"].to_numpy(float),
        "y_over_total_vs_closing": test_df["y_over_total_vs_closing"].to_numpy(float),
    }

    # Preprocess
    imputer, scaler = fit_preprocessor(X_train)
    arts = PreprocessArtifacts(feature_names=feature_names, imputer=imputer, scaler=scaler)
    X_tr_std = scaler.transform(imputer.transform(X_train))
    X_va_std = scaler.transform(imputer.transform(X_val))
    X_te_std = scaler.transform(imputer.transform(X_test))

    # Train
    tc = TrainConfig(
        lr=cfg["model"].get("lr", 1e-3),
        batch_size=cfg["model"].get("batch_size", 256),
        max_epochs=cfg["model"].get("max_epochs", 100),
        early_stopping_patience=cfg["model"].get("early_stopping_patience", 15),
    )
    # Optional class weights per head
    cw_cfg = cfg.get("class_weights", {})
    head_weights = {
        "p_home_win": float(cw_cfg.get("home_win", 1.0)),
        "p_home_cover": float(cw_cfg.get("home_cover", 1.0)),
        "p_over": float(cw_cfg.get("over_total", 1.0)),
    }
    model, metrics = train_model(X_tr_std, y_train, X_va_std, y_val, tc, class_weights=head_weights)

    # Calibrate on validation
    import torch

    model.eval()
    with torch.no_grad():
        val_out = model(torch.tensor(X_va_std, dtype=torch.float32))
    p_val = {
        "p_home_win": val_out["p_home_win"].numpy(),
        "p_home_cover": val_out["p_home_cover"].numpy(),
        "p_over": val_out["p_over"].numpy(),
    }
    calibrators = fit_isotonic_per_head(y_val, p_val)

    # Evaluate on test
    with torch.no_grad():
        te_out = model(torch.tensor(X_te_std, dtype=torch.float32))
    p_test = {
        "p_home_win": te_out["p_home_win"].numpy(),
        "p_home_cover": te_out["p_home_cover"].numpy(),
        "p_over": te_out["p_over"].numpy(),
    }

    test_metrics = {
        "win_auc": auc_score(y_test["y_home_win"], p_test["p_home_win"]),
        "win_brier": brier_score(y_test["y_home_win"], p_test["p_home_win"]),
        "win_logloss": log_loss(y_test["y_home_win"], p_test["p_home_win"]),
        "cover_brier": brier_score(y_test["y_home_cover_vs_closing"], p_test["p_home_cover"]),
        "over_brier": brier_score(y_test["y_over_total_vs_closing"], p_test["p_over"]),
        "val_brier_mean": metrics.get("val_brier_mean"),
    }

    # Save artifacts
    run_dir = artifacts_root / f"run_{test_season}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_artifacts(arts, run_dir)
    import torch

    torch.save(model.state_dict(), run_dir / "model.pt")
    save_calibrators(calibrators, run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print(f"Saved artifacts to {run_dir}")


if __name__ == "__main__":
    main()


