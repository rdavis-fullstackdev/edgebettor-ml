from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import os
from .datasets import PreprocessArtifacts, build_feature_matrix, load_artifacts, transform_with
from .model_nn import MultiHeadNet
from .calibration import load_calibrators


def load_model(artifacts_dir: str | Path) -> MultiHeadNet:
    artifacts_dir = Path(artifacts_dir)
    feature_names = json.loads((artifacts_dir / "feature_names.json").read_text(encoding="utf-8"))
    state_path = artifacts_dir / "model.pt"
    # Prefer to infer input_dim from checkpoint to avoid mismatch
    ckpt = torch.load(state_path, map_location="cpu") if state_path.exists() else None
    if ckpt is not None and isinstance(ckpt, dict) and "trunk.0.weight" in ckpt:
        input_dim = int(ckpt["trunk.0.weight"].shape[1])
    else:
        input_dim = len(feature_names)
    model = MultiHeadNet(input_dim=input_dim)
    if ckpt is not None:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def predict_proba(features_df, artifacts_dir: str | Path) -> Dict[str, np.ndarray]:
    arts: PreprocessArtifacts = load_artifacts(artifacts_dir)
    model = load_model(artifacts_dir)
    # Ensure required columns exist in the same order as training
    df = features_df.copy()
    for col in arts.feature_names:
        if col not in df.columns:
            df[col] = np.nan
    df = df[arts.feature_names]
    X = build_feature_matrix(df, arts.feature_names)
    X_std = transform_with(arts, X)
    with torch.no_grad():
        out = model(torch.tensor(X_std, dtype=torch.float32))
    p = {k: v.numpy() for k, v in out.items()}
    if os.environ.get("APPLY_CALIBRATION", "0") == "1":
        calib_path = Path(artifacts_dir) / "calibrators.pkl"
        if calib_path.exists():
            calibrators = load_calibrators(artifacts_dir)
            for k, iso in calibrators.items():
                p[k] = iso.predict(p[k])
    return p


