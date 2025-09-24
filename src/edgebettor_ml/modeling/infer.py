from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .datasets import PreprocessArtifacts, build_feature_matrix, load_artifacts, transform_with
from .model_nn import MultiHeadNet
from .calibration import load_calibrators


def load_model(artifacts_dir: str | Path) -> MultiHeadNet:
    artifacts_dir = Path(artifacts_dir)
    feature_names = json.loads((artifacts_dir / "feature_names.json").read_text(encoding="utf-8"))
    input_dim = len(feature_names)
    model = MultiHeadNet(input_dim=input_dim)
    state_path = artifacts_dir / "model.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    return model


def predict_proba(features_df, artifacts_dir: str | Path) -> Dict[str, np.ndarray]:
    arts: PreprocessArtifacts = load_artifacts(artifacts_dir)
    model = load_model(artifacts_dir)
    X = build_feature_matrix(features_df, arts.feature_names)
    X_std = transform_with(arts, X)
    with torch.no_grad():
        out = model(torch.tensor(X_std, dtype=torch.float32))
    p = {k: v.numpy() for k, v in out.items()}
    calib_path = Path(artifacts_dir) / "calibrators.pkl"
    if calib_path.exists():
        calibrators = load_calibrators(artifacts_dir)
        for k, iso in calibrators.items():
            p[k] = iso.predict(p[k])
    return p


