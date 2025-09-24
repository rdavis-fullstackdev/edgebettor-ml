from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessArtifacts:
    feature_names: List[str]
    imputer: SimpleImputer
    scaler: StandardScaler


def build_feature_matrix(df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    X = df[feature_names].to_numpy(dtype=float)
    return X


def fit_preprocessor(X_train: np.ndarray) -> Tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_train)
    scaler.fit(X_imp)
    return imputer, scaler


def transform_with(art: PreprocessArtifacts, X: np.ndarray) -> np.ndarray:
    X_imp = art.imputer.transform(X)
    X_std = art.scaler.transform(X_imp)
    return X_std


def save_artifacts(art: PreprocessArtifacts, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_names.json").write_text(json.dumps(art.feature_names, indent=2), encoding="utf-8")
    with (out_dir / "scalers.pkl").open("wb") as f:
        pickle.dump({"imputer": art.imputer, "scaler": art.scaler}, f)


def load_artifacts(path: str | Path) -> PreprocessArtifacts:
    path = Path(path)
    feature_names = json.loads((path / "feature_names.json").read_text(encoding="utf-8"))
    with (path / "scalers.pkl").open("rb") as f:
        obj = pickle.load(f)
    return PreprocessArtifacts(feature_names=feature_names, imputer=obj["imputer"], scaler=obj["scaler"])


