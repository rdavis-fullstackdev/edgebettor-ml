from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_per_head(y_val: Dict[str, np.ndarray], p_val: Dict[str, np.ndarray]) -> Dict[str, IsotonicRegression]:
    calibrators: Dict[str, IsotonicRegression] = {}
    for key_map in [
        ("y_home_win", "p_home_win"),
        ("y_home_cover_vs_closing", "p_home_cover"),
        ("y_over_total_vs_closing", "p_over"),
    ]:
        yk, pk = key_map
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_val[pk], y_val[yk])
        calibrators[pk] = iso
    return calibrators


def save_calibrators(calibrators: Dict[str, IsotonicRegression], path: str | Path) -> None:
    path = Path(path)
    with (path / "calibrators.pkl").open("wb") as f:
        pickle.dump(calibrators, f)


def load_calibrators(path: str | Path) -> Dict[str, IsotonicRegression]:
    path = Path(path)
    with (path / "calibrators.pkl").open("rb") as f:
        return pickle.load(f)


