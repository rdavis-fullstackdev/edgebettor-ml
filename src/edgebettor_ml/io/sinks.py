from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

from .schemas import EvRow, PredictionRow


def write_predictions_csv(path: str | Path, rows: Iterable[PredictionRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in rows])
    df.to_csv(path, index=False)


def write_predictions_json(path: str | Path, rows: Iterable[PredictionRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump() for r in rows]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_ev_csv(path: str | Path, rows: Iterable[EvRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in rows])
    df.to_csv(path, index=False)


def maybe_post_to_api(json_rows: Iterable[dict]) -> None:
    api_url = os.environ.get("API_URL")
    if not api_url:
        return
    token = os.environ.get("API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.post(api_url.rstrip("/"), json=list(json_rows), headers=headers, timeout=30)
    resp.raise_for_status()


