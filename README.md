## NFL EV Model

Predict NFL outcomes and compute expected value (EV) for moneyline, spread, and totals.

### Stack
- Python 3.11
- PyTorch, scikit-learn, xgboost
- pandas, numpy, scipy, pydantic, PyYAML
- FastAPI service
- Tooling: uv (locking), ruff + black, pytest

### Quickstart
1) Create venv and install deps
```
make setup
```
2) Lock and export dependencies
```
make lock
```
3) Lint and test
```
make lint
make test
```
4) Run service
```
make service
```

If make is unavailable on Windows PowerShell:
```
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
# Lock with uv
uv lock && uv export -o requirements.lock
pytest -q
uvicorn edgebettor_ml.service.app:app --reload --host 0.0.0.0 --port 8000
```

### Environment
Create a `.env` file based on `.env.example` with:
- `PYTHONPATH=src`
- `API_URL` (optional)
- `API_TOKEN` (optional)
- `ODDS_API_KEY` (for The Odds API)
- `TZ=America/Chicago`

### Outputs
- Artifacts under `artifacts/{run_id}/`
- Data cache under `.data/raw`
- Predictions under `outputs/`

See `edgebettor-ml-prompt.txt` for the full spec.

# edgebettor-ml