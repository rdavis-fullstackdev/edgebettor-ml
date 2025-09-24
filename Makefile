PY := python
UV := uv

.PHONY: setup lock lint format test service predict train data

setup:
	$(UV) venv .venv || $(PY) -m venv .venv
	. .venv/Scripts/activate && $(UV) pip install -e .[dev] || . .venv/Scripts/activate && pip install -e .[dev]

lock:
	$(UV) lock || echo "Install 'uv' to generate lockfile"
	$(UV) export -o requirements.lock || echo "Install 'uv' to export requirements.lock"

lint:
	. .venv/Scripts/activate && ruff check . && black --check .

format:
	. .venv/Scripts/activate && ruff check --fix . && black .

test:
	. .venv/Scripts/activate && pytest

service:
	. .venv/Scripts/activate && uvicorn edgebettor_ml.service.app:app --reload --host 0.0.0.0 --port 8000

data:
	. .venv/Scripts/activate && $(PY) -m edgebettor_ml.data.acquire

train:
	. .venv/Scripts/activate && $(PY) scripts/train_nn.py --config configs/train.yaml

predict:
	. .venv/Scripts/activate && $(PY) scripts/predict_week.py --season $(SEASON) --week $(WEEK) --odds-source $(ODDS) --odds-csv-path $(PATH)


