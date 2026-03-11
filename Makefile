PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .[dev]

ingest:
	$(PYTHON) scripts/ingest.py --dataset beir/scifact

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

eval:
	$(PYTHON) scripts/evaluate.py --dataset beir/scifact --strategy hybrid --k 10

test:
	pytest -q

lint:
	ruff check .
