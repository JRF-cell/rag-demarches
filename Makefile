PYTHON ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: install data explore split questions questions-dry review review-sample test lint format eval api

install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -m src.ingest.download
	$(PY) -m src.ingest.build_corpus
	$(PY) -m src.ingest.chunk

explore:
	$(PY) -m src.ingest.explore

split:
	$(PY) -m src.evaluation.split

questions-dry:
	$(PY) -m src.evaluation.generate_questions

questions:
	$(PY) -m src.evaluation.generate_questions --call

review:
	$(PY) -m src.evaluation.review

review-sample:
	$(PY) -m src.evaluation.review --sample 80 --seed 42

test:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

format:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

eval:
	@echo "Not implemented yet (EXP=$(EXP))"; exit 1

api:
	@echo "Not implemented yet"; exit 1
