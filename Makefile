# ============================================================
# Credit Risk Scoring Engine — Makefile
# Usage: make <target>
# ============================================================

.PHONY: help setup data features train shap app test lint clean

PYTHON := python
SRC := src
CONFIG := config.yaml

help:
	@echo "Credit Risk Scoring Engine — Available Commands"
	@echo "================================================"
	@echo "  make setup     Install all dependencies"
	@echo "  make data      Run data preprocessing pipeline"
	@echo "  make features  Run feature engineering"
	@echo "  make train     Train + evaluate XGBoost model"
	@echo "  make shap      Generate SHAP explanations"
	@echo "  make pipeline  Run data + features + train + shap"
	@echo "  make app       Launch Streamlit app locally"
	@echo "  make test      Run pytest with coverage"
	@echo "  make lint      Run flake8 + black check"
	@echo "  make clean     Remove generated artefacts"

setup:
	pip install -r requirements-dev.txt

data:
	$(PYTHON) $(SRC)/data_pipeline/preprocess.py --config $(CONFIG)

features:
	$(PYTHON) $(SRC)/features/engineer.py --config $(CONFIG)

train:
	$(PYTHON) $(SRC)/models/train.py --config $(CONFIG)

shap:
	$(PYTHON) $(SRC)/explainability/generate_shap.py --config $(CONFIG)

pipeline: data features train shap
	@echo "Full pipeline complete!"

app:
	streamlit run streamlit_app/app.py

test:
	pytest tests/ -v --cov=$(SRC) --cov-report=html --cov-report=term-missing

lint:
	flake8 $(SRC)/ streamlit_app/ tests/ --max-line-length=100
	black --check $(SRC)/ streamlit_app/ tests/

format:
	black $(SRC)/ streamlit_app/ tests/
	isort $(SRC)/ streamlit_app/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/ htmlcov/ .coverage
	@echo "Cleaned up generated files."
