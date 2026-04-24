# ─────────────────────────────────────────────────────────────
# Makefile — Geo Customer Segmentation Project
# Usage: make <target>
# ─────────────────────────────────────────────────────────────

.PHONY: help install setup data validate features cluster profile \
        recommend geo pipeline api dashboard mlflow notebook \
        test test-cov lint format docker-up docker-down clean

PYTHON := python

help:
	@echo ""
	@echo "  geo-customer-segmentation"
	@echo "  ──────────────────────────────────────────────────"
	@echo "  Setup"
	@echo "    make install       Install Python dependencies"
	@echo "    make setup         First-time project setup"
	@echo ""
	@echo "  Pipeline (run in order)"
	@echo "    make data          Generate 30k synthetic subscriber data"
	@echo "    make validate      Validate raw data quality"
	@echo "    make features      Run feature engineering pipeline"
	@echo "    make cluster       Run KMeans clustering (MLflow tracked)"
	@echo "    make cluster-tune  Run clustering with Optuna HPO"
	@echo "    make profile       Auto-name and profile segments"
	@echo "    make recommend     Train recommendation engine"
	@echo "    make geo           Generate H3 segment density map"
	@echo "    make pipeline      Run ALL steps end-to-end"
	@echo ""
	@echo "  Optional real data"
	@echo "    make osm           Download Jakarta OSM POIs (free)"
	@echo "    make opencellid    Download OpenCelliD tower data"
	@echo ""
	@echo "  Notebooks"
	@echo "    make notebook      Launch Jupyter Lab"
	@echo "    make run-notebooks Run all notebooks as scripts"
	@echo ""
	@echo "  Serving"
	@echo "    make api           Start FastAPI service :8000"
	@echo "    make dashboard     Start Streamlit dashboard :8501"
	@echo "    make mlflow        Start MLflow UI :5000"
	@echo ""
	@echo "  Quality"
	@echo "    make test          Run pytest suite"
	@echo "    make test-cov      Tests with coverage report"
	@echo "    make lint          flake8 + black check"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-up     Start full stack (API + Dashboard + MLflow)"
	@echo "    make docker-down   Stop all containers"
	@echo ""
	@echo "    make clean         Remove Python cache files"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	pip install -e .

setup: install
	cp -n .env.example .env || true
	mkdir -p data/raw data/processed data/external data/models logs docs
	@echo "✅ Setup complete. Run: make pipeline"

# ── Pipeline ──────────────────────────────────────────────────
data:
	$(PYTHON) src/data_engineering/generate_data.py \
		--n_subscribers 30000 --output data/raw/

validate:
	$(PYTHON) src/data_engineering/data_validation.py \
		--input data/raw/subscribers.parquet

features:
	$(PYTHON) src/features/feature_pipeline.py \
		--input data/raw --output data/processed

cluster:
	$(PYTHON) src/models/clustering.py \
		--algorithm kmeans

cluster-tune:
	$(PYTHON) src/models/clustering.py \
		--algorithm kmeans --tune --n_trials 40

cluster-dbscan:
	$(PYTHON) src/models/clustering.py \
		--algorithm dbscan

profile:
	$(PYTHON) src/models/segment_profiler.py

recommend:
	$(PYTHON) src/models/recommender.py --train

geo:
	$(PYTHON) src/models/geo_density.py \
		--output data/processed/segment_density.geojson

pipeline: data validate features cluster profile recommend geo
	@echo ""
	@echo "✅ Full pipeline complete!"
	@echo "   make dashboard   → View results at :8501"
	@echo "   make api         → Start prediction API at :8000"
	@echo "   make mlflow      → View experiments at :5000"

# ── Real data ingestion ───────────────────────────────────────
osm:
	$(PYTHON) src/data_engineering/ingest_osm.py --city "Jakarta"

opencellid:
	$(PYTHON) src/data_engineering/ingest_opencellid.py --country ID

# ── Serving ───────────────────────────────────────────────────
api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboards/streamlit_app.py \
		--server.port 8501 --server.address 0.0.0.0

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# ── Notebooks ─────────────────────────────────────────────────
notebook:
	jupyter lab notebooks/

run-notebooks:
	$(PYTHON) notebooks/01_eda_mobility_patterns.py
	$(PYTHON) notebooks/02_feature_engineering.py
	$(PYTHON) notebooks/03_clustering_experiments.py
	$(PYTHON) notebooks/04_segment_profiling.py
	@echo "✅ All notebooks complete. Charts in docs/"

# ── Testing ───────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ── Code quality ──────────────────────────────────────────────
lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503 || true
	black src/ --check --line-length=100 || true

format:
	black src/ --line-length=100
	isort src/ --profile=black

# ── Docker ────────────────────────────────────────────────────
docker-up:
	cd mlops/docker && docker-compose up --build -d
	@echo "API       : http://localhost:8000"
	@echo "Dashboard : http://localhost:8501"
	@echo "MLflow    : http://localhost:5000"

docker-down:
	cd mlops/docker && docker-compose down

# ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/
	@echo "✅ Cleaned."

clean-data:
	rm -f data/raw/*.parquet data/raw/*.csv
	rm -f data/processed/*.parquet data/processed/*.geojson data/processed/*.html data/processed/*.txt
	rm -f data/models/*.pkl data/models/*.png
	@echo "✅ Data removed. Run: make pipeline to regenerate."
