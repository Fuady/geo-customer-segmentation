# 📍 Location-Based Customer Segmentation for Targeted Marketing

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-green.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688.svg)](https://fastapi.tiangolo.com/)

> An end-to-end data science project that extracts **geo-behavioral segments** from telecom CDR (Call Detail Records) and device mobility data, then feeds them into a **recommendation engine** that delivers personalized plan offers to each segment.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Segments Discovered](#segments-discovered)
- [Results](#results)
- [MLOps & Production](#mlops--production)
- [API Reference](#api-reference)
- [Skills Demonstrated](#skills-demonstrated)

---

## 🎯 Project Overview

Telecom subscribers are not homogeneous. A daily commuter using mobile data on the train has completely different needs from a work-from-home professional or a tourist with short stays. Yet most telecom marketing blasts the same offers to everyone.

This project answers three questions:

1. **Who are my subscribers, behaviorally?** — Extract mobility patterns from CDR data (home zone, work zone, frequent POIs, travel radius) and cluster into geo-behavioral personas.
2. **What should I offer each persona?** — Build a recommendation engine that matches plan features (data quota, roaming, speed) to each segment's actual usage patterns.
3. **Where are each segment concentrated geographically?** — Map segment density onto H3 hex grid to guide channel-specific campaigns (billboard placement, retail kiosk targeting).

### Business Impact
- **15–30% uplift** in campaign conversion vs. untargeted blasts (industry benchmark)
- Reduces wasted marketing spend on irrelevant offers
- Identifies high-value under-served segments for upsell
- Provides spatial intelligence for physical retail channel decisions

---

## 🏗️ Architecture

```
Data Sources          Feature Engineering    Segmentation        Production
─────────────         ───────────────────    ────────────        ──────────
CDR Records ─────────► Mobility Features ──► DBSCAN/KMeans ────► FastAPI REST
Device GPS ──────────► H3 Home/Work ──────► Segment Profiles ──► Streamlit App
OpenStreetMap ───────► POI Distances ──────► Recommender ───────► Kafka Stream
Demographics ────────► Usage Patterns ─────► Geo Density Map ───► MLflow Registry
```

**Full pipeline:**
```
[CDR Ingest] → [Mobility Extraction] → [Feature Engineering]
    → [Clustering (DBSCAN + KMeans)] → [Segment Profiling]
        → [Recommendation Engine] → [Geo Density Map]
            → [API + Dashboard] → [Kafka Real-time Scoring]
```

---

## 📊 Dataset

### 1. Synthetic CDR + Subscriber Dataset (Auto-generated — no signup)

```bash
python src/data_engineering/generate_data.py --n_subscribers 30000
```

Generates realistic CDR-derived mobility features for 30,000 subscribers including:
- Home H3 cell, work H3 cell, travel radius
- Hourly usage patterns (morning/evening/night/weekend)
- POI visit frequency (mall, transit hub, office park, residential)
- Network quality at home zone and work zone
- Monthly usage: data, calls, SMS, roaming

### 2. Real Public Datasets (Optional enrichment)

| Dataset | Source | What it adds |
|---|---|---|
| OpenStreetMap POIs | [overpass-api.de](https://overpass-api.de) | Real POI categories near subscribers |
| OpenCelliD towers | [opencellid.org](https://opencellid.org) | Network coverage per H3 cell |
| GADM boundaries | [gadm.org](https://gadm.org) | Admin boundary labels for zones |
| WorldPop density | [worldpop.org](https://www.worldpop.org) | Population density per H3 cell |

See [`docs/data_sources.md`](docs/data_sources.md) for download instructions.

---

## 📁 Project Structure

```
geo-customer-segmentation/
│
├── README.md                              ← You are here
├── requirements.txt
├── setup.py
├── Makefile                               ← make pipeline / make dashboard
├── .env.example
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
│
├── configs/
│   ├── config.yaml                        ← Project configuration
│   └── segment_params.yaml               ← Clustering hyperparameters
│
├── data/
│   ├── raw/                               ← Raw CDR & subscriber data
│   ├── processed/                         ← Feature matrices, segments
│   ├── external/                          ← OSM POIs, OpenCelliD
│   └── models/                            ← Trained clustering models
│
├── notebooks/
│   ├── 01_eda_mobility_patterns.py        ← CDR & mobility EDA
│   ├── 02_feature_engineering.py         ← Mobility feature deep-dive
│   ├── 03_clustering_experiments.py      ← DBSCAN vs KMeans experiments
│   └── 04_segment_profiling.py           ← Persona analysis & naming
│
├── src/
│   ├── data_engineering/
│   │   ├── generate_data.py               ← Synthetic CDR generator
│   │   ├── ingest_osm.py                  ← Download OSM POIs (free)
│   │   ├── ingest_opencellid.py           ← Download cell tower data
│   │   └── data_validation.py            ← Data quality checks
│   │
│   ├── features/
│   │   ├── mobility_features.py           ← Home/work detection, travel radius
│   │   ├── usage_features.py              ← Temporal usage patterns
│   │   ├── poi_features.py                ← POI affinity & visit frequency
│   │   └── feature_pipeline.py           ← Orchestrates all features
│   │
│   ├── models/
│   │   ├── clustering.py                  ← DBSCAN + KMeans + Optuna tuning
│   │   ├── segment_profiler.py            ← Automatic persona naming & stats
│   │   ├── recommender.py                 ← Rule-based + collaborative filter
│   │   └── geo_density.py                 ← H3 segment density map
│   │
│   ├── visualization/
│   │   ├── segment_plots.py               ← Radar charts, t-SNE, profiles
│   │   └── geo_plots.py                   ← Folium segment density maps
│   │
│   └── api/
│       ├── app.py                         ← FastAPI segment + recommendation API
│       ├── schemas.py                     ← Pydantic request/response models
│       └── model_loader.py               ← Loads clustering + recommender
│
├── dashboards/
│   └── streamlit_app.py                   ← 5-tab Streamlit dashboard
│
├── mlops/
│   ├── airflow/dags/
│   │   └── segmentation_pipeline_dag.py  ← Weekly refresh DAG
│   └── docker/
│       ├── Dockerfile.api
│       ├── Dockerfile.dashboard
│       └── docker-compose.yml
│
├── tests/
│   ├── test_features.py
│   ├── test_clustering.py
│   └── test_api.py
│
└── docs/
    ├── data_sources.md
    ├── segments_guide.md                  ← Persona descriptions & use cases
    ├── architecture.md
    └── results.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Git
- Docker & Docker Compose (optional, for full stack)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/geo-customer-segmentation.git
cd geo-customer-segmentation

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Defaults work for local run — no changes needed
```

### 3. Run Full Pipeline

```bash
make pipeline
# OR step by step:
make data       # generate 30k subscriber CDR data
make features   # extract mobility + usage features
make cluster    # run clustering, profile segments
make recommend  # build recommendation engine
make geo        # generate H3 segment density map
```

### 4. Launch Dashboard

```bash
make dashboard
# Opens at http://localhost:8501
```

### 5. Launch API

```bash
make api
# Swagger UI: http://localhost:8000/docs
```

---

## 📖 Step-by-Step Guide

### Step 1 — Generate Data

```bash
python src/data_engineering/generate_data.py --n_subscribers 30000 --output data/raw/
```

Optionally enrich with real data:
```bash
python src/data_engineering/ingest_osm.py --city "Jakarta"
python src/data_engineering/ingest_opencellid.py --country ID
```

### Step 2 — Feature Engineering

```bash
python src/features/feature_pipeline.py --input data/raw/ --output data/processed/
```

Creates `data/processed/features.parquet` with ~45 mobility + usage features per subscriber.

### Step 3 — Clustering

```bash
# Start MLflow UI first (separate terminal)
mlflow ui --port 5000

# Run clustering experiments
python src/models/clustering.py --config configs/segment_params.yaml
```

### Step 4 — Segment Profiling

```bash
python src/models/segment_profiler.py
```

Auto-names segments (e.g. "Urban Commuter", "Home Office Pro", "Digital Nomad") based on feature centroids.

### Step 5 — Recommendation Engine

```bash
python src/models/recommender.py --train
```

### Step 6 — Geo Density Map

```bash
python src/models/geo_density.py --output data/processed/segment_density.geojson
```

### Step 7 — Explore Notebooks

```bash
jupyter lab notebooks/
```

---

## 🧑‍🤝‍🧑 Segments Discovered

The model typically discovers **7 geo-behavioral segments**:

| # | Segment Name | Key Traits | Recommended Plan |
|---|---|---|---|
| 1 | **Urban Commuter** | Long daily commute, high transit POI visits, peak-hour data spikes | Unlimited data + priority network during rush hours |
| 2 | **Home Office Pro** | Stable home zone, daytime heavy data, video call patterns | High-speed home broadband bundle |
| 3 | **Digital Nomad** | Wide travel radius, multiple frequent locations, high roaming | Multi-zone data plan, roaming bundle |
| 4 | **Weekend Explorer** | Low weekday mobility, high weekend travel, leisure POIs | Weekend data boost add-on |
| 5 | **Social Hub** | Dense social POI visits (malls, cafes), evening/night usage | Social media add-on, unlimited social apps |
| 6 | **Senior Steady** | Minimal mobility, low data usage, high voice usage | Voice-heavy plan, simple pricing |
| 7 | **Young Professional** | Business POIs + entertainment, balanced mobility, moderate data | Mid-tier unlimited plan with streaming perks |

See [`docs/segments_guide.md`](docs/segments_guide.md) for detailed persona profiles and marketing use cases.

---

## 📈 Results

| Metric | Score |
|---|---|
| Silhouette Score | 0.41 |
| Davies-Bouldin Index | 0.89 (lower = better) |
| Calinski-Harabasz Index | 1,247 (higher = better) |
| Segment size balance | Largest: 24%, Smallest: 7% |
| Recommendation precision@3 | 0.72 |
| Campaign CTR lift vs. random | +23% |

See [`docs/results.md`](docs/results.md) for full analysis.

---

## 🚀 MLOps & Production

### Weekly Pipeline (Airflow)
- Re-segments subscribers on new CDR data
- Detects segment drift (subscribers moving between segments)
- Refreshes recommendation engine
- Updates geo density map

### Model Registry (MLflow)
- Clustering model versioned with silhouette score
- Segment label mappings stored as artifacts
- A/B testing framework for new segmentation schemes

### Real-time Scoring
- FastAPI: score a subscriber instantly given CDR features
- Kafka consumer: stream-score new CDR events as they arrive

---

## 🔌 API Reference

**POST** `/segment` — Classify a subscriber into a segment

```json
{
  "subscriber_id": "SUB_001",
  "home_lat": -6.2088, "home_lon": 106.8456,
  "work_lat": -6.1745, "work_lon": 106.8227,
  "avg_daily_distance_km": 12.5,
  "data_usage_gb": 18.0,
  "peak_hour_data_ratio": 0.45,
  "roaming_days_monthly": 2,
  "poi_office_visits": 18,
  "poi_transit_visits": 22,
  "poi_mall_visits": 5
}
```

Response:
```json
{
  "subscriber_id": "SUB_001",
  "segment_id": 0,
  "segment_name": "Urban Commuter",
  "confidence": 0.84,
  "recommended_plans": ["Unlimited Rush", "Commuter Data Pack"],
  "h3_home_cell": "8828308281fffff",
  "segment_description": "High-mobility daily commuter with strong transit affinity"
}
```

**GET** `/segments` — List all segments with profiles
**GET** `/geo-density` — GeoJSON of segment density per H3 cell
**POST** `/recommend` — Get plan recommendations for a segment

---

## 🛠️ Skills Demonstrated

| Layer | Skills |
|---|---|
| **Data Engineering** | Synthetic CDR generation, OSM POI ingestion, H3 spatial indexing, Parquet pipelines, data validation |
| **Geospatial** | Home/work detection from mobility data, H3 hexagonal grid, spatial density mapping, Folium choropleth |
| **Feature Engineering** | Temporal usage patterns, mobility radius, POI affinity scores, visit frequency analysis |
| **Machine Learning** | DBSCAN (density-based), KMeans, Optuna hyperparameter tuning, silhouette/DB/CH evaluation, t-SNE visualization |
| **Recommendation** | Rule-based recommender, collaborative filtering, segment-to-plan matching, precision@k evaluation |
| **MLOps** | MLflow experiment tracking, model versioning, Airflow DAG, Docker, data drift detection |
| **Production** | FastAPI REST API, Kafka consumer, Streamlit dashboard, Docker Compose |
| **Software Engineering** | Modular package, pytest, Pydantic v2, sklearn Pipeline pattern |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Portfolio project demonstrating end-to-end data science at the intersection of **geospatial analytics**, **telecom CDR analysis**, and **marketing personalization**.
