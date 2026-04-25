# Architecture Documentation

## System Overview

End-to-end ML platform for geo-behavioral customer segmentation. Uses CDR-derived mobility features to cluster subscribers into personas and deliver personalized plan recommendations.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐   │
│  │ CDR / GPS Data  │   │  OpenCelliD     │   │  OpenStreetMap      │   │
│  │ (mobility logs) │   │  (cell towers)  │   │  POIs (free API)    │   │
│  └────────┬────────┘   └────────┬────────┘   └────────────┬────────┘   │
└───────────┼─────────────────────┼─────────────────────────┼────────────┘
            │                     │                         │
            ▼                     ▼                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                                   │
│  mobility_features.py → home/work detection, travel radius, entropy     │
│  usage_features.py    → temporal patterns, time-of-day ratios           │
│  poi_features.py      → POI affinity scores, dominant POI type          │
│  feature_pipeline.py  → orchestrates all, outputs scaled matrix         │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    CLUSTERING LAYER (MLflow tracked)                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  clustering.py                                                   │    │
│  │  • KMeans (default)  — fast, interpretable, optimal K via Optuna│    │
│  │  • DBSCAN             — density-based, noise-aware               │    │
│  │  • HDBSCAN            — hierarchical, soft membership            │    │
│  │  Evaluation: Silhouette + Davies-Bouldin + Calinski-Harabasz     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  segment_profiler.py → auto-names clusters via archetype z-score match  │
│  geo_density.py      → H3 hex map of segment spatial distribution       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
┌────────────────────────────┐   ┌────────────────────────────────────────┐
│  Recommendation Engine     │   │  MLflow Model Registry                 │
│  recommender.py            │   │  ──────────────────────                │
│  • Rule-based (segment map)│   │  None → Staging → Production           │
│  • Implicit ALS (CF)       │   │  Silhouette ≥ 0.30 to auto-promote    │
│  • Hybrid blend            │   └────────────────────────────────────────┘
└────────────┬───────────────┘
             │
    ┌────────┴─────────────┐
    ▼                      ▼
┌──────────────┐  ┌────────────────────────┐
│ FastAPI REST │  │  Streamlit Dashboard   │
│ :8000        │  │  :8501                 │
│              │  │  • Segment overview    │
│ /segment     │  │  • Folium geo map      │
│ /recommend   │  │  • Radar profiles      │
│ /geo-density │  │  • Recommender UI      │
│ /segments    │  │  • Live classifier     │
└──────────────┘  └────────────────────────┘

━━━━━━━━━━━━━━━━━ MLOPS ━━━━━━━━━━━━━━━━━━

Airflow DAG (weekly):
  validate → features → cluster → profile → recommend → geo → drift_check
                                                              ↓
                                                         promote / alert
```

## Technology Decisions

| Choice | Rationale |
|---|---|
| **KMeans over DBSCAN** | For behavioral data with overlapping clusters, KMeans produces more balanced, interpretable segments. DBSCAN finds noise-robust clusters but often unbalanced. |
| **H3 resolution 8** | ~0.74 km² — city-block level. Fine enough for neighbourhood targeting, coarse enough for ≥5 subscribers/cell. |
| **UMAP over t-SNE** | UMAP preserves global structure better than t-SNE and is significantly faster on 30k subscribers. |
| **Implicit ALS** | Industry-standard for implicit feedback (no explicit ratings). Suitable for learning from CDR interaction patterns. |
| **Archetype matching** | Rule-based name assignment via z-score direction matching is reproducible and interpretable — no "black box" naming. |

## Data Flow

```
Raw CDR data (per subscriber monthly summary)
        │
        ▼
Feature engineering (3 transformers in sequence)
  1. Mobility → home/work ratios, commute flags
  2. Temporal → time-of-day patterns, business/leisure split
  3. POI → composite affinity scores
        │
        ▼
Standard scaling → clustering feature matrix (35 features)
        │
        ▼
KMeans clustering → segment_id per subscriber
        │
        ▼
Segment profiler → auto-name via archetype matching
        │
   ┌────┴──────┐
   ▼           ▼
API          Geo density map (H3 GeoJSON)
```
