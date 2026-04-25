"""
src/features/feature_pipeline.py
──────────────────────────────────
Orchestrates the complete feature engineering pipeline:
  1. Load raw subscriber data
  2. Run mobility feature engineering
  3. Run temporal usage feature engineering
  4. Run POI affinity feature engineering
  5. Optionally enrich with real OSM POI density
  6. Scale features for clustering
  7. Save processed feature matrix

Usage:
    python src/features/feature_pipeline.py
    python src/features/feature_pipeline.py --input data/raw/ --output data/processed/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.mobility_features import MobilityFeatureEngineer, TemporalUsageEngineer
from src.features.poi_features import POIFeatureEngineer, add_h3_poi_density


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_segment_params(path: str = "configs/segment_params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_scaler(scaler_name: str):
    """Return the configured scaler object."""
    return {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "robust":   RobustScaler(),
    }.get(scaler_name, StandardScaler())


def run_pipeline(
    config: dict,
    seg_params: dict,
    input_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    logger.info("Loading raw subscriber data...")
    df = pd.read_parquet(input_dir / "subscribers.parquet")
    logger.info(f"  {len(df):,} subscribers, {len(df.columns)} columns")

    # Load optional OSM POI data
    pois_df = None
    poi_paths = list(Path("data/external").glob("osm_pois_*.parquet"))
    if poi_paths:
        logger.info(f"Loading OSM POI data: {poi_paths[0]}")
        pois_df = pd.read_parquet(poi_paths[0])
        logger.info(f"  {len(pois_df):,} POIs loaded")

    # ── 2. Mobility features ──────────────────────────────────────────────────
    logger.info("Step 2: Mobility feature engineering...")
    mob_eng = MobilityFeatureEngineer()
    df = mob_eng.fit_transform(df)

    # ── 3. Temporal usage features ────────────────────────────────────────────
    logger.info("Step 3: Temporal usage feature engineering...")
    temp_eng = TemporalUsageEngineer()
    df = temp_eng.fit_transform(df)

    # ── 4. POI affinity features ──────────────────────────────────────────────
    logger.info("Step 4: POI affinity feature engineering...")
    poi_eng = POIFeatureEngineer()
    df = poi_eng.fit_transform(df)

    # ── 5. OSM POI density enrichment (if data available) ────────────────────
    h3_res = config["data_generation"]["h3_resolution"]
    if pois_df is not None:
        logger.info("Step 5: Adding OSM H3 POI density features...")
        df = add_h3_poi_density(df, pois_df, h3_col="home_h3", resolution=h3_res)
    else:
        logger.info("Step 5: OSM data not found — skipping density enrichment")
        logger.info("  Run: python src/data_engineering/ingest_osm.py --city Jakarta")

    # ── 6. Select clustering features ─────────────────────────────────────────
    logger.info("Step 6: Selecting and scaling clustering features...")
    all_cluster_feats = (
        seg_params["clustering_features"]["mobility"] +
        seg_params["clustering_features"]["usage_volume"] +
        seg_params["clustering_features"]["usage_temporal"] +
        seg_params["clustering_features"]["poi_affinity"] +
        seg_params["clustering_features"]["network"]
    )

    # Add engineered features that are in the dataframe
    engineered_extras = [
        "home_work_ratio", "mobility_ratio", "commute_intensity",
        "location_concentration", "business_leisure_ratio",
        "poi_work_affinity", "poi_leisure_affinity", "poi_stay_affinity",
        "poi_diversity_score", "gb_per_dollar",
    ]

    cluster_feats = [f for f in all_cluster_feats + engineered_extras
                     if f in df.columns]

    # Remove duplicates while preserving order
    seen = set()
    cluster_feats = [f for f in cluster_feats
                     if not (f in seen or seen.add(f))]

    logger.info(f"  Clustering features: {len(cluster_feats)}")

    X_cluster = df[cluster_feats].fillna(df[cluster_feats].median())

    # Scale
    scaler_name = config["segmentation"].get("scaler", "standard")
    scaler = get_scaler(scaler_name)
    X_scaled = scaler.fit_transform(X_cluster)
    X_scaled_df = pd.DataFrame(X_scaled, columns=cluster_feats, index=df.index)

    # ── 7. Save outputs ────────────────────────────────────────────────────────
    # Full feature matrix (all engineered features)
    features_path = output_dir / "features.parquet"
    df.to_parquet(features_path, index=False)
    logger.success(f"Full features → {features_path}")

    # Scaled clustering matrix
    scaled_path = output_dir / "features_scaled.parquet"
    X_scaled_df.to_parquet(scaled_path, index=False)
    logger.success(f"Scaled features → {scaled_path}")

    # Save scaler and feature list for inference
    models_dir = Path("data/models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")
    joblib.dump(cluster_feats, models_dir / "cluster_feature_list.pkl")
    joblib.dump({"mob_eng": mob_eng, "temp_eng": temp_eng, "poi_eng": poi_eng},
                models_dir / "feature_transformers.pkl")

    # Save cluster feature list as text
    feat_txt = output_dir / "cluster_features.txt"
    feat_txt.write_text("\n".join(cluster_feats))

    print("\n" + "=" * 55)
    print("FEATURE PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Subscribers    : {len(df):,}")
    print(f"  Total features : {len(df.columns)}")
    print(f"  Cluster feats  : {len(cluster_feats)}")
    print(f"  Scaler         : {scaler_name}")
    print(f"  Output dir     : {output_dir.resolve()}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--input",  default="data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--params", default="configs/segment_params.yaml")
    args = parser.parse_args()

    config     = load_config(args.config)
    seg_params = load_segment_params(args.params)
    run_pipeline(config, seg_params, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
