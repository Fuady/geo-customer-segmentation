"""
src/data_engineering/generate_data.py
──────────────────────────────────────
Generates a realistic synthetic telecom subscriber dataset simulating
CDR-derived mobility and usage features for 30,000 subscribers.

Simulates 7 behavioral personas:
  0. Urban Commuter       — long daily commute, peak-hour data
  1. Home Office Pro      — stable home zone, heavy daytime data
  2. Digital Nomad        — wide travel radius, roaming, multi-location
  3. Weekend Explorer     — low weekday mobility, high weekend travel
  4. Social Hub           — dense POI visits (malls/cafes), evening usage
  5. Senior Steady        — minimal mobility, high voice, low data
  6. Young Professional   — business + entertainment mix, moderate mobility

Usage:
    python src/data_engineering/generate_data.py
    python src/data_engineering/generate_data.py --n_subscribers 50000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Persona definitions ────────────────────────────────────────────────────────
# Each persona has distributions for every feature.
# Format: (mean, std) or list of choices with probabilities.
PERSONAS = {
    0: {
        "name": "Urban Commuter",
        "weight": 0.20,
        "avg_daily_distance_km":  (22, 6),
        "travel_radius_km":       (18, 5),
        "home_work_distance_km":  (15, 5),
        "n_frequent_locations":   (3, 1),
        "pct_time_at_home":       (0.52, 0.07),
        "pct_time_at_work":       (0.28, 0.06),
        "mobility_entropy":       (1.8, 0.3),
        "data_usage_gb":          (16, 5),
        "call_minutes_monthly":   (280, 80),
        "sms_monthly":            (60, 30),
        "roaming_days_monthly":   (0.5, 1.0),
        "morning_data_ratio":     (0.28, 0.05),
        "afternoon_data_ratio":   (0.18, 0.04),
        "evening_data_ratio":     (0.35, 0.06),
        "night_data_ratio":       (0.19, 0.04),
        "weekend_data_ratio":     (0.30, 0.05),
        "peak_hour_data_ratio":   (0.45, 0.07),
        "poi_office_score":       (0.55, 0.12),
        "poi_transit_score":      (0.75, 0.10),
        "poi_mall_score":         (0.30, 0.10),
        "poi_residential_score":  (0.40, 0.10),
        "poi_entertainment_score":(0.25, 0.10),
        "home_zone_rsrq":         (-11, 2),
        "work_zone_rsrq":         (-12, 3),
        "avg_throughput_mbps":    (28, 8),
        "monthly_charges":        (75, 15),
        "tenure_months":          (36, 18),
        "senior_citizen":         0.05,
        "internet_service":       "fiber_optic",
    },
    1: {
        "name": "Home Office Pro",
        "weight": 0.18,
        "avg_daily_distance_km":  (4, 2),
        "travel_radius_km":       (5, 2),
        "home_work_distance_km":  (1.5, 1),
        "n_frequent_locations":   (2, 1),
        "pct_time_at_home":       (0.75, 0.06),
        "pct_time_at_work":       (0.10, 0.04),
        "mobility_entropy":       (0.8, 0.2),
        "data_usage_gb":          (45, 12),
        "call_minutes_monthly":   (380, 100),
        "sms_monthly":            (40, 25),
        "roaming_days_monthly":   (0.2, 0.5),
        "morning_data_ratio":     (0.25, 0.05),
        "afternoon_data_ratio":   (0.40, 0.06),
        "evening_data_ratio":     (0.22, 0.05),
        "night_data_ratio":       (0.13, 0.04),
        "weekend_data_ratio":     (0.26, 0.05),
        "peak_hour_data_ratio":   (0.30, 0.06),
        "poi_office_score":       (0.15, 0.08),
        "poi_transit_score":      (0.10, 0.06),
        "poi_mall_score":         (0.20, 0.08),
        "poi_residential_score":  (0.85, 0.08),
        "poi_entertainment_score":(0.20, 0.08),
        "home_zone_rsrq":         (-9, 2),
        "work_zone_rsrq":         (-9, 2),
        "avg_throughput_mbps":    (55, 15),
        "monthly_charges":        (68, 12),
        "tenure_months":          (48, 20),
        "senior_citizen":         0.08,
        "internet_service":       "fiber_optic",
    },
    2: {
        "name": "Digital Nomad",
        "weight": 0.10,
        "avg_daily_distance_km":  (40, 15),
        "travel_radius_km":       (60, 25),
        "home_work_distance_km":  (30, 15),
        "n_frequent_locations":   (7, 2),
        "pct_time_at_home":       (0.30, 0.08),
        "pct_time_at_work":       (0.15, 0.06),
        "mobility_entropy":       (2.5, 0.3),
        "data_usage_gb":          (28, 10),
        "call_minutes_monthly":   (250, 80),
        "sms_monthly":            (80, 40),
        "roaming_days_monthly":   (8, 4),
        "morning_data_ratio":     (0.22, 0.06),
        "afternoon_data_ratio":   (0.28, 0.06),
        "evening_data_ratio":     (0.28, 0.06),
        "night_data_ratio":       (0.22, 0.05),
        "weekend_data_ratio":     (0.40, 0.07),
        "peak_hour_data_ratio":   (0.28, 0.06),
        "poi_office_score":       (0.35, 0.12),
        "poi_transit_score":      (0.65, 0.12),
        "poi_mall_score":         (0.40, 0.12),
        "poi_residential_score":  (0.20, 0.08),
        "poi_entertainment_score":(0.50, 0.12),
        "home_zone_rsrq":         (-13, 3),
        "work_zone_rsrq":         (-14, 3),
        "avg_throughput_mbps":    (20, 8),
        "monthly_charges":        (90, 18),
        "tenure_months":          (28, 16),
        "senior_citizen":         0.02,
        "internet_service":       "DSL",
    },
    3: {
        "name": "Weekend Explorer",
        "weight": 0.14,
        "avg_daily_distance_km":  (8, 3),
        "travel_radius_km":       (35, 12),
        "home_work_distance_km":  (6, 3),
        "n_frequent_locations":   (4, 1),
        "pct_time_at_home":       (0.60, 0.07),
        "pct_time_at_work":       (0.20, 0.05),
        "mobility_entropy":       (1.5, 0.3),
        "data_usage_gb":          (12, 4),
        "call_minutes_monthly":   (200, 60),
        "sms_monthly":            (70, 35),
        "roaming_days_monthly":   (2, 2),
        "morning_data_ratio":     (0.20, 0.05),
        "afternoon_data_ratio":   (0.20, 0.05),
        "evening_data_ratio":     (0.28, 0.06),
        "night_data_ratio":       (0.18, 0.05),
        "weekend_data_ratio":     (0.55, 0.08),
        "peak_hour_data_ratio":   (0.25, 0.06),
        "poi_office_score":       (0.30, 0.10),
        "poi_transit_score":      (0.25, 0.10),
        "poi_mall_score":         (0.45, 0.12),
        "poi_residential_score":  (0.55, 0.10),
        "poi_entertainment_score":(0.55, 0.12),
        "home_zone_rsrq":         (-11, 2),
        "work_zone_rsrq":         (-12, 2),
        "avg_throughput_mbps":    (30, 8),
        "monthly_charges":        (52, 10),
        "tenure_months":          (32, 18),
        "senior_citizen":         0.10,
        "internet_service":       "DSL",
    },
    4: {
        "name": "Social Hub",
        "weight": 0.15,
        "avg_daily_distance_km":  (10, 4),
        "travel_radius_km":       (12, 4),
        "home_work_distance_km":  (8, 3),
        "n_frequent_locations":   (5, 2),
        "pct_time_at_home":       (0.45, 0.07),
        "pct_time_at_work":       (0.22, 0.05),
        "mobility_entropy":       (1.9, 0.3),
        "data_usage_gb":          (22, 7),
        "call_minutes_monthly":   (320, 80),
        "sms_monthly":            (150, 50),
        "roaming_days_monthly":   (1, 2),
        "morning_data_ratio":     (0.15, 0.04),
        "afternoon_data_ratio":   (0.22, 0.05),
        "evening_data_ratio":     (0.42, 0.07),
        "night_data_ratio":       (0.21, 0.05),
        "weekend_data_ratio":     (0.45, 0.07),
        "peak_hour_data_ratio":   (0.35, 0.06),
        "poi_office_score":       (0.20, 0.08),
        "poi_transit_score":      (0.30, 0.10),
        "poi_mall_score":         (0.80, 0.10),
        "poi_residential_score":  (0.40, 0.10),
        "poi_entertainment_score":(0.75, 0.10),
        "home_zone_rsrq":         (-10, 2),
        "work_zone_rsrq":         (-11, 2),
        "avg_throughput_mbps":    (35, 10),
        "monthly_charges":        (48, 12),
        "tenure_months":          (22, 14),
        "senior_citizen":         0.03,
        "internet_service":       "fiber_optic",
    },
    5: {
        "name": "Senior Steady",
        "weight": 0.11,
        "avg_daily_distance_km":  (3, 1.5),
        "travel_radius_km":       (4, 2),
        "home_work_distance_km":  (2, 1),
        "n_frequent_locations":   (2, 1),
        "pct_time_at_home":       (0.80, 0.06),
        "pct_time_at_work":       (0.05, 0.03),
        "mobility_entropy":       (0.5, 0.2),
        "data_usage_gb":          (3, 2),
        "call_minutes_monthly":   (550, 120),
        "sms_monthly":            (20, 15),
        "roaming_days_monthly":   (0.2, 0.5),
        "morning_data_ratio":     (0.30, 0.06),
        "afternoon_data_ratio":   (0.35, 0.06),
        "evening_data_ratio":     (0.25, 0.05),
        "night_data_ratio":       (0.10, 0.04),
        "weekend_data_ratio":     (0.28, 0.05),
        "peak_hour_data_ratio":   (0.20, 0.05),
        "poi_office_score":       (0.05, 0.04),
        "poi_transit_score":      (0.08, 0.05),
        "poi_mall_score":         (0.15, 0.07),
        "poi_residential_score":  (0.90, 0.06),
        "poi_entertainment_score":(0.10, 0.06),
        "home_zone_rsrq":         (-10, 2),
        "work_zone_rsrq":         (-10, 2),
        "avg_throughput_mbps":    (22, 8),
        "monthly_charges":        (35, 8),
        "tenure_months":          (68, 24),
        "senior_citizen":         0.75,
        "internet_service":       "DSL",
    },
    6: {
        "name": "Young Professional",
        "weight": 0.12,
        "avg_daily_distance_km":  (14, 5),
        "travel_radius_km":       (20, 6),
        "home_work_distance_km":  (12, 4),
        "n_frequent_locations":   (4, 1),
        "pct_time_at_home":       (0.55, 0.06),
        "pct_time_at_work":       (0.25, 0.05),
        "mobility_entropy":       (1.7, 0.3),
        "data_usage_gb":          (30, 8),
        "call_minutes_monthly":   (300, 80),
        "sms_monthly":            (100, 40),
        "roaming_days_monthly":   (1.5, 2),
        "morning_data_ratio":     (0.20, 0.05),
        "afternoon_data_ratio":   (0.22, 0.05),
        "evening_data_ratio":     (0.38, 0.06),
        "night_data_ratio":       (0.20, 0.05),
        "weekend_data_ratio":     (0.38, 0.06),
        "peak_hour_data_ratio":   (0.38, 0.07),
        "poi_office_score":       (0.60, 0.12),
        "poi_transit_score":      (0.45, 0.10),
        "poi_mall_score":         (0.55, 0.12),
        "poi_residential_score":  (0.45, 0.10),
        "poi_entertainment_score":(0.60, 0.12),
        "home_zone_rsrq":         (-10, 2),
        "work_zone_rsrq":         (-11, 2),
        "avg_throughput_mbps":    (38, 10),
        "monthly_charges":        (62, 12),
        "tenure_months":          (24, 12),
        "senior_citizen":         0.01,
        "internet_service":       "fiber_optic",
    },
}

NUMERIC_FEATURES = [
    "avg_daily_distance_km", "travel_radius_km", "home_work_distance_km",
    "n_frequent_locations", "pct_time_at_home", "pct_time_at_work",
    "mobility_entropy", "data_usage_gb", "call_minutes_monthly", "sms_monthly",
    "roaming_days_monthly", "morning_data_ratio", "afternoon_data_ratio",
    "evening_data_ratio", "night_data_ratio", "weekend_data_ratio",
    "peak_hour_data_ratio", "poi_office_score", "poi_transit_score",
    "poi_mall_score", "poi_residential_score", "poi_entertainment_score",
    "home_zone_rsrq", "work_zone_rsrq", "avg_throughput_mbps",
    "monthly_charges", "tenure_months",
]


def generate_subscriber_from_persona(
    persona_id: int,
    persona: dict,
    rng: np.random.Generator,
) -> dict:
    """Sample a single subscriber's features from a persona distribution."""
    row = {"true_persona_id": persona_id, "true_persona_name": persona["name"]}

    for feat in NUMERIC_FEATURES:
        if feat in persona:
            mu, sigma = persona[feat]
            val = rng.normal(mu, sigma)
            # Apply sensible clipping
            clips = {
                "pct_time_at_home":   (0.0, 1.0),
                "pct_time_at_work":   (0.0, 1.0),
                "morning_data_ratio": (0.0, 1.0),
                "afternoon_data_ratio":(0.0, 1.0),
                "evening_data_ratio": (0.0, 1.0),
                "night_data_ratio":   (0.0, 1.0),
                "weekend_data_ratio": (0.0, 1.0),
                "peak_hour_data_ratio":(0.0, 1.0),
                "poi_office_score":   (0.0, 1.0),
                "poi_transit_score":  (0.0, 1.0),
                "poi_mall_score":     (0.0, 1.0),
                "poi_residential_score":(0.0, 1.0),
                "poi_entertainment_score":(0.0, 1.0),
                "data_usage_gb":      (0.1, 200.0),
                "call_minutes_monthly":(0.0, 2000.0),
                "sms_monthly":        (0.0, 1000.0),
                "roaming_days_monthly":(0.0, 30.0),
                "n_frequent_locations":(1.0, 15.0),
                "avg_daily_distance_km":(0.0, 200.0),
                "travel_radius_km":   (0.5, 300.0),
                "home_work_distance_km":(0.0, 100.0),
                "mobility_entropy":   (0.0, 3.5),
                "home_zone_rsrq":     (-25.0, -3.0),
                "work_zone_rsrq":     (-25.0, -3.0),
                "avg_throughput_mbps":(0.5, 200.0),
                "monthly_charges":    (10.0, 300.0),
                "tenure_months":      (1.0, 120.0),
            }
            lo, hi = clips.get(feat, (-1e9, 1e9))
            row[feat] = float(np.clip(val, lo, hi))

    # Categorical features
    row["senior_citizen"] = int(rng.random() < persona["senior_citizen"])
    row["internet_service"] = persona["internet_service"]

    return row


def add_geography(
    df: pd.DataFrame,
    geo_bounds: dict,
    h3_res: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Assign lat/lon home and work coordinates, plus H3 cells."""
    n = len(df)

    # Create clustered locations (realistic urban clusters)
    n_clusters = 15
    cluster_lats = rng.uniform(geo_bounds["lat_min"], geo_bounds["lat_max"], n_clusters)
    cluster_lons = rng.uniform(geo_bounds["lon_min"], geo_bounds["lon_max"], n_clusters)
    weights = rng.dirichlet(np.ones(n_clusters) * 3)
    assigned = rng.choice(n_clusters, n, p=weights)

    home_lats = np.clip(
        rng.normal(cluster_lats[assigned], 0.04),
        geo_bounds["lat_min"], geo_bounds["lat_max"]
    )
    home_lons = np.clip(
        rng.normal(cluster_lons[assigned], 0.04),
        geo_bounds["lon_min"], geo_bounds["lon_max"]
    )

    # Work location: distance from home based on home_work_distance_km
    angle = rng.uniform(0, 2 * np.pi, n)
    dist_deg = df["home_work_distance_km"].values / 111.0   # approx km→degrees
    work_lats = np.clip(home_lats + dist_deg * np.sin(angle),
                        geo_bounds["lat_min"], geo_bounds["lat_max"])
    work_lons = np.clip(home_lons + dist_deg * np.cos(angle),
                        geo_bounds["lon_min"], geo_bounds["lon_max"])

    df = df.copy()
    df["home_lat"] = home_lats.round(6)
    df["home_lon"] = home_lons.round(6)
    df["work_lat"] = work_lats.round(6)
    df["work_lon"] = work_lons.round(6)

    # H3 indexing
    try:
        import h3
        df["home_h3"] = df.apply(
            lambda r: h3.geo_to_h3(r["home_lat"], r["home_lon"], h3_res), axis=1
        )
        df["work_h3"] = df.apply(
            lambda r: h3.geo_to_h3(r["work_lat"], r["work_lon"], h3_res), axis=1
        )
        df["same_h3_zone"] = (df["home_h3"] == df["work_h3"]).astype(int)
    except ImportError:
        df["home_h3"] = "h3_unavailable"
        df["work_h3"] = "h3_unavailable"
        df["same_h3_zone"] = 0

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CDR subscriber dataset")
    parser.add_argument("--n_subscribers", type=int, default=30000)
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed or config["data_generation"]["random_seed"]
    rng = np.random.default_rng(seed)
    geo_bounds = config["data_generation"]["geo_bounds"]
    h3_res = config["data_generation"]["h3_resolution"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_subscribers
    logger.info(f"Generating {n:,} subscribers across {len(PERSONAS)} personas...")

    # Assign persona IDs based on weights
    persona_ids = list(PERSONAS.keys())
    persona_weights = np.array([PERSONAS[p]["weight"] for p in persona_ids])
    persona_weights /= persona_weights.sum()
    assigned_personas = rng.choice(persona_ids, size=n, p=persona_weights)

    rows = []
    for i, pid in enumerate(assigned_personas):
        row = generate_subscriber_from_persona(pid, PERSONAS[pid], rng)
        row["subscriber_id"] = f"SUB_{i+1:08d}"
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add geography
    logger.info("Adding geographic coordinates and H3 indexing...")
    df = add_geography(df, geo_bounds, h3_res, rng)

    # Reorder columns
    front_cols = ["subscriber_id", "true_persona_id", "true_persona_name",
                  "home_lat", "home_lon", "work_lat", "work_lon", "home_h3", "work_h3"]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    # Save
    out_parquet = output_dir / "subscribers.parquet"
    out_csv_sample = output_dir / "subscribers_sample.csv"
    df.to_parquet(out_parquet, index=False)
    df.head(2000).to_csv(out_csv_sample, index=False)

    logger.success(f"Saved {len(df):,} subscribers → {out_parquet}")
    logger.info(f"CSV sample (2000 rows) → {out_csv_sample}")

    # Persona summary
    print("\n" + "=" * 65)
    print("DATASET GENERATION COMPLETE")
    print("=" * 65)
    summary = df.groupby(["true_persona_id", "true_persona_name"]).size().reset_index()
    summary.columns = ["ID", "Persona", "Count"]
    summary["Pct"] = (summary["Count"] / n * 100).round(1)
    for _, row in summary.iterrows():
        print(f"  [{row['ID']}] {row['Persona']:22s}: {row['Count']:5,} ({row['Pct']:.1f}%)")
    print(f"\n  Total subscribers  : {n:,}")
    print(f"  Features per row   : {len(df.columns)}")
    print(f"  Output             : {out_parquet.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
