"""
src/models/segment_profiler.py
────────────────────────────────
Automatically profiles and names each cluster by comparing
feature centroids against persona archetypes.

For each segment:
  - Computes feature means vs. population means (z-scores)
  - Identifies top distinguishing features
  - Assigns a persona name based on archetype similarity
  - Generates a plain-English description

Outputs:
  - data/processed/segment_profiles.parquet  — per-segment statistics
  - data/models/segment_names.pkl            — {segment_id: name} mapping
  - data/models/segment_descriptions.pkl    — {segment_id: description}

Usage:
    python src/models/segment_profiler.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── Archetype definitions for auto-naming ─────────────────────────────────────
# Each archetype has signature z-score directions for key features.
# (+) = above average, (−) = below average, (0) = neutral
ARCHETYPES = {
    "Urban Commuter": {
        "avg_daily_distance_km":   +1,
        "poi_transit_score":       +1,
        "home_work_distance_km":   +1,
        "peak_hour_data_ratio":    +1,
        "pct_time_at_home":        -1,
        "data_usage_gb":           +1,
    },
    "Home Office Pro": {
        "pct_time_at_home":        +1,
        "data_usage_gb":           +1,
        "avg_daily_distance_km":   -1,
        "travel_radius_km":        -1,
        "afternoon_data_ratio":    +1,
        "poi_residential_score":   +1,
    },
    "Digital Nomad": {
        "travel_radius_km":        +1,
        "roaming_days_monthly":    +1,
        "n_frequent_locations":    +1,
        "mobility_entropy":        +1,
        "pct_time_at_home":        -1,
        "poi_transit_score":       +1,
    },
    "Weekend Explorer": {
        "weekend_data_ratio":      +1,
        "avg_daily_distance_km":   -1,
        "poi_entertainment_score": +1,
        "poi_mall_score":          +1,
        "morning_data_ratio":      -1,
    },
    "Social Hub": {
        "poi_mall_score":          +1,
        "poi_entertainment_score": +1,
        "evening_data_ratio":      +1,
        "sms_monthly":             +1,
        "n_frequent_locations":    +1,
        "weekend_data_ratio":      +1,
    },
    "Senior Steady": {
        "call_minutes_monthly":    +1,
        "data_usage_gb":           -1,
        "avg_daily_distance_km":   -1,
        "pct_time_at_home":        +1,
        "morning_data_ratio":      +1,
        "poi_residential_score":   +1,
    },
    "Young Professional": {
        "poi_office_score":        +1,
        "poi_entertainment_score": +1,
        "data_usage_gb":           +1,
        "evening_data_ratio":      +1,
        "monthly_charges":         +1,
        "tenure_months":           -1,
    },
}

ARCHETYPE_DESCRIPTIONS = {
    "Urban Commuter":    "High-mobility daily commuter with strong transit affinity and peak-hour data spikes. Needs reliable coverage on-the-go.",
    "Home Office Pro":   "Predominantly home-based heavy data user with daytime usage patterns. Likely working from home with video calls.",
    "Digital Nomad":     "Wide travel radius, multi-location lifestyle, significant roaming activity. Values seamless connectivity everywhere.",
    "Weekend Explorer":  "Stable weekday patterns with notable weekend mobility spikes. Uses leisure and entertainment POIs on weekends.",
    "Social Hub":        "Dense POI visits to malls and entertainment venues. Evening and weekend power user with high social app usage.",
    "Senior Steady":     "Minimal mobility, voice-heavy usage, very low data. Values simplicity and reliability over data features.",
    "Young Professional":"Balanced work and entertainment POI affinity, moderate mobility, growing data needs. Seeks value + streaming perks.",
}

ARCHETYPE_PLANS = {
    "Urban Commuter":    ["unlimited_rush", "data_boost", "young_pro"],
    "Home Office Pro":   ["home_pro", "unlimited_rush", "data_boost"],
    "Digital Nomad":     ["nomad_global", "unlimited_rush", "data_boost"],
    "Weekend Explorer":  ["weekend_explorer", "social_unlimited", "data_boost"],
    "Social Hub":        ["social_unlimited", "young_pro", "weekend_explorer"],
    "Senior Steady":     ["voice_classic", "weekend_explorer", "data_boost"],
    "Young Professional":["young_pro", "unlimited_rush", "social_unlimited"],
}


def compute_segment_zscores(
    df: pd.DataFrame,
    segment_col: str,
    feature_cols: list,
) -> pd.DataFrame:
    """Compute per-segment mean and z-score vs. population for each feature."""
    pop_mean = df[feature_cols].mean()
    pop_std  = df[feature_cols].std().replace(0, 1)

    profiles = []
    for seg_id in sorted(df[segment_col].unique()):
        if seg_id < 0:  # skip noise
            continue
        sub = df[df[segment_col] == seg_id][feature_cols]
        seg_mean = sub.mean()
        seg_std  = sub.std()
        z_scores = (seg_mean - pop_mean) / pop_std
        row = {
            "segment_id":     seg_id,
            "n_subscribers":  len(sub),
            "pct_of_total":   round(len(sub) / len(df) * 100, 1),
        }
        for feat in feature_cols:
            row[f"mean_{feat}"]    = round(float(seg_mean[feat]), 3)
            row[f"zscore_{feat}"]  = round(float(z_scores[feat]), 3)
        profiles.append(row)

    return pd.DataFrame(profiles)


def match_archetype(
    z_scores: dict,
    archetypes: dict,
    feature_cols: list,
) -> str:
    """Find the closest archetype for a segment based on z-score direction matching."""
    best_name = "Unknown Segment"
    best_score = -np.inf

    for name, signature in archetypes.items():
        score = 0.0
        for feat, direction in signature.items():
            if feat in z_scores and feat in feature_cols:
                z = z_scores[feat]
                # Add score if z-score direction matches archetype direction
                score += direction * z
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def profile_segments(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    feature_cols: list = None,
) -> tuple:
    """
    Profile all segments and return:
      - profiles_df: DataFrame with per-segment statistics
      - segment_names: {segment_id: name}
      - segment_descriptions: {segment_id: description}
      - segment_plans: {segment_id: [plan_ids]}
    """
    if feature_cols is None:
        # Use numeric non-ID columns
        exclude = ["segment_id", "true_persona_id", "senior_citizen",
                   "same_h3_zone", "is_home_worker", "is_heavy_commuter",
                   "is_frequent_traveler", "is_roamer"]
        feature_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude
            and not c.startswith("h3_poi_")
        ]

    profiles_df = compute_segment_zscores(df, segment_col, feature_cols)

    segment_names = {}
    segment_descriptions = {}
    segment_plans = {}
    used_names = {}  # track usage count for deduplication

    for _, row in profiles_df.iterrows():
        seg_id = int(row["segment_id"])
        z_scores = {
            col.replace("zscore_", ""): row[col]
            for col in profiles_df.columns
            if col.startswith("zscore_")
        }
        name = match_archetype(z_scores, ARCHETYPES, feature_cols)

        # Deduplicate names if multiple segments match same archetype
        if name in used_names:
            used_names[name] += 1
            name_display = f"{name} {used_names[name]}"
        else:
            used_names[name] = 1
            name_display = name

        segment_names[seg_id]        = name_display
        segment_descriptions[seg_id] = ARCHETYPE_DESCRIPTIONS.get(name, "Behavioral segment identified from mobility and usage patterns.")
        segment_plans[seg_id]        = ARCHETYPE_PLANS.get(name, ["data_boost", "young_pro", "unlimited_rush"])

    # Add names to profiles
    profiles_df["segment_name"]        = profiles_df["segment_id"].map(segment_names)
    profiles_df["segment_description"] = profiles_df["segment_id"].map(segment_descriptions)

    return profiles_df, segment_names, segment_descriptions, segment_plans


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Profile and name clusters")
    parser.add_argument("--data",   default="data/processed/features_segmented.parquet")
    parser.add_argument("--output", default="data/processed/segment_profiles.parquet")
    parser.add_argument("--models", default="data/models")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Segmented data not found: {data_path}")
        logger.error("Run: python src/models/clustering.py")
        sys.exit(1)

    logger.info(f"Loading segmented data: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"  {len(df):,} subscribers, {df['segment_id'].nunique()} segments")

    # Load cluster feature list
    feat_path = Path("data/processed/cluster_features.txt")
    feature_cols = feat_path.read_text().strip().split("\n") if feat_path.exists() else None

    profiles_df, seg_names, seg_descs, seg_plans = profile_segments(
        df, segment_col="segment_id", feature_cols=feature_cols
    )

    # Save
    output_path = Path(args.output)
    profiles_df.to_parquet(output_path, index=False)
    logger.success(f"Segment profiles → {output_path}")

    models_dir = Path(args.models)
    models_dir.mkdir(exist_ok=True)
    joblib.dump(seg_names,  models_dir / "segment_names.pkl")
    joblib.dump(seg_descs,  models_dir / "segment_descriptions.pkl")
    joblib.dump(seg_plans,  models_dir / "segment_plans.pkl")
    logger.success(f"Segment metadata → {models_dir}")

    # Print summary
    print("\n" + "=" * 65)
    print("SEGMENT PROFILES")
    print("=" * 65)
    for _, row in profiles_df.sort_values("n_subscribers", ascending=False).iterrows():
        print(f"  [{int(row['segment_id'])}] {row['segment_name']:25s} "
              f"{int(row['n_subscribers']):6,} ({row['pct_of_total']:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
