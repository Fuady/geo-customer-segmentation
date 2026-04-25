"""
tests/test_features.py
───────────────────────
Unit tests for the feature engineering modules.
Run: pytest tests/test_features.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.mobility_features import MobilityFeatureEngineer, TemporalUsageEngineer
from src.features.poi_features import POIFeatureEngineer


def make_sample_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "subscriber_id":          [f"SUB_{i:08d}" for i in range(n)],
        "avg_daily_distance_km":  rng.uniform(1, 60, n),
        "travel_radius_km":       rng.uniform(2, 100, n),
        "home_work_distance_km":  rng.uniform(0.5, 50, n),
        "n_frequent_locations":   rng.uniform(1, 10, n),
        "pct_time_at_home":       rng.uniform(0.2, 0.9, n),
        "pct_time_at_work":       rng.uniform(0.05, 0.4, n),
        "mobility_entropy":       rng.uniform(0, 3, n),
        "data_usage_gb":          rng.uniform(0.5, 80, n),
        "call_minutes_monthly":   rng.uniform(0, 1000, n),
        "sms_monthly":            rng.uniform(0, 300, n),
        "roaming_days_monthly":   rng.uniform(0, 20, n),
        "morning_data_ratio":     rng.uniform(0.1, 0.4, n),
        "afternoon_data_ratio":   rng.uniform(0.1, 0.4, n),
        "evening_data_ratio":     rng.uniform(0.1, 0.5, n),
        "night_data_ratio":       rng.uniform(0.05, 0.3, n),
        "weekend_data_ratio":     rng.uniform(0.15, 0.6, n),
        "peak_hour_data_ratio":   rng.uniform(0.15, 0.6, n),
        "poi_office_score":       rng.uniform(0, 1, n),
        "poi_transit_score":      rng.uniform(0, 1, n),
        "poi_mall_score":         rng.uniform(0, 1, n),
        "poi_residential_score":  rng.uniform(0, 1, n),
        "poi_entertainment_score":rng.uniform(0, 1, n),
        "home_zone_rsrq":         rng.uniform(-20, -5, n),
        "work_zone_rsrq":         rng.uniform(-20, -5, n),
        "avg_throughput_mbps":    rng.uniform(1, 100, n),
        "monthly_charges":        rng.uniform(20, 150, n),
        "tenure_months":          rng.uniform(1, 100, n),
        "same_h3_zone":           rng.integers(0, 2, n),
    })


# ── MobilityFeatureEngineer ───────────────────────────────────────────────────
class TestMobilityFeatureEngineer:

    def test_fit_transform_runs(self):
        df = make_sample_df()
        eng = MobilityFeatureEngineer()
        result = eng.fit_transform(df)
        assert len(result) == len(df)

    def test_expected_columns_created(self):
        df = make_sample_df()
        result = MobilityFeatureEngineer().fit_transform(df)
        for col in ["home_work_ratio", "mobility_ratio", "commute_intensity",
                    "location_concentration", "is_home_worker",
                    "is_heavy_commuter", "is_frequent_traveler"]:
            assert col in result.columns, f"Missing: {col}"

    def test_binary_flags_are_0_or_1(self):
        df = make_sample_df()
        result = MobilityFeatureEngineer().fit_transform(df)
        for col in ["is_home_worker", "is_heavy_commuter", "is_frequent_traveler", "is_roamer"]:
            assert set(result[col].unique()).issubset({0, 1}), f"{col} has non-binary values"

    def test_home_worker_flag_logic(self):
        df = make_sample_df(100)
        df["pct_time_at_home"] = 0.85  # above threshold
        result = MobilityFeatureEngineer().fit_transform(df)
        assert result["is_home_worker"].all()

    def test_location_concentration_in_range(self):
        df = make_sample_df()
        result = MobilityFeatureEngineer().fit_transform(df)
        assert result["location_concentration"].between(0, 1).all()

    def test_no_nulls_in_derived_features(self):
        df = make_sample_df()
        result = MobilityFeatureEngineer().fit_transform(df)
        new_cols = ["home_work_ratio", "mobility_ratio", "commute_intensity",
                    "location_concentration"]
        for col in new_cols:
            assert result[col].isnull().sum() == 0, f"Nulls in {col}"


# ── TemporalUsageEngineer ─────────────────────────────────────────────────────
class TestTemporalUsageEngineer:

    def test_transform_runs(self):
        df = make_sample_df()
        result = TemporalUsageEngineer().fit_transform(df)
        assert len(result) == len(df)

    def test_expected_columns_created(self):
        df = make_sample_df()
        result = TemporalUsageEngineer().fit_transform(df)
        for col in ["business_leisure_ratio", "is_evening_heavy",
                    "is_weekend_heavy", "is_peak_hour_user", "data_tier"]:
            assert col in result.columns, f"Missing: {col}"

    def test_evening_heavy_flag_logic(self):
        df = make_sample_df(50)
        df["evening_data_ratio"] = 0.50  # above threshold
        result = TemporalUsageEngineer().fit_transform(df)
        assert result["is_evening_heavy"].all()

    def test_data_tier_categories(self):
        df = make_sample_df()
        result = TemporalUsageEngineer().fit_transform(df)
        valid_tiers = {"very_low", "low", "medium", "high", "very_high"}
        assert set(result["data_tier"].unique()).issubset(valid_tiers)

    def test_gb_per_dollar_non_negative(self):
        df = make_sample_df()
        result = TemporalUsageEngineer().fit_transform(df)
        assert (result["gb_per_dollar"] >= 0).all()


# ── POIFeatureEngineer ────────────────────────────────────────────────────────
class TestPOIFeatureEngineer:

    def test_transform_runs(self):
        df = make_sample_df()
        result = POIFeatureEngineer().fit_transform(df)
        assert len(result) == len(df)

    def test_composite_scores_in_range(self):
        df = make_sample_df()
        result = POIFeatureEngineer().fit_transform(df)
        for col in ["poi_work_affinity", "poi_leisure_affinity", "poi_stay_affinity"]:
            assert result[col].between(0, 1).all(), f"{col} out of [0,1]"

    def test_dominant_poi_type_set(self):
        df = make_sample_df()
        result = POIFeatureEngineer().fit_transform(df)
        valid_types = {"office", "transit", "mall", "residential", "entertainment"}
        assert set(result["dominant_poi_type"].unique()).issubset(valid_types)

    def test_poi_diversity_non_negative(self):
        df = make_sample_df()
        result = POIFeatureEngineer().fit_transform(df)
        assert (result["poi_diversity_score"] >= 0).all()

    def test_binary_flags_are_valid(self):
        df = make_sample_df()
        result = POIFeatureEngineer().fit_transform(df)
        for col in ["is_office_worker_poi", "is_mall_visitor",
                    "is_transit_user", "is_residential_anchor"]:
            assert set(result[col].unique()).issubset({0, 1}), f"{col} not binary"
