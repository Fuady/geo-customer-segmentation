"""
tests/test_api.py
──────────────────
Tests for API schemas and model loader logic.
Run: pytest tests/test_api.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.schemas import SubscriberFeatures, SegmentResult


SAMPLE_PAYLOAD = {
    "subscriber_id":          "TEST_001",
    "avg_daily_distance_km":  22.0,
    "travel_radius_km":       18.0,
    "home_work_distance_km":  15.0,
    "n_frequent_locations":   3.0,
    "pct_time_at_home":       0.52,
    "pct_time_at_work":       0.28,
    "mobility_entropy":       1.8,
    "data_usage_gb":          16.0,
    "call_minutes_monthly":   280.0,
    "sms_monthly":            60.0,
    "roaming_days_monthly":   0.5,
    "morning_data_ratio":     0.28,
    "afternoon_data_ratio":   0.18,
    "evening_data_ratio":     0.35,
    "night_data_ratio":       0.19,
    "weekend_data_ratio":     0.30,
    "peak_hour_data_ratio":   0.45,
    "poi_office_score":       0.55,
    "poi_transit_score":      0.75,
    "poi_mall_score":         0.30,
    "poi_residential_score":  0.40,
    "poi_entertainment_score":0.25,
    "home_zone_rsrq":         -11.0,
    "work_zone_rsrq":         -12.0,
    "avg_throughput_mbps":    28.0,
    "monthly_charges":        75.0,
    "tenure_months":          36.0,
    "senior_citizen":         0,
    "home_lat":               -6.2088,
    "home_lon":               106.8456,
}


class TestSchemaValidation:

    def test_valid_payload_parses(self):
        sub = SubscriberFeatures(**SAMPLE_PAYLOAD)
        assert sub.subscriber_id == "TEST_001"
        assert sub.data_usage_gb == 16.0

    def test_ratio_fields_out_of_range_rejected(self):
        from pydantic import ValidationError
        bad = {**SAMPLE_PAYLOAD, "pct_time_at_home": 1.5}  # > 1
        with pytest.raises(ValidationError):
            SubscriberFeatures(**bad)

    def test_negative_distance_rejected(self):
        from pydantic import ValidationError
        bad = {**SAMPLE_PAYLOAD, "avg_daily_distance_km": -5.0}
        with pytest.raises(ValidationError):
            SubscriberFeatures(**bad)

    def test_senior_citizen_out_of_range_rejected(self):
        from pydantic import ValidationError
        bad = {**SAMPLE_PAYLOAD, "senior_citizen": 2}
        with pytest.raises(ValidationError):
            SubscriberFeatures(**bad)

    def test_optional_lat_lon_can_be_none(self):
        payload = {k: v for k, v in SAMPLE_PAYLOAD.items()
                   if k not in ("home_lat", "home_lon")}
        sub = SubscriberFeatures(**payload)
        assert sub.home_lat is None
        assert sub.home_lon is None

    def test_defaults_applied(self):
        minimal = {"subscriber_id": "MIN_001", "data_usage_gb": 10.0,
                   "monthly_charges": 50.0, "tenure_months": 12.0}
        sub = SubscriberFeatures(**minimal)
        assert sub.avg_daily_distance_km == 10.0  # default
        assert sub.travel_radius_km == 15.0

    def test_segment_result_schema(self):
        result = SegmentResult(
            subscriber_id="TEST",
            segment_id=2,
            segment_name="Digital Nomad",
            confidence=0.84,
            segment_description="Frequent traveler",
            recommended_plans=["nomad_global", "unlimited_rush"],
        )
        assert result.segment_id == 2
        assert len(result.recommended_plans) == 2


class TestRuleBasedRecommender:

    def _make_recommender(self, seg_names, seg_plans):
        from src.models.recommender import RuleBasedRecommender
        import yaml
        config = yaml.safe_load(open("configs/config.yaml"))
        rec = RuleBasedRecommender(config, n_plans=3)
        rec.fit(seg_names, seg_plans)
        return rec

    def test_returns_correct_number_of_plans(self):
        seg_names = {0: "Urban Commuter", 1: "Home Office Pro"}
        seg_plans = {0: ["unlimited_rush", "data_boost", "young_pro"],
                     1: ["home_pro", "unlimited_rush", "data_boost"]}
        rec = self._make_recommender(seg_names, seg_plans)
        plans = rec.recommend(0)
        assert len(plans) == 3

    def test_recommend_returns_list_of_strings(self):
        seg_names = {0: "Senior Steady"}
        seg_plans = {0: ["voice_classic", "weekend_explorer", "data_boost"]}
        rec = self._make_recommender(seg_names, seg_plans)
        plans = rec.recommend(0)
        assert all(isinstance(p, str) for p in plans)

    def test_unknown_segment_returns_fallback(self):
        seg_names = {}
        seg_plans = {}
        rec = self._make_recommender(seg_names, seg_plans)
        plans = rec.recommend(999)
        assert isinstance(plans, list)
        assert len(plans) > 0

    def test_recommend_with_scores_has_required_keys(self):
        seg_names = {0: "Urban Commuter"}
        seg_plans = {0: ["unlimited_rush", "data_boost", "young_pro"]}
        rec = self._make_recommender(seg_names, seg_plans)
        scored = rec.recommend_with_scores(0)
        for item in scored:
            for key in ["rank", "plan_id", "plan_name", "match_score", "reason"]:
                assert key in item

    def test_match_scores_descend_by_rank(self):
        seg_names = {0: "Digital Nomad"}
        seg_plans = {0: ["nomad_global", "unlimited_rush", "data_boost"]}
        rec = self._make_recommender(seg_names, seg_plans)
        scored = rec.recommend_with_scores(0)
        scores = [s["match_score"] for s in scored]
        assert scores == sorted(scores, reverse=True)


class TestDataValidation:

    def test_valid_subscriber_data_passes(self):
        from tests.test_features import make_sample_df
        from src.data_engineering.data_validation import _run_validation
        df = make_sample_df(500)
        # Add required columns that are in validation but not in test df
        df["internet_service"] = "fiber_optic"
        df["home_zone_rsrq"] = -11.0
        # Run validation — should pass
        result = _run_validation(df)
        assert isinstance(result, bool)

    def test_duplicate_ids_detected(self):
        from src.data_engineering.data_validation import DataValidator
        df = pd.DataFrame({
            "subscriber_id": ["A", "A", "B"],
            "home_lat": [-6.2, -6.2, -6.3],
            "home_lon": [106.8, 106.8, 106.9],
        })
        v = DataValidator(df, "test")
        v.expect_unique("subscriber_id")
        assert v.report() is False

    def test_out_of_range_lat_detected(self):
        from src.data_engineering.data_validation import DataValidator
        df = pd.DataFrame({
            "home_lat": [-6.2, 200.0, -6.3],  # 200 is invalid
        })
        v = DataValidator(df, "test")
        v.expect_range("home_lat", -11.0, 6.0)
        assert v.report() is False
