"""
tests/test_clustering.py
─────────────────────────
Unit and integration tests for clustering and segment profiling.
Run: pytest tests/test_clustering.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.clustering import (
    compute_cluster_metrics, fit_kmeans, fit_dbscan
)
from src.models.segment_profiler import (
    profile_segments, compute_segment_zscores, match_archetype, ARCHETYPES
)


def make_feature_matrix(n: int = 500, n_features: int = 15, n_clusters: int = 5,
                         seed: int = 42) -> tuple:
    """Create a well-separated synthetic feature matrix for testing."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-3, 3, (n_clusters, n_features))
    labels_true = rng.integers(0, n_clusters, n)
    X = centres[labels_true] + rng.normal(0, 0.4, (n, n_features))
    return X, labels_true


def make_segmented_df(n: int = 500, n_segs: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    seg_ids = rng.integers(0, n_segs, n)
    return pd.DataFrame({
        "subscriber_id":          [f"SUB_{i}" for i in range(n)],
        "segment_id":             seg_ids,
        "true_persona_name":      [f"Persona {s}" for s in seg_ids],
        "avg_daily_distance_km":  rng.uniform(2, 50, n) + seg_ids * 5,
        "data_usage_gb":          rng.uniform(2, 50, n) + seg_ids * 3,
        "roaming_days_monthly":   rng.uniform(0, 10, n),
        "poi_office_score":       rng.uniform(0, 1, n),
        "poi_transit_score":      rng.uniform(0, 1, n),
        "poi_mall_score":         rng.uniform(0, 1, n),
        "poi_residential_score":  rng.uniform(0, 1, n),
        "morning_data_ratio":     rng.uniform(0.1, 0.4, n),
        "evening_data_ratio":     rng.uniform(0.15, 0.5, n),
        "monthly_charges":        rng.uniform(30, 120, n),
        "mobility_entropy":       rng.uniform(0.5, 3, n),
        "call_minutes_monthly":   rng.uniform(50, 800, n) + seg_ids * 30,
        "pct_time_at_home":       rng.uniform(0.3, 0.9, n),
        "travel_radius_km":       rng.uniform(3, 80, n),
    })


# ── compute_cluster_metrics ───────────────────────────────────────────────────
class TestComputeClusterMetrics:

    def test_returns_all_keys(self):
        X, _ = make_feature_matrix(n=200, n_clusters=4)
        labels = np.tile(range(4), 50)
        metrics = compute_cluster_metrics(X, labels)
        for k in ["silhouette", "davies_bouldin", "calinski_harabasz",
                  "n_clusters", "n_noise", "largest_cluster_pct"]:
            assert k in metrics

    def test_silhouette_in_range(self):
        X, _ = make_feature_matrix(n=300, n_clusters=4)
        km = fit_kmeans(X, {"n_clusters": 4, "random_state": 42})[0]
        labels = km.labels_
        metrics = compute_cluster_metrics(X, labels)
        assert -1.0 <= metrics["silhouette"] <= 1.0

    def test_noise_points_counted(self):
        X, _ = make_feature_matrix(n=200, n_clusters=3)
        labels = np.array([-1] * 20 + [0] * 60 + [1] * 60 + [2] * 60)
        metrics = compute_cluster_metrics(X, labels)
        assert metrics["n_noise"] == 20

    def test_well_separated_clusters_high_silhouette(self):
        """Tightly defined clusters should yield silhouette > 0.5."""
        rng = np.random.default_rng(42)
        # 4 very well-separated clusters
        X = np.vstack([
            rng.normal([0, 0], 0.1, (100, 2)),
            rng.normal([5, 0], 0.1, (100, 2)),
            rng.normal([0, 5], 0.1, (100, 2)),
            rng.normal([5, 5], 0.1, (100, 2)),
        ])
        labels = np.repeat([0, 1, 2, 3], 100)
        metrics = compute_cluster_metrics(X, labels)
        assert metrics["silhouette"] > 0.5


# ── fit_kmeans ────────────────────────────────────────────────────────────────
class TestFitKMeans:

    def test_returns_model_and_labels(self):
        X, _ = make_feature_matrix(n=300, n_clusters=5)
        model, labels = fit_kmeans(X, {"n_clusters": 5, "random_state": 42})
        assert hasattr(model, "cluster_centers_")
        assert len(labels) == len(X)

    def test_correct_number_of_clusters(self):
        X, _ = make_feature_matrix(n=300, n_clusters=6)
        _, labels = fit_kmeans(X, {"n_clusters": 6, "random_state": 42})
        assert len(np.unique(labels)) == 6

    def test_labels_are_integers(self):
        X, _ = make_feature_matrix(n=200, n_clusters=4)
        _, labels = fit_kmeans(X, {"n_clusters": 4, "random_state": 42})
        assert labels.dtype in [np.int32, np.int64, int]

    def test_all_points_assigned(self):
        X, _ = make_feature_matrix(n=250, n_clusters=5)
        _, labels = fit_kmeans(X, {"n_clusters": 5, "random_state": 42})
        assert (labels >= 0).all()


# ── fit_dbscan ────────────────────────────────────────────────────────────────
class TestFitDBSCAN:

    def test_returns_model_and_labels(self):
        X, _ = make_feature_matrix(n=200, n_clusters=4)
        model, labels = fit_dbscan(X, {"eps": 1.5, "min_samples": 10})
        assert hasattr(model, "labels_")
        assert len(labels) == len(X)

    def test_noise_label_is_minus_one(self):
        X, _ = make_feature_matrix(n=200, n_clusters=3)
        _, labels = fit_dbscan(X, {"eps": 0.01, "min_samples": 200})
        # Very restrictive params → many noise points
        assert -1 in labels


# ── profile_segments ──────────────────────────────────────────────────────────
class TestProfileSegments:

    def test_returns_profiles_and_metadata(self):
        df = make_segmented_df()
        profiles, names, descs, plans = profile_segments(df, "segment_id")
        assert len(profiles) > 0
        assert len(names) > 0
        assert len(descs) > 0
        assert len(plans) > 0

    def test_profile_has_required_columns(self):
        df = make_segmented_df()
        profiles, _, _, _ = profile_segments(df, "segment_id")
        assert "segment_id" in profiles.columns
        assert "n_subscribers" in profiles.columns
        assert "pct_of_total" in profiles.columns
        assert "segment_name" in profiles.columns

    def test_pct_of_total_sums_to_100(self):
        df = make_segmented_df(n=500, n_segs=5)
        profiles, _, _, _ = profile_segments(df, "segment_id")
        total_pct = profiles["pct_of_total"].sum()
        assert abs(total_pct - 100.0) < 0.5

    def test_segment_names_are_strings(self):
        df = make_segmented_df()
        _, names, _, _ = profile_segments(df, "segment_id")
        for seg_id, name in names.items():
            assert isinstance(name, str)
            assert len(name) > 0


# ── match_archetype ───────────────────────────────────────────────────────────
class TestMatchArchetype:

    def test_returns_known_archetype_name(self):
        # Feed strong z-scores for "Senior Steady" archetype
        z_scores = {
            "call_minutes_monthly": 2.0,   # very high calls
            "data_usage_gb": -2.0,         # very low data
            "avg_daily_distance_km": -2.0, # stays home
            "pct_time_at_home": 2.0,       # mostly at home
        }
        feat_cols = list(z_scores.keys())
        result = match_archetype(z_scores, ARCHETYPES, feat_cols)
        assert result == "Senior Steady"

    def test_all_archetypes_matchable(self):
        """Every archetype should be matchable with its own signature."""
        for archetype_name, signature in ARCHETYPES.items():
            z_scores = {feat: direction * 2.0 for feat, direction in signature.items()}
            feat_cols = list(z_scores.keys())
            result = match_archetype(z_scores, ARCHETYPES, feat_cols)
            assert result == archetype_name, \
                f"Expected {archetype_name}, got {result}"

    def test_handles_missing_features_gracefully(self):
        z_scores = {"unknown_feature": 2.0}
        result = match_archetype(z_scores, ARCHETYPES, list(z_scores.keys()))
        assert isinstance(result, str)
