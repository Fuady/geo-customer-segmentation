"""
src/features/mobility_features.py
───────────────────────────────────
Extracts mobility-based features from raw CDR/GPS data.

In a real system, these would be computed from raw CDR records (lat/lon per call/data event).
For this project, the raw data already contains CDR-derived summaries which we enrich here.

Features engineered:
  - home_work_symmetry: ratio of home vs work time balance
  - mobility_ratio: travel radius relative to home-work distance
  - commute_intensity: daily distance relative to city median
  - location_concentration: 1 - mobility_entropy/max_entropy (higher = more concentrated)
  - is_home_worker: flag for WFH pattern
  - is_heavy_commuter: flag for long daily commuter
  - is_frequent_traveler: flag for high travel radius
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger


class MobilityFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives mobility pattern features from raw CDR location summaries."""

    CITY_MEDIAN_DISTANCE_KM = 10.0  # Jakarta approximate median commute

    def fit(self, X: pd.DataFrame, y=None):
        # Learn city-level stats from training data for normalization
        if "avg_daily_distance_km" in X.columns:
            self.median_daily_dist_ = X["avg_daily_distance_km"].median()
        else:
            self.median_daily_dist_ = self.CITY_MEDIAN_DISTANCE_KM
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── Time allocation ratio ─────────────────────────────────────────────
        # How symmetrically split is time between home and other places?
        home_t = df.get("pct_time_at_home", pd.Series(0.5, index=df.index))
        work_t = df.get("pct_time_at_work", pd.Series(0.2, index=df.index))
        other_t = np.clip(1 - home_t - work_t, 0, 1)
        df["home_work_ratio"] = (home_t / (work_t + 1e-6)).clip(0, 20).round(3)
        df["other_time_ratio"] = other_t.round(3)

        # ── Mobility intensity ────────────────────────────────────────────────
        # How far do they travel relative to their home-work distance?
        hw_dist = df.get("home_work_distance_km", pd.Series(5.0, index=df.index))
        radius  = df.get("travel_radius_km",       pd.Series(10.0, index=df.index))
        df["mobility_ratio"] = (radius / (hw_dist + 1.0)).clip(0, 20).round(3)

        # Normalised daily distance vs. city median
        daily_dist = df.get("avg_daily_distance_km", pd.Series(8.0, index=df.index))
        df["commute_intensity"] = (daily_dist / self.median_daily_dist_).clip(0, 10).round(3)

        # ── Location concentration ────────────────────────────────────────────
        entropy = df.get("mobility_entropy", pd.Series(1.5, index=df.index))
        MAX_ENTROPY = 3.5
        df["location_concentration"] = (1 - entropy / MAX_ENTROPY).clip(0, 1).round(3)

        # ── Same zone flag ────────────────────────────────────────────────────
        # If home_h3 == work_h3, subscriber rarely leaves their neighbourhood
        if "same_h3_zone" in df.columns:
            df["is_local_worker"] = df["same_h3_zone"].astype(int)
        else:
            df["is_local_worker"] = (hw_dist < 2.0).astype(int)

        # ── Behavioral flags ──────────────────────────────────────────────────
        df["is_home_worker"]      = (home_t > 0.70).astype(int)
        df["is_heavy_commuter"]   = (daily_dist > 18.0).astype(int)
        df["is_frequent_traveler"]= (radius > 40.0).astype(int)
        df["is_roamer"]           = (
            df.get("roaming_days_monthly", pd.Series(0, index=df.index)) > 3
        ).astype(int)

        logger.debug(f"Mobility features: {df.shape[1] - X.shape[1]} new columns")
        return df


class TemporalUsageEngineer(BaseEstimator, TransformerMixin):
    """Derives temporal usage pattern features."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        morning   = df.get("morning_data_ratio",   pd.Series(0.25, index=df.index))
        afternoon = df.get("afternoon_data_ratio", pd.Series(0.25, index=df.index))
        evening   = df.get("evening_data_ratio",   pd.Series(0.25, index=df.index))
        night     = df.get("night_data_ratio",     pd.Series(0.25, index=df.index))
        weekend   = df.get("weekend_data_ratio",   pd.Series(0.30, index=df.index))
        peak      = df.get("peak_hour_data_ratio", pd.Series(0.30, index=df.index))
        data_gb   = df.get("data_usage_gb",        pd.Series(10.0, index=df.index))

        # Primary usage time window
        time_matrix = pd.DataFrame({
            "morning":   morning,
            "afternoon": afternoon,
            "evening":   evening,
            "night":     night,
        })
        df["primary_usage_window"] = time_matrix.idxmax(axis=1)

        # Business hours vs leisure hours ratio
        business = morning + afternoon
        leisure  = evening + night
        df["business_leisure_ratio"] = (business / (leisure + 1e-6)).clip(0, 10).round(3)

        # Evening-heavy flag (Social Hub pattern)
        df["is_evening_heavy"] = (evening > 0.38).astype(int)

        # Weekend power user flag
        df["is_weekend_heavy"] = (weekend > 0.45).astype(int)

        # Peak-hour commuter flag
        df["is_peak_hour_user"] = (peak > 0.42).astype(int)

        # Data usage tier
        df["data_tier"] = pd.cut(
            data_gb,
            bins=[0, 5, 15, 30, 60, 1000],
            labels=["very_low", "low", "medium", "high", "very_high"],
        ).astype(str)

        # Data per charge ratio (value perception)
        charges = df.get("monthly_charges", pd.Series(50.0, index=df.index))
        df["gb_per_dollar"] = (data_gb / (charges + 1.0)).clip(0, 5).round(3)

        logger.debug(f"Temporal features: {df.shape[1] - X.shape[1]} new columns")
        return df
