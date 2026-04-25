"""
src/features/usage_features.py
────────────────────────────────
Extracts usage-based features for telecom subscriber segmentation.

In a real CDR system these would be derived from raw event records.
Here we enrich the already-summarised synthetic data with:
  - Usage intensity ratios
  - Voice vs data behaviour
  - Subscription value metrics
  - Tenure-adjusted usage trends
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger


class UsageFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives composite usage features from summarised CDR metrics.
    sklearn-compatible — fit() is stateless, safe to call anywhere.
    """

    def fit(self, X: pd.DataFrame, y=None):
        # Learn population medians for normalisation
        self.median_data_    = X["data_usage_gb"].median()    if "data_usage_gb"        in X.columns else 15.0
        self.median_calls_   = X["call_minutes_monthly"].median() if "call_minutes_monthly" in X.columns else 300.0
        self.median_charges_ = X["monthly_charges"].median()  if "monthly_charges"       in X.columns else 60.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        data_gb   = df.get("data_usage_gb",        pd.Series(self.median_data_,    index=df.index))
        calls_min = df.get("call_minutes_monthly", pd.Series(self.median_calls_,   index=df.index))
        sms       = df.get("sms_monthly",           pd.Series(50.0,                index=df.index))
        charges   = df.get("monthly_charges",       pd.Series(self.median_charges_, index=df.index))
        tenure    = df.get("tenure_months",         pd.Series(24.0,                index=df.index))
        roaming   = df.get("roaming_days_monthly",  pd.Series(0.0,                 index=df.index))

        # ── Voice vs data balance ─────────────────────────────────────────────
        # Normalise each to 0-1 against population medians
        data_norm  = (data_gb  / (self.median_data_    + 1e-6)).clip(0, 5)
        voice_norm = (calls_min / (self.median_calls_   + 1e-6)).clip(0, 5)
        df["voice_data_ratio"] = (voice_norm / (data_norm + 1e-6)).clip(0, 10).round(3)
        df["is_voice_dominant"] = (df["voice_data_ratio"] > 1.5).astype(int)
        df["is_data_dominant"]  = (df["voice_data_ratio"] < 0.5).astype(int)

        # ── Value metrics ─────────────────────────────────────────────────────
        df["gb_per_dollar"]     = (data_gb / (charges + 1.0)).clip(0, 5).round(3)
        df["calls_per_dollar"]  = (calls_min / (charges + 1.0)).clip(0, 30).round(3)
        df["arpu_normalised"]   = (charges / (self.median_charges_ + 1e-6)).clip(0, 5).round(3)

        # ── Tenure-adjusted usage ─────────────────────────────────────────────
        df["data_per_tenure_month"]  = (data_gb  / tenure.clip(1)).round(3)
        df["calls_per_tenure_month"] = (calls_min / tenure.clip(1)).round(3)

        # ── Roaming intensity ─────────────────────────────────────────────────
        df["roaming_intensity"] = (roaming / 30.0).clip(0, 1).round(3)
        df["is_roaming_user"]   = (roaming > 2).astype(int)

        # ── Engagement score (composite) ─────────────────────────────────────
        # High usage across channels → more engaged subscriber
        sms_norm = (sms / 100.0).clip(0, 3)
        df["engagement_score"] = (
            0.40 * data_norm.clip(0, 1) +
            0.30 * voice_norm.clip(0, 1) +
            0.20 * sms_norm.clip(0, 1) +
            0.10 * df["roaming_intensity"]
        ).clip(0, 1).round(3)

        # ── Usage tier flags ──────────────────────────────────────────────────
        df["data_tier"] = pd.cut(
            data_gb,
            bins=[0, 5, 15, 30, 60, 1000],
            labels=["very_low", "low", "medium", "high", "very_high"],
        ).astype(str)

        df["arpu_tier"] = pd.cut(
            charges,
            bins=[0, 35, 55, 75, 100, 10000],
            labels=["budget", "standard", "mid", "premium", "enterprise"],
        ).astype(str)

        # ── SMS intensity ─────────────────────────────────────────────────────
        df["sms_intensity"] = (sms / (calls_min.clip(1) / 10)).clip(0, 10).round(3)

        logger.debug(f"Usage features: {df.shape[1] - X.shape[1]} new columns")
        return df


def encode_usage_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode usage tier columns for clustering."""
    cat_cols = ["data_tier", "arpu_tier"]
    existing = [c for c in cat_cols if c in df.columns]
    return pd.get_dummies(df, columns=existing, drop_first=False, dtype=int)
