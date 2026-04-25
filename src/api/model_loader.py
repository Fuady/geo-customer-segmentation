"""
src/api/model_loader.py
────────────────────────
Loads all trained artifacts for the API:
  - Clustering model
  - Feature scaler
  - Feature transformers
  - Segment names/descriptions
  - Recommender engine
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.mobility_features import MobilityFeatureEngineer, TemporalUsageEngineer
from src.features.poi_features import POIFeatureEngineer

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ModelLoader:
    """Singleton loader for all segmentation artifacts."""

    def __init__(self):
        self.model          = None
        self.scaler         = None
        self.feature_cols: list   = []
        self.segment_names: dict  = {}
        self.segment_descs: dict  = {}
        self.segment_plans: dict  = {}
        self.recommender    = None
        self.profiles_df    = None
        self.config         = {}
        self._transformers  = {}
        self.n_segments     = 0

    def load(self) -> None:
        """Load all model artifacts from disk."""
        self.config = load_config()

        models_dir    = Path("data/models")
        processed_dir = Path("data/processed")

        # ── Clustering model ──────────────────────────────────────────────────
        for algo in ["kmeans", "dbscan", "hdbscan"]:
            p = models_dir / f"clustering_{algo}.pkl"
            if p.exists():
                self.model = joblib.load(p)
                logger.info(f"Clustering model loaded: {p}")
                break

        # ── Feature scaler ────────────────────────────────────────────────────
        scaler_path = models_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info("Feature scaler loaded")

        # ── Feature column list ───────────────────────────────────────────────
        feat_path = processed_dir / "cluster_features.txt"
        if feat_path.exists():
            self.feature_cols = feat_path.read_text().strip().split("\n")

        # ── Feature transformers ──────────────────────────────────────────────
        trans_path = models_dir / "feature_transformers.pkl"
        if trans_path.exists():
            self._transformers = joblib.load(trans_path)
        else:
            self._transformers = {
                "mob_eng":  MobilityFeatureEngineer().fit(pd.DataFrame()),
                "temp_eng": TemporalUsageEngineer().fit(pd.DataFrame()),
                "poi_eng":  POIFeatureEngineer().fit(pd.DataFrame()),
            }

        # ── Segment metadata ──────────────────────────────────────────────────
        for fname, attr in [
            ("segment_names.pkl",        "segment_names"),
            ("segment_descriptions.pkl", "segment_descs"),
            ("segment_plans.pkl",        "segment_plans"),
        ]:
            p = models_dir / fname
            if p.exists():
                setattr(self, attr, joblib.load(p))

        # ── Segment profiles ──────────────────────────────────────────────────
        profiles_path = processed_dir / "segment_profiles.parquet"
        if profiles_path.exists():
            self.profiles_df = pd.read_parquet(profiles_path)

        # ── Recommender ───────────────────────────────────────────────────────
        rec_path = models_dir / "recommender.pkl"
        if rec_path.exists():
            self.recommender = joblib.load(rec_path)

        self.n_segments = len(self.segment_names)

        if self.model is None:
            logger.warning("No clustering model found — run: python src/models/clustering.py")
        else:
            logger.success(f"All models loaded. Segments: {self.n_segments}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def _build_features(self, raw: dict) -> pd.DataFrame:
        """Apply feature engineering to a raw input dict."""
        df = pd.DataFrame([raw])

        # Apply transformers
        for eng in self._transformers.values():
            try:
                df = eng.transform(df)
            except Exception:
                pass

        # Align to expected feature columns
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_cols].fillna(0.0)
        return X

    def predict_segment(self, raw: dict) -> dict:
        """Classify a subscriber and return segment + recommendations."""
        subscriber_id = raw.get("subscriber_id", "unknown")

        X = self._build_features(raw)
        X_scaled = self.scaler.transform(X) if self.scaler else X.values

        # Predict segment
        if hasattr(self.model, "predict"):
            seg_id = int(self.model.predict(X_scaled)[0])
        else:
            seg_id = 0  # fallback

        # Confidence: distance to nearest centroid (KMeans) → normalised
        confidence = 0.75
        if hasattr(self.model, "transform"):
            distances = self.model.transform(X_scaled)[0]
            nearest_dist = distances.min()
            confidence = float(np.clip(1 - nearest_dist / (distances.max() + 1e-6), 0, 1))

        seg_name  = self.segment_names.get(seg_id, f"Segment {seg_id}")
        seg_desc  = self.segment_descs.get(seg_id, "Behavioral segment")
        seg_plans = self.segment_plans.get(seg_id, ["data_boost"])

        # H3 cell
        h3_cell = None
        if H3_AVAILABLE and raw.get("home_lat") and raw.get("home_lon"):
            try:
                h3_cell = h3.geo_to_h3(raw["home_lat"], raw["home_lon"], 8)
            except Exception:
                pass

        return {
            "subscriber_id":       subscriber_id,
            "segment_id":          seg_id,
            "segment_name":        seg_name,
            "confidence":          round(confidence, 3),
            "segment_description": seg_desc,
            "recommended_plans":   seg_plans[:3],
            "h3_home_cell":        h3_cell,
        }

    def get_recommendations(
        self,
        segment_id: int,
        subscriber_id: Optional[str] = None,
    ) -> list[dict]:
        if self.recommender:
            return self.recommender.recommend_with_details(segment_id, subscriber_id)
        # Fallback: rule-based
        from src.models.recommender import RuleBasedRecommender
        rec = RuleBasedRecommender(self.config)
        rec.fit(self.segment_names, self.segment_plans)
        return rec.recommend_with_scores(segment_id)

    def get_segment_name(self, segment_id: int) -> str:
        return self.segment_names.get(segment_id, f"Segment {segment_id}")

    def get_all_segment_profiles(self) -> list[dict]:
        profiles = []
        for seg_id, name in sorted(self.segment_names.items()):
            n_subs = 0
            pct = 0.0
            if self.profiles_df is not None and "segment_id" in self.profiles_df.columns:
                row = self.profiles_df[self.profiles_df["segment_id"] == seg_id]
                if len(row) > 0:
                    n_subs = int(row["n_subscribers"].iloc[0])
                    pct    = float(row["pct_of_total"].iloc[0])

            profiles.append({
                "segment_id":    seg_id,
                "segment_name":  name,
                "description":   self.segment_descs.get(seg_id, ""),
                "n_subscribers": n_subs,
                "pct_of_total":  pct,
                "top_plans":     self.segment_plans.get(seg_id, [])[:3],
                "key_traits":    [],
            })
        return profiles
