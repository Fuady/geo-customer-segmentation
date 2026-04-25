"""
src/features/poi_features.py
──────────────────────────────
Builds POI affinity and visit pattern features.

In production, these come from counting H3-cell-level POI visits from CDR.
Here we enrich the synthetic POI scores with derived features.

Features:
  - poi_work_affinity: combined office + transit score (commuter proxy)
  - poi_leisure_affinity: mall + entertainment score (lifestyle proxy)
  - poi_stay_affinity: residential + park score (homebody proxy)
  - dominant_poi_type: the POI category with highest affinity
  - poi_diversity_score: entropy of POI visits (high = well-rounded lifestyle)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger


class POIFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives composite POI affinity features."""

    POI_COLS = [
        "poi_office_score", "poi_transit_score", "poi_mall_score",
        "poi_residential_score", "poi_entertainment_score",
    ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Safely get POI columns with fallback defaults
        office  = df.get("poi_office_score",       pd.Series(0.3, index=df.index))
        transit = df.get("poi_transit_score",      pd.Series(0.3, index=df.index))
        mall    = df.get("poi_mall_score",          pd.Series(0.3, index=df.index))
        resi    = df.get("poi_residential_score",  pd.Series(0.5, index=df.index))
        entert  = df.get("poi_entertainment_score",pd.Series(0.3, index=df.index))

        # ── Composite affinity scores ─────────────────────────────────────────
        # Work affinity: office + transit → Urban Commuter / Young Professional
        df["poi_work_affinity"] = (0.55 * office + 0.45 * transit).clip(0, 1).round(3)

        # Leisure affinity: mall + entertainment → Social Hub / Weekend Explorer
        df["poi_leisure_affinity"] = (0.50 * mall + 0.50 * entert).clip(0, 1).round(3)

        # Stay affinity: home-centric → Home Office Pro / Senior Steady
        df["poi_stay_affinity"] = (0.70 * resi + 0.30 * (1 - transit)).clip(0, 1).round(3)

        # Transit affinity standalone
        df["poi_transit_affinity"] = transit.clip(0, 1).round(3)

        # ── Dominant POI type ─────────────────────────────────────────────────
        poi_matrix = pd.DataFrame({
            "office":        office,
            "transit":       transit,
            "mall":          mall,
            "residential":   resi,
            "entertainment": entert,
        })
        df["dominant_poi_type"] = poi_matrix.idxmax(axis=1)

        # ── POI diversity (Shannon entropy) ──────────────────────────────────
        # High entropy = visits many POI types evenly → Digital Nomad / Young Pro
        def poi_entropy(row):
            vals = np.array([row["poi_office_score"], row["poi_transit_score"],
                             row["poi_mall_score"], row["poi_residential_score"],
                             row["poi_entertainment_score"]])
            vals = vals / (vals.sum() + 1e-10)
            vals = vals[vals > 0]
            return float(-np.sum(vals * np.log(vals)))

        df["poi_diversity_score"] = df.apply(poi_entropy, axis=1).round(3)

        # ── Behavioral flags from POI ─────────────────────────────────────────
        df["is_office_worker_poi"]  = (df["poi_work_affinity"] > 0.55).astype(int)
        df["is_mall_visitor"]       = (mall > 0.60).astype(int)
        df["is_transit_user"]       = (transit > 0.55).astype(int)
        df["is_residential_anchor"] = (resi > 0.70).astype(int)

        logger.debug(f"POI features: {df.shape[1] - X.shape[1]} new columns")
        return df


def add_h3_poi_density(
    df: pd.DataFrame,
    pois_df: pd.DataFrame,
    h3_col: str = "home_h3",
    resolution: int = 8,
) -> pd.DataFrame:
    """
    Count POIs of each type within each subscriber's home H3 cell.
    Used when real OSM POI data is available (from ingest_osm.py).
    """
    if pois_df is None or len(pois_df) == 0 or h3_col not in df.columns:
        logger.warning("POI density enrichment skipped — no POI data or H3 column")
        return df

    logger.info("Computing H3-level POI density from OSM data...")

    try:
        import h3
    except ImportError:
        logger.warning("H3 not installed — skipping POI density")
        return df

    # Assign each OSM POI to an H3 cell
    def assign_h3(row):
        try:
            return h3.geo_to_h3(row["latitude"], row["longitude"], resolution)
        except Exception:
            return None

    pois_df = pois_df.copy()
    pois_df["h3_cell"] = pois_df.apply(assign_h3, axis=1)
    pois_df = pois_df.dropna(subset=["h3_cell"])

    # Pivot: count of each POI type per H3 cell
    poi_counts = pois_df.groupby(["h3_cell", "poi_type"]).size().unstack(fill_value=0)
    poi_counts.columns = [f"h3_poi_{c}_count" for c in poi_counts.columns]
    poi_counts = poi_counts.reset_index().rename(columns={"h3_cell": h3_col})

    df = df.merge(poi_counts, on=h3_col, how="left")
    new_cols = [c for c in poi_counts.columns if c != h3_col]
    df[new_cols] = df[new_cols].fillna(0)

    logger.info(f"Added {len(new_cols)} H3 POI density columns")
    return df
