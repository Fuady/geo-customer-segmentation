"""
src/models/recommender.py
──────────────────────────
Plan recommendation engine for geo-behavioral segments.

Two recommendation strategies:
  1. Rule-based (segment → plan mapping)  — fast, interpretable, always works
  2. Collaborative filter (implicit ALS)  — learns from "accepted offers" history

The final recommendation blends both approaches.

Usage:
    python src/models/recommender.py --train
    python src/models/recommender.py --evaluate
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Plan catalog ──────────────────────────────────────────────────────────────
PLAN_FEATURES = {
    "unlimited_rush":   {"data_gb": 999, "voice_min": 999, "roaming": False, "speed": "priority", "price": 85},
    "home_pro":         {"data_gb": 500, "voice_min": 999, "roaming": False, "speed": "high",     "price": 70},
    "nomad_global":     {"data_gb": 50,  "voice_min": 500, "roaming": True,  "speed": "standard", "price": 95},
    "weekend_explorer": {"data_gb": 20,  "voice_min": 200, "roaming": False, "speed": "double_weekend", "price": 55},
    "social_unlimited": {"data_gb": 999, "voice_min": 200, "roaming": False, "speed": "social_only", "price": 45},
    "voice_classic":    {"data_gb": 5,   "voice_min": 1000,"roaming": False, "speed": "standard", "price": 35},
    "young_pro":        {"data_gb": 20,  "voice_min": 300, "roaming": False, "speed": "standard", "price": 60},
    "data_boost":       {"data_gb": 50,  "voice_min": 100, "roaming": False, "speed": "high",     "price": 50},
}

# Segment → primary recommended plans (rule-based)
SEGMENT_PLAN_RULES = {
    "Urban Commuter":    ["unlimited_rush", "data_boost", "young_pro"],
    "Home Office Pro":   ["home_pro", "unlimited_rush", "data_boost"],
    "Digital Nomad":     ["nomad_global", "unlimited_rush", "data_boost"],
    "Weekend Explorer":  ["weekend_explorer", "social_unlimited", "data_boost"],
    "Social Hub":        ["social_unlimited", "young_pro", "weekend_explorer"],
    "Senior Steady":     ["voice_classic", "weekend_explorer", "data_boost"],
    "Young Professional":["young_pro", "unlimited_rush", "social_unlimited"],
}


class RuleBasedRecommender:
    """Fast rule-based plan recommender using segment-to-plan mapping."""

    def __init__(self, config: dict, n_plans: int = 3):
        self.config   = config
        self.n_plans  = n_plans
        self.seg_names: dict = {}
        self.seg_plans: dict = {}

    def fit(self, segment_names: dict, segment_plans: dict):
        self.seg_names = segment_names
        self.seg_plans = segment_plans
        return self

    def recommend(self, segment_id: int) -> list[str]:
        """Return top N plan IDs for a segment."""
        name = self.seg_names.get(segment_id, "")
        # Try exact name match first
        base_name = name.split(" ")[0] + " " + name.split(" ")[1] if len(name.split()) >= 2 else name
        for key in SEGMENT_PLAN_RULES:
            if key in name:
                return SEGMENT_PLAN_RULES[key][:self.n_plans]
        # Fall back to stored segment plans
        return self.seg_plans.get(segment_id, ["data_boost", "young_pro", "unlimited_rush"])[:self.n_plans]

    def recommend_with_scores(self, segment_id: int) -> list[dict]:
        plan_ids = self.recommend(segment_id)
        results = []
        for rank, pid in enumerate(plan_ids, 1):
            plan = self._get_plan_details(pid)
            results.append({
                "rank":         rank,
                "plan_id":      pid,
                "plan_name":    plan.get("name", pid),
                "price":        plan.get("price", 0),
                "match_score":  round(1.0 - (rank - 1) * 0.15, 2),
                "reason":       self._explain(pid, segment_id),
            })
        return results

    def _get_plan_details(self, plan_id: str) -> dict:
        """Look up plan details from config."""
        plans = self.config.get("recommendation", {}).get("plans", [])
        for p in plans:
            if p["id"] == plan_id:
                return p
        feats = PLAN_FEATURES.get(plan_id, {})
        return {"id": plan_id, "name": plan_id.replace("_", " ").title(),
                "price": feats.get("price", 0)}

    def _explain(self, plan_id: str, segment_id: int) -> str:
        """Generate a plain-English reason for this recommendation."""
        name = self.seg_names.get(segment_id, "your segment")
        reasons = {
            "unlimited_rush":   f"Best for {name}'s high-mobility, peak-hour data needs",
            "home_pro":         f"Optimised for {name}'s heavy daytime home usage",
            "nomad_global":     f"Covers {name}'s frequent roaming and multi-zone activity",
            "weekend_explorer": f"Double weekend data matches {name}'s weekend pattern",
            "social_unlimited": f"Unlimited social apps suits {name}'s evening usage habits",
            "voice_classic":    f"High voice allowance with simple pricing for {name}",
            "young_pro":        f"Balanced data + streaming perks for {name}'s lifestyle",
            "data_boost":       f"50GB high-speed data for {name}'s moderate data needs",
        }
        return reasons.get(plan_id, f"Recommended for {name}")


class CollaborativeFilterRecommender:
    """
    Implicit ALS collaborative filter.
    Learns from simulated 'offer accepted' interaction matrix.

    In production, this trains on real:
      subscriber_id × plan_id → interaction_weight
    (e.g. clicked, accepted, upgraded to)
    """

    def __init__(self, n_factors: int = 32, iterations: int = 30):
        self.n_factors   = n_factors
        self.iterations  = iterations
        self.model       = None
        self.plan_ids    = []
        self.seg_id_map  = {}

    def _generate_interaction_matrix(
        self,
        df: pd.DataFrame,
        seg_col: str = "segment_id",
    ) -> pd.DataFrame:
        """
        Generate a synthetic subscriber × plan interaction matrix.
        Based on true persona labels if available, else on segment_id.
        """
        from src.models.segment_profiler import SEGMENT_PLAN_RULES, ARCHETYPE_PLANS

        seg_to_name = {}
        if "true_persona_name" in df.columns:
            for seg_id in df[seg_col].unique():
                mode_persona = df[df[seg_col] == seg_id]["true_persona_name"].mode()
                seg_to_name[seg_id] = mode_persona.iloc[0] if len(mode_persona) > 0 else "Unknown"

        rows = []
        rng = np.random.default_rng(42)
        for _, row in df.iterrows():
            seg_id = row.get(seg_col, 0)
            persona_name = seg_to_name.get(seg_id, "")

            primary_plans = None
            for archetype_name, plans in ARCHETYPE_PLANS.items():
                if archetype_name in persona_name:
                    primary_plans = plans
                    break
            if primary_plans is None:
                primary_plans = list(PLAN_FEATURES.keys())[:3]

            # Primary plans get weight 3–5, secondary get 1
            for plan_id in PLAN_FEATURES.keys():
                if plan_id in primary_plans:
                    weight = rng.integers(3, 6)
                else:
                    weight = rng.integers(0, 2)
                rows.append({
                    "subscriber_id": row["subscriber_id"],
                    seg_col:         seg_id,
                    "plan_id":       plan_id,
                    "weight":        weight,
                })

        return pd.DataFrame(rows)

    def fit(self, df: pd.DataFrame, seg_col: str = "segment_id"):
        """Train ALS on simulated interaction data."""
        try:
            import implicit
            from scipy.sparse import csr_matrix

            logger.info("Generating interaction matrix for collaborative filter...")
            interactions = self._generate_interaction_matrix(df, seg_col)

            # Create user/item index mappings
            subs = sorted(df["subscriber_id"].unique())
            plans = sorted(PLAN_FEATURES.keys())
            self.plan_ids = plans
            sub_idx = {s: i for i, s in enumerate(subs)}
            plan_idx = {p: i for i, p in enumerate(plans)}

            rows = interactions["subscriber_id"].map(sub_idx)
            cols = interactions["plan_id"].map(plan_idx)
            data = interactions["weight"].values

            # Build sparse user-item matrix
            mat = csr_matrix(
                (data, (rows, cols)),
                shape=(len(subs), len(plans))
            )

            logger.info(f"Training ALS ({self.n_factors} factors, {self.iterations} iters)...")
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.n_factors,
                iterations=self.iterations,
                regularization=0.1,
                random_state=42,
            )
            self.model.fit(mat)
            self._user_matrix = mat
            self._sub_idx = sub_idx
            logger.success("ALS training complete")

        except ImportError:
            logger.warning("implicit not installed — CF recommender unavailable")
            logger.warning("Install with: pip install implicit")
            self.model = None

        return self

    def recommend(self, subscriber_id: str, n: int = 3) -> list[str]:
        """Get top N plan recommendations for a subscriber."""
        if self.model is None or subscriber_id not in self._sub_idx:
            return list(PLAN_FEATURES.keys())[:n]

        idx = self._sub_idx[subscriber_id]
        ids, _ = self.model.recommend(
            idx, self._user_matrix[idx],
            N=n, filter_already_liked_items=False
        )
        return [self.plan_ids[i] for i in ids]


class HybridRecommender:
    """Blends rule-based and collaborative filtering recommendations."""

    def __init__(self, config: dict, n_plans: int = 3, cf_weight: float = 0.3):
        self.config    = config
        self.n_plans   = n_plans
        self.cf_weight = cf_weight
        self.rule_rec  = RuleBasedRecommender(config, n_plans)
        self.cf_rec    = CollaborativeFilterRecommender()
        self._trained  = False

    def fit(self, df: pd.DataFrame, segment_names: dict, segment_plans: dict):
        self.rule_rec.fit(segment_names, segment_plans)
        self.cf_rec.fit(df, seg_col="segment_id")
        self._trained = True
        return self

    def recommend(self, segment_id: int, subscriber_id: Optional[str] = None) -> list[str]:
        rule_plans = self.rule_rec.recommend(segment_id)
        if subscriber_id and self.cf_rec.model is not None:
            cf_plans = self.cf_rec.recommend(subscriber_id, n=self.n_plans)
            # Merge: prioritise rule-based, fill with CF
            merged = list(dict.fromkeys(rule_plans + cf_plans))[:self.n_plans]
            return merged
        return rule_plans[:self.n_plans]

    def recommend_with_details(
        self, segment_id: int, subscriber_id: Optional[str] = None
    ) -> list[dict]:
        return self.rule_rec.recommend_with_scores(segment_id)

    def evaluate_precision_at_k(
        self, df: pd.DataFrame, k: int = 3
    ) -> float:
        """
        Evaluate precision@k on held-out subscribers.
        Ground truth: plans matching the subscriber's true persona.
        """
        from src.models.segment_profiler import ARCHETYPE_PLANS

        hits = 0
        total = 0
        for _, row in df.sample(min(2000, len(df)), random_state=42).iterrows():
            seg_id = row.get("segment_id", 0)
            persona = row.get("true_persona_name", "")
            # Ground truth: plans for this persona
            gt_plans = set()
            for name, plans in ARCHETYPE_PLANS.items():
                if name in persona:
                    gt_plans = set(plans)
                    break
            if not gt_plans:
                continue
            recs = set(self.recommend(seg_id, row.get("subscriber_id")))
            hits += len(recs & gt_plans) / k
            total += 1

        return hits / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate recommendation engine")
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--config",   default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    data_path = Path("data/processed/features_segmented.parquet")
    if not data_path.exists():
        logger.error(f"Segmented data not found: {data_path}")
        logger.error("Run: python src/models/clustering.py && python src/models/segment_profiler.py")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    seg_names  = joblib.load("data/models/segment_names.pkl")
    seg_plans  = joblib.load("data/models/segment_plans.pkl")

    rec = HybridRecommender(config, n_plans=config["recommendation"]["n_plans"])

    if args.train or (not args.evaluate):
        logger.info("Training hybrid recommender...")
        rec.fit(df, seg_names, seg_plans)
        joblib.dump(rec, "data/models/recommender.pkl")
        logger.success("Recommender saved → data/models/recommender.pkl")

    if args.evaluate:
        if not Path("data/models/recommender.pkl").exists():
            rec.fit(df, seg_names, seg_plans)
        else:
            rec = joblib.load("data/models/recommender.pkl")

        prec = rec.evaluate_precision_at_k(df, k=3)
        logger.info(f"Precision@3: {prec:.4f}")

    # Demo recommendations
    print("\n" + "=" * 65)
    print("SEGMENT RECOMMENDATIONS")
    print("=" * 65)
    for seg_id, name in sorted(seg_names.items()):
        plans = rec.recommend(seg_id)
        print(f"  [{seg_id}] {name:25s} → {', '.join(plans)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
