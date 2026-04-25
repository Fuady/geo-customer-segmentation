"""
src/models/clustering.py
─────────────────────────
Customer segmentation via geo-behavioral clustering.

Supports:
  - KMeans        (default — fast, deterministic, interpretable)
  - DBSCAN        (density-based — handles noise, no K needed)
  - HDBSCAN       (hierarchical density — soft cluster membership)
  - Optuna HPO    (auto-tune n_clusters for KMeans via silhouette score)

Evaluation metrics logged per run:
  - Silhouette Score  (higher = better, range −1 to 1)
  - Davies-Bouldin    (lower = better)
  - Calinski-Harabasz (higher = better)
  - Inertia           (KMeans only, lower = better)

All experiments tracked in MLflow.

Usage:
    python src/models/clustering.py
    python src/models/clustering.py --algorithm dbscan
    python src/models/clustering.py --tune --n_trials 30
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import yaml
from loguru import logger
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_params(path: str = "configs/segment_params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute all clustering evaluation metrics."""
    # Filter out noise points (label == -1 for DBSCAN)
    mask = labels >= 0
    X_valid = X[mask]
    labels_valid = labels[mask]

    n_clusters = len(np.unique(labels_valid))

    if n_clusters < 2 or len(X_valid) < n_clusters + 1:
        return {
            "silhouette":         -1.0,
            "davies_bouldin":     999.0,
            "calinski_harabasz":  0.0,
            "n_clusters":         n_clusters,
            "n_noise":            int((labels == -1).sum()),
            "largest_cluster_pct":1.0,
        }

    sil = silhouette_score(X_valid, labels_valid, sample_size=min(5000, len(X_valid)))
    db  = davies_bouldin_score(X_valid, labels_valid)
    ch  = calinski_harabasz_score(X_valid, labels_valid)

    counts = np.bincount(labels_valid)
    largest_pct = counts.max() / len(labels_valid)

    return {
        "silhouette":         round(float(sil), 4),
        "davies_bouldin":     round(float(db), 4),
        "calinski_harabasz":  round(float(ch), 2),
        "n_clusters":         int(n_clusters),
        "n_noise":            int((labels == -1).sum()),
        "largest_cluster_pct":round(float(largest_pct), 3),
    }


def fit_kmeans(X: np.ndarray, params: dict) -> tuple:
    model = KMeans(
        n_clusters=params.get("n_clusters", 7),
        init=params.get("init", "k-means++"),
        n_init=params.get("n_init", 20),
        max_iter=params.get("max_iter", 500),
        random_state=params.get("random_state", 42),
    )
    labels = model.fit_predict(X)
    return model, labels


def fit_dbscan(X: np.ndarray, params: dict) -> tuple:
    model = DBSCAN(
        eps=params.get("eps", 0.5),
        min_samples=params.get("min_samples", 30),
        metric=params.get("metric", "euclidean"),
        n_jobs=-1,
    )
    labels = model.fit_predict(X)
    return model, labels


def fit_hdbscan(X: np.ndarray, params: dict) -> tuple:
    try:
        import hdbscan
        model = hdbscan.HDBSCAN(
            min_cluster_size=params.get("min_cluster_size", 50),
            min_samples=params.get("min_samples", 10),
            cluster_selection_epsilon=params.get("cluster_selection_epsilon", 0.3),
            metric=params.get("metric", "euclidean"),
            cluster_selection_method=params.get("cluster_selection_method", "eom"),
            prediction_data=True,
        )
        labels = model.fit_predict(X)
        return model, labels
    except ImportError:
        logger.warning("hdbscan not installed — falling back to DBSCAN")
        return fit_dbscan(X, params)


def tune_kmeans_optuna(
    X: np.ndarray,
    config: dict,
    seg_params: dict,
    n_trials: int = 40,
) -> dict:
    """Auto-tune n_clusters using Optuna to maximise silhouette score."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    k_min, k_max = seg_params["optuna"]["search_space"]["n_clusters"]
    base_params  = seg_params["kmeans"].copy()

    def objective(trial):
        k = trial.suggest_int("n_clusters", k_min, k_max)
        params = {**base_params, "n_clusters": k}
        _, labels = fit_kmeans(X, params)
        metrics = compute_cluster_metrics(X, labels)
        return metrics["silhouette"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials,
                   timeout=seg_params["optuna"].get("timeout_seconds", 600))

    best_k = study.best_params["n_clusters"]
    best_sil = study.best_value
    logger.info(f"Optuna best: n_clusters={best_k}, silhouette={best_sil:.4f}")
    return {**base_params, "n_clusters": best_k}


def reduce_dimensions_umap(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce to 2D for visualization using UMAP."""
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=-1)
        return reducer.fit_transform(X)
    except ImportError:
        logger.warning("umap-learn not installed — using PCA for 2D reduction")
        return PCA(n_components=n_components, random_state=42).fit_transform(X)


def main():
    parser = argparse.ArgumentParser(description="Train geo-behavioral clustering model")
    parser.add_argument("--algorithm", default="kmeans",
                        choices=["kmeans", "dbscan", "hdbscan"])
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna HPO for n_clusters (KMeans only)")
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--params", default="configs/segment_params.yaml")
    parser.add_argument("--scaled_data", default="data/processed/features_scaled.parquet")
    parser.add_argument("--full_data",   default="data/processed/features.parquet")
    args = parser.parse_args()

    config     = load_config(args.config)
    seg_params = load_params(args.params)

    # ── Load scaled feature matrix ─────────────────────────────────────────────
    scaled_path = Path(args.scaled_data)
    full_path   = Path(args.full_data)

    if not scaled_path.exists():
        logger.error(f"Scaled features not found: {scaled_path}")
        logger.error("Run: python src/features/feature_pipeline.py")
        sys.exit(1)

    logger.info(f"Loading scaled features: {scaled_path}")
    X_df = pd.read_parquet(scaled_path)
    X = X_df.values
    logger.info(f"Feature matrix: {X.shape}")

    df_full = pd.read_parquet(full_path) if full_path.exists() else None

    # ── MLflow setup ───────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ── Optuna HPO ─────────────────────────────────────────────────────────────
    if args.tune and args.algorithm == "kmeans":
        logger.info(f"Running Optuna HPO ({args.n_trials} trials)...")
        best_params = tune_kmeans_optuna(X, config, seg_params, args.n_trials)
        seg_params["kmeans"].update(best_params)
        logger.info(f"Tuned params: {best_params}")

    # ── Train model ────────────────────────────────────────────────────────────
    algo_params = seg_params.get(args.algorithm, {})
    if args.algorithm == "kmeans":
        algo_params.setdefault("n_clusters",
                               config["segmentation"]["n_segments_default"])

    logger.info(f"Training {args.algorithm.upper()} with params: {algo_params}")

    with mlflow.start_run(run_name=f"{args.algorithm}_segmentation"):
        mlflow.log_param("algorithm", args.algorithm)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_subscribers", X.shape[0])
        mlflow.log_params({f"param_{k}": v for k, v in algo_params.items()})

        if args.algorithm == "kmeans":
            model, labels = fit_kmeans(X, algo_params)
            mlflow.log_metric("inertia", round(float(model.inertia_), 2))
        elif args.algorithm == "dbscan":
            model, labels = fit_dbscan(X, algo_params)
        else:
            model, labels = fit_hdbscan(X, algo_params)

        metrics = compute_cluster_metrics(X, labels)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        logger.info(f"Metrics: {metrics}")

        # ── 2D reduction for visualization ────────────────────────────────────
        logger.info("Computing 2D UMAP projection for visualization...")
        X_2d = reduce_dimensions_umap(X, n_components=2)
        viz_df = pd.DataFrame(X_2d, columns=["umap_x", "umap_y"])
        viz_df["segment_id"] = labels
        if df_full is not None and "subscriber_id" in df_full.columns:
            viz_df["subscriber_id"] = df_full["subscriber_id"].values
        if df_full is not None and "true_persona_name" in df_full.columns:
            viz_df["true_persona"] = df_full["true_persona_name"].values

        # ── Attach segment labels to full feature dataframe ───────────────────
        if df_full is not None:
            df_full = df_full.copy()
            df_full["segment_id"] = labels

        # ── Save artifacts ────────────────────────────────────────────────────
        models_dir = Path("data/models")
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f"clustering_{args.algorithm}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        labels_path = Path("data/processed") / "segment_labels.parquet"
        pd.DataFrame({"subscriber_id": df_full["subscriber_id"] if df_full is not None
                      else range(len(labels)),
                      "segment_id": labels}).to_parquet(labels_path, index=False)
        mlflow.log_artifact(str(labels_path))

        viz_path = Path("data/processed") / "umap_projection.parquet"
        viz_df.to_parquet(viz_path, index=False)
        mlflow.log_artifact(str(viz_path))

        if df_full is not None:
            segmented_path = Path("data/processed") / "features_segmented.parquet"
            df_full.to_parquet(segmented_path, index=False)

        # Save metadata
        metadata = {
            "algorithm":    args.algorithm,
            "n_segments":   metrics["n_clusters"],
            "silhouette":   metrics["silhouette"],
            "feature_cols": list(X_df.columns),
        }
        joblib.dump(metadata, models_dir / "clustering_metadata.pkl")

        run_id = mlflow.active_run().info.run_id
        logger.success(f"MLflow run ID: {run_id}")

        # Register model if silhouette above threshold
        min_sil = config["monitoring"]["min_silhouette_score"]
        if metrics["silhouette"] >= min_sil:
            mlflow.sklearn.log_model(
                model if hasattr(model, "fit") else None,
                "model",
                registered_model_name=config["mlflow"]["registered_model_name"],
            )
            logger.success(f"Model registered (silhouette={metrics['silhouette']:.3f} ≥ {min_sil})")
        else:
            logger.warning(f"Model NOT registered (silhouette={metrics['silhouette']:.3f} < {min_sil})")

    print("\n" + "=" * 55)
    print("CLUSTERING COMPLETE")
    print("=" * 55)
    print(f"  Algorithm      : {args.algorithm.upper()}")
    print(f"  N segments     : {metrics['n_clusters']}")
    print(f"  Silhouette     : {metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin : {metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-H     : {metrics['calinski_harabasz']:.1f}")
    print(f"  Noise points   : {metrics['n_noise']:,}")
    print(f"\n  MLflow UI      : {config['mlflow']['tracking_uri']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
