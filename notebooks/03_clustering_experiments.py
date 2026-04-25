# %% [markdown]
# # 03 — Clustering Experiments
# **Project:** Location-Based Customer Segmentation for Targeted Marketing
#
# This notebook compares clustering approaches:
# - KMeans: interpretable, fast, assumes spherical clusters
# - DBSCAN: density-based, discovers noise, no K needed
# - HDBSCAN: hierarchical density, soft cluster membership
# - Elbow & Silhouette analysis for optimal K selection
# - 2D visualization with UMAP / t-SNE

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 120
Path("docs").mkdir(exist_ok=True)

from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from sklearn.decomposition import PCA

# Load scaled features
X_df = pd.read_parquet("data/processed/features_scaled.parquet")
df_full = pd.read_parquet("data/processed/features.parquet")
X = X_df.values
print(f"Feature matrix: {X.shape}")
print(f"True personas: {df_full['true_persona_name'].nunique()}")

# %% [markdown]
# ## 1. Elbow Curve & Silhouette Analysis for KMeans

# %%
K_RANGE = range(3, 13)
inertias    = []
silhouettes = []
db_scores   = []
ch_scores   = []

print("Testing K values...")
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
    silhouettes.append(sil)
    db_scores.append(davies_bouldin_score(X, labels))
    ch_scores.append(calinski_harabasz_score(X, labels))
    print(f"  K={k:2d} | Inertia={km.inertia_:,.0f} | Sil={sil:.3f} | DB={db_scores[-1]:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

axes[0, 0].plot(K_RANGE, inertias, "o-", color="#e74c3c", linewidth=2)
axes[0, 0].set_title("Elbow Curve (Inertia)"); axes[0, 0].set_xlabel("K")
axes[0, 0].set_ylabel("Inertia")

axes[0, 1].plot(K_RANGE, silhouettes, "s-", color="#2ecc71", linewidth=2)
axes[0, 1].set_title("Silhouette Score (↑ better)"); axes[0, 1].set_xlabel("K")
axes[0, 1].axhline(max(silhouettes), color="gray", linestyle="--", linewidth=1)
best_k = list(K_RANGE)[np.argmax(silhouettes)]
axes[0, 1].axvline(best_k, color="red", linestyle="--",
                   label=f"Best K={best_k}")
axes[0, 1].legend()

axes[1, 0].plot(K_RANGE, db_scores, "^-", color="#9b59b6", linewidth=2)
axes[1, 0].set_title("Davies-Bouldin Index (↓ better)")
axes[1, 0].set_xlabel("K")

axes[1, 1].plot(K_RANGE, ch_scores, "D-", color="#185FA5", linewidth=2)
axes[1, 1].set_title("Calinski-Harabasz Index (↑ better)")
axes[1, 1].set_xlabel("K")

plt.suptitle(f"KMeans Evaluation — Optimal K = {best_k}", fontsize=13)
plt.tight_layout()
plt.savefig("docs/cluster_elbow_silhouette.png", bbox_inches="tight")
plt.show()
print(f"\nOptimal K by silhouette: {best_k}")

# %% [markdown]
# ## 2. Train Final KMeans Model

# %%
final_km = KMeans(n_clusters=best_k, n_init=20, max_iter=500, random_state=42)
labels_km = final_km.fit_predict(X)

print(f"KMeans (K={best_k}) metrics:")
print(f"  Silhouette  : {silhouette_score(X, labels_km, sample_size=5000):.4f}")
print(f"  Davies-Bouldin: {davies_bouldin_score(X, labels_km):.4f}")
print(f"  Calinski-H  : {calinski_harabasz_score(X, labels_km):.1f}")
print(f"\nCluster sizes:")
unique, counts = np.unique(labels_km, return_counts=True)
for k, c in zip(unique, counts):
    print(f"  Cluster {k}: {c:,} ({c/len(labels_km)*100:.1f}%)")

# %% [markdown]
# ## 3. UMAP 2D Visualization

# %%
umap_path = Path("data/processed/umap_projection.parquet")
if umap_path.exists():
    viz_df = pd.read_parquet(umap_path)
    X_2d = viz_df[["umap_x", "umap_y"]].values
    plot_labels = viz_df["segment_id"].values if "segment_id" in viz_df.columns else labels_km
else:
    # Fallback to PCA if UMAP data not available
    print("UMAP projection not found — using PCA for visualization")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    plot_labels = labels_km

COLORS = ["#e74c3c","#3498db","#9b59b6","#2ecc71","#f39c12","#95a5a6","#1abc9c","#e67e22","#c0392b","#2980b9"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Coloured by discovered segments
scatter1 = axes[0].scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=[COLORS[l % len(COLORS)] for l in plot_labels],
    s=4, alpha=0.5
)
axes[0].set_title(f"Discovered Segments (K={best_k})")
axes[0].set_xlabel("UMAP Dim 1"); axes[0].set_ylabel("UMAP Dim 2")

# Coloured by true persona
if "true_persona_name" in df_full.columns:
    persona_list = sorted(df_full["true_persona_name"].unique())
    persona_map = {p: i for i, p in enumerate(persona_list)}
    true_colors = [COLORS[persona_map[p] % len(COLORS)]
                   for p in df_full["true_persona_name"]]
    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=true_colors, s=4, alpha=0.5)
    axes[1].set_title("True Persona Labels")
    axes[1].set_xlabel("UMAP Dim 1"); axes[1].set_ylabel("UMAP Dim 2")
    for persona in persona_list:
        mask = df_full["true_persona_name"] == persona
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        axes[1].annotate(persona.split()[0], (cx, cy), fontsize=8,
                        ha="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

plt.tight_layout()
plt.savefig("docs/cluster_umap_projection.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. DBSCAN Comparison

# %%
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=30, n_jobs=-1)
labels_db = dbscan.fit_predict(X)
n_db_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = (labels_db == -1).sum()

mask = labels_db >= 0
db_sil = silhouette_score(X[mask], labels_db[mask], sample_size=min(5000, mask.sum())) if mask.sum() > 100 else -1

print(f"\nDBSCAN Results:")
print(f"  Clusters found  : {n_db_clusters}")
print(f"  Noise points    : {n_noise:,} ({n_noise/len(labels_db):.1%})")
print(f"  Silhouette (non-noise): {db_sil:.4f}")

# %% [markdown]
# ## 5. Adjusted Rand Index vs True Personas

# %%
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
true_labels = le.fit_transform(df_full["true_persona_name"])

ari_kmeans = adjusted_rand_score(true_labels, labels_km)
ari_dbscan = adjusted_rand_score(true_labels[labels_db >= 0], labels_db[labels_db >= 0])

print(f"\nAdjusted Rand Index vs True Personas:")
print(f"  KMeans (K={best_k}): ARI = {ari_kmeans:.3f}  (1.0 = perfect match)")
print(f"  DBSCAN:        ARI = {ari_dbscan:.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
methods = [f"KMeans\nK={best_k}", f"DBSCAN\n(eps=0.5)"]
aris = [ari_kmeans, ari_dbscan]
colors = ["#2ecc71" if a > 0.4 else "#f39c12" if a > 0.2 else "#e74c3c" for a in aris]
ax.bar(methods, aris, color=colors, edgecolor="white", width=0.5)
ax.set_title("Adjusted Rand Index vs True Persona Labels\n(Higher = discovered segments align better with true personas)")
ax.set_ylabel("ARI Score")
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(0.4, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Good (>0.4)")
ax.legend()
for i, (m, v) in enumerate(zip(methods, aris)):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("docs/cluster_ari_comparison.png", bbox_inches="tight")
plt.show()

print("\n✅ Clustering experiments complete.")
