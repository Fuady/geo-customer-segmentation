# %% [markdown]
# # 02 — Feature Engineering Deep Dive
# **Project:** Location-Based Customer Segmentation for Targeted Marketing
#
# This notebook walks through every feature transformation:
# - Mobility ratio and concentration features
# - Temporal usage pattern features
# - POI composite affinity scores
# - Feature importance pre-screening via variance and mutual information
# - Feature selection for clustering

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
sns.set_theme(style="whitegrid")
Path("docs").mkdir(exist_ok=True)

from src.features.mobility_features import MobilityFeatureEngineer, TemporalUsageEngineer
from src.features.poi_features import POIFeatureEngineer

df_raw = pd.read_parquet("data/raw/subscribers.parquet")
print(f"Raw features: {df_raw.shape}")

# ── Apply all feature engineering ─────────────────────────────────────────────
mob_eng  = MobilityFeatureEngineer().fit(df_raw)
temp_eng = TemporalUsageEngineer().fit(df_raw)
poi_eng  = POIFeatureEngineer().fit(df_raw)

df = mob_eng.transform(df_raw)
df = temp_eng.transform(df)
df = poi_eng.transform(df)

print(f"After engineering: {df.shape} — {df.shape[1] - df_raw.shape[1]} new features")

# %% [markdown]
# ## 1. Mobility Derived Features

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
derived_mobility = [
    ("home_work_ratio",        "Home/Work Time Ratio"),
    ("mobility_ratio",         "Mobility Ratio (radius/hw_dist)"),
    ("commute_intensity",      "Commute Intensity (vs. city median)"),
    ("location_concentration", "Location Concentration (1=stays home)"),
    ("is_heavy_commuter",      "Is Heavy Commuter (flag)"),
    ("is_home_worker",         "Is Home Worker (flag)"),
]
for ax, (col, label) in zip(axes.flat, derived_mobility):
    if col not in df.columns:
        ax.set_visible(False)
        continue
    for persona in sorted(df["true_persona_name"].unique()):
        data = df[df["true_persona_name"] == persona][col]
        if data.nunique() <= 2:
            # Binary flag — use bar chart
            rates = df.groupby("true_persona_name")[col].mean()
            colors = sns.color_palette("tab10", len(rates))
            ax.bar(rates.index, rates.values, color=colors)
            ax.set_title(label, fontsize=9)
            ax.tick_params(axis="x", rotation=35, labelsize=7)
            break
        data = data.clip(data.quantile(0.02), data.quantile(0.98))
        ax.hist(data, bins=35, alpha=0.4, density=True, label=persona)
    ax.set_title(label, fontsize=9)

plt.suptitle("Derived Mobility Features", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("docs/feat_mobility_derived.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Temporal Pattern Features

# %%
print("\nTemporal feature stats by persona:")
temporal_feats = ["business_leisure_ratio", "is_evening_heavy",
                  "is_weekend_heavy", "is_peak_hour_user", "data_tier"]
for feat in temporal_feats:
    if feat in df.columns:
        print(f"\n{feat}:")
        print(df.groupby("true_persona_name")[feat].mean().round(3).to_string()
              if pd.api.types.is_numeric_dtype(df[feat])
              else df.groupby("true_persona_name")[feat].value_counts(normalize=True).unstack().fillna(0).round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Business/Leisure ratio
if "business_leisure_ratio" in df.columns:
    bl = df.groupby("true_persona_name")["business_leisure_ratio"].median().sort_values()
    axes[0].barh(bl.index, bl.values, color=sns.color_palette("Blues_d", len(bl)))
    axes[0].set_title("Business/Leisure Ratio by Persona\n(>1 = more business hours data)")
    axes[0].axvline(1.0, color="red", linestyle="--", linewidth=1, label="Equal")
    axes[0].legend()

# Evening heavy flag
if "is_evening_heavy" in df.columns:
    ev = df.groupby("true_persona_name")["is_evening_heavy"].mean().sort_values(ascending=False)
    axes[1].bar(ev.index, ev.values * 100,
                color=sns.color_palette("Oranges_d", len(ev)))
    axes[1].set_title("% of Subscribers Who Are Evening-Heavy Users")
    axes[1].set_ylabel("% of Segment")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(ev.values * 100):
        axes[1].text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("docs/feat_temporal_derived.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. POI Composite Features

# %%
poi_composites = {
    "poi_work_affinity":    "Office + Transit Score",
    "poi_leisure_affinity": "Mall + Entertainment Score",
    "poi_stay_affinity":    "Residential + Park Score",
    "poi_diversity_score":  "POI Visit Diversity (Entropy)",
}
available_pois = {k: v for k, v in poi_composites.items() if k in df.columns}

if available_pois:
    fig, axes = plt.subplots(1, len(available_pois), figsize=(14, 4))
    if len(available_pois) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, available_pois.items()):
        persona_means = df.groupby("true_persona_name")[col].mean().sort_values()
        colors = sns.color_palette("tab10", len(persona_means))
        ax.barh(persona_means.index, persona_means.values, color=colors)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Score")
    plt.suptitle("POI Composite Affinity Features", fontsize=12)
    plt.tight_layout()
    plt.savefig("docs/feat_poi_composites.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 4. Feature Variance Analysis
# Low-variance features contribute little to clustering — we identify and remove them.

# %%
from sklearn.preprocessing import StandardScaler

# Select numeric clustering candidates
exclude = ["segment_id", "senior_citizen", "same_h3_zone", "true_persona_id"]
exclude += [c for c in df.columns if c.startswith("is_") or c.startswith("h3_poi_")]
num_cols = [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in exclude
            and not c.endswith("_id")]

X = df[num_cols].fillna(df[num_cols].median())
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=num_cols)

variance = X_scaled.var().sort_values(ascending=False)
LOW_VARIANCE_THRESHOLD = 0.05
low_var = variance[variance < LOW_VARIANCE_THRESHOLD]
print(f"Low-variance features ({len(low_var)}):")
print(low_var.to_string())

fig, ax = plt.subplots(figsize=(14, 6))
variance.head(40).plot.bar(ax=ax, color="#5E5BE0", alpha=0.8)
ax.axhline(LOW_VARIANCE_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
           label=f"Threshold ({LOW_VARIANCE_THRESHOLD})")
ax.set_title("Feature Variance after Standard Scaling\n(Features below threshold removed from clustering)")
ax.set_ylabel("Variance")
ax.tick_params(axis="x", rotation=40, labelsize=7)
ax.legend()
plt.tight_layout()
plt.savefig("docs/feat_variance_analysis.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Feature Correlation with Persona Labels

# %%
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(df["true_persona_name"])
X_mi = X.copy()

print("Computing mutual information scores...")
mi_scores = mutual_info_classif(X_mi, y, random_state=42)
mi_series = pd.Series(mi_scores, index=num_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
mi_series.head(25).sort_values().plot.barh(ax=ax, color="#e74c3c", alpha=0.85)
ax.set_title("Top 25 Features by Mutual Information with True Persona Labels")
ax.set_xlabel("Mutual Information Score")
plt.tight_layout()
plt.savefig("docs/feat_mutual_information.png", bbox_inches="tight")
plt.show()

print(f"\nTop 15 most informative features for segmentation:")
print(mi_series.head(15).round(4).to_string())
print("\n✅ Feature engineering notebook complete.")
