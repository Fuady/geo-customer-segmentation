# %% [markdown]
# # 01 — EDA: Mobility Patterns & CDR Analysis
# **Project:** Location-Based Customer Segmentation for Targeted Marketing
#
# This notebook explores:
# - Subscriber mobility distributions (travel radius, commute distance)
# - Temporal usage patterns (when do different users consume data?)
# - POI affinity scores (where do they spend time?)
# - Network quality distribution
# - Correlation between mobility and usage behaviour
#
# Run: `python notebooks/01_eda_mobility_patterns.py`

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
sns.set_theme(style="whitegrid", palette="muted")
Path("docs").mkdir(exist_ok=True)

# Load data
df = pd.read_parquet("data/raw/subscribers.parquet")
print(f"Loaded {len(df):,} subscribers with {len(df.columns)} columns")
print(f"\nPersona distribution:")
print(df["true_persona_name"].value_counts().to_string())

# %% [markdown]
# ## 1. Mobility Distributions

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

mobility_pairs = [
    ("avg_daily_distance_km", "Avg Daily Distance (km)"),
    ("travel_radius_km",      "Travel Radius (km)"),
    ("home_work_distance_km", "Home-Work Distance (km)"),
    ("n_frequent_locations",  "Frequent Locations Count"),
    ("pct_time_at_home",      "% Time at Home"),
    ("mobility_entropy",      "Mobility Entropy"),
]

persona_palette = sns.color_palette("tab10", len(df["true_persona_name"].unique()))

for ax, (col, label) in zip(axes.flat, mobility_pairs):
    for i, persona in enumerate(sorted(df["true_persona_name"].unique())):
        subset = df[df["true_persona_name"] == persona][col]
        subset = subset.clip(subset.quantile(0.01), subset.quantile(0.99))
        ax.hist(subset, bins=40, alpha=0.4, color=persona_palette[i],
                label=persona, density=True)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("")

axes[0, 0].legend(fontsize=7, loc="upper right")
plt.suptitle("Mobility Feature Distributions by Persona", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("docs/eda_mobility_distributions.png", bbox_inches="tight")
plt.show()
print("Saved docs/eda_mobility_distributions.png")

# %% [markdown]
# ## 2. Temporal Usage Heatmap
# When does each persona use data during the day?

# %%
time_cols = ["morning_data_ratio", "afternoon_data_ratio",
             "evening_data_ratio", "night_data_ratio"]
time_labels = ["Morning\n06-12", "Afternoon\n12-18",
               "Evening\n18-22", "Night\n22-06"]

persona_time = df.groupby("true_persona_name")[time_cols].mean()
persona_time.columns = time_labels

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(persona_time, annot=True, fmt=".2f", cmap="YlOrRd",
            ax=ax, linewidths=0.5, cbar_kws={"label": "Data Usage Ratio"})
ax.set_title("Average Data Usage by Time of Day & Persona\n"
             "(Each row sums to 1.0 — shows relative time-of-day preference)")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig("docs/eda_temporal_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. POI Affinity Profiles

# %%
poi_cols = ["poi_office_score", "poi_transit_score", "poi_mall_score",
            "poi_residential_score", "poi_entertainment_score"]
poi_labels = ["Office", "Transit", "Mall", "Residential", "Entertainment"]

persona_poi = df.groupby("true_persona_name")[poi_cols].mean()
persona_poi.columns = poi_labels

fig, ax = plt.subplots(figsize=(11, 5))
persona_poi.plot(kind="bar", ax=ax, width=0.75,
                 color=sns.color_palette("Set2", len(poi_labels)))
ax.set_title("POI Affinity Scores by Persona\n(Higher = more visits to that POI type)")
ax.set_xlabel("")
ax.set_ylabel("Affinity Score (0–1)")
ax.legend(title="POI Type", bbox_to_anchor=(1, 1))
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("docs/eda_poi_affinity.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Usage Volume Comparison

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
usage_pairs = [
    ("data_usage_gb",        "Data Usage (GB/month)"),
    ("call_minutes_monthly", "Call Minutes/Month"),
    ("roaming_days_monthly", "Roaming Days/Month"),
]
for ax, (col, label) in zip(axes, usage_pairs):
    persona_means = df.groupby("true_persona_name")[col].median().sort_values(ascending=True)
    colors = sns.color_palette("tab10", len(persona_means))
    ax.barh(persona_means.index, persona_means.values, color=colors)
    ax.set_title(f"Median {label}")
    ax.set_xlabel(label)
plt.tight_layout()
plt.savefig("docs/eda_usage_comparison.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Geographic Distribution

# %%
try:
    import folium

    colours = {
        "Urban Commuter": "#e74c3c",
        "Home Office Pro": "#3498db",
        "Digital Nomad": "#9b59b6",
        "Weekend Explorer": "#2ecc71",
        "Social Hub": "#f39c12",
        "Senior Steady": "#95a5a6",
        "Young Professional": "#1abc9c",
    }

    centre_lat = df["home_lat"].mean()
    centre_lon = df["home_lon"].mean()
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=9,
                   tiles="CartoDB positron")

    sample = df.sample(min(3000, len(df)), random_state=42)
    for _, row in sample.iterrows():
        color = colours.get(row["true_persona_name"], "#888")
        folium.CircleMarker(
            location=[row["home_lat"], row["home_lon"]],
            radius=3, color=color, fill=True, fill_opacity=0.5, weight=0.5,
            tooltip=row["true_persona_name"],
        ).add_to(m)

    m.save("docs/eda_geo_distribution.html")
    print("Map saved → docs/eda_geo_distribution.html")
except ImportError:
    print("folium not installed — skipping geo map")

# %% [markdown]
# ## 6. Correlation Analysis

# %%
CLUSTER_FEATURES = [
    "avg_daily_distance_km", "travel_radius_km", "home_work_distance_km",
    "data_usage_gb", "call_minutes_monthly", "roaming_days_monthly",
    "morning_data_ratio", "evening_data_ratio", "weekend_data_ratio",
    "poi_office_score", "poi_transit_score", "poi_mall_score",
    "poi_residential_score", "home_zone_rsrq", "monthly_charges",
]
available = [c for c in CLUSTER_FEATURES if c in df.columns]
corr = df[available].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            ax=ax, annot_kws={"size": 8}, linewidths=0.3)
ax.set_title("Feature Correlation Matrix (Mobility + Usage + POI)", fontsize=12)
plt.tight_layout()
plt.savefig("docs/eda_correlation_matrix.png", bbox_inches="tight")
plt.show()

# Key anti-correlations
churn_corr = corr.abs().unstack().sort_values(ascending=False)
churn_corr = churn_corr[churn_corr < 1.0].drop_duplicates()
print("\nTop 10 feature correlations:")
print(churn_corr.head(10).to_string())

print("\n✅ EDA complete. Charts saved to docs/")
