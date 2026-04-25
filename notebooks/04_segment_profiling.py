# %% [markdown]
# # 04 — Segment Profiling & Marketing Insights
# **Project:** Location-Based Customer Segmentation for Targeted Marketing
#
# This notebook turns raw cluster numbers into actionable marketing personas:
# - Feature centroid analysis per segment
# - Auto-named persona profiles
# - Radar chart comparison
# - Revenue opportunity analysis
# - Campaign targeting recommendations

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
import joblib
warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")
Path("docs").mkdir(exist_ok=True)

COLORS = ["#e74c3c","#3498db","#9b59b6","#2ecc71","#f39c12","#95a5a6","#1abc9c","#e67e22"]

# Load segmented data
df = pd.read_parquet("data/processed/features_segmented.parquet")
profiles = pd.read_parquet("data/processed/segment_profiles.parquet")
seg_names = joblib.load("data/models/segment_names.pkl")
seg_plans = joblib.load("data/models/segment_plans.pkl")

print(f"Loaded {len(df):,} subscribers across {len(seg_names)} segments")
df["segment_name"] = df["segment_id"].map(seg_names)

# %% [markdown]
# ## 1. Segment Size and Revenue Distribution

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Subscriber count
seg_size = df.groupby("segment_name").size().sort_values(ascending=True)
axes[0].barh(seg_size.index, seg_size.values,
             color=[COLORS[i % len(COLORS)] for i in range(len(seg_size))])
axes[0].set_title("Subscribers per Segment")
axes[0].set_xlabel("Count")
for i, v in enumerate(seg_size.values):
    axes[0].text(v + 50, i, f"{v:,}", va="center", fontsize=8)

# Avg monthly charges
arpu = df.groupby("segment_name")["monthly_charges"].mean().sort_values(ascending=True)
colors_arpu = plt.cm.Greens(np.linspace(0.4, 0.9, len(arpu)))
axes[1].barh(arpu.index, arpu.values, color=colors_arpu)
axes[1].set_title("Avg Monthly Charges (ARPU)")
axes[1].set_xlabel("$/month")
for i, v in enumerate(arpu.values):
    axes[1].text(v + 0.5, i, f"${v:.0f}", va="center", fontsize=8)

# Total monthly revenue per segment
total_rev = df.groupby("segment_name")["monthly_charges"].sum().sort_values(ascending=True)
axes[2].barh(total_rev.index, total_rev.values / 1000,
             color=plt.cm.Blues(np.linspace(0.4, 0.9, len(total_rev))))
axes[2].set_title("Total Monthly Revenue (K$)")
axes[2].set_xlabel("Revenue ($K/month)")
for i, v in enumerate(total_rev.values):
    axes[2].text(v/1000 + 0.2, i, f"${v/1000:.0f}K", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("docs/profile_revenue_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Multi-Feature Segment Comparison Heatmap

# %%
PROFILE_FEATURES = {
    "Avg Daily Dist (km)":  "avg_daily_distance_km",
    "Travel Radius (km)":   "travel_radius_km",
    "Data Usage (GB)":      "data_usage_gb",
    "Roaming Days/mo":      "roaming_days_monthly",
    "Call Mins/mo":         "call_minutes_monthly",
    "Evening Data %":       "evening_data_ratio",
    "Weekend Data %":       "weekend_data_ratio",
    "Office POI Score":     "poi_office_score",
    "Transit POI Score":    "poi_transit_score",
    "Mall POI Score":       "poi_mall_score",
    "Residential POI":      "poi_residential_score",
    "ARPU ($)":             "monthly_charges",
}

avail = {k: v for k, v in PROFILE_FEATURES.items() if v in df.columns}
seg_matrix = df.groupby("segment_name")[list(avail.values())].mean()
seg_matrix.columns = list(avail.keys())

# Z-score normalize for heatmap
from sklearn.preprocessing import StandardScaler
seg_matrix_z = pd.DataFrame(
    StandardScaler().fit_transform(seg_matrix),
    index=seg_matrix.index, columns=seg_matrix.columns
)

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(seg_matrix_z, annot=True, fmt=".1f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9},
            cbar_kws={"label": "Z-score vs population mean"})
ax.set_title("Segment Feature Profiles (Z-scores)\n"
             "+2 = well above average | -2 = well below average", fontsize=12)
ax.set_xlabel("")
plt.tight_layout()
plt.savefig("docs/profile_segment_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Radar Charts — One Per Segment

# %%
RADAR_DIMS = {
    "Mobility":     "avg_daily_distance_km",
    "Data Usage":   "data_usage_gb",
    "Evening":      "evening_data_ratio",
    "Weekend":      "weekend_data_ratio",
    "Work POI":     "poi_office_score",
    "Leisure POI":  "poi_mall_score",
    "Roaming":      "roaming_days_monthly",
    "Voice":        "call_minutes_monthly",
}
radar_avail = {k: v for k, v in RADAR_DIMS.items() if v in df.columns}
categories = list(radar_avail.keys())
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

pop_means = df[[v for v in radar_avail.values()]].mean()
pop_stds  = df[[v for v in radar_avail.values()]].std().replace(0, 1)

def norm_val(val, mean, std):
    return float(np.clip((val - (mean - 2*std)) / (4*std + 1e-6), 0, 1))

n_segs = len(seg_names)
n_cols = min(4, n_segs)
n_rows = -(-n_segs // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows),
                         subplot_kw=dict(polar=True))
if n_rows == 1:
    axes = [axes] if n_cols == 1 else list(axes)
else:
    axes = [ax for row in axes for ax in row]

for i, (seg_id, seg_name) in enumerate(sorted(seg_names.items())):
    ax = axes[i]
    seg_data = df[df["segment_id"] == seg_id]
    seg_means = seg_data[[v for v in radar_avail.values()]].mean()

    values = [norm_val(seg_means[v], pop_means[v], pop_stds[v])
              for v in radar_avail.values()]
    values += values[:1]
    pop_vals = [0.5] * N + [0.5]

    color = COLORS[seg_id % len(COLORS)]
    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.plot(angles, pop_vals, color="#aaaaaa", linewidth=1, linestyle="--")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title(f"[{seg_id}] {seg_name}", size=9, pad=12, fontweight="bold")

for ax in axes[n_segs:]:
    ax.set_visible(False)

plt.suptitle("Segment Radar Profiles (vs. Population Average)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("docs/profile_radar_charts.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Marketing Action Table

# %%
print("\n" + "=" * 80)
print("MARKETING ACTION PLAN BY SEGMENT")
print("=" * 80)

plan_names = {
    "unlimited_rush": "Unlimited Rush ($85)",
    "home_pro":       "Home Pro ($70)",
    "nomad_global":   "Nomad Global ($95)",
    "weekend_explorer":"Weekend Explorer ($55)",
    "social_unlimited":"Social Unlimited ($45)",
    "voice_classic":  "Voice Classic ($35)",
    "young_pro":      "Young Professional ($60)",
    "data_boost":     "Data Boost 50GB ($50)",
}

for seg_id, seg_name in sorted(seg_names.items()):
    seg_data = df[df["segment_id"] == seg_id]
    n_subs  = len(seg_data)
    avg_arpu = seg_data["monthly_charges"].mean()
    total_rev = avg_arpu * n_subs
    plans = [plan_names.get(p, p) for p in seg_plans.get(seg_id, [])[:2]]

    print(f"\n  [{seg_id}] {seg_name}")
    print(f"      Subscribers : {n_subs:,} ({n_subs/len(df)*100:.1f}%)")
    print(f"      Avg ARPU    : ${avg_arpu:.0f}/month")
    print(f"      Total Rev   : ${total_rev:,.0f}/month")
    print(f"      Top Offers  : {' | '.join(plans)}")

print("\n" + "=" * 80)

# Revenue uplift opportunity
print("\nUpsell Opportunity Analysis:")
for seg_id, seg_name in sorted(seg_names.items()):
    seg_data = df[df["segment_id"] == seg_id]
    if len(seg_data) == 0:
        continue
    cur_arpu = seg_data["monthly_charges"].mean()
    plan_id  = seg_plans.get(seg_id, ["data_boost"])[0]
    plan_price_map = {
        "unlimited_rush": 85, "home_pro": 70, "nomad_global": 95,
        "weekend_explorer": 55, "social_unlimited": 45,
        "voice_classic": 35, "young_pro": 60, "data_boost": 50,
    }
    suggested_price = plan_price_map.get(plan_id, 60)
    uplift_per_sub  = max(0, suggested_price - cur_arpu)
    total_uplift    = uplift_per_sub * len(seg_data)
    if uplift_per_sub > 0:
        print(f"  {seg_name:25s}: ${uplift_per_sub:.0f}/sub × {len(seg_data):,} = ${total_uplift:,.0f}/mo potential")

print("\n✅ Segment profiling notebook complete.")
