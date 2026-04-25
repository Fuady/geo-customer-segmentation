"""
dashboards/streamlit_app.py
─────────────────────────────
Interactive Streamlit dashboard for Geo-Behavioral Customer Segmentation.

Tabs:
  1. 📊 Overview        — KPI cards, segment distribution, usage comparison
  2. 🗺️  Geo Map          — H3 segment density Folium map
  3. 🧑‍🤝‍🧑 Segment Deep Dive — Radar charts, feature profiles per segment
  4. 🎯 Recommender      — Interactive plan recommendation tool
  5. 🔍 Subscriber Lookup — Single subscriber classification

Usage:
    streamlit run dashboards/streamlit_app.py
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yaml
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geo Segmentation Intelligence",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEGMENT_COLORS = [
    "#e74c3c", "#3498db", "#9b59b6", "#2ecc71",
    "#f39c12", "#95a5a6", "#1abc9c", "#e67e22",
]


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data():
    p = Path("data/processed/features_segmented.parquet")
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_profiles():
    p = Path("data/processed/segment_profiles.parquet")
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_density():
    p = Path("data/processed/segment_density.parquet")
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource
def load_models():
    models = {}
    for name, path in [
        ("seg_names",  "data/models/segment_names.pkl"),
        ("seg_descs",  "data/models/segment_descriptions.pkl"),
        ("seg_plans",  "data/models/segment_plans.pkl"),
        ("scaler",     "data/models/feature_scaler.pkl"),
        ("rec",        "data/models/recommender.pkl"),
        ("feat_cols",  "data/models/cluster_feature_list.pkl"),
    ]:
        p = Path(path)
        if p.exists():
            models[name] = joblib.load(p)
    return models


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/marker.png", width=55)
    st.title("Geo Segmentation")
    st.markdown("---")

    cfg     = load_config()
    df      = load_data()
    models  = load_models()
    profiles = load_profiles()
    density  = load_density()

    seg_names = models.get("seg_names", {})
    seg_descs = models.get("seg_descs", {})
    seg_plans = models.get("seg_plans", {})

    if df is None:
        st.error("No segmented data found.\n\nRun:\n```\nmake pipeline\n```")
        st.stop()

    n_subs = len(df)
    n_segs = len(seg_names) or df["segment_id"].nunique()
    st.success(f"✅ {n_subs:,} subscribers")
    st.success(f"✅ {n_segs} segments")
    st.markdown("---")
    st.caption("Portfolio Project\nGeospatial × Telecom × ML")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🗺️ Geo Map", "🧑‍🤝‍🧑 Segment Profiles",
    "🎯 Recommender", "🔍 Subscriber Lookup"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Segmentation Overview")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Subscribers", f"{n_subs:,}")
    col2.metric("Segments Discovered", n_segs)
    col3.metric("Avg Monthly Charges",
                f"${df['monthly_charges'].mean():.0f}" if "monthly_charges" in df.columns else "—")
    col4.metric("Avg Data Usage",
                f"{df['data_usage_gb'].mean():.1f} GB" if "data_usage_gb" in df.columns else "—")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Segment distribution pie
    with col_a:
        seg_counts = df["segment_id"].value_counts().reset_index()
        seg_counts.columns = ["segment_id", "count"]
        if seg_names:
            seg_counts["name"] = seg_counts["segment_id"].map(seg_names)
        else:
            seg_counts["name"] = "Segment " + seg_counts["segment_id"].astype(str)
        fig = px.pie(seg_counts, values="count", names="name",
                     title="Subscriber Distribution by Segment",
                     color_discrete_sequence=SEGMENT_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Avg data usage by segment
    with col_b:
        if "data_usage_gb" in df.columns:
            usage_by_seg = df.groupby("segment_id")["data_usage_gb"].mean().reset_index()
            if seg_names:
                usage_by_seg["name"] = usage_by_seg["segment_id"].map(seg_names)
            else:
                usage_by_seg["name"] = "Seg " + usage_by_seg["segment_id"].astype(str)
            usage_by_seg = usage_by_seg.sort_values("data_usage_gb", ascending=True)
            fig2 = px.bar(usage_by_seg, x="data_usage_gb", y="name",
                          orientation="h",
                          title="Avg Monthly Data Usage by Segment (GB)",
                          color="data_usage_gb",
                          color_continuous_scale="Blues",
                          labels={"data_usage_gb": "GB", "name": ""})
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Travel radius distribution
    with col_c:
        if "travel_radius_km" in df.columns and seg_names:
            df["segment_name"] = df["segment_id"].map(seg_names).fillna("Unknown")
            fig3 = px.box(df, x="segment_name", y="travel_radius_km",
                          title="Travel Radius by Segment (km)",
                          color="segment_name",
                          color_discrete_sequence=SEGMENT_COLORS,
                          labels={"travel_radius_km": "Travel Radius (km)", "segment_name": ""})
            fig3.update_layout(showlegend=False)
            fig3.update_xaxes(tickangle=20)
            st.plotly_chart(fig3, use_container_width=True)

    # Monthly charges vs data usage scatter
    with col_d:
        if "monthly_charges" in df.columns and "data_usage_gb" in df.columns:
            sample = df.sample(min(3000, len(df)), random_state=42)
            if seg_names:
                sample["segment_name"] = sample["segment_id"].map(seg_names)
            else:
                sample["segment_name"] = "Seg " + sample["segment_id"].astype(str)
            fig4 = px.scatter(sample, x="data_usage_gb", y="monthly_charges",
                              color="segment_name",
                              title="Data Usage vs Monthly Charges",
                              opacity=0.5, size_max=5,
                              color_discrete_sequence=SEGMENT_COLORS,
                              labels={"data_usage_gb": "Data Usage (GB)",
                                      "monthly_charges": "Monthly Charges ($)",
                                      "segment_name": "Segment"})
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEO MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Geographic Segment Density Map")

    map_html = Path("data/processed/segment_density_map.html")
    geojson_path = Path("data/processed/segment_density.geojson")

    if map_html.exists():
        col_stats, _ = st.columns([3, 1])
        with col_stats:
            if density is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("H3 Cells Mapped", f"{len(density):,}")
                c2.metric("Covered Subscribers",
                          f"{density['n_subscribers'].sum():,}" if "n_subscribers" in density else "—")
                c3.metric("Avg ARPU",
                          f"${density['avg_monthly_charges'].mean():.0f}" if "avg_monthly_charges" in density else "—")

        from streamlit.components.v1 import html as st_html
        with open(map_html) as f:
            map_content = f.read()
        st_html(map_content, height=560, scrolling=False)
    else:
        st.info(
            "Geo density map not found.\n\n"
            "Generate it with:\n```\npython src/models/geo_density.py\n```"
        )
        # Fallback scatter map from subscriber coordinates
        if "home_lat" in df.columns:
            st.subheader("Subscriber Location Scatter (fallback)")
            sample = df.sample(min(5000, len(df)), random_state=42)
            if seg_names:
                sample["segment_name"] = sample["segment_id"].map(seg_names)
            fig_scatter = px.scatter_mapbox(
                sample, lat="home_lat", lon="home_lon",
                color="segment_name" if seg_names else "segment_id",
                zoom=9, height=500,
                mapbox_style="carto-positron",
                color_discrete_sequence=SEGMENT_COLORS,
                opacity=0.6,
                title="Subscriber Home Locations by Segment",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Segment presence heatmap table
    if density is not None:
        pct_cols = [c for c in density.columns if c.startswith("pct_seg_")]
        if pct_cols and seg_names:
            st.subheader("Segment Presence per H3 Cell (Top 20 cells)")
            renamed = {c: seg_names.get(int(c.replace("pct_seg_", "")), c)
                       for c in pct_cols}
            heat_df = density[pct_cols].rename(columns=renamed).head(20)
            fig_heat = px.imshow(
                heat_df.values,
                x=list(heat_df.columns),
                y=[f"Cell {i+1}" for i in range(len(heat_df))],
                color_continuous_scale="Blues",
                title="Segment Distribution Heatmap (H3 Cells)",
                aspect="auto",
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENT PROFILES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Segment Deep Dive")

    if not seg_names:
        st.warning("Segment metadata not found. Run the full pipeline first.")
    else:
        selected_seg = st.selectbox(
            "Select Segment",
            options=sorted(seg_names.keys()),
            format_func=lambda x: f"[{x}] {seg_names.get(x, 'Unknown')}",
        )

        seg_name = seg_names.get(selected_seg, "Unknown")
        seg_desc = seg_descs.get(selected_seg, "")
        seg_plan_list = seg_plans.get(selected_seg, [])

        col_info, col_plans = st.columns([2, 1])
        with col_info:
            st.subheader(seg_name)
            st.write(seg_desc)
            seg_data = df[df["segment_id"] == selected_seg]
            st.caption(f"{len(seg_data):,} subscribers ({len(seg_data)/n_subs*100:.1f}% of total)")

        with col_plans:
            st.subheader("Recommended Plans")
            for i, plan_id in enumerate(seg_plan_list[:3], 1):
                plan_name = plan_id.replace("_", " ").title()
                st.markdown(f"**{i}.** {plan_name}")

        st.markdown("---")

        # Radar chart comparing this segment vs all segments
        RADAR_FEATURES = {
            "Data Usage":      "data_usage_gb",
            "Mobility":        "avg_daily_distance_km",
            "Travel Radius":   "travel_radius_km",
            "Evening Usage":   "evening_data_ratio",
            "Transit Affinity":"poi_transit_score",
            "Office Affinity": "poi_office_score",
            "Social POI":      "poi_mall_score",
            "Roaming":         "roaming_days_monthly",
        }

        available_feats = {k: v for k, v in RADAR_FEATURES.items() if v in df.columns}

        if available_feats:
            pop_means = {k: df[v].mean() for k, v in available_feats.items()}
            pop_stds  = {k: df[v].std().clip(1e-6) for k, v in available_feats.items()}
            seg_means = {k: seg_data[v].mean() for k, v in available_feats.items()}

            # Normalise 0-1
            def normalise(val, mean, std):
                return float(np.clip((val - (mean - 2*std)) / (4*std + 1e-6), 0, 1))

            seg_norm = [normalise(seg_means[k], pop_means[k], pop_stds[k]) for k in available_feats]
            pop_norm = [0.5] * len(available_feats)  # population avg is always 0.5

            categories = list(available_feats.keys())
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=seg_norm + [seg_norm[0]],
                theta=categories + [categories[0]],
                fill="toself", name=seg_name,
                line_color=SEGMENT_COLORS[selected_seg % len(SEGMENT_COLORS)],
                fillcolor=SEGMENT_COLORS[selected_seg % len(SEGMENT_COLORS)],
                opacity=0.4,
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=pop_norm + [pop_norm[0]],
                theta=categories + [categories[0]],
                fill="toself", name="Population Avg",
                line_color="#aaaaaa",
                fillcolor="#dddddd",
                opacity=0.2,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"Segment Profile Radar: {seg_name}",
                showlegend=True,
                height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Temporal usage pattern
        if all(c in df.columns for c in ["morning_data_ratio", "afternoon_data_ratio",
                                          "evening_data_ratio", "night_data_ratio"]):
            st.subheader("Usage Time Pattern")
            time_data = {
                "Time Window": ["Morning\n(6-12)", "Afternoon\n(12-18)",
                                "Evening\n(18-22)", "Night\n(22-6)"],
                "Segment":  [seg_data["morning_data_ratio"].mean(),
                             seg_data["afternoon_data_ratio"].mean(),
                             seg_data["evening_data_ratio"].mean(),
                             seg_data["night_data_ratio"].mean()],
                "All Subs": [df["morning_data_ratio"].mean(),
                             df["afternoon_data_ratio"].mean(),
                             df["evening_data_ratio"].mean(),
                             df["night_data_ratio"].mean()],
            }
            time_df = pd.DataFrame(time_data).melt("Time Window", var_name="Group", value_name="Data Ratio")
            fig_time = px.bar(time_df, x="Time Window", y="Data Ratio",
                              color="Group", barmode="group",
                              title="Data Usage by Time of Day",
                              color_discrete_map={"Segment": SEGMENT_COLORS[selected_seg % len(SEGMENT_COLORS)],
                                                  "All Subs": "#aaaaaa"})
            st.plotly_chart(fig_time, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Plan Recommendation Engine")
    st.markdown("Select a segment to see tailored plan recommendations with match scores.")

    if not seg_names:
        st.warning("Segment metadata not loaded.")
    else:
        plans_catalog = cfg.get("recommendation", {}).get("plans", [])
        plan_map = {p["id"]: p for p in plans_catalog}

        col_seg, col_rec = st.columns([1, 2])

        with col_seg:
            rec_seg = st.selectbox(
                "Segment",
                options=sorted(seg_names.keys()),
                format_func=lambda x: f"[{x}] {seg_names.get(x, 'Unknown')}",
                key="rec_segment",
            )
            seg_plan_ids = seg_plans.get(rec_seg, [])

            st.markdown(f"**{seg_names.get(rec_seg, '')}**")
            st.write(seg_descs.get(rec_seg, ""))

            seg_size = (df["segment_id"] == rec_seg).sum()
            st.metric("Segment Size", f"{seg_size:,} subscribers")

        with col_rec:
            st.subheader("Top Recommended Plans")
            for rank, plan_id in enumerate(seg_plan_ids[:3], 1):
                plan = plan_map.get(plan_id, {"name": plan_id, "monthly_price": 0,
                                               "description": ""})
                match_score = round(1.0 - (rank - 1) * 0.15, 2)
                with st.container():
                    rc1, rc2 = st.columns([3, 1])
                    with rc1:
                        st.markdown(f"**{rank}. {plan.get('name', plan_id)}**")
                        st.caption(plan.get("description", ""))
                    with rc2:
                        st.metric("Price", f"${plan.get('monthly_price', '—')}/mo")
                        st.progress(match_score, text=f"Match: {match_score:.0%}")
                    st.divider()

        # All segments comparison table
        st.subheader("All Segments — Plan Matrix")
        rec_data = []
        for seg_id, name in sorted(seg_names.items()):
            plans = seg_plans.get(seg_id, [])
            rec_data.append({
                "Segment": name,
                "Plan 1": plans[0].replace("_", " ").title() if len(plans) > 0 else "—",
                "Plan 2": plans[1].replace("_", " ").title() if len(plans) > 1 else "—",
                "Plan 3": plans[2].replace("_", " ").title() if len(plans) > 2 else "—",
                "Subscribers": (df["segment_id"] == seg_id).sum(),
            })
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SUBSCRIBER LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Single Subscriber Classification")
    st.markdown("Enter subscriber features to classify them in real time.")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Mobility & Usage")
        daily_dist  = st.slider("Avg Daily Distance (km)", 0.0, 80.0, 22.0, step=0.5)
        radius      = st.slider("Travel Radius (km)", 0.0, 120.0, 18.0, step=1.0)
        hw_dist     = st.slider("Home-Work Distance (km)", 0.0, 60.0, 15.0, step=0.5)
        data_gb     = st.slider("Data Usage (GB/month)", 0.0, 100.0, 16.0, step=0.5)
        roaming     = st.slider("Roaming Days/Month", 0.0, 30.0, 0.5, step=0.5)

        st.subheader("Time Pattern")
        evening_r = st.slider("Evening Data Ratio", 0.0, 1.0, 0.35, step=0.01)
        weekend_r = st.slider("Weekend Data Ratio", 0.0, 1.0, 0.30, step=0.01)
        peak_r    = st.slider("Peak Hour Ratio",    0.0, 1.0, 0.45, step=0.01)

        st.subheader("POI Affinity")
        office_s  = st.slider("Office POI Score",  0.0, 1.0, 0.55, step=0.05)
        transit_s = st.slider("Transit POI Score", 0.0, 1.0, 0.75, step=0.05)
        mall_s    = st.slider("Mall POI Score",    0.0, 1.0, 0.30, step=0.05)

        classify_btn = st.button("🔮 Classify Subscriber", use_container_width=True)

    with col_result:
        st.subheader("Classification Result")

        if classify_btn:
            from src.api.model_loader import ModelLoader
            from src.features.mobility_features import MobilityFeatureEngineer, TemporalUsageEngineer
            from src.features.poi_features import POIFeatureEngineer

            ldr = ModelLoader()
            ldr.load()

            if not ldr.is_loaded():
                st.error("Models not loaded. Run `make pipeline` first.")
            else:
                remaining_time = 1.0 - evening_r - weekend_r * 0.3
                payload = {
                    "subscriber_id": "DEMO_001",
                    "avg_daily_distance_km":  daily_dist,
                    "travel_radius_km":       radius,
                    "home_work_distance_km":  hw_dist,
                    "n_frequent_locations":   3.0,
                    "pct_time_at_home":       max(0.1, 0.7 - daily_dist / 100),
                    "pct_time_at_work":       0.22,
                    "mobility_entropy":       min(2.5, daily_dist / 15),
                    "data_usage_gb":          data_gb,
                    "call_minutes_monthly":   280.0,
                    "sms_monthly":            60.0,
                    "roaming_days_monthly":   roaming,
                    "morning_data_ratio":     0.28,
                    "afternoon_data_ratio":   0.18,
                    "evening_data_ratio":     evening_r,
                    "night_data_ratio":       max(0.05, 1 - 0.28 - 0.18 - evening_r),
                    "weekend_data_ratio":     weekend_r,
                    "peak_hour_data_ratio":   peak_r,
                    "poi_office_score":       office_s,
                    "poi_transit_score":      transit_s,
                    "poi_mall_score":         mall_s,
                    "poi_residential_score":  0.40,
                    "poi_entertainment_score":0.25,
                    "home_zone_rsrq":         -11.0,
                    "work_zone_rsrq":         -12.0,
                    "avg_throughput_mbps":    28.0,
                    "monthly_charges":        75.0,
                    "tenure_months":          36.0,
                    "senior_citizen":         0,
                    "home_lat":               -6.2088,
                    "home_lon":               106.8456,
                }

                result = ldr.predict_segment(payload)
                seg_id = result["segment_id"]
                seg_name_r = result["segment_name"]
                conf = result["confidence"]

                color = SEGMENT_COLORS[seg_id % len(SEGMENT_COLORS)]

                st.markdown(
                    f"<div style='background:{color};color:white;padding:16px;"
                    f"border-radius:8px;font-size:20px;font-weight:600;"
                    f"text-align:center;'>{seg_name_r}</div>",
                    unsafe_allow_html=True,
                )

                st.metric("Confidence", f"{conf:.0%}")
                st.write(result.get("segment_description", ""))

                if result.get("h3_home_cell"):
                    st.code(f"Home H3 cell: {result['h3_home_cell']}")

                st.subheader("Recommended Plans")
                for i, plan_id in enumerate(result.get("recommended_plans", [])[:3], 1):
                    plan_name = plan_id.replace("_", " ").title()
                    st.markdown(f"**{i}.** {plan_name}")
        else:
            st.info("Adjust the sliders and click **Classify Subscriber** to see results.")
