"""
src/visualization/geo_plots.py
────────────────────────────────
Geospatial visualization for segment density maps.
Builds layered Folium maps showing subscriber segments on H3 hex grids.
"""

from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from loguru import logger

SEGMENT_COLORS = [
    "#e74c3c", "#3498db", "#9b59b6", "#2ecc71",
    "#f39c12", "#95a5a6", "#1abc9c", "#e67e22",
]


def make_base_map(
    centre_lat: float = -6.2,
    centre_lon: float = 106.85,
    zoom: int = 10,
) -> folium.Map:
    return folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )


def add_h3_segment_layer(
    m: folium.Map,
    geojson: dict,
    segment_names: Optional[dict] = None,
) -> folium.Map:
    """Add H3 hex polygons coloured by dominant segment."""
    for feature in geojson.get("features", []):
        props = feature["properties"]
        color = props.get("color", "#888888")
        seg_name = props.get("segment_name", "Unknown")
        n_subs   = props.get("n_subscribers", 0)
        charges  = props.get("avg_monthly_charges", 0)
        data_gb  = props.get("avg_data_usage_gb", 0)
        rsrq     = props.get("avg_rsrq", 0)

        try:
            folium.GeoJson(
                feature,
                style_function=lambda f, c=color: {
                    "fillColor":   c,
                    "color":       "#444",
                    "weight":      0.4,
                    "fillOpacity": 0.65,
                },
                tooltip=folium.Tooltip(
                    f"<b>{seg_name}</b><br>"
                    f"Subscribers: {n_subs:,}<br>"
                    f"Avg ARPU: ${charges:.0f}<br>"
                    f"Avg Data: {data_gb:.1f} GB<br>"
                    f"RSRQ: {rsrq:.1f} dB"
                ),
            ).add_to(m)
        except Exception:
            pass

    return m


def add_segment_legend(
    m: folium.Map,
    segment_names: dict,
) -> folium.Map:
    rows = "".join([
        f'<tr><td style="background:{SEGMENT_COLORS[i % len(SEGMENT_COLORS)]};'
        f'width:16px;height:12px;border-radius:3px;"></td>'
        f'<td style="padding-left:7px;font-size:12px;">[{seg_id}] {name}</td></tr>'
        for i, (seg_id, name) in enumerate(sorted(segment_names.items()))
    ])
    legend_html = f"""
    <div style="position:fixed;bottom:25px;left:25px;z-index:1000;
                background:#fff;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;box-shadow:2px 2px 6px rgba(0,0,0,.15);">
      <b style="font-size:13px;">Subscriber Segments</b>
      <table style="margin-top:5px;border-spacing:4px;">{rows}</table>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def add_subscriber_scatter(
    m: folium.Map,
    df: pd.DataFrame,
    segment_names: Optional[dict] = None,
    max_points: int = 4000,
) -> folium.Map:
    """Add subscriber home location scatter points."""
    sample = df.sample(min(max_points, len(df)), random_state=42)
    for _, row in sample.iterrows():
        seg_id = int(row.get("segment_id", 0))
        color  = SEGMENT_COLORS[seg_id % len(SEGMENT_COLORS)]
        name   = segment_names.get(seg_id, f"Seg {seg_id}") if segment_names else f"Seg {seg_id}"
        folium.CircleMarker(
            location=[row["home_lat"], row["home_lon"]],
            radius=2.5,
            color=color,
            fill=True, fill_color=color, fill_opacity=0.55, weight=0.3,
            tooltip=f"Segment: {name}",
        ).add_to(m)
    return m


def add_data_usage_heatmap(
    m: folium.Map,
    df: pd.DataFrame,
    weight_col: str = "data_usage_gb",
) -> folium.Map:
    """Overlay a data usage heatmap."""
    col = weight_col if weight_col in df.columns else "data_usage_gb"
    max_val = df[col].quantile(0.95) if col in df.columns else 1.0
    heat_data = [
        [row["home_lat"], row["home_lon"], row.get(col, 0) / max_val]
        for _, row in df.iterrows()
        if pd.notna(row.get("home_lat")) and pd.notna(row.get("home_lon"))
    ]
    if heat_data:
        HeatMap(
            heat_data,
            radius=10, blur=8, max_zoom=13,
            gradient={"0.0": "#2ecc71", "0.4": "#f39c12", "0.7": "#e74c3c", "1.0": "#8e44ad"},
            name="Data Usage Heatmap",
        ).add_to(m)
    return m


def create_full_segment_map(
    geojson: dict,
    df: Optional[pd.DataFrame] = None,
    segment_names: Optional[dict] = None,
    output_path: Optional[Path] = None,
    include_heatmap: bool = False,
) -> folium.Map:
    """
    Build a complete layered segment map:
      - H3 hex polygons (dominant segment choropleth)
      - Subscriber scatter (optional)
      - Data usage heatmap (optional)
      - Segment legend
    """
    # Compute centre from features
    lats, lons = [], []
    for feat in geojson.get("features", [])[:50]:
        for coord in feat["geometry"]["coordinates"][0]:
            lons.append(coord[0])
            lats.append(coord[1])

    centre = [np.mean(lats) if lats else -6.2, np.mean(lons) if lons else 106.85]
    m = make_base_map(centre[0], centre[1])

    m = add_h3_segment_layer(m, geojson, segment_names)

    if df is not None and "home_lat" in df.columns:
        m = add_subscriber_scatter(m, df, segment_names)

    if include_heatmap and df is not None:
        m = add_data_usage_heatmap(m, df)

    if segment_names:
        m = add_segment_legend(m, segment_names)

    folium.LayerControl(collapsed=False).add_to(m)

    if output_path:
        m.save(str(output_path))
        logger.success(f"Map saved → {output_path}")

    return m
