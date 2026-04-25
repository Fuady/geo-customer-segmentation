"""
src/models/geo_density.py
──────────────────────────
Builds a geospatial density map of subscriber segments.

For each H3 hex cell, computes:
  - Dominant segment (most common segment in that cell)
  - Segment distribution (% of each segment)
  - Avg monthly charges per cell
  - Avg data usage per cell
  - Coverage quality (avg RSRQ)

Outputs:
  - data/processed/segment_density.geojson  — H3 polygons for Folium/Kepler
  - data/processed/segment_density.parquet  — tabular version

Usage:
    python src/models/geo_density.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import json
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


SEGMENT_COLORS = [
    "#e74c3c",  # 0 Urban Commuter       — red
    "#3498db",  # 1 Home Office Pro       — blue
    "#9b59b6",  # 2 Digital Nomad         — purple
    "#2ecc71",  # 3 Weekend Explorer      — green
    "#f39c12",  # 4 Social Hub            — orange
    "#95a5a6",  # 5 Senior Steady         — gray
    "#1abc9c",  # 6 Young Professional    — teal
    "#e67e22",  # 7+ extra                — dark orange
]


def build_segment_density(
    df: pd.DataFrame,
    h3_col: str = "home_h3",
    segment_col: str = "segment_id",
    min_subs: int = 5,
) -> pd.DataFrame:
    """Aggregate subscriber segments to H3 cell level."""
    # Filter noise
    df_valid = df[df[segment_col] >= 0].copy()

    agg = df_valid.groupby(h3_col).agg(
        n_subscribers=(segment_col, "count"),
        dominant_segment=(segment_col, lambda x: x.mode()[0]),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_data_usage_gb=("data_usage_gb", "mean"),
        avg_rsrq=("home_zone_rsrq", "mean"),
        segment_entropy=(segment_col, lambda x: _segment_entropy(x)),
    ).reset_index()

    # Segment distribution columns
    n_segs = df_valid[segment_col].nunique()
    for seg_id in sorted(df_valid[segment_col].unique()):
        col = f"pct_seg_{seg_id}"
        seg_counts = df_valid.groupby(h3_col)[segment_col].apply(
            lambda x: (x == seg_id).sum() / len(x)
        ).reset_index()
        seg_counts.columns = [h3_col, col]
        agg = agg.merge(seg_counts, on=h3_col, how="left")
        agg[col] = agg[col].fillna(0).round(3)

    agg = agg[agg["n_subscribers"] >= min_subs].copy()
    agg["avg_monthly_charges"] = agg["avg_monthly_charges"].round(2)
    agg["avg_data_usage_gb"]   = agg["avg_data_usage_gb"].round(2)
    agg["avg_rsrq"]            = agg["avg_rsrq"].round(2)

    logger.info(f"Segment density grid: {len(agg):,} H3 cells")
    return agg


def _segment_entropy(series: pd.Series) -> float:
    """Shannon entropy of segment distribution in a cell."""
    counts = series.value_counts(normalize=True)
    return float(-np.sum(counts * np.log(counts + 1e-10)))


def density_to_geojson(
    density_df: pd.DataFrame,
    h3_col: str = "home_h3",
    segment_names: dict = None,
) -> dict:
    """Convert H3 density DataFrame to GeoJSON FeatureCollection."""
    try:
        import h3
    except ImportError:
        logger.error("h3 not installed — cannot generate GeoJSON. pip install h3")
        return {"type": "FeatureCollection", "features": []}

    features = []
    for _, row in density_df.iterrows():
        h3_cell = row[h3_col]
        try:
            boundary = h3.h3_to_geo_boundary(h3_cell, geo_json=True)
            seg_id = int(row["dominant_segment"])
            color_idx = seg_id % len(SEGMENT_COLORS)

            seg_name = "Unknown"
            if segment_names:
                seg_name = segment_names.get(seg_id, f"Segment {seg_id}")

            props = {
                "h3_cell":           h3_cell,
                "dominant_segment":  seg_id,
                "segment_name":      seg_name,
                "n_subscribers":     int(row["n_subscribers"]),
                "avg_monthly_charges": float(row["avg_monthly_charges"]),
                "avg_data_usage_gb": float(row["avg_data_usage_gb"]),
                "avg_rsrq":          float(row["avg_rsrq"]),
                "segment_entropy":   float(row["segment_entropy"]),
                "color":             SEGMENT_COLORS[color_idx],
            }
            # Add segment percentage columns
            for col in density_df.columns:
                if col.startswith("pct_seg_"):
                    props[col] = float(row[col])

            features.append({
                "type": "Feature",
                "geometry": {
                    "type":        "Polygon",
                    "coordinates": [boundary],
                },
                "properties": props,
            })
        except Exception:
            pass

    return {"type": "FeatureCollection", "features": features}


def create_folium_map(
    geojson: dict,
    output_path: Path,
    segment_names: dict = None,
) -> None:
    """Create interactive Folium map of segment density."""
    try:
        import folium
    except ImportError:
        logger.warning("folium not installed — skipping interactive map")
        return

    # Compute centre from features
    lats, lons = [], []
    for feat in geojson["features"]:
        coords = feat["geometry"]["coordinates"][0]
        for lon, lat in coords:
            lats.append(lat)
            lons.append(lon)

    centre = [np.mean(lats) if lats else -6.2,
              np.mean(lons) if lons else 106.85]

    m = folium.Map(location=centre, zoom_start=10, tiles="CartoDB positron")

    # Build legend HTML
    if segment_names:
        rows = "".join([
            f'<tr><td style="background:{SEGMENT_COLORS[i % len(SEGMENT_COLORS)]};'
            f'width:16px;height:12px;border-radius:3px;"></td>'
            f'<td style="padding-left:7px;font-size:12px;">{name}</td></tr>'
            for i, name in sorted(segment_names.items())
        ])
        legend_html = f"""
        <div style="position:fixed;bottom:25px;left:25px;z-index:1000;
                    background:#fff;padding:10px 14px;border-radius:8px;
                    border:1px solid #ccc;box-shadow:2px 2px 6px rgba(0,0,0,.15);">
          <b style="font-size:13px;">Subscriber Segments</b>
          <table style="margin-top:5px;border-spacing:4px;">{rows}</table>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    # Add H3 polygons
    for feat in geojson["features"]:
        color = feat["properties"].get("color", "#888")
        seg_name = feat["properties"].get("segment_name", "Unknown")
        n_subs   = feat["properties"].get("n_subscribers", 0)
        charges  = feat["properties"].get("avg_monthly_charges", 0)
        data_gb  = feat["properties"].get("avg_data_usage_gb", 0)
        rsrq     = feat["properties"].get("avg_rsrq", 0)

        folium.GeoJson(
            feat,
            style_function=lambda f, c=color: {
                "fillColor":   c,
                "color":       "#555",
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

    m.save(str(output_path))
    logger.success(f"Interactive map saved → {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate geo segment density map")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--data",    default="data/processed/features_segmented.parquet")
    parser.add_argument("--output",  default="data/processed/segment_density.geojson")
    args = parser.parse_args()

    config = load_config(args.config)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Segmented data not found: {data_path}")
        logger.error("Run clustering.py and segment_profiler.py first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)

    # Load segment names if available
    seg_names_path = Path("data/models/segment_names.pkl")
    segment_names = None
    if seg_names_path.exists():
        import joblib
        segment_names = joblib.load(seg_names_path)

    # Build density
    density_df = build_segment_density(
        df, h3_col="home_h3", segment_col="segment_id", min_subs=5
    )

    # Save parquet
    parquet_path = Path(args.output).with_suffix(".parquet")
    density_df.to_parquet(parquet_path, index=False)
    logger.success(f"Density parquet → {parquet_path}")

    # Convert to GeoJSON
    geojson = density_to_geojson(density_df, segment_names=segment_names)

    with open(args.output, "w") as f:
        json.dump(geojson, f)
    logger.success(f"GeoJSON → {args.output}")

    # Interactive map
    map_path = Path(args.output).with_name("segment_density_map.html")
    create_folium_map(geojson, map_path, segment_names)

    print("\n" + "=" * 55)
    print("GEO DENSITY MAP COMPLETE")
    print("=" * 55)
    print(f"  H3 cells (≥5 subs)  : {len(density_df):,}")
    print(f"  Dominant segment map :")
    if "dominant_segment" in density_df.columns and segment_names:
        for seg_id, count in density_df["dominant_segment"].value_counts().items():
            name = segment_names.get(seg_id, f"Segment {seg_id}")
            print(f"    [{seg_id}] {name:22s}: {count:4d} H3 cells")
    print("=" * 55)


if __name__ == "__main__":
    main()
