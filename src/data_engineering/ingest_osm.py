"""
src/data_engineering/ingest_osm.py
────────────────────────────────────
Downloads Points of Interest from OpenStreetMap via the Overpass API.
Completely FREE — no API key or account needed.

POIs are used to build POI affinity scores for geo-behavioral segmentation:
  - Office/business zones → identify work locations
  - Transit hubs → identify commuters
  - Shopping malls/entertainment → Social Hub segment
  - Residential areas → home zone quality
  - Universities/schools → identify student segment

Usage:
    python src/data_engineering/ingest_osm.py --city "Jakarta"
    python src/data_engineering/ingest_osm.py --city "Singapore"
"""

import argparse
import time
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Named city bounding boxes [south, west, north, east]
CITY_BOUNDS = {
    "Jakarta":      [-6.50, 106.60, -5.90, 107.10],
    "Surabaya":     [-7.40, 112.55, -7.10, 112.85],
    "Bandung":      [-6.98, 107.50, -6.83, 107.72],
    "Singapore":    [1.20,  103.60,  1.48, 104.05],
    "Kuala Lumpur": [2.95,  101.55,  3.28, 101.85],
    "Bangkok":      [13.55, 100.40, 13.95, 100.90],
    "Manila":       [14.45, 120.85, 14.75, 121.20],
}

# POI queries — Overpass QL filters
POI_QUERIES = {
    "office":       'node["office"]["office"!="diplomatic"]',
    "transit":      'node["public_transport"~"station|stop_position"]',
    "train_station":'node["railway"~"station|subway_entrance"]',
    "mall":         'node["shop"~"mall|department_store|supermarket"]',
    "entertainment":'node["amenity"~"cinema|theatre|nightclub|bar|pub"]',
    "restaurant":   'node["amenity"~"restaurant|cafe|fast_food"]',
    "education":    'node["amenity"~"university|college|school"]',
    "healthcare":   'node["amenity"~"hospital|clinic|pharmacy"]',
    "residential":  'node["place"~"neighbourhood|suburb|residential"]',
    "park":         'node["leisure"~"park|garden|sports_centre"]',
}


def fetch_pois(bbox: list, poi_type: str, query_filter: str) -> list:
    """Query Overpass API for one POI type in a bounding box."""
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:60];
    (
      {query_filter}({south},{west},{north},{east});
    );
    out body;
    """
    logger.info(f"  Fetching '{poi_type}'...")
    try:
        r = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=90,
            headers={"User-Agent": "TelecomSegmentationResearch/1.0"},
        )
        r.raise_for_status()
        elements = r.json().get("elements", [])
        logger.info(f"    → {len(elements):,} POIs")
        return elements
    except requests.exceptions.Timeout:
        logger.warning(f"    → timeout for '{poi_type}' — skipping")
        return []
    except Exception as e:
        logger.warning(f"    → error for '{poi_type}': {e} — skipping")
        return []


def elements_to_gdf(elements: list, poi_type: str) -> gpd.GeoDataFrame:
    rows = []
    for el in elements:
        if el.get("type") == "node" and "lat" in el and "lon" in el:
            tags = el.get("tags", {})
            rows.append({
                "osm_id":   el["id"],
                "poi_type": poi_type,
                "name":     tags.get("name", ""),
                "latitude": el["lat"],
                "longitude": el["lon"],
            })
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_id", "poi_type", "name",
                                          "latitude", "longitude", "geometry"])
    df = pd.DataFrame(rows)
    geom = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def main():
    parser = argparse.ArgumentParser(description="Download OSM POIs for segmentation enrichment")
    parser.add_argument("--city", type=str, default="Jakarta",
                        help=f"City: {list(CITY_BOUNDS.keys())}")
    parser.add_argument("--bbox", type=str, default=None,
                        help="Custom bbox: south,west,north,east")
    parser.add_argument("--output", type=str, default="data/external")
    parser.add_argument("--poi_types", nargs="+", default=list(POI_QUERIES.keys()))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.bbox:
        bbox = [float(x) for x in args.bbox.split(",")]
        city_slug = "custom"
    elif args.city in CITY_BOUNDS:
        bbox = CITY_BOUNDS[args.city]
        city_slug = args.city.lower().replace(" ", "_")
    else:
        logger.error(f"Unknown city '{args.city}'. Available: {list(CITY_BOUNDS.keys())}")
        return

    logger.info(f"Downloading POIs for: {args.city} | bbox: {bbox}")

    gdfs = []
    for poi_type in args.poi_types:
        if poi_type not in POI_QUERIES:
            continue
        gdf = elements_to_gdf(fetch_pois(bbox, poi_type, POI_QUERIES[poi_type]), poi_type)
        if len(gdf) > 0:
            gdfs.append(gdf)
        time.sleep(2)  # polite rate-limiting for the free API

    if not gdfs:
        logger.error("No POIs collected.")
        return

    gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")

    parquet_path = output_dir / f"osm_pois_{city_slug}.parquet"
    geojson_path = output_dir / f"osm_pois_{city_slug}.geojson"
    gdf_all.drop(columns=["geometry"]).to_parquet(parquet_path, index=False)
    gdf_all.to_file(geojson_path, driver="GeoJSON")

    logger.success(f"Saved {len(gdf_all):,} POIs → {parquet_path}")
    print("\nPOI breakdown:")
    print(gdf_all["poi_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
