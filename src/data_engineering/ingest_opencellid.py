"""
src/data_engineering/ingest_opencellid.py
──────────────────────────────────────────
Downloads and processes REAL cell tower data from OpenCelliD.
Used to enrich H3 cells with tower density and signal quality estimates.

HOW TO GET THE DATA (free):
  1. Register at https://opencellid.org/register
  2. Get your API token from your profile page
  3. Add to .env:  OPENCELLID_TOKEN=your_token_here
  4. Run: python src/data_engineering/ingest_opencellid.py --country ID

Alternative (no account):
  Download the full CSV manually from opencellid.org/downloads.php
  Place at data/external/cell_towers_raw.csv, then run with --local flag.

Usage:
    python src/data_engineering/ingest_opencellid.py --country ID
    python src/data_engineering/ingest_opencellid.py --local data/external/cell_towers_raw.csv
"""

import os
import sys
import argparse
import gzip
import shutil
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Country MCC codes
COUNTRY_MCC = {
    "ID": [510],        # Indonesia
    "SG": [525],        # Singapore
    "MY": [502],        # Malaysia
    "AU": [505],        # Australia
    "GB": [234, 235],   # UK
    "US": [310, 311],   # USA
}

# Indonesia bounding box
DEFAULT_BOUNDS = {
    "ID": {"lat_min": -11.0, "lat_max": 6.0, "lon_min": 95.0, "lon_max": 141.0},
    "SG": {"lat_min": 1.15,  "lat_max": 1.48, "lon_min": 103.6, "lon_max": 104.1},
    "MY": {"lat_min": 0.8,   "lat_max": 7.4,  "lon_min": 99.6,  "lon_max": 119.3},
}


def download_opencellid(token: str, output_dir: Path) -> Path:
    """Download full OpenCelliD database (~1 GB compressed)."""
    url = (
        f"https://download.opencellid.org/ocid/downloads"
        f"?token={token}&type=full&file=cell_towers.csv.gz"
    )
    gz_path = output_dir / "cell_towers.csv.gz"
    logger.info(f"Downloading OpenCelliD database (~1 GB)...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=131072):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}% ({downloaded/1e6:.0f} MB)", end="")
    print()
    logger.success(f"Downloaded → {gz_path}")
    return gz_path


def filter_by_country(gz_path: Path, output_dir: Path, country: str, bounds: dict) -> pd.DataFrame:
    """Decompress and filter towers by country MCC and bounding box."""
    mcc_list = COUNTRY_MCC.get(country, [])
    logger.info(f"Filtering for {country} (MCC={mcc_list})...")

    csv_path = output_dir / "cell_towers_raw.csv"
    logger.info("Decompressing...")
    with gzip.open(gz_path, "rb") as fin, open(csv_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)

    cols = ["radio", "mcc", "net", "area", "cell", "unit",
            "lon", "lat", "range", "samples", "changeable",
            "created", "updated", "averageSignal"]
    chunks = []
    for chunk in pd.read_csv(csv_path, names=cols, chunksize=500_000, low_memory=False):
        if mcc_list:
            chunk = chunk[chunk["mcc"].isin(mcc_list)]
        chunk = chunk[
            (chunk["lat"] >= bounds["lat_min"]) & (chunk["lat"] <= bounds["lat_max"]) &
            (chunk["lon"] >= bounds["lon_min"]) & (chunk["lon"] <= bounds["lon_max"])
        ]
        if len(chunk):
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    logger.info(f"Filtered: {len(df):,} towers for {country}")
    return df


def process_towers(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Clean tower data and convert to GeoDataFrame."""
    df = df.rename(columns={
        "net": "mnc", "area": "lac", "cell": "cell_id",
        "lon": "longitude", "lat": "latitude",
        "range": "range_m", "averageSignal": "avg_signal",
    })
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

    for col in ["created", "updated"]:
        df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")

    radio_map = {"GSM": "2G", "UMTS": "3G", "LTE": "4G", "NR": "5G"}
    df["generation"] = df["radio"].map(radio_map).fillna("Unknown")

    geom = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def add_h3_tower_density(gdf: gpd.GeoDataFrame, resolution: int = 8) -> pd.DataFrame:
    """Count towers per H3 cell for spatial enrichment."""
    try:
        import h3
        gdf = gdf.copy()
        gdf["h3_cell"] = gdf.apply(
            lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], resolution),
            axis=1
        )
        density = gdf.groupby("h3_cell").agg(
            tower_count=("h3_cell", "count"),
            lte_count=("generation", lambda x: (x == "4G").sum()),
            nr_count=("generation",  lambda x: (x == "5G").sum()),
            avg_range_m=("range_m", "mean"),
        ).reset_index()
        logger.info(f"H3 tower density: {len(density):,} cells at resolution {resolution}")
        return density
    except ImportError:
        logger.warning("H3 not installed — skipping H3 density calculation")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Ingest OpenCelliD cell tower data")
    parser.add_argument("--country", default="ID", help="ISO country code")
    parser.add_argument("--token", default=None, help="OpenCelliD API token")
    parser.add_argument("--local", default=None, help="Path to local CSV file")
    parser.add_argument("--output", default="data/external")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.getenv("OPENCELLID_TOKEN")
    bounds = DEFAULT_BOUNDS.get(args.country,
                                {"lat_min": -11, "lat_max": 6, "lon_min": 95, "lon_max": 141})

    if args.local:
        logger.info(f"Loading local file: {args.local}")
        cols = ["radio", "mcc", "net", "area", "cell", "unit",
                "lon", "lat", "range", "samples", "changeable",
                "created", "updated", "averageSignal"]
        df_raw = pd.read_csv(args.local, names=cols, low_memory=False)
        mcc_list = COUNTRY_MCC.get(args.country, [])
        if mcc_list:
            df_raw = df_raw[df_raw["mcc"].isin(mcc_list)]
        df_raw = df_raw[
            (df_raw["lat"].between(bounds["lat_min"], bounds["lat_max"])) &
            (df_raw["lon"].between(bounds["lon_min"], bounds["lon_max"]))
        ]
    elif token:
        gz_path = download_opencellid(token, output_dir)
        df_raw = filter_by_country(gz_path, output_dir, args.country, bounds)
    else:
        logger.error(
            "No data source provided.\n"
            "  Option 1: Set OPENCELLID_TOKEN in .env and run without --local\n"
            "  Option 2: Download manually from opencellid.org and use --local\n"
            "  Get a free token at: https://opencellid.org/register"
        )
        sys.exit(1)

    gdf = process_towers(df_raw)

    # Save outputs
    parquet_path = output_dir / f"cell_towers_{args.country}.parquet"
    geojson_path = output_dir / f"cell_towers_{args.country}.geojson"
    gdf.drop(columns=["geometry"]).to_parquet(parquet_path, index=False)
    gdf.to_file(geojson_path, driver="GeoJSON")

    # H3 density table
    density = add_h3_tower_density(gdf)
    if len(density):
        density.to_parquet(output_dir / f"tower_density_h3_{args.country}.parquet", index=False)

    logger.success(f"Saved {len(gdf):,} towers → {parquet_path}")
    print("\nRadio type breakdown:")
    print(gdf["radio"].value_counts().to_string())


if __name__ == "__main__":
    main()
