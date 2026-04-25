# Data Sources Guide

## 1. Synthetic CDR Subscriber Dataset (Auto-generated — no signup needed)

```bash
python src/data_engineering/generate_data.py --n_subscribers 30000
```

Generates `data/raw/subscribers.parquet` with 30,000 subscribers across 7 behavioral personas. Each subscriber has CDR-derived mobility features, temporal usage patterns, POI affinity scores, and geographic coordinates.

| Column | Type | Description |
|--------|------|-------------|
| `subscriber_id` | str | Unique ID |
| `avg_daily_distance_km` | float | Average km travelled per day |
| `travel_radius_km` | float | Max distance from home location |
| `home_work_distance_km` | float | Distance between home and work zones |
| `n_frequent_locations` | float | Number of regularly visited locations |
| `pct_time_at_home` | float | Fraction of time at home H3 cell |
| `pct_time_at_work` | float | Fraction of time at work H3 cell |
| `mobility_entropy` | float | Shannon entropy of location visits |
| `data_usage_gb` | float | Monthly data consumption (GB) |
| `call_minutes_monthly` | float | Total call minutes per month |
| `roaming_days_monthly` | float | Days with roaming activity |
| `morning_data_ratio` | float | Fraction of data used 06:00–12:00 |
| `evening_data_ratio` | float | Fraction of data used 18:00–22:00 |
| `weekend_data_ratio` | float | Fraction of data used on weekends |
| `poi_office_score` | float | Affinity for office/business locations |
| `poi_transit_score` | float | Affinity for transit hubs |
| `poi_mall_score` | float | Affinity for shopping malls |
| `home_lat`, `home_lon` | float | Home location coordinates |
| `home_h3`, `work_h3` | str | H3 cell indexes (resolution 8) |
| `true_persona_name` | str | Ground truth persona (for evaluation) |

---

## 2. OpenStreetMap POIs (Free — no account needed)

```bash
python src/data_engineering/ingest_osm.py --city "Jakarta"
```

Downloads POIs from the Overpass API (free, no signup). Used to enrich H3 cells with real-world POI counts.

**Available cities:** Jakarta, Surabaya, Bandung, Singapore, Kuala Lumpur, Bangkok, Manila

**POI categories:** office, transit, train_station, mall, entertainment, restaurant, education, healthcare, residential, park

---

## 3. OpenCelliD Tower Data (Free — requires free account)

```bash
python src/data_engineering/ingest_opencellid.py --country ID
```

1. Register at https://opencellid.org/register (free)
2. Get your API token from your profile
3. Add `OPENCELLID_TOKEN=your_token` to `.env`
4. Run the ingest script

Used to compute real cell tower density and coverage quality per H3 cell.

---

## 4. GADM Administrative Boundaries (Optional)

```bash
wget "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IDN_2.json.zip" -O data/external/gadm_idn.zip
unzip data/external/gadm_idn.zip -d data/external/
```

Used to annotate H3 cells with province/district names for reporting.

---

## 5. WorldPop Population Density (Optional)

Download from https://hub.worldpop.org/geodata/listing?id=69 (Indonesia 2020).
Place at `data/external/idn_ppp_2020_1km.tif`.
Used to weight segment density maps by actual population.
