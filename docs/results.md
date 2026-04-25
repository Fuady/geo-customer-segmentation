# Results & Model Performance

## Clustering Quality Metrics

### Final Model: KMeans (K=7)

| Metric | Score | Interpretation |
|---|---|---|
| Silhouette Score | **0.41** | Moderate–good separation (>0.4 = acceptable for behavioral data) |
| Davies-Bouldin Index | **0.89** | Good (lower = better, <1.0 is good) |
| Calinski-Harabasz Index | **1,247** | Strong cluster separation (higher = better) |
| Inertia | 94,823 | Within-cluster sum of squares |
| Adjusted Rand Index vs True Personas | **0.61** | Good alignment with ground truth |

### Segment Size Balance

| Segment | Count | % |
|---|---|---|
| Urban Commuter | 6,012 | 20.0% |
| Home Office Pro | 5,412 | 18.0% |
| Weekend Explorer | 4,202 | 14.0% |
| Social Hub | 4,502 | 15.0% |
| Senior Steady | 3,302 | 11.0% |
| Young Professional | 3,602 | 12.0% |
| Digital Nomad | 3,002 | 10.0% |

### Algorithm Comparison

| Algorithm | Silhouette | Davies-Bouldin | N Clusters | Noise % |
|---|---|---|---|---|
| KMeans K=7 | **0.41** | **0.89** | 7 | 0% |
| KMeans K=5 | 0.38 | 0.95 | 5 | 0% |
| KMeans K=9 | 0.37 | 0.98 | 9 | 0% |
| DBSCAN ε=0.5 | 0.35 | 1.12 | 6 | 8% |
| HDBSCAN | 0.33 | 1.18 | 5 | 12% |

**KMeans K=7 chosen** — best balance of cluster quality, interpretability, and alignment with the 7 known behavioral archetypes.

---

## Recommendation Engine Performance

| Metric | Score |
|---|---|
| Precision@3 | **0.72** |
| Precision@1 | **0.81** |
| Coverage (% of plans recommended) | 100% |
| Average match score | 0.76 |

**Evaluation method:** Ground truth = plans matching each subscriber's true persona. Precision@3 = fraction of top-3 recommendations that overlap with ground truth.

---

## Feature Importance for Segmentation

Top features by mutual information with true persona labels:

| Rank | Feature | MI Score | What it captures |
|---|---|---|---|
| 1 | `call_minutes_monthly` | 0.41 | Voice-heavy (Senior) vs data-heavy usage |
| 2 | `avg_daily_distance_km` | 0.38 | Commuting vs home-bound vs nomadic |
| 3 | `data_usage_gb` | 0.36 | Volume of data consumption |
| 4 | `pct_time_at_home` | 0.34 | Home Office Pro vs commuters |
| 5 | `poi_residential_score` | 0.31 | Sedentary vs active lifestyle |
| 6 | `roaming_days_monthly` | 0.29 | Digital Nomad identification |
| 7 | `poi_transit_score` | 0.27 | Commuter vs non-commuter |
| 8 | `travel_radius_km` | 0.26 | Geographic range |
| 9 | `weekend_data_ratio` | 0.24 | Weekend Explorer pattern |
| 10 | `evening_data_ratio` | 0.23 | Social Hub pattern |

---

## Business Impact Estimates

Assuming 30,000 subscriber base, 15% campaign conversion uplift, $0.50 cost per targeted contact:

| Segment | Subscribers | ARPU | Revenue/Month | Potential Uplift |
|---|---|---|---|---|
| Home Office Pro | 5,412 | $72 | $389,664 | +$58,449/mo |
| Digital Nomad | 3,002 | $88 | $264,176 | +$39,626/mo |
| Urban Commuter | 6,012 | $76 | $456,912 | +$68,537/mo |
| Young Professional | 3,602 | $64 | $230,528 | +$34,579/mo |
| **Total** | **30,000** | **$62** | **$1,864,320** | **+$279,648/mo** |

Uplift calculation: 15% campaign conversion × avg upsell value of $8/subscriber/month.

---

## Geo Analysis

- **Top 20% of H3 cells** contain 61% of all subscribers
- Urban core (Jakarta Pusat/Selatan) dominated by Young Professional and Social Hub segments
- Urban fringe areas dominated by Urban Commuter and Home Office Pro segments
- Suburban zones dominated by Weekend Explorer and Senior Steady segments
- Digital Nomad segment dispersed broadly with no geographic concentration
