"""
src/api/schemas.py
───────────────────
Pydantic v2 schemas for the segmentation API.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class SubscriberFeatures(BaseModel):
    """Input features for geo-behavioral segment classification."""

    subscriber_id: str = Field(..., example="SUB_00000001")

    # Mobility features
    avg_daily_distance_km:  float = Field(10.0, ge=0, example=22.0)
    travel_radius_km:       float = Field(15.0, ge=0, example=18.0)
    home_work_distance_km:  float = Field(8.0,  ge=0, example=15.0)
    n_frequent_locations:   float = Field(3.0,  ge=1, example=3.0)
    pct_time_at_home:       float = Field(0.55, ge=0, le=1, example=0.52)
    pct_time_at_work:       float = Field(0.22, ge=0, le=1, example=0.28)
    mobility_entropy:       float = Field(1.5,  ge=0, example=1.8)

    # Usage volume
    data_usage_gb:          float = Field(15.0, ge=0, example=16.0)
    call_minutes_monthly:   float = Field(300.0,ge=0, example=280.0)
    sms_monthly:            float = Field(50.0, ge=0, example=60.0)
    roaming_days_monthly:   float = Field(0.5,  ge=0, example=0.5)

    # Temporal patterns
    morning_data_ratio:     float = Field(0.25, ge=0, le=1, example=0.28)
    afternoon_data_ratio:   float = Field(0.22, ge=0, le=1, example=0.18)
    evening_data_ratio:     float = Field(0.30, ge=0, le=1, example=0.35)
    night_data_ratio:       float = Field(0.18, ge=0, le=1, example=0.19)
    weekend_data_ratio:     float = Field(0.30, ge=0, le=1, example=0.30)
    peak_hour_data_ratio:   float = Field(0.35, ge=0, le=1, example=0.45)

    # POI affinity (0–1 scores)
    poi_office_score:       float = Field(0.40, ge=0, le=1, example=0.55)
    poi_transit_score:      float = Field(0.40, ge=0, le=1, example=0.75)
    poi_mall_score:         float = Field(0.30, ge=0, le=1, example=0.30)
    poi_residential_score:  float = Field(0.50, ge=0, le=1, example=0.40)
    poi_entertainment_score:float = Field(0.30, ge=0, le=1, example=0.25)

    # Network quality
    home_zone_rsrq:         float = Field(-11.0, example=-11.0)
    work_zone_rsrq:         float = Field(-12.0, example=-12.0)
    avg_throughput_mbps:    float = Field(25.0,  ge=0, example=28.0)

    # Other
    monthly_charges:        float = Field(60.0, ge=0, example=75.0)
    tenure_months:          float = Field(24.0, ge=0, example=36.0)
    senior_citizen:         int   = Field(0, ge=0, le=1, example=0)

    # Optional geography
    home_lat: Optional[float] = Field(None, example=-6.2088)
    home_lon: Optional[float] = Field(None, example=106.8456)

    class Config:
        json_schema_extra = {"example": {
            "subscriber_id": "SUB_00000001",
            "avg_daily_distance_km": 22.0,
            "travel_radius_km": 18.0,
            "home_work_distance_km": 15.0,
            "n_frequent_locations": 3.0,
            "pct_time_at_home": 0.52,
            "pct_time_at_work": 0.28,
            "mobility_entropy": 1.8,
            "data_usage_gb": 16.0,
            "call_minutes_monthly": 280.0,
            "sms_monthly": 60.0,
            "roaming_days_monthly": 0.5,
            "morning_data_ratio": 0.28,
            "afternoon_data_ratio": 0.18,
            "evening_data_ratio": 0.35,
            "night_data_ratio": 0.19,
            "weekend_data_ratio": 0.30,
            "peak_hour_data_ratio": 0.45,
            "poi_office_score": 0.55,
            "poi_transit_score": 0.75,
            "poi_mall_score": 0.30,
            "poi_residential_score": 0.40,
            "poi_entertainment_score": 0.25,
            "home_zone_rsrq": -11.0,
            "work_zone_rsrq": -12.0,
            "avg_throughput_mbps": 28.0,
            "monthly_charges": 75.0,
            "tenure_months": 36.0,
            "senior_citizen": 0,
            "home_lat": -6.2088,
            "home_lon": 106.8456,
        }}


class PlanRecommendation(BaseModel):
    rank:        int
    plan_id:     str
    plan_name:   str
    price:       float
    match_score: float
    reason:      str


class SegmentResult(BaseModel):
    subscriber_id:       str
    segment_id:          int
    segment_name:        str
    confidence:          float
    segment_description: str
    recommended_plans:   List[str]
    h3_home_cell:        Optional[str] = None


class BatchSegmentRequest(BaseModel):
    subscribers: List[SubscriberFeatures] = Field(..., max_length=500)


class BatchSegmentResponse(BaseModel):
    results: List[dict]
    total:   int


class RecommendRequest(BaseModel):
    segment_id:    int
    subscriber_id: Optional[str] = None


class RecommendResponse(BaseModel):
    segment_id:      int
    segment_name:    str
    recommendations: List[PlanRecommendation]


class SegmentProfile(BaseModel):
    segment_id:   int
    segment_name: str
    description:  str
    n_subscribers: int
    pct_of_total:  float
    top_plans:    List[str]
    key_traits:   List[str]


class HealthResponse(BaseModel):
    status:       str
    models_loaded:bool
    n_segments:   int
