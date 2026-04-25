"""
src/api/app.py
───────────────
FastAPI REST API for geo-behavioral segment classification and plan recommendation.

Endpoints:
  POST /segment          — Classify a subscriber into a segment
  POST /segment/batch    — Classify a batch (up to 500)
  GET  /segments         — List all segments with profiles
  POST /recommend        — Get plan recommendations for a segment
  GET  /geo-density      — Return GeoJSON segment density map
  GET  /health           — Health check

Usage:
    uvicorn src.api.app:app --reload --port 8000
    Swagger UI: http://localhost:8000/docs
"""

import sys
import json
from pathlib import Path

import pandas as pd
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.schemas import (
    SubscriberFeatures,
    SegmentResult,
    BatchSegmentRequest,
    BatchSegmentResponse,
    RecommendRequest,
    RecommendResponse,
    SegmentProfile,
    HealthResponse,
)
from src.api.model_loader import ModelLoader

app = FastAPI(
    title="Geo Customer Segmentation API",
    description=(
        "Real-time geo-behavioral customer segmentation and plan recommendation. "
        "Classifies subscribers into mobility personas and recommends personalized offers."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = ModelLoader()


@app.on_event("startup")
async def startup():
    logger.info("Loading models on startup...")
    loader.load()
    logger.info("Models loaded.")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy" if loader.is_loaded() else "degraded",
        models_loaded=loader.is_loaded(),
        n_segments=loader.n_segments,
    )


@app.get("/segments", response_model=list[SegmentProfile], tags=["Segments"])
async def list_segments():
    """Return all segments with profiles and recommended plans."""
    if not loader.is_loaded():
        raise HTTPException(503, "Models not loaded")
    return loader.get_all_segment_profiles()


@app.post("/segment", response_model=SegmentResult, tags=["Segmentation"])
async def segment_subscriber(subscriber: SubscriberFeatures):
    """Classify a single subscriber into a geo-behavioral segment."""
    if not loader.is_loaded():
        raise HTTPException(503, "Models not loaded")
    try:
        result = loader.predict_segment(subscriber.model_dump())
        return SegmentResult(**result)
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(500, str(e))


@app.post("/segment/batch", response_model=BatchSegmentResponse, tags=["Segmentation"])
async def segment_batch(request: BatchSegmentRequest):
    """Classify a batch of subscribers (max 500)."""
    if not loader.is_loaded():
        raise HTTPException(503, "Models not loaded")
    if len(request.subscribers) > 500:
        raise HTTPException(400, "Batch size exceeds maximum of 500")
    try:
        results = [loader.predict_segment(s.model_dump())
                   for s in request.subscribers]
        return BatchSegmentResponse(
            results=results,
            total=len(results),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
async def recommend_plans(request: RecommendRequest):
    """Get plan recommendations for a segment or subscriber."""
    if not loader.is_loaded():
        raise HTTPException(503, "Models not loaded")
    try:
        recommendations = loader.get_recommendations(
            segment_id=request.segment_id,
            subscriber_id=request.subscriber_id,
        )
        return RecommendResponse(
            segment_id=request.segment_id,
            segment_name=loader.get_segment_name(request.segment_id),
            recommendations=recommendations,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/geo-density", tags=["Geospatial"])
async def geo_density():
    """Return GeoJSON of segment density per H3 hex cell."""
    geojson_path = Path("data/processed/segment_density.geojson")
    if not geojson_path.exists():
        raise HTTPException(
            404,
            "Geo density map not found. Run: python src/models/geo_density.py"
        )
    with open(geojson_path) as f:
        return json.load(f)


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Geo Customer Segmentation API",
        "version": "1.0.0",
        "docs": "/docs",
        "segments": "/segments",
        "health": "/health",
    }
