"""
src/api/kafka_consumer.py
──────────────────────────
Real-time CDR event consumer for streaming segment classification.

Consumes CDR event batches from a Kafka topic, scores each subscriber
against the latest segmentation model, and publishes results to an
output topic for downstream marketing automation systems.

Architecture:
  CDR Events (Kafka) → Consumer → Feature Engineering → Segment Model
      → Enriched Events (Kafka) → CRM / Campaign System

Event schema (input topic: cdr-events):
  {
    "subscriber_id": "SUB_00000001",
    "event_time": "2024-01-15T08:30:00Z",
    "data_usage_gb": 1.2,
    "call_minutes": 15,
    "location_lat": -6.2088,
    "location_lon": 106.8456,
    ...
  }

Output schema (output topic: segment-scores):
  {
    "subscriber_id": "SUB_00000001",
    "segment_id": 0,
    "segment_name": "Urban Commuter",
    "confidence": 0.84,
    "recommended_plans": ["unlimited_rush", "data_boost"],
    "scored_at": "2024-01-15T08:30:05Z"
  }

Setup:
  1. Start Kafka: docker run -p 9092:9092 confluentinc/cp-kafka
  2. Create topics: kafka-topics.sh --create --topic cdr-events ...
  3. Run consumer: python src/api/kafka_consumer.py

Configuration in .env:
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092
  KAFKA_INPUT_TOPIC=cdr-events
  KAFKA_OUTPUT_TOPIC=segment-scores
  KAFKA_GROUP_ID=segmentation-consumer
"""

import os
import sys
import json
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Configuration from environment ────────────────────────────────────────────
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC       = os.getenv("KAFKA_INPUT_TOPIC",       "cdr-events")
OUTPUT_TOPIC      = os.getenv("KAFKA_OUTPUT_TOPIC",      "segment-scores")
GROUP_ID          = os.getenv("KAFKA_GROUP_ID",          "segmentation-consumer")
BATCH_SIZE        = int(os.getenv("KAFKA_BATCH_SIZE",    "100"))


class SegmentationConsumer:
    """
    Kafka consumer that classifies CDR events into geo-behavioral segments.
    Thread-safe, graceful shutdown supported.
    """

    def __init__(self):
        self.loader    = None
        self._running  = threading.Event()
        self._running.set()
        self._consumer = None
        self._producer = None
        self._stats    = {"processed": 0, "errors": 0, "start_time": datetime.now()}

    def load_models(self) -> None:
        """Load segmentation models at startup."""
        from src.api.model_loader import ModelLoader
        self.loader = ModelLoader()
        self.loader.load()
        if not self.loader.is_loaded():
            raise RuntimeError("Failed to load segmentation models")
        logger.success(f"Models loaded — {self.loader.n_segments} segments")

    def start(self) -> None:
        """Start the Kafka consumer loop."""
        try:
            from kafka import KafkaConsumer, KafkaProducer
        except ImportError:
            logger.error("kafka-python not installed: pip install kafka-python")
            sys.exit(1)

        logger.info(f"Connecting to Kafka: {BOOTSTRAP_SERVERS}")
        logger.info(f"Consuming: {INPUT_TOPIC} → {OUTPUT_TOPIC}")

        self._consumer = KafkaConsumer(
            INPUT_TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id=GROUP_ID,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000,
        )
        self._producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )

        logger.success("Kafka consumer started. Waiting for events...")
        self._consume_loop()

    def _consume_loop(self) -> None:
        """Main consumption loop — processes messages until shutdown."""
        batch = []
        while self._running.is_set():
            try:
                for message in self._consumer:
                    if not self._running.is_set():
                        break
                    batch.append(message.value)
                    if len(batch) >= BATCH_SIZE:
                        self._process_batch(batch)
                        batch = []

            except StopIteration:
                # consumer_timeout_ms hit — flush pending batch
                if batch:
                    self._process_batch(batch)
                    batch = []

            except Exception as e:
                logger.error(f"Consumer error: {e}")
                self._stats["errors"] += 1

        # Flush remaining
        if batch:
            self._process_batch(batch)

        logger.info("Consumer loop stopped.")

    def _process_batch(self, events: list) -> None:
        """Score a batch of CDR events and publish results."""
        if not self.loader or not self.loader.is_loaded():
            logger.warning("Model not loaded — skipping batch")
            return

        results = []
        for event in events:
            try:
                result = self._score_event(event)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error scoring event {event.get('subscriber_id', '?')}: {e}")
                self._stats["errors"] += 1

        # Publish results
        for result in results:
            self._producer.send(OUTPUT_TOPIC, value=result)
            self._stats["processed"] += 1

        if results:
            self._producer.flush()
            logger.debug(f"Published {len(results)} segment scores")

        # Log periodic stats
        if self._stats["processed"] % 1000 == 0 and self._stats["processed"] > 0:
            elapsed = (datetime.now() - self._stats["start_time"]).seconds
            rate = self._stats["processed"] / max(elapsed, 1)
            logger.info(
                f"Stats: {self._stats['processed']:,} processed | "
                f"{self._stats['errors']} errors | "
                f"{rate:.1f} events/sec"
            )

    def _score_event(self, event: dict) -> dict:
        """
        Map a raw CDR event to segment classification input and score it.
        In production, events would have pre-computed mobility features.
        Here we use whatever features are present and fill defaults.
        """
        # Build scoring payload — use event data where available
        payload = {
            "subscriber_id":          event.get("subscriber_id", "unknown"),
            "avg_daily_distance_km":  event.get("avg_daily_distance_km", 10.0),
            "travel_radius_km":       event.get("travel_radius_km", 15.0),
            "home_work_distance_km":  event.get("home_work_distance_km", 8.0),
            "n_frequent_locations":   event.get("n_frequent_locations", 3.0),
            "pct_time_at_home":       event.get("pct_time_at_home", 0.55),
            "pct_time_at_work":       event.get("pct_time_at_work", 0.22),
            "mobility_entropy":       event.get("mobility_entropy", 1.5),
            "data_usage_gb":          event.get("data_usage_gb", 15.0),
            "call_minutes_monthly":   event.get("call_minutes_monthly", 300.0),
            "sms_monthly":            event.get("sms_monthly", 50.0),
            "roaming_days_monthly":   event.get("roaming_days_monthly", 0.5),
            "morning_data_ratio":     event.get("morning_data_ratio", 0.25),
            "afternoon_data_ratio":   event.get("afternoon_data_ratio", 0.22),
            "evening_data_ratio":     event.get("evening_data_ratio", 0.30),
            "night_data_ratio":       event.get("night_data_ratio", 0.18),
            "weekend_data_ratio":     event.get("weekend_data_ratio", 0.30),
            "peak_hour_data_ratio":   event.get("peak_hour_data_ratio", 0.35),
            "poi_office_score":       event.get("poi_office_score", 0.40),
            "poi_transit_score":      event.get("poi_transit_score", 0.40),
            "poi_mall_score":         event.get("poi_mall_score", 0.30),
            "poi_residential_score":  event.get("poi_residential_score", 0.50),
            "poi_entertainment_score":event.get("poi_entertainment_score", 0.30),
            "home_zone_rsrq":         event.get("home_zone_rsrq", -11.0),
            "work_zone_rsrq":         event.get("work_zone_rsrq", -12.0),
            "avg_throughput_mbps":    event.get("avg_throughput_mbps", 25.0),
            "monthly_charges":        event.get("monthly_charges", 60.0),
            "tenure_months":          event.get("tenure_months", 24.0),
            "senior_citizen":         event.get("senior_citizen", 0),
            "home_lat":               event.get("location_lat"),
            "home_lon":               event.get("location_lon"),
        }

        result = self.loader.predict_segment(payload)
        result["event_time"]  = event.get("event_time", "")
        result["scored_at"]   = datetime.now(timezone.utc).isoformat()
        return result

    def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down consumer...")
        self._running.clear()
        if self._consumer:
            self._consumer.close()
        if self._producer:
            self._producer.close()
        logger.info(f"Final stats: {self._stats}")


# ── Standalone demo (no Kafka needed) ─────────────────────────────────────────
def run_demo_without_kafka() -> None:
    """
    Demonstrate the scoring pipeline without a real Kafka cluster.
    Generates synthetic CDR events and scores them locally.
    """
    import numpy as np
    logger.info("Running demo mode (no Kafka required)")

    consumer = SegmentationConsumer()
    consumer.load_models()

    rng = np.random.default_rng(42)
    n_demo = 20
    demo_events = []
    for i in range(n_demo):
        demo_events.append({
            "subscriber_id":        f"DEMO_{i:04d}",
            "data_usage_gb":        float(rng.uniform(2, 50)),
            "call_minutes_monthly": float(rng.uniform(50, 800)),
            "avg_daily_distance_km":float(rng.uniform(2, 50)),
            "travel_radius_km":     float(rng.uniform(5, 80)),
            "poi_office_score":     float(rng.uniform(0.1, 0.9)),
            "poi_transit_score":    float(rng.uniform(0.1, 0.9)),
            "poi_mall_score":       float(rng.uniform(0.1, 0.9)),
            "evening_data_ratio":   float(rng.uniform(0.15, 0.55)),
            "weekend_data_ratio":   float(rng.uniform(0.2, 0.6)),
            "monthly_charges":      float(rng.uniform(30, 120)),
            "location_lat":         float(rng.uniform(-6.5, -5.9)),
            "location_lon":         float(rng.uniform(106.6, 107.1)),
        })

    print(f"\nScoring {n_demo} demo CDR events...")
    print("-" * 70)
    for event in demo_events:
        result = consumer._score_event(event)
        print(
            f"  {result['subscriber_id']:12s} → "
            f"[{result['segment_id']}] {result['segment_name']:22s} "
            f"(conf={result['confidence']:.2f}) "
            f"→ {result['recommended_plans'][0]}"
        )
    print("-" * 70)
    print(f"Demo complete. {n_demo} events scored successfully.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kafka CDR event segmentation consumer")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (no Kafka required)")
    args = parser.parse_args()

    if args.demo:
        run_demo_without_kafka()
        return

    consumer = SegmentationConsumer()
    consumer.load_models()

    # Register graceful shutdown
    def handler(sig, frame):
        consumer.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    consumer.start()


if __name__ == "__main__":
    main()
