"""
src/data_engineering/data_validation.py
────────────────────────────────────────
Data quality checks for the subscriber/CDR dataset.
Lightweight rule-based validation — fails fast before feature engineering.

Usage:
    python src/data_engineering/data_validation.py --input data/raw/subscribers.parquet
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    rule: str
    passed: bool
    message: str
    severity: str = "ERROR"

    def __str__(self):
        icon = "✓" if self.passed else ("✗" if self.severity == "ERROR" else "⚠")
        return f"  [{icon}] {self.rule}: {self.message}"


class DataValidator:
    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.df = df
        self.name = name
        self.results: list[ValidationResult] = []

    def _add(self, rule, passed, msg, severity="ERROR"):
        self.results.append(ValidationResult(rule, passed, msg, severity))

    def expect_row_count_above(self, min_rows: int):
        n = len(self.df)
        self._add("row_count", n >= min_rows,
                  f"{n:,} rows (need ≥ {min_rows:,})")
        return self

    def expect_columns(self, cols: list):
        missing = [c for c in cols if c not in self.df.columns]
        self._add("required_columns", len(missing) == 0,
                  f"Missing: {missing}" if missing else f"All {len(cols)} required columns present")
        return self

    def expect_no_nulls(self, cols: list):
        for col in cols:
            if col not in self.df.columns:
                continue
            n = self.df[col].isnull().sum()
            self._add(f"no_nulls:{col}", n == 0,
                      f"{n:,} nulls" if n else "no nulls")
        return self

    def expect_unique(self, col: str):
        if col not in self.df.columns:
            return self
        dups = self.df[col].duplicated().sum()
        self._add(f"unique:{col}", dups == 0,
                  f"{dups:,} duplicates" if dups else "all unique")
        return self

    def expect_range(self, col: str, lo: float, hi: float, severity="ERROR"):
        if col not in self.df.columns:
            return self
        out = ((self.df[col] < lo) | (self.df[col] > hi)).sum()
        self._add(f"range:{col}", out == 0,
                  f"{out:,} values outside [{lo}, {hi}]" if out else f"all in [{lo}, {hi}]",
                  severity=severity)
        return self

    def expect_values_in(self, col: str, valid: set):
        if col not in self.df.columns:
            return self
        bad = ~self.df[col].isin(valid)
        n = bad.sum()
        self._add(f"values_in:{col}", n == 0,
                  f"{n:,} unexpected: {self.df.loc[bad, col].unique()[:5]}" if n else f"all in {valid}")
        return self

    def expect_positive(self, col: str):
        if col not in self.df.columns:
            return self
        neg = (self.df[col] < 0).sum()
        self._add(f"positive:{col}", neg == 0,
                  f"{neg:,} negative values" if neg else "all positive")
        return self

    def report(self) -> bool:
        print(f"\n{'='*55}")
        print(f"VALIDATION REPORT: {self.name}")
        print(f"{'='*55}")
        errors   = [r for r in self.results if not r.passed and r.severity == "ERROR"]
        warnings = [r for r in self.results if not r.passed and r.severity == "WARNING"]
        passed   = [r for r in self.results if r.passed]
        for r in self.results:
            print(r)
        print(f"\n  {len(passed)} passed | {len(warnings)} warnings | {len(errors)} errors")
        print("=" * 55)
        if errors:
            logger.error(f"Validation FAILED: {len(errors)} errors")
        elif warnings:
            logger.warning(f"Validation PASSED with {len(warnings)} warnings")
        else:
            logger.success("Validation PASSED")
        return len(errors) == 0


def validate_subscribers(df: pd.DataFrame) -> bool:
    REQUIRED = [
        "subscriber_id", "home_lat", "home_lon", "work_lat", "work_lon",
        "avg_daily_distance_km", "travel_radius_km", "data_usage_gb",
        "call_minutes_monthly", "morning_data_ratio", "evening_data_ratio",
        "weekend_data_ratio", "poi_office_score", "poi_transit_score",
        "poi_mall_score", "home_zone_rsrq", "monthly_charges",
    ]
    RATIO_COLS = [
        "pct_time_at_home", "pct_time_at_work", "morning_data_ratio",
        "afternoon_data_ratio", "evening_data_ratio", "night_data_ratio",
        "weekend_data_ratio", "peak_hour_data_ratio",
        "poi_office_score", "poi_transit_score", "poi_mall_score",
        "poi_residential_score", "poi_entertainment_score",
    ]
    return (
        DataValidator(df, "subscribers.parquet")
        .expect_row_count_above(1000)
        .expect_columns(REQUIRED)
        .expect_unique("subscriber_id")
        .expect_no_nulls(["subscriber_id", "home_lat", "home_lon",
                          "data_usage_gb", "monthly_charges"])
        .expect_range("home_lat", -11.0, 6.0)
        .expect_range("home_lon", 95.0, 141.0)
        .expect_range("data_usage_gb", 0, 500)
        .expect_range("home_zone_rsrq", -30, -3)
        .expect_positive("avg_daily_distance_km")
        .expect_positive("monthly_charges")
        .expect_values_in("internet_service", {"fiber_optic", "DSL", "none"})
        *[DataValidator(df, "").expect_range(c, 0, 1, severity="WARNING")
          for c in RATIO_COLS]  # type: ignore
        .report()
    ) if False else (
        # Simpler chained approach
        v := DataValidator(df, "subscribers.parquet"),
        v.expect_row_count_above(1000),
        v.expect_columns(REQUIRED),
        v.expect_unique("subscriber_id"),
        v.expect_no_nulls(["subscriber_id", "home_lat", "home_lon"]),
        v.expect_range("home_lat", -11.0, 6.0),
        v.expect_range("home_lon", 95.0, 141.0),
        v.expect_range("data_usage_gb", 0, 500),
        v.expect_range("home_zone_rsrq", -30, -3),
        v.expect_positive("avg_daily_distance_km"),
        v.expect_positive("monthly_charges"),
        [v.expect_range(c, 0, 1, severity="WARNING") for c in RATIO_COLS],
        v.report()
    )[-1]


def _run_validation(df: pd.DataFrame) -> bool:
    """Simple flat validation without chaining tricks."""
    v = DataValidator(df, "subscribers.parquet")
    v.expect_row_count_above(1000)
    v.expect_unique("subscriber_id")
    v.expect_no_nulls(["subscriber_id", "home_lat", "home_lon",
                       "data_usage_gb", "monthly_charges"])
    v.expect_range("home_lat", -11.0, 6.0)
    v.expect_range("home_lon", 95.0, 141.0)
    v.expect_range("data_usage_gb", 0, 500)
    v.expect_range("home_zone_rsrq", -30, -3)
    v.expect_positive("avg_daily_distance_km")
    v.expect_positive("monthly_charges")
    ratio_cols = [
        "pct_time_at_home", "pct_time_at_work",
        "morning_data_ratio", "afternoon_data_ratio",
        "evening_data_ratio", "night_data_ratio",
        "weekend_data_ratio", "peak_hour_data_ratio",
        "poi_office_score", "poi_transit_score", "poi_mall_score",
    ]
    for col in ratio_cols:
        if col in df.columns:
            v.expect_range(col, 0.0, 1.0, severity="WARNING")
    return v.report()


def main():
    parser = argparse.ArgumentParser(description="Validate subscriber dataset")
    parser.add_argument("--input", default="data/raw/subscribers.parquet")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        logger.error(f"File not found: {path}")
        logger.info("Run: python src/data_engineering/generate_data.py")
        sys.exit(1)

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    ok = _run_validation(df)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
