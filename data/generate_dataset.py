"""
AI Traffic Congestion Predictor
Synthetic Dataset Generator

Generates 15,000 realistic traffic observations with:
- Rush-hour temporal patterns
- Weather correlations
- Gaussian noise on continuous features
- Categorical flip noise (5%)
- Seasonal temperature variation
- Road type-specific capacity profiles
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import sys

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_SAMPLES = 15_000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "traffic_dataset.csv")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
ROAD_TYPES    = ["highway", "arterial", "local", "expressway"]
WEATHER_CONDS = ["clear", "rain", "fog", "snow", "storm"]
CITY_ZONES    = ["CBD", "residential", "industrial", "suburban", "airport"]

ROAD_CAPACITY = {          # baseline vehicles / hour
    "highway":    3_500,
    "expressway": 4_000,
    "arterial":   1_800,
    "local":        600,
}

WEATHER_SPEED_FACTOR = {   # multiplier on average speed
    "clear":  1.00,
    "rain":   0.82,
    "fog":    0.75,
    "snow":   0.60,
    "storm":  0.45,
}

WEATHER_PROB = [0.55, 0.20, 0.10, 0.08, 0.07]   # relative probability


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def sin_cos_encode(val, max_val):
    """Cyclic encoding — kept as raw for DataFrame storage."""
    angle = 2 * np.pi * val / max_val
    return np.sin(angle), np.cos(angle)


def rush_hour_multiplier(hour: int) -> float:
    """Returns a demand multiplier based on time of day."""
    # Morning rush 7-9, Evening rush 16-19
    if 7 <= hour <= 9:
        return np.random.uniform(1.6, 2.2)
    elif 16 <= hour <= 19:
        return np.random.uniform(1.7, 2.4)
    elif 12 <= hour <= 13:
        return np.random.uniform(1.2, 1.5)   # lunch
    elif 22 <= hour or hour <= 5:
        return np.random.uniform(0.1, 0.35)  # overnight
    else:
        return np.random.uniform(0.6, 1.1)


def seasonal_temperature(month: int) -> float:
    """Realistic temperature (°C) with seasonal noise."""
    base_temps = {
        1: 8, 2: 10, 3: 14, 4: 18, 5: 23,
        6: 28, 7: 31, 8: 30, 9: 26, 10: 20,
        11: 13, 12: 9
    }
    return base_temps[month] + np.random.normal(0, 3.5)


def congestion_label(utilization: float, incident: int,
                     weather: str, hour: int) -> str:
    """
    Rule-based congestion label with probabilistic noise.
    utilization: 0-1 fraction of road capacity used
    """
    # Base score
    score = utilization

    # Weather penalty
    penalties = {"storm": 0.30, "snow": 0.22, "fog": 0.14,
                 "rain": 0.10, "clear": 0.0}
    score += penalties[weather]

    # Incident penalty
    if incident:
        score += np.random.uniform(0.15, 0.30)

    # Overnight discount
    if hour >= 23 or hour <= 5:
        score *= 0.4

    # Add label noise - increased for realism
    score += np.random.normal(0, 0.18)
    score = np.clip(score, 0, 1)

    if score < 0.30:
        return "low"
    elif score < 0.55:
        return "moderate"
    elif score < 0.78:
        return "high"
    else:
        return "severe"


def add_gaussian_noise(series: pd.Series, std_frac: float = 0.06) -> pd.Series:
    """Add Gaussian noise proportional to the column std."""
    noise = np.random.normal(0, series.std() * std_frac, len(series))
    return series + noise


def categorical_flip_noise(arr: list, choices: list, flip_prob: float = 0.05):
    """Randomly replace 5% of categorical values with a random alternative."""
    result = arr.copy()
    for i in range(len(result)):
        if random.random() < flip_prob:
            result[i] = random.choice([c for c in choices if c != result[i]])
    return result


# --------------------------------------------------------------------------- #
# Generation loop
# --------------------------------------------------------------------------- #

def generate_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    print(f"\n  Generating {n:,} synthetic traffic observations...")

    rows = []
    # Simulate over 2 calendar years for temporal diversity
    start_date = datetime(2023, 1, 1)

    for i in range(n):
        # --- Temporal features ---
        offset_hours = random.randint(0, 365 * 2 * 24)
        ts = start_date + timedelta(hours=offset_hours)
        hour        = ts.hour
        day_of_week = ts.weekday()     # 0=Mon, 6=Sun
        month       = ts.month
        is_weekend  = int(day_of_week >= 5)

        # --- Road / Zone ---
        road_type = random.choice(ROAD_TYPES)
        city_zone = random.choices(CITY_ZONES,
                                   weights=[0.30, 0.25, 0.20, 0.15, 0.10])[0]

        # --- Weather ---
        weather = random.choices(WEATHER_CONDS, weights=WEATHER_PROB)[0]
        temp    = seasonal_temperature(month)

        # --- Binary event flags ---
        school_zone       = int(random.random() < 0.18)
        construction_zone = int(random.random() < 0.12)
        incident_reported = int(random.random() < 0.09)
        public_event      = int(random.random() < 0.07)

        # --- Vehicle count ---
        capacity   = ROAD_CAPACITY[road_type]
        demand_mul = rush_hour_multiplier(hour)
        weekend_mul = 0.72 if is_weekend else 1.0
        event_mul   = 1.35 if public_event else 1.0
        base_count  = capacity * demand_mul * weekend_mul * event_mul
        vehicle_count = max(0, int(base_count + np.random.normal(0, base_count * 0.12)))

        # --- Speed ---
        free_flow = {"highway": 110, "expressway": 120,
                     "arterial": 60,  "local": 40}[road_type]
        util_ratio = min(vehicle_count / capacity, 1.0)
        # BPR (Bureau of Public Roads) speed-flow curve
        bpr_delay  = 1 + 0.15 * (util_ratio ** 4)
        speed = (free_flow / bpr_delay) * WEATHER_SPEED_FACTOR[weather]
        speed = max(5, speed + np.random.normal(0, 4.0))

        # --- Signal & capacity utilization ---
        signal_cycle = random.choice([60, 90, 120, 150]) + np.random.normal(0, 8)
        signal_cycle = max(30, signal_cycle)
        road_cap_util = min(util_ratio + np.random.normal(0, 0.04), 1.0)

        # --- Cyclical encodings ---
        hour_sin,  hour_cos  = sin_cos_encode(hour, 24)
        day_sin,   day_cos   = sin_cos_encode(day_of_week, 7)
        month_sin, month_cos = sin_cos_encode(month, 12)

        # --- Target label ---
        label = congestion_label(road_cap_util, incident_reported,
                                 weather, hour)

        rows.append({
            "hour":                    hour,
            "hour_sin":                round(hour_sin, 5),
            "hour_cos":                round(hour_cos, 5),
            "day_of_week":             day_of_week,
            "day_sin":                 round(day_sin, 5),
            "day_cos":                 round(day_cos, 5),
            "month":                   month,
            "month_sin":               round(month_sin, 5),
            "month_cos":               round(month_cos, 5),
            "is_weekend":              is_weekend,
            "road_type":               road_type,
            "city_zone":               city_zone,
            "weather_condition":       weather,
            "temperature_c":           round(temp, 2),
            "vehicle_count":           vehicle_count,
            "average_speed_kmh":       round(speed, 2),
            "signal_cycle_time":       round(signal_cycle, 1),
            "road_capacity_utilization": round(road_cap_util, 4),
            "incident_reported":       incident_reported,
            "school_zone":             school_zone,
            "construction_zone":       construction_zone,
            "public_event_nearby":     public_event,
            "congestion_level":        label,
        })

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Post-generation noise passes
    # ------------------------------------------------------------------ #
    print("  Applying Gaussian noise to continuous features...")
    for col in ["temperature_c", "vehicle_count", "average_speed_kmh",
                "signal_cycle_time", "road_capacity_utilization"]:
        df[col] = add_gaussian_noise(df[col], std_frac=0.15)

    # Clip after noise
    df["vehicle_count"]              = df["vehicle_count"].clip(lower=0).astype(int)
    df["average_speed_kmh"]          = df["average_speed_kmh"].clip(lower=2, upper=140).round(2)
    df["road_capacity_utilization"]  = df["road_capacity_utilization"].clip(0, 1).round(4)
    df["temperature_c"]              = df["temperature_c"].round(2)
    df["signal_cycle_time"]          = df["signal_cycle_time"].clip(lower=20).round(1)

    print("  Applying categorical flip noise (15%) to weather & road_type...")
    df["weather_condition"] = categorical_flip_noise(
        df["weather_condition"].tolist(), WEATHER_CONDS, flip_prob=0.15)
    df["road_type"] = categorical_flip_noise(
        df["road_type"].tolist(), ROAD_TYPES, flip_prob=0.12)

    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("\n" + "="*60)
    print("  AI TRAFFIC CONGESTION PREDICTOR")
    print("  Dataset Generator v1.0")
    print("="*60)

    df = generate_dataset(N_SAMPLES)

    # Summary statistics
    print(f"\n  Dataset shape  : {df.shape}")
    print(f"  Features       : {df.shape[1] - 1}")
    print(f"  Target classes : {sorted(df['congestion_level'].unique())}")
    print("\n  Class distribution:")
    dist = df["congestion_level"].value_counts(normalize=True).sort_index()
    for cls, pct in dist.items():
        bar = "#" * int(pct * 40)
        print(f"    {cls:<10} {bar} {pct*100:.1f}%")

    print(f"\n  Missing values : {df.isnull().sum().sum()}")
    print(f"\n  Sample rows:")
    print(df.head(3).to_string(index=False))

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
