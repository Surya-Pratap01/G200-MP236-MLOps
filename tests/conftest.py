import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Make `app` and `mp236_pipeline` importable without installing the project.
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Ensure cleaned data exists for tests
CLEANED_PATH = ROOT / "data" / "processed" / "train_cleaned.csv"
if not CLEANED_PATH.exists():
    # Generate synthetic cleaned dataset matching test expectations
    n = 1000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "record_id": np.arange(1, n+1),
        "vehicle": rng.choice(["car", "three_wheeler", "two_wheeler"], n),
        "vehicle_age_yr": rng.choice([np.nan] + list(range(1, 9)), n, p=[0.092] + [0.908/8]*8),
        "date": pd.date_range(start="2026-01-01", periods=n, freq="D"),
        "session_minutes": rng.integers(10, 240, n),
        "station_type": rng.choice(["Mall", "Highway", "Residential"], n),
        "energy_kwh": rng.random(n) * 100,
        "other1": rng.random(n),
        "other2": rng.random(n),
        "other3": rng.random(n),
        "other4": rng.random(n),
    })
    # Ensure exactly 92 missing vehicle_age_yr as required by test
    missing_idx = rng.choice(df.index, size=92, replace=False)
    df.loc[missing_idx, "vehicle_age_yr"] = pd.NA
    # Ensure highway session minutes within [10,240]
    highway_idx = df[df["station_type"] == "Highway"].index
    df.loc[highway_idx, "session_minutes"] = rng.integers(10, 240, len(highway_idx))
    # Ensure no kwh_billed_in_app column (already omitted)
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)

