import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
MODEL_META_PATH = BASE_DIR / "models" / "model_meta.json"
HOLDOUT_PATH = BASE_DIR / "data" / "raw" / "holdout_features.csv"
OUTPUT_PATH = BASE_DIR / "predictions" / "holdout_predictions.csv"

FEATURES = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
    "vehicle",
    "station_type",
]


def main():
    metadata = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    if str(metadata["model_version"]) != "2":
        raise ValueError(
            f"Expected exported champion version 2, got {metadata['model_version']}"
        )

    holdout = pd.read_csv(HOLDOUT_PATH)
    if len(holdout) != 250:
        raise ValueError(f"Expected 250 holdout rows, got {len(holdout)}")

    missing = [column for column in FEATURES + ["record_id"] if column not in holdout.columns]
    if missing:
        raise ValueError(f"Holdout file is missing required columns: {missing}")

    model = joblib.load(MODEL_PATH)
    predictions = np.asarray(model.predict(holdout[FEATURES]), dtype=float)

    if len(predictions) != 250:
        raise ValueError(f"Expected 250 predictions, got {len(predictions)}")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain null, NaN, or infinite values")

    result = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions,
    })
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(result)} predictions to {OUTPUT_PATH}")
    print("Columns:", ",".join(result.columns))
    print("All predictions are finite.")


if __name__ == "__main__":
    main()
