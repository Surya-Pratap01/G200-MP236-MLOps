import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Reuse FEATURE_COLUMNS from app.main if needed, but define locally
FEATURE_COLUMNS = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
    "vehicle",
    "station_type",
]

NUMERIC = ["battery_start_pct", "session_minutes", "charger_kw", "temp_c", "vehicle_age_yr", "queue_before"]
CATEGORICAL = ["vehicle", "station_type"]

def make_model():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "battery_start_pct": rng.uniform(5, 95, n),
        "session_minutes": rng.uniform(10, 240, n),
        "charger_kw": rng.choice([3.3, 7.4, 22.0], n),
        "temp_c": rng.uniform(15, 40, n),
        "vehicle_age_yr": rng.uniform(0, 8, n),
        "queue_before": rng.integers(0, 15, n),
        "vehicle": rng.choice(["car", "three_wheeler", "two_wheeler"], n),
        "station_type": rng.choice(["Mall", "Highway", "Residential"], n),
    })
    # introduce some missing ages
    df.loc[:5, "vehicle_age_yr"] = np.nan
    y = df["charger_kw"] * df["session_minutes"] / 60 * 0.8
    pre = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocessor", pre),
        ("regressor", RandomForestRegressor(n_estimators=10, random_state=0))
    ])
    model.fit(df[FEATURE_COLUMNS], y)
    return model

if __name__ == "__main__":
    model = make_model()
    # Save model
    # Write model and metadata to the repository's top‑level models folder
    import pathlib, json
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    model_path = repo_root / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    (model_path.parent / "model_meta.json").write_text(json.dumps({"model_version": "1"}))
    print(f"Dummy model written to {model_path}")
