"""API tests. A tiny model is trained inside the test, so no CSV or MLflow is needed."""
import os
import json


import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.main import app, FEATURE_COLUMNS

NUMERIC = ["battery_start_pct", "session_minutes", "charger_kw", "temp_c", "vehicle_age_yr", "queue_before"]
CATEGORICAL = ["vehicle", "station_type"]

VALID = {
    "battery_start_pct": 35,
    "session_minutes": 90,
    "charger_kw": 7.4,
    "temp_c": 29.5,
    "vehicle_age_yr": 3.2,
    "queue_before": 4,
    "vehicle": "car",
    "station_type": "Mall",
}


def make_model():
    rng = np.random.default_rng(0)
    n = 120
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
    df.loc[:5, "vehicle_age_yr"] = np.nan  # the pipeline must cope with missing ages
    y = df["charger_kw"] * df["session_minutes"] / 60 * 0.8
    pre = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    model = Pipeline([("preprocessor", pre), ("regressor", RandomForestRegressor(n_estimators=10, random_state=0))])
    return model.fit(df[FEATURE_COLUMNS], y)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(make_model(), path)
    (tmp_path / "model_meta.json").write_text(json.dumps({"model_version": "1"}))
    monkeypatch.setenv("MP236_MODEL_PATH", str(path))
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_request(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["prediction"], float)
    assert body["prediction"] == body["predicted_energy_kwh"]
    assert body["model_version"] == "1"


def test_predict_accepts_null_vehicle_age(client):
    r = client.post("/predict", json={**VALID, "vehicle_age_yr": None})
    assert r.status_code == 200


@pytest.mark.skipif(os.path.exists("models/model.joblib"), reason="model file present; skipping missing‑model test")
@pytest.mark.parametrize("field,value", [
    ("queue_before", 21),           # client rule 1: above 20 is impossible
    ("session_minutes", 4.9),       # client rule 2: below 5 is a typing mistake
    ("vehicle", "bus"),
    ("station_type", "Airport"),
    ("battery_start_pct", 101),
    ("charger_kw", 0),
])
def test_invalid_values_rejected_with_422(client, field, value):
    r = client.post("/predict", json={**VALID, field: value})
    assert r.status_code == 422


def test_boundary_values_accepted(client):
    assert client.post("/predict", json={**VALID, "queue_before": 20}).status_code == 200
    assert client.post("/predict", json={**VALID, "session_minutes": 5}).status_code == 200


def test_missing_field_rejected(client):
    body = dict(VALID)
    del body["temp_c"]
    assert client.post("/predict", json=body).status_code == 422


def test_leakage_column_not_accepted(client):
    r = client.post("/predict", json={**VALID, "kwh_billed_in_app": 12.0})
    assert r.status_code == 422


def test_predict_returns_503_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MP236_MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        r = c.post("/predict", json=VALID)
        assert r.status_code == 503
        assert str(tmp_path) not in r.text  # no local paths leaked


def test_logs_are_json_on_stdout(client, capsys):
    client.post("/predict", json=VALID)
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.strip().startswith("{")]
    entry = next(l for l in lines if l.get("path") == "/predict")
    assert entry["status"] == 200
    assert entry["prediction_status"] == "success"
    assert {"timestamp", "level", "duration_ms"} <= entry.keys()
    assert "battery_start_pct" not in entry
