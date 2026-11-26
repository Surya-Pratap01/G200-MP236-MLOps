"""FastAPI service for MP236 (EV charging session energy, Guwahati).

POST /predict  -> predicted energy_kwh for one charging session
GET  /health   -> {"status": "ok", ...}

The model is a scikit-learn Pipeline saved with joblib at
models/model.joblib (override with env MP236_MODEL_PATH). It is loaded
once at startup, never per request, and never from a developer's local
MLflow registry, so the same file works locally, in Docker and in k8s.
"""
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from app.schemas import PredictionRequest, PredictionResponse

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.joblib"
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


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with timestamp and level."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            # Exception class only: no stack trace / local paths in logs.
            payload["exception"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str)


class StdoutHandler(logging.StreamHandler):
    """Always writes to the *current* sys.stdout (container logs, test capture)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("mp236")
    if not logger.handlers:
        handler = StdoutHandler()  # stdout only, never a file
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    # We emit our own per-request JSON line; silence uvicorn's plain-text one.
    logging.getLogger("uvicorn.access").disabled = True
    return logger


logger = setup_logging()


def load_model(path: Path):
    model = joblib.load(path)
    version = os.getenv("MP236_MODEL_VERSION")
    if not version:
        meta = path.with_name("model_meta.json")
        version = "unknown"
        if meta.exists():
            version = str(json.loads(meta.read_text()).get("model_version", "unknown"))
    return model, version


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MP236_MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.model = None
    app.state.model_version = "unknown"
    try:
        app.state.model, app.state.model_version = load_model(path)
        logger.info("service started", extra={"fields": {
            "event": "startup", "model_loaded": True, "model_version": app.state.model_version}})
    except Exception as exc:  # model missing / unreadable: stay up, report it
        logger.error("model load failed", exc_info=True, extra={"fields": {
            "event": "startup", "model_loaded": False, "error": type(exc).__name__}})
    yield
    logger.info("service stopping", extra={"fields": {"event": "shutdown"}})


app = FastAPI(title="MP236 EV Charging Energy API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    request.state.log_fields = {}
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as exc:
        request.state.log_fields["error"] = type(exc).__name__
        raise
    finally:
        fields = {
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        }
        fields.update(request.state.log_fields)
        level = logging.ERROR if status >= 500 else logging.INFO
        logger.log(level, "request", extra={"fields": fields})


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
        "model_version": request.app.state.model_version,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    model = request.app.state.model
    if model is None:
        request.state.log_fields.update({"prediction_status": "model_unavailable"})
        raise HTTPException(status_code=503, detail="Model is not available")

    # Exactly the eight training columns, in training order.
    row = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
    try:
        value = float(model.predict(row)[0])
    except Exception as exc:
        request.state.log_fields.update({
            "prediction_status": "failed", "error": type(exc).__name__})
        raise HTTPException(status_code=500, detail="Prediction failed") from None

    value = round(value, 4)
    # Log only non-personal modelling inputs' categories and the result.
    request.state.log_fields.update({
        "prediction_status": "success",
        "prediction_kwh": value,
        "vehicle": payload.vehicle,
        "station_type": payload.station_type,
        "model_version": request.app.state.model_version,
    })
    return PredictionResponse(
        prediction=value,
        predicted_energy_kwh=value,
        model_version=request.app.state.model_version,
    )
