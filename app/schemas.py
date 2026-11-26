"""Request / response models for the MP236 prediction API.

The client's two rules live here, in the schema, so a bad request is
rejected with HTTP 422 and never reaches the model:
  1. queue_before above 20 is impossible  -> le=20
  2. session_minutes below 5 is a typo    -> ge=5
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    # Exactly the eight fields from PROBLEM.md section 5. Anything else
    # (for example the leaking column kwh_billed_in_app) is rejected.
    model_config = ConfigDict(extra="forbid")

    battery_start_pct: float = Field(..., ge=0, le=100, description="Battery level when plugged in (%)")
    session_minutes: float = Field(..., ge=5, description="Planned session length in MINUTES (Highway too)")
    charger_kw: float = Field(..., gt=0, description="Charger power (kW)")
    temp_c: float = Field(..., description="Air temperature (deg C)")
    vehicle_age_yr: Optional[float] = Field(None, ge=0, description="Vehicle age in years; null is imputed by the model")
    queue_before: float = Field(..., ge=0, le=20, description="Vehicles waiting before this one")
    vehicle: Literal["car", "three_wheeler", "two_wheeler"]
    station_type: Literal["Mall", "Highway", "Residential"]


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted energy in kWh")
    predicted_energy_kwh: float = Field(..., description="Same value, explicit name")
    model_version: str
