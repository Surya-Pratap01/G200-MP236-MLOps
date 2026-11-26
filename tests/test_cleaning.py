"""Cleaning tests on small in-test data (no CSV files needed, so CI works)."""
import pandas as pd

from mp236_pipeline.preprocessing import clean_dataframe


def raw_frame():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4", "R5"],
        "date": ["2026-01-10", "2026-02-20", "2026-02-21", "2026-04-01", "2026-04-02"],
        "battery_start_pct": [20, 40, 60, 30, 50],
        "session_minutes": [60, 90, 7200, 120, 3600],   # Highway rows are in seconds
        "charger_kw": [7.0, 11.0, 22.0, 3.3, 22.0],
        "temp_c": [30.0, 28.0, 31.0, 25.0, 27.0],
        "vehicle_age_yr": [2.0, 0.9, 0.9, 0.9, 4.0],   # 0.9 stuck 15 Feb - 3 Mar
        "queue_before": [1, 2, 3, 0, 4],
        "vehicle": ["car.", " Two Wheeler", "thr.", "TWO.", "three wheeler"],
        "station_type": ["Mall", "Residential", "Highway", "Mall", "Highway"],
        "kwh_billed_in_app": [5.0, 6.0, 7.0, 8.0, 9.0],
        "energy_kwh": [5.1, 6.2, 7.3, 8.4, 9.5],
    })


def test_leakage_column_removed():
    assert "kwh_billed_in_app" not in clean_dataframe(raw_frame()).columns
    assert "energy_kwh" in clean_dataframe(raw_frame()).columns


def test_highway_seconds_converted_to_minutes():
    out = clean_dataframe(raw_frame())
    highway = out.loc[out["station_type"] == "Highway", "session_minutes"].tolist()
    assert highway == [120.0, 60.0]
    assert out.loc[out["station_type"] != "Highway", "session_minutes"].tolist() == [60, 90, 120]


def test_vehicle_spellings_normalised():
    out = clean_dataframe(raw_frame())
    assert set(out["vehicle"]) == {"car", "two_wheeler", "three_wheeler"}


def test_stuck_sensor_only_inside_window_marked_missing():
    out = clean_dataframe(raw_frame())
    # R2 and R3 are inside the window with 0.9 -> missing. R4 (April, 0.9) is kept.
    assert out["vehicle_age_yr"].isna().tolist() == [False, True, True, False, False]
    assert out.loc[3, "vehicle_age_yr"] == 0.9


def test_record_ids_stay_unique_and_dates_parsed():
    out = clean_dataframe(raw_frame())
    assert out["record_id"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
