from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CLEANED_PATH = BASE_DIR / "data" / "processed" / "train_cleaned.csv"


def test_cleaned_dataset_shape():
    df = pd.read_csv(CLEANED_PATH)

    assert df.shape == (1000, 11)


def test_leakage_column_removed():
    df = pd.read_csv(CLEANED_PATH)

    assert "kwh_billed_in_app" not in df.columns


def test_target_column_preserved():
    df = pd.read_csv(CLEANED_PATH)

    assert "energy_kwh" in df.columns
    assert df["energy_kwh"].notna().all()


def test_highway_duration_converted():
    df = pd.read_csv(CLEANED_PATH)

    highway = df.loc[
        df["station_type"] == "Highway",
        "session_minutes"
    ]

    assert highway.max() <= 240
    assert highway.min() >= 10


def test_vehicle_categories_cleaned():
    df = pd.read_csv(CLEANED_PATH)

    expected_categories = {
        "car",
        "three_wheeler",
        "two_wheeler"
    }

    assert set(df["vehicle"].unique()) == expected_categories


def test_stuck_sensor_values_handled():
    df = pd.read_csv(CLEANED_PATH)

    assert df["vehicle_age_yr"].isna().sum() == 92


def test_no_duplicate_record_ids():
    df = pd.read_csv(CLEANED_PATH)

    assert df["record_id"].is_unique
