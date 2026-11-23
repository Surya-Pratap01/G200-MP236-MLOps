from pathlib import Path
import pandas as pd


# Dataset paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "train_cleaned.csv"


def clean_data():

    # Load original training dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # 1. Remove data leakage column
    df = df.drop(columns=["kwh_billed_in_app"])

    # 2. Fix session duration unit mix-up
    highway_rows = df["station_type"] == "Highway"

    df.loc[highway_rows, "session_minutes"] = (
        df.loc[highway_rows, "session_minutes"] / 60
    )

    # 3. Handle stuck sensor values
    stuck_sensor_rows = (
        (df["date"] >= "2026-02-15")
        & (df["date"] <= "2026-03-03")
        & (df["vehicle_age_yr"] == 0.9)
    )

    df.loc[stuck_sensor_rows, "vehicle_age_yr"] = pd.NA

    # 4. Normalize vehicle category spellings
    df["vehicle"] = (
        df["vehicle"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    vehicle_mapping = {
        "car.": "car",
        "thr.": "three_wheeler",
        "three wheeler": "three_wheeler",
        "two wheeler": "two_wheeler",
        "two.": "two_wheeler"
    }

    df["vehicle"] = df["vehicle"].replace(vehicle_mapping)

    # 5. Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n========== DATA CLEANING COMPLETE ==========")
    print("Original dataset shape:", pd.read_csv(RAW_DATA_PATH).shape)
    print("Cleaned dataset shape:", df.shape)
    print("Remaining missing values:")
    print(df.isnull().sum())
    print("\nVehicle categories:")
    print(df["vehicle"].value_counts())
    print("\nCleaned dataset saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    clean_data()
