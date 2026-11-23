from pathlib import Path
import pandas as pd
import mlflow.sklearn


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "train_cleaned.csv"

MODEL_URI = "models:/MP236_EV_Energy_Prediction_Model/1"


def verify_model():

    # Load registered model from MLflow
    model = mlflow.sklearn.load_model(MODEL_URI)

    # Load sample data
    df = pd.read_csv(DATA_PATH)

    # Select model input features
    features = [
        "battery_start_pct",
        "session_minutes",
        "charger_kw",
        "temp_c",
        "vehicle_age_yr",
        "queue_before",
        "vehicle",
        "station_type"
    ]

    sample_data = df[features].head(5)

    # Generate predictions
    predictions = model.predict(sample_data)

    print("\n========== REGISTERED MODEL VERIFICATION ==========")
    print("Model loaded successfully!")
    print("Model URI:", MODEL_URI)

    print("\nSample predictions:")

    for i, prediction in enumerate(predictions):
        print(f"Record {i + 1}: {prediction:.2f} kWh")


if __name__ == "__main__":
    verify_model()
