import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mp236_pipeline.preprocessing import clean_dataframe


BASE_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = BASE_DIR / "params.yaml"
RAW_TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
CLEAN_TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_cleaned.csv"
LATER_PATH = BASE_DIR / "data" / "raw" / "later.csv"
V1_MODEL_PATH = BASE_DIR / "models" / "model.joblib"
MODEL_META_PATH = BASE_DIR / "models" / "model_meta.json"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

MODEL_NAME = "MP236_EV_Energy_Prediction_Model"
EXPECTED_TRAIN_SHA256 = (
    "ebd05b68a4971458c13ffbf3098b7db7"
    "e027300ae66fce6a76ed3d85cbdb04c0"
)

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
NUMERIC_FEATURES = FEATURES[:6]
CATEGORICAL_FEATURES = FEATURES[6:]


def main():
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_hash = hashlib.sha256(RAW_TRAIN_PATH.read_bytes()).hexdigest()
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"train.csv hash mismatch: {train_hash}")

    train = pd.read_csv(CLEAN_TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)

    if len(train) != 1000 or len(later) != 300:
        raise ValueError(
            f"Expected 1000 train and 300 later rows; got {len(train)} and {len(later)}"
        )

    later["date"] = pd.to_datetime(later["date"])
    later["session_minutes"] = pd.to_numeric(
        later["session_minutes"]
    ).astype(float)
    later = later.sort_values("date", kind="mergesort").reset_index(drop=True)

    # Split chronologically before fitting the model's imputers and encoder.
    later_train = clean_dataframe(later.iloc[:210].copy())
    check = clean_dataframe(later.iloc[210:].copy())

    target = params["target_column"]
    train_rows = pd.concat([train, later_train], ignore_index=True)

    X_train = train_rows[FEATURES]
    y_train = train_rows[target]
    X_check = check[FEATURES]
    y_check = check[target]

    if len(X_train) != 1210 or len(X_check) != 90:
        raise ValueError(
            f"Expected 1210 fit rows and 90 check rows; got {len(X_train)} and {len(X_check)}"
        )

    model_settings = params["model"]
    random_state = params["training"]["random_state"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    v2_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=model_settings["n_estimators"],
            max_depth=model_settings["max_depth"],
            min_samples_split=model_settings["min_samples_split"],
            min_samples_leaf=model_settings["min_samples_leaf"],
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    # Compare V1 and V2 on the identical final 90 later rows.
    v1_model = joblib.load(V1_MODEL_PATH)
    v2_model.fit(X_train, y_train)

    mae_v1 = float(mean_absolute_error(y_check, v1_model.predict(X_check)))
    mae_v2 = float(mean_absolute_error(y_check, v2_model.predict(X_check)))
    promote = mae_v2 <= 0.88 * mae_v1

    print(f"V1 MAE on same 90 rows: {mae_v1:.6f}")
    print(f"V2 MAE on same 90 rows: {mae_v2:.6f}")
    print(f"Promotion decision: {'promote' if promote else 'keep'}")

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )

    try:
        client.get_registered_model(MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not find the registered V1 model {MODEL_NAME!r}. "
            "Stop and check the MLflow registry before continuing."
        ) from exc

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not any(str(version.version) == "1" for version in versions):
        raise RuntimeError("The registry does not contain expected V1 / Version 1.")
    if any(str(version.version) == "2" for version in versions):
        raise RuntimeError("Version 2 already exists; inspect MLflow before rerunning.")

    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="stage4_v2_chronological") as run:
        mlflow.log_params({
            "model_name": model_settings["name"],
            "n_estimators": model_settings["n_estimators"],
            "max_depth": model_settings["max_depth"],
            "min_samples_split": model_settings["min_samples_split"],
            "min_samples_leaf": model_settings["min_samples_leaf"],
            "random_state": random_state,
            "v2_training_rows": len(X_train),
            "v2_check_rows": len(X_check),
            "later_training_rows": 210,
            "data_sha256": train_hash,
            "later_sha256": hashlib.sha256(LATER_PATH.read_bytes()).hexdigest(),
        })
        mlflow.log_metrics({
            "mae_v1_on_check": mae_v1,
            "mae_v2_on_check": mae_v2,
        })
        mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="v2_model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )
        run_id = run.info.run_id

    registered_v2 = mlflow.register_model(
        model_uri=f"runs:/{run_id}/v2_model",
        name=MODEL_NAME,
        await_registration_for=300,
    )
    if str(registered_v2.version) != "2":
        raise RuntimeError(
            f"Expected V2 to register as Version 2, got {registered_v2.version}"
        )

    champion_version = "2" if promote else "1"
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        champion_version,
    )

    champion_model = v2_model if promote else v1_model
    joblib.dump(champion_model, V1_MODEL_PATH)
    MODEL_META_PATH.write_text(
        json.dumps({
            "model_name": MODEL_NAME,
            "model_version": champion_version,
            "source_uri": f"models:/{MODEL_NAME}@champion",
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Registered V2 as version {registered_v2.version}")
    print(f"Champion version: {champion_version}")
    print(f"Exported champion to {V1_MODEL_PATH}")


if __name__ == "__main__":
    main()
