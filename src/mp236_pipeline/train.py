from pathlib import Path
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = BASE_DIR / "params.yaml"

def train_model():

    # Load model configuration
    with open(PARAMS_PATH, "r") as file:
        params = yaml.safe_load(file)

    data_path = BASE_DIR / params["data"]["train_path"]
    df = pd.read_csv(data_path)

    target_column = params["target_column"]

    # Select input features and target
    numeric_features = [
        "battery_start_pct",
        "session_minutes",
        "charger_kw",
        "temp_c",
        "vehicle_age_yr",
        "queue_before"
    ]

    categorical_features = [
        "vehicle",
        "station_type"
    ]

    feature_columns = numeric_features + categorical_features

    X = df[feature_columns]
    y = df[target_column]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["training"]["test_size"],
        random_state=params["training"]["random_state"]
    )

    # Preprocess numerical features
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Preprocess categorical features
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # Create model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=params["model"]["n_estimators"],
            max_depth=params["model"]["max_depth"],
            min_samples_split=params["model"]["min_samples_split"],
            min_samples_leaf=params["model"]["min_samples_leaf"],
            random_state=params["training"]["random_state"],
            n_jobs=-1
        ))
    ])

    # Configure MLflow
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="random_forest_baseline"):

        # Log model parameters
        mlflow.log_params({
            "model_name": params["model"]["name"],
            "n_estimators": params["model"]["n_estimators"],
            "max_depth": params["model"]["max_depth"],
            "min_samples_split": params["model"]["min_samples_split"],
            "min_samples_leaf": params["model"]["min_samples_leaf"],
            "test_size": params["training"]["test_size"],
            "random_state": params["training"]["random_state"]
        })

        # Train model
        model.fit(X_train, y_train)

        # Generate predictions
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        # Log metrics
        mlflow.log_metrics({
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_pipeline",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        print("\n========== MODEL TRAINING COMPLETE ==========")
        print("Experiment:", params["mlflow"]["experiment_name"])
        print("Model: Random Forest Regressor")
        print("Training rows:", len(X_train))
        print("Testing rows:", len(X_test))
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2 Score:", round(r2, 4))
        print("MLflow Run ID:", mlflow.active_run().info.run_id)

if __name__ == "__main__":
    train_model()