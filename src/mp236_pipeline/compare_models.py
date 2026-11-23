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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = BASE_DIR / "params.yaml"


def create_pipeline(regressor, numeric_features, categorical_features):

    # Numerical preprocessing
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])


def compare_models():

    # Load configuration
    with open(PARAMS_PATH, "r") as file:
        params = yaml.safe_load(file)

    # Load cleaned dataset
    data_path = BASE_DIR / params["data"]["train_path"]
    df = pd.read_csv(data_path)

    # Select features
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
    y = df[params["target_column"]]

    # Use the same split for both models
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["training"]["test_size"],
        random_state=params["training"]["random_state"]
    )

    model_settings = params["model"]
    random_state = params["training"]["random_state"]

    models = {
        "random_forest_comparison": RandomForestRegressor(
            n_estimators=model_settings["n_estimators"],
            max_depth=model_settings["max_depth"],
            min_samples_split=model_settings["min_samples_split"],
            min_samples_leaf=model_settings["min_samples_leaf"],
            random_state=random_state,
            n_jobs=-1
        ),

        "extra_trees_candidate": ExtraTreesRegressor(
            n_estimators=model_settings["n_estimators"],
            max_depth=model_settings["max_depth"],
            min_samples_split=model_settings["min_samples_split"],
            min_samples_leaf=model_settings["min_samples_leaf"],
            random_state=random_state,
            n_jobs=-1
        )
    }

    # Configure MLflow
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    results = []

    for run_name, regressor in models.items():

        model = create_pipeline(
            regressor,
            numeric_features,
            categorical_features
        )

        with mlflow.start_run(run_name=run_name):

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Evaluate model
            mae = mean_absolute_error(y_test, predictions)
            rmse = mean_squared_error(y_test, predictions) ** 0.5
            r2 = r2_score(y_test, predictions)

            # Log parameters
            mlflow.log_params({
                "model_name": type(regressor).__name__,
                "n_estimators": model_settings["n_estimators"],
                "max_depth": model_settings["max_depth"],
                "random_state": random_state,
                "test_size": params["training"]["test_size"]
            })

            # Log metrics
            mlflow.log_metrics({
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            # Log model artifact
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model_pipeline",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree"
                ]
            )

            run_id = mlflow.active_run().info.run_id

            results.append({
                "model": type(regressor).__name__,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "run_id": run_id
            })

            print(f"\nCompleted: {type(regressor).__name__}")
            print("MAE:", round(mae, 4))
            print("RMSE:", round(rmse, 4))
            print("R2:", round(r2, 4))
            print("Run ID:", run_id)

    # Compare model results
    results_df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    best_model = results_df.loc[results_df["MAE"].idxmin()]

    print("\n========== BEST MODEL BASED ON MAE ==========")
    print("Model:", best_model["model"])
    print("MAE:", round(best_model["MAE"], 4))
    print("Run ID:", best_model["run_id"])


if __name__ == "__main__":
    compare_models()
