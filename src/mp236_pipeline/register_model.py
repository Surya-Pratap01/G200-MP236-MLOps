import mlflow


# Winning model details
BEST_RUN_ID = "ad1dbf0ae5484cc1be61bad57af4c553"
MODEL_PATH = "model_pipeline"
MODEL_NAME = "MP236_EV_Energy_Prediction_Model"


def register_best_model():

    # Get the model from the successful MLflow run
    model_uri = f"runs:/{BEST_RUN_ID}/{MODEL_PATH}"

    # Register the model
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("\n========== MODEL REGISTRATION COMPLETE ==========")
    print("Model name:", registered_model.name)
    print("Model version:", registered_model.version)
    print("Source run:", BEST_RUN_ID)
    print("Status:", registered_model.status)


if __name__ == "__main__":
    register_best_model()
