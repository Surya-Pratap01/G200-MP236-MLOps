from pathlib import Path
import hashlib

from mlflow.tracking import MlflowClient


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"

EXPECTED_SHA256 = (
    "ebd05b68a4971458c13ffbf3098b7db7"
    "e027300ae66fce6a76ed3d85cbdb04c0"
)


def repair_run_metadata():

    # Calculate hash of the original training file
    data_hash = hashlib.sha256(
        TRAIN_PATH.read_bytes()
    ).hexdigest()

    if data_hash != EXPECTED_SHA256:
        raise ValueError(
            "Training dataset hash does not match "
            "the original MP236 dataset."
        )

    client = MlflowClient()

    experiment = next(
        (
            exp for exp in client.search_experiments()
            if exp.name == "MP236_EV_Energy_Prediction"
        ),
        None
    )

    if experiment is None:
        raise ValueError("MP236 experiment not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000
    )

    print("Original dataset SHA-256:", data_hash)
    print("Runs found:", len(runs))

    for run in runs:

        run_id = run.info.run_id

        # Add missing dataset fingerprint
        if "data_sha256" not in run.data.params:
            client.log_param(
                run_id,
                "data_sha256",
                data_hash
            )

        # Add lowercase MAE metric
        old_mae = run.data.metrics.get("MAE")

        if old_mae is not None and "mae" not in run.data.metrics:
            client.log_metric(
                run_id,
                "mae",
                old_mae
            )

        print(
            "Updated:",
            run_id,
            "| Status:",
            run.info.status,
            "| MAE:",
            old_mae
        )

    print("\nMetadata repair completed.")


if __name__ == "__main__":
    repair_run_metadata()