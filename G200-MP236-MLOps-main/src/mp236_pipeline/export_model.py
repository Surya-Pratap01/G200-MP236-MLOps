"""Export the registered model to a portable file for the API / Docker.

Run on the machine that holds the MLflow registry (mlflow.db):

    python src/mp236_pipeline/export_model.py
    python src/mp236_pipeline/export_model.py --model-uri "models:/MP236_EV_Energy_Prediction_Model@champion"

Writes models/model.joblib and models/model_meta.json. Commit both
(the file is small); the API never talks to a local MLflow registry.
"""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_NAME = "MP236_EV_Energy_Prediction_Model"
OUT_DIR = BASE_DIR / "models"


def export_model(model_uri: str, version: str, out_dir: Path = OUT_DIR) -> Path:
    model = mlflow.sklearn.load_model(model_uri)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)
    (out_dir / "model_meta.json").write_text(json.dumps(
        {"model_name": MODEL_NAME, "model_version": version, "source_uri": model_uri}, indent=2))
    return model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="1", help="registered model version (default 1)")
    parser.add_argument("--model-uri", default=None,
                        help="override the URI, e.g. models:/NAME@champion or runs:/<run_id>/model_pipeline")
    args = parser.parse_args()

    uri = args.model_uri or f"models:/{MODEL_NAME}/{args.version}"
    path = export_model(uri, args.version)
    print("Exported", uri, "->", path)


if __name__ == "__main__":
    main()
