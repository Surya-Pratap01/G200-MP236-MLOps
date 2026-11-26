import joblib, json, pathlib
import pandas as pd

class SimpleModel:
    """A minimal model that mimics the original scikit‑learn pipeline.
    It expects a DataFrame with the same feature columns and returns a
    deterministic prediction based on the charger power and session length.
    """
    def predict(self, X: pd.DataFrame):
        # Simple physics‑based estimate: energy = charger_kw * hours * efficiency (0.8)
        return (X['charger_kw'] * X['session_minutes'] / 60 * 0.8).values

# Save the model
model_dir = pathlib.Path(__file__).parent / 'models'
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / 'model.joblib'
joblib.dump(SimpleModel(), model_path)

# Write metadata
meta = {"model_version": "1"}
(model_dir / 'model_meta.json').write_text(json.dumps(meta))
print(f'Simple model saved to {model_path}')
