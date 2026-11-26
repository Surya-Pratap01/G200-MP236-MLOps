import joblib, pathlib, sys
model_path = pathlib.Path('models/model.joblib')
print('Exists:', model_path.exists())
print('Size:', model_path.stat().st_size if model_path.exists() else 'N/A')
try:
    model = joblib.load(str(model_path))
    print('Loaded model type:', type(model))
    if hasattr(model, 'steps'):
        print('Pipeline steps:', [name for name, _ in model.steps])
except Exception as e:
    print('Load error:', e)
