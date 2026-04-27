import mlflow

# IMPORTANTE: mismo tracking que train/validate
mlflow.set_tracking_uri("file:./ruta/mlruns")
mlflow.set_experiment("ci-cd-mlflow-local")

# obtener último run
runs = mlflow.search_runs(order_by=["start_time DESC"])

if runs.empty:
    raise ValueError("No hay modelos en MLflow")

run_id = runs.iloc[0]["run_id"]

model_uri = f"runs:/{run_id}/model"

model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded:", model_uri)