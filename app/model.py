import mlflow
import mlflow.pyfunc

# IMPORTANTE: mismo tracking que train/validate
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ci-cd-mlflow-local")

# obtener último run
runs = mlflow.search_runs(order_by=["start_time DESC"])

if runs.empty:
    raise ValueError("No hay modelos en MLflow")

run_id = runs.iloc[0]["run_id"]

MODEL_URI = f"runs:/{run_id}/model"

model = mlflow.pyfunc.load_model(MODEL_URI)