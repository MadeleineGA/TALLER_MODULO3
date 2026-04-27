import mlflow.pyfunc

MODEL_PATH = "ruta/mlruns/495601362237022937/models/m-97e2ccdbae4a4a8eb4d835590d556b18/artifacts"

model = mlflow.pyfunc.load_model(MODEL_PATH)