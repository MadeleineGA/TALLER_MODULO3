import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import sys

THRESHOLD = 5000.0

# --- Dataset ---
print("--- Debug: Cargando dataset load_diabetes ---")
data = load_diabetes(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")

# --- Cargar último modelo desde MLflow ---
print("--- Debug: Cargando modelo desde MLflow ---")

runs = mlflow.search_runs(order_by=["start_time DESC"])

if runs.empty:
    print("--- ERROR: No hay runs en MLflow ---")
    sys.exit(1)

run_id = runs.iloc[0]["run_id"]
model_uri = f"runs:/{run_id}/model"

print(f"--- Debug: Model URI: {model_uri} ---")

model = mlflow.pyfunc.load_model(model_uri)

# --- Predicción ---
print("--- Debug: Realizando predicciones ---")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

# --- Validación ---
if mse <= THRESHOLD:
    print("✅ Modelo aprobado")
    sys.exit(0)
else:
    print("❌ Modelo rechazado")
    sys.exit(1)