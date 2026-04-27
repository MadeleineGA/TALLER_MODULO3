import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 🔥 UNA sola configuración
mlflow.set_tracking_uri("file:./ruta/mlruns")
mlflow.set_experiment("ci-cd-mlflow-local")

# Dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Métrica
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# MLflow run
with mlflow.start_run() as run:
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    # 🔥 clave para CI
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    print("MSE:", mse)
    print("RUN ID:", run.info.run_id)