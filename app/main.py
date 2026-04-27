from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from app.model import model

app = FastAPI()

# 🔹 Definir input
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "API MLflow + FastAPI funcionando 🚀"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)

    prediction = model.predict(X)

    return {
        "prediction": prediction.tolist()
    }