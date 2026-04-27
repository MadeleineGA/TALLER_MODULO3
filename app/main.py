from fastapi import FastAPI
import pandas as pd
from app.model import model

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API MLflow + FastAPI funcionando 🚀"}

@app.post("/predict")
def predict(data: list):
    df = pd.DataFrame(data)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}