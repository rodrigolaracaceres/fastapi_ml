from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Cargar modelo
model = joblib.load("model.pkl")

@app.get("/")
def read_root():
    return {"message": "API de Regresión Lineal"}

@app.get("/predict")
def predict(x: float):
    prediction = model.predict(np.array([[x]]))
    return {"prediction": float(prediction[0][0])}