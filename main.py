from fastapi import FastAPI
from fastapi.responses import Response
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = FastAPI()

# Cargar modelo y datos originales
data = joblib.load("model.pkl")
model = data["model"]
X_train = data["X_train"]
y_train = data["y_train"]

@app.get("/")
def read_root():
    return {"message": "API de Regresión Lineal"}

@app.get("/predict")
def predict(x: float):
    prediction = model.predict(np.array([[x]]))
    return {"prediction": float(prediction[0][0])}

@app.get("/plot")
def plot_regression():
    """Genera y devuelve un gráfico de la regresión lineal usando los datos originales"""
    # Predicciones del modelo
    y_pred = model.predict(X_train)

    # Crear el gráfico
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train, y_train, label="Datos reales", alpha=0.6)
    plt.plot(X_train, y_pred, color="red", label="Regresión Lineal")
    plt.xlabel("X")
    plt.ylabel("Predicción")
    plt.title("Regresión Lineal")
    plt.legend()
    plt.grid()

    # Guardar la imagen en memoria
    img_io = io.BytesIO()
    plt.savefig(img_io, format="png")
    img_io.seek(0)

    return Response(content=img_io.getvalue(), media_type="image/png")

@app.get("/metrics")
def get_metrics():
    """Calcula y devuelve métricas del modelo basado en los datos originales"""
    # Predicciones del modelo
    y_pred = model.predict(X_train)
    
    rmse = mean_squared_error(y_train, y_pred) ** 0.5  # Raíz del Error Cuadrático Medio
    mae = mean_absolute_error(y_train, y_pred)  # Error Absoluto Medio
    r2 = r2_score(y_train, y_pred)  # Coeficiente de Determinación (R²)

    return {
        "R² Score": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4)
    }
