from fastapi import FastAPI
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io

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

@app.get("/plot")
def plot_regression():
    """Genera y devuelve un gráfico de la regresión lineal"""
    # Simulación de datos (mismos usados en el modelo)
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2.5 * X + np.random.randn(100, 1) * 2

    # Predicciones del modelo
    y_pred = model.predict(X)

    # Crear el gráfico
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, label="Datos reales", alpha=0.6)
    plt.plot(X, y_pred, color="red", label="Regresión Lineal")
    plt.xlabel("X")
    plt.ylabel("Predicción")
    plt.title("Regresión Lineal")
    plt.legend()
    plt.grid()

    # Guardar la imagen en memoria
    img_io = io.BytesIO()
    plt.savefig(img_io, format="png")
    img_io.seek(0)

    # Devolver la imagen como respuesta
    return Response(content=img_io.getvalue(), media_type="image/png")