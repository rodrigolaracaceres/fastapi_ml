# 📌 Linear Regression Model

Este proyecto implementa un modelo **Regresión Lineal** utilizando **FastAPI** y lo despliega en **Render**.

## 🔹 Crear un archivo `requirements.txt`

Para asegurar que la API funcione correctamente, crea un archivo `requirements.txt` con las siguientes dependencias:

```bash
fastapi
uvicorn
numpy
joblib
scikit-learn
matplotlib
```

Esto garantizará que la API tenga las librerías necesarias para ejecutarse en Render.

---

## 🔹 Nombre del archivo principal

El nombre del archivo de la API **debe llamarse `main.py`** para que Render lo detecte automáticamente.

---

## 🔹 Iniciar sesión en Render

Para desplegar la API en **Render**, sigue estos pasos:

1️⃣ **Inicia sesión en [Render](https://render.com/)** utilizando tu cuenta de **GitHub**.  

2️⃣ **Crea un nuevo servicio web**.  

3️⃣ **Conecta el repositorio de GitHub donde tienes este proyecto**.  

4️⃣ **Configura el Start Command** con:  

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

5️⃣ **Selecciona Python 3.9+ como versión del runtime.**  

6️⃣ **Haz clic en Deploy** y Render generará la URL de tu API.  

---

## 🔹 Pruebas de la API

Una vez desplegado en Render, puedes probar la API con los siguientes endpoints:

🔹 **[Realizar una predicción con Regresión Lineal](https://fastapi-ml-kogt.onrender.com/predict?x=5.0)**

🔹 **[Obtener métricas del modelo](https://fastapi-ml-kogt.onrender.com/metrics)**

🔹 **[Ver el gráfico de la regresión](https://fastapi-ml-kogt.onrender.com/plot)**

---

## 🔹 Estructura del Proyecto

```
/tu-repo
│── main.py               # Código de la API FastAPI
│── train_model.py        # Entrena y guarda el modelo de Regresión Lineal
│── requirements.txt      # Dependencias del proyecto
│── README.md             # Documentación del proyecto
```

---

## 🎯 Autor

📌 **Rodrigo Alberto Lara Cáceres**\
🔗 **[LinkedIn](https://www.linkedin.com/in/rodrigo-lara-caceres/)**\
📧 **[rodlarca@gmail.com](mailto:rodlarca@gmail.com)**

🚀 **¡Listo para hacer predicciones con Regresión Lineal en la nube!** 🎯🔥