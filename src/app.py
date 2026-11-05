## app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Modelo de Regresión Lineal con Streamlit")

# 1. Crear o cargar datos de ejemplo
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.flatten() + np.random.randn(100) * 2 + 5

df = pd.DataFrame({'Variable X': X.flatten(), 'Variable Y': y})

# 2. Entrenar el modelo
model = LinearRegression()
model.fit(X, y)

# 3. Mostrar métricas y ecuación
st.subheader("Resultados del Modelo")
st.write(f"**Pendiente (B₁):** {model.coef_[0]:.2f}")
st.write(f"**Intersección (B₀):** {model.intercept_:.2f}")
st.write(f"Ecuación: Y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * X")

# 4. Gráfico de dispersión y línea de regresión
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Datos reales')
ax.plot(X, model.predict(X), color='red', label='Línea de regresión')
ax.set_xlabel('Variable X')
ax.set_ylabel('Variable Y')
ax.legend()
st.pyplot(fig)

# 5. Interfaz para hacer predicciones
st.subheader("Hacer una Predicción")
input_x = st.number_input("Ingresa un valor para X:", value=5.0)
prediction = model.predict([[input_x]])[0]
st.success(f"El valor predicho para Y es: **{prediction:.2f}**")
