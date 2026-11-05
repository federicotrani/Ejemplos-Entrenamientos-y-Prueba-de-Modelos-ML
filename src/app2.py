import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Regresión Lineal", layout="wide")
st.title("📊 Modelo de Regresión Lineal")

# 1. Carga del archivo CSV
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # 2. Selección de variables
    st.sidebar.header("2. Configuración del Modelo")
    all_columns = df.columns.tolist()
    
    target_var = st.sidebar.selectbox("Selecciona la variable objetivo (Y):", all_columns)
    
    remaining_columns = [col for col in all_columns if col != target_var]
    features = st.sidebar.multiselect(
        "Selecciona las variables predictoras (X):", 
        remaining_columns, 
        default=remaining_columns[:2] if len(remaining_columns) >= 2 else remaining_columns
    )

    if len(features) > 0:
        X = df[features]
        y = df[target_var]

        # 3. Entrenamiento del Modelo
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # 4. Mostrar Métricas del Modelo
        st.subheader("📈 Rendimiento del Modelo")
        col1, col2, col3 = st.columns(3)
        
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        col1.metric("Coeficiente de Determinación (R2)", f"{r2:.4f}")
        col2.metric("Raíz del Error Cuadrático Medio (RMSE)", f"{rmse:.4f}")
        col3.metric("Intersección (B0)", f"{model.intercept_:.4f}")

        # 5. Coeficientes de cada variable
        st.subheader("📋 Coeficientes de las Variables Predictoras")
        coef_df = pd.DataFrame({
            'Variable Predictora (X)': features,
            'Coeficiente (Peso)': model.coef_
        })
        st.table(coef_df)

        # 6. Sección de Predicciones en Vivo
        st.subheader("🔮 Hacer una Predicción")
        st.write("Introduce los valores para generar una estimación:")
        
        # Generar inputs dinámicos basados en las variables seleccionadas
        input_data = {}
        input_cols = st.columns(min(len(features), 4)) # Distribuye los inputs en columnas
        
        for idx, feat in enumerate(features):
            col_to_use = input_cols[idx % 4]
            default_val = float(df[feat].mean())
            input_data[feat] = col_to_use.number_input(f"Valor para {feat}:", value=default_val)

        # Botón de predicción
        if st.button("Calcular Predicción"):
            input_df = pd.DataFrame([input_data])
            live_pred = model.predict(input_df)[0]
            st.success(f"El valor estimado para **{target_var}** es: **{live_pred:.4f}**")
    else:
        st.warning("Por favor, selecciona al menos una variable predictora (X) en la barra lateral.")
else:
    st.info("👋 Por favor, sube un archivo CSV en la barra lateral izquierda para comenzar.")
