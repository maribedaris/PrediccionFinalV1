import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Predictor de Red Neuronal", layout="centered")

# ==============================
# CARGAR MODELOS Y ARTEFACTOS
# ==============================

@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load('modelo_red_neuronal.joblib')
    scaler = joblib.load('scaler.joblib')
    cols_escalar = joblib.load('columnas_escalar.joblib')
    cols_categoricas = joblib.load('columnas_categoricas.joblib')
    le_binarios = joblib.load('label_encoders_binarios.joblib')
    ohe = joblib.load('one_hot_encoder.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    
    return modelo, scaler, cols_escalar, cols_categoricas, le_binarios, ohe, feature_columns


try:
    modelo, scaler, cols_escalar, cols_categoricas, le_binarios, ohe, feature_columns = cargar_artefactos()
except Exception as e:
    st.error(f"Error al cargar los artefactos: {e}")
    st.stop()


# ==============================
# INTERFAZ
# ==============================

st.title("Clasificador - Red Neuronal")
st.markdown("Ingresa los datos para obtener una predicción del modelo.")

with st.form("formulario_prediccion"):

    input_data = {}

    st.subheader("Variables Numéricas")
    for col in cols_escalar:
        input_data[col] = st.number_input(col, value=0.0)

    st.subheader("Variables Categóricas (Binarias)")
    for col, le in le_binarios.items():
        if col in feature_columns:
            opciones = list(le.classes_)
            input_data[col] = st.selectbox(col, opciones)

    st.subheader("Variables Categóricas (Múltiples)")
    for i, col in enumerate(cols_categoricas):
        opciones = list(ohe.categories_[i])
        input_data[col] = st.selectbox(col, opciones)

    submit_button = st.form_submit_button("Realizar Predicción")


# ==============================
# PREDICCIÓN
# ==============================

if submit_button:

    try:
        # 1️⃣ Convertir a DataFrame
        df_input = pd.DataFrame([input_data])

        # 2️⃣ Escalar numéricas
        df_input[cols_escalar] = scaler.transform(df_input[cols_escalar])

        # 3️⃣ Codificar binarias
        for col, le in le_binarios.items():
            if col in df_input.columns:
                df_input[col] = le.transform(df_input[col])

        # 4️⃣ One Hot Encoding
        df_cat = df_input[cols_categoricas]
        cat_encoded = ohe.transform(df_cat)

        if hasattr(cat_encoded, "toarray"):
            cat_encoded = cat_encoded.toarray()

        try:
            ohe_col_names = ohe.get_feature_names_out(cols_categoricas)
        except:
            ohe_col_names = ohe.get_feature_names(cols_categoricas)

        df_cat_encoded = pd.DataFrame(cat_encoded, columns=ohe_col_names)

        # 5️⃣ Eliminar categóricas originales
        df_input = df_input.drop(columns=cols_categoricas)

        # 6️⃣ Unir todo
        df_final = pd.concat(
            [df_input.reset_index(drop=True),
             df_cat_encoded.reset_index(drop=True)],
            axis=1
        )

        # 7️⃣ Asegurar orden correcto de columnas
        df_final = df_final.reindex(columns=feature_columns, fill_value=0)

        # 8️⃣ Predicción
        prediccion = modelo.predict(df_final)[0]
        probabilidad = modelo.predict_proba(df_final)[0][1]

        # ==============================
        # RESULTADOS
        # ==============================

        st.subheader("Resultado")

        if prediccion == 1:
            st.success("Ingresará al proceso ✅")
        else:
            st.error("No ingresará al proceso ❌")

        st.write(f"Probabilidad estimada de ingreso: {probabilidad:.2%}")

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")
