import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Predictor de Red Neuronal", layout="centered")

# Cargar modelos y transformadores en caché para optimizar velocidad
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
    st.error(f"Error al cargar los artefactos: {e}. Verifica que todos los archivos .joblib estén en la misma carpeta.")
    st.stop()

st.title("Clasificador - Red Neuronal")
st.markdown("Ingresa los datos en el siguiente formulario para obtener una predicción del modelo.")

# Creación del formulario de entrada
with st.form("formulario_prediccion"):
    st.subheader("Variables Numéricas")
    input_data = {}
    
    # Cuadros editables numéricos generados dinámicamente
    col1, col2 = st.columns(2)
    with col1:
        if 'Año desmovilización' in cols_escalar:
            input_data['Año desmovilización'] = st.number_input('Año desmovilización', min_value=1950, max_value=2050, value=2010, step=1)
    with col2:
        if 'Total Integrantes grupo familiar' in cols_escalar:
            input_data['Total Integrantes grupo familiar'] = st.number_input('Total Integrantes grupo familiar', min_value=1, max_value=30, value=1, step=1)

    # Resto de numéricas si existieran
    for col in cols_escalar:
        if col not in input_data:
            input_data[col] = st.number_input(col, value=0.0)

    st.subheader("Variables Categóricas (Binarias)")
    # Cuadros desplegables para binarios (se excluyen variables target si estuvieran guardadas ahí)
    for col, le in le_binarios.items():
        if col in feature_columns:  # Solo mostramos los que son características predictoras
            opciones = list(le.classes_)
            input_data[col] = st.selectbox(col, opciones)

    st.subheader("Variables Categóricas (Múltiples)")
    # Cuadros desplegables extraídos del OneHotEncoder
    for i, col in enumerate(cols_categoricas):
        opciones = list(ohe.categories_[i])
        input_data[col] = st.selectbox(col, opciones)

    # Botón de predicción
    submit_button = st.form_submit_button(label='Realizar Predicción')

# Lógica una vez se hace clic en el botón
if submit_button:
    # 1. Convertir los datos a DataFrame
    df_input = pd.DataFrame([input_data])

    # 2. Transformar Numéricas (Escalado)
    df_input[cols_escalar] = scaler.transform(df_input[cols_escalar])

    # 3. Transformar Binarias (Label Encoding)
    for col, le in le_binarios.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])

    # 4. Transformar Categóricas Múltiples (One-Hot Encoding)
    df_cat = df_input[cols_categoricas]
    cat_encoded = ohe.transform(df_cat)
    if hasattr(cat_encoded, "toarray"):
        cat_encoded = cat_encoded.toarray()
    
    try:
        ohe_col_names = ohe.get_feature_names_out(cols_categoricas)
    except AttributeError: # Para versiones anteriores de scikit-learn
        ohe_col_names = ohe.get_feature_names(cols_categoricas)