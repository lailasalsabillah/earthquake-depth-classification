import streamlit as st
import numpy as np
import joblib

# ===========================
# LOAD SCALER & MODEL XGBOOST
# ===========================
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")

# ===========================
# TITLE
# ===========================
st.title("🌋 Prediksi Kedalaman Gempa Bumi Menggunakan XGBoost")
st.write("""
Aplikasi ini memprediksi kelas kedalaman gempa (Shallow, Intermediate, Deep)  
berdasarkan model **XGBoost**.  
Versi ini dioptimalkan untuk **deploy Streamlit Cloud**.
""")

# ===========================
# INPUT FORM
# ===========================
st.sidebar.header("Masukkan Data Gempa")

latitude = st.sidebar.number_input("Latitude", -10.0, 10.0, 0.0)
longitude = st.sidebar.number_input("Longitude", 90.0, 150.0, 120.0)
mag = st.sidebar.number_input("Magnitude", 3.0, 9.0, 5.0)
gap = st.sidebar.number_input("Gap", 0, 300, 50)
dmin = st.sidebar.number_input("Dmin", 0.0, 30.0, 2.0)
rms = st.sidebar.number_input("RMS", 0.0, 3.0, 0.5)
horizontalError = st.sidebar.number_input("Horizontal Error", 0.0, 30.0, 5.0)
depthError = st.sidebar.number_input("Depth Error", 0.0, 20.0, 3.0)
magError = st.sidebar.number_input("Magnitude Error", 0.0, 1.0, 0.1)
year = st.sidebar.slider("Year", 2020, 2024, 2023)

# ===========================
# PREDICTION FUNCTION
# ===========================
def predict_xgb(data):
    # scaling
    scaled = scaler.transform(data)
    pred = xgb_model.predict(scaled)[0]
    return pred

# Mapping label kelas
label_mapping = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

# ===========================
# BUTTON
# ===========================
if st.sidebar.button("Prediksi"):
    input_data = np.array([[
        latitude, longitude, mag,
        gap, dmin, rms,
        horizontalError, depthError, magError,
        year
    ]])

    result = predict_xgb(input_data)

    st.subheader("🔍 Hasil Prediksi Kedalaman Gempa")
    st.write(f"**XGBoost Prediction:** {label_mapping[result]}")
    st.success("Prediksi berhasil dilakukan!")
