import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# LOAD MODEL & SCALER
# ===========================
scaler = joblib.load("models/scaler.pkl")
lstm_model = load_model("models/lstm_depth_class.keras")
xgb_model = joblib.load("models/xgb_depth_class.pkl")

# TITLE
# ===========================
st.title("🌋 Prediksi Kedalaman Gempa Bumi")
st.write("""
Aplikasi ini memprediksi kelas kedalaman gempa (Shallow, Intermediate, Deep)  
berdasarkan model **LSTM** dan **XGBoost**.
""")

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

# PREDICTION
# ===========================
def predict(data):
    data_scaled = scaler.transform(data)

    # LSTM requires 3D
    data_lstm = data_scaled.reshape((1, 1, data_scaled.shape[1]))
    lstm_pred = np.argmax(lstm_model.predict(data_lstm), axis=1)[0]

    xgb_pred = xgb_model.predict(data)[0]

    return lstm_pred, xgb_pred

label_mapping = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

if st.sidebar.button("Prediksi"):
    input_data = np.array([[
        latitude, longitude, mag,
        gap, dmin, rms,
        horizontalError, depthError, magError,
        year
    ]])

    lstm_result, xgb_result = predict(input_data)

    st.subheader("🔍 Hasil Prediksi")
    st.write(f"**LSTM Model:** {label_mapping[lstm_result]}")
    st.write(f"**XGBoost Model:** {label_mapping[xgb_result]}")

    st.success("Prediksi berhasil dilakukan!")
