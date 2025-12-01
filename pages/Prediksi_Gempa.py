import streamlit as st
import numpy as np
import pandas as pd
import joblib

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

st.title("🔍 Prediksi Kedalaman Gempa")

st.write(
    "Masukkan parameter gempa di bawah ini untuk memprediksi apakah gempa termasuk "
    "**Shallow**, **Intermediate**, atau **Deep**."
)

# =========================
# INPUT FORM
# =========================
col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", -15.0, 15.0, -6.5, help="Lintang episenter gempa.")
    longitude = st.number_input("Longitude", 90.0, 150.0, 107.0, help="Bujur episenter gempa.")
    mag = st.number_input("Magnitudo (Mw)", 3.0, 9.5, 5.0, help="Kekuatan gempa (skala Mw).")
    gap = st.number_input("Gap (derajat)", 0, 360, 80, help="Sudut celah distribusi stasiun.")

with col2:
    dmin = st.number_input("Dmin (derajat)", 0.0, 30.0, 2.1, help="Jarak minimum ke stasiun terdekat.")
    rms = st.number_input("RMS Residual", 0.0, 3.0, 0.7, help="RMS residual time.")
    horizontalError = st.number_input("Horizontal Error (km)", 0.0, 50.0, 8.0)
    depthError = st.number_input("Depth Error (km)", 0.0, 30.0, 6.0)
    magError = st.number_input("Magnitude Error", 0.0, 1.0, 0.12)
    year = st.number_input("Tahun", 2000, 2100, 2023)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_scaler_and_models():
    scaler = None
    xgb_model = None
    lstm_model = None
    scaler_err = xgb_err = lstm_err = None

    try:
        scaler = joblib.load("models/scaler.pkl")
    except Exception as e:
        scaler_err = str(e)

    try:
        xgb_model = joblib.load("models/xgb_depth_class.pkl")
    except Exception as e:
        xgb_err = str(e)

    if TENSORFLOW_AVAILABLE:
        try:
            lstm_model = load_model("models/lstm_depth_class.keras")
        except Exception as e:
            lstm_err = str(e)
    else:
        lstm_err = "TensorFlow tidak tersedia di environment ini."

    return scaler, xgb_model, lstm_model, scaler_err, xgb_err, lstm_err


scaler, xgb_model, lstm_model, scaler_err, xgb_err, lstm_err = load_scaler_and_models()

label_map = {
    0: "Shallow (< 70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (> 300 km)"
}

# =========================
# PREDICTION
# =========================
if st.button("🚀 Jalankan Prediksi"):

    data = np.array([[
        latitude, longitude, mag, gap, dmin, rms,
        horizontalError, depthError, magError, year
    ]])

    df_input = pd.DataFrame(data, columns=[
        "latitude", "longitude", "mag", "gap", "dmin", "rms",
        "horizontalError", "depthError", "magError", "year"
    ])

    if scaler is None:
        st.error("Scaler belum tersedia. Pastikan file `models/scaler.pkl` sudah diupload.\n\n"
                 f"Detail error: {scaler_err}")
    else:
        scaled = scaler.transform(data)

        col_pred1, col_pred2 = st.columns(2)

        # XGBoost prediction
        with col_pred1:
            st.subheader("🧠 Model XGBoost")
            if xgb_model is None:
                st.warning("Model XGBoost belum tersedia. Upload `models/xgb_depth_class.pkl`.")
                if xgb_err:
                    st.caption(f"Detail: {xgb_err}")
            else:
                xgb_pred = xgb_model.predict(scaled)[0]
                st.success(f"Prediksi: **{label_map.get(int(xgb_pred), 'Tidak diketahui')}**")

        # LSTM prediction
        with col_pred2:
            st.subheader("🧠 Model LSTM")
            if not TENSORFLOW_AVAILABLE:
                st.warning("TensorFlow tidak terinstal di environment ini.")
            elif lstm_model is None:
                st.warning("Model LSTM belum tersedia. Upload `models/lstm_depth_class.keras`.")
                if lstm_err:
                    st.caption(f"Detail: {lstm_err}")
            else:
                lstm_input = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))
                lstm_prob = lstm_model.predict(lstm_input)
                lstm_class = int(np.argmax(lstm_prob, axis=1)[0])
                st.success(f"Prediksi: **{label_map.get(lstm_class, 'Tidak diketahui')}**")

        st.markdown("---")
        st.subheader("📄 Data Input")
        st.dataframe(df_input, use_container_width=True)

        # Simple bar chart magnitude vs dummy depth category index
        st.subheader("📈 Visualisasi Magnitudo")
        st.bar_chart(pd.DataFrame({"Magnitude": [mag]}))
