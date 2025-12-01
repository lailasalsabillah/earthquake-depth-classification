import streamlit as st

st.set_page_config(
    page_title="Prediksi Kedalaman Gempa",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 Sistem Prediksi Kedalaman Gempa Bumi")

st.write("""
Aplikasi ini dibangun untuk **memprediksi kelas kedalaman gempa bumi** (Shallow, Intermediate, Deep)
berdasarkan parameter seismik seperti lokasi, magnitudo, dan error pengukuran.

Gunakan menu di sebelah kiri untuk:
- 🔍 Halaman **Prediksi Gempa** — memasukkan parameter dan melihat hasil prediksi model **XGBoost & LSTM**  
- 📊 Halaman **Visualisasi Data** — melihat distribusi magnitude, kedalaman, dan korelasi fitur  
- 🗺️ Halaman **Peta Lokasi Gempa** — menampilkan sebaran episenter gempa dalam peta interaktif  
- 📥 Halaman **Unduh Hasil** — mengunduh dataset dalam format CSV

> **Catatan penting:** sebelum deploy, pastikan kamu sudah:
> - Mengganti file `dataset_gempa.csv` dengan dataset gempa asli milikmu  
> - Mengisi folder `models/` dengan file:
>   - `scaler.pkl`
>   - `xgb_depth_class.pkl`
>   - `lstm_depth_class.keras`
""")