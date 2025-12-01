import streamlit as st
import pandas as pd

st.title("📥 Unduh Dataset atau Hasil Prediksi")

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("dataset_gempa.csv")
        return df, None
    except Exception as e:
        return None, str(e)

df, err = load_dataset()

if df is None:
    st.error("Gagal memuat `dataset_gempa.csv`. Pastikan file tersebut sudah diupload.")
    st.caption(f"Detail error: {err}")
else:
    st.write("Klik tombol di bawah untuk mengunduh dataset gempa yang digunakan dalam pelatihan model.")

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Dataset Gempa (CSV)",
        data=csv_bytes,
        file_name="dataset_gempa.csv",
        mime="text/csv"
    )

    st.info(
        "Untuk menyertakan hasil prediksi dalam file terpisah, kamu bisa menambahkan "
        "logika penyimpanan hasil prediksi ke CSV di halaman Prediksi."
    )