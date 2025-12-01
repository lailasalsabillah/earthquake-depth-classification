import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Visualisasi Dataset Gempa")

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
    st.write("Berikut beberapa visualisasi sederhana dari dataset gempa yang digunakan untuk pelatihan model.")

    st.subheader("Distribusi Kedalaman Gempa")
    fig1, ax1 = plt.subplots(figsize=(7,4))
    sns.histplot(df["depth"], bins=40, kde=True, ax=ax1)
    ax1.set_xlabel("Depth (km)")
    st.pyplot(fig1)

    st.subheader("Distribusi Magnitudo")
    fig2, ax2 = plt.subplots(figsize=(7,4))
    sns.histplot(df["mag"], bins=40, kde=True, color="orange", ax=ax2)
    ax2.set_xlabel("Magnitude (Mw)")
    st.pyplot(fig2)

    st.subheader("Sebaran Lokasi Gempa")
    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.scatter(df["longitude"], df["latitude"], s=5, alpha=0.4)
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    st.pyplot(fig3)

    st.subheader("Heatmap Korelasi Fitur Numerik")
    num_cols = df.select_dtypes(include="number").columns
    corr = df[num_cols].corr()
    fig4, ax4 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)