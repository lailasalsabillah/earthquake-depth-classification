import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.title("🗺️ Peta Lokasi Gempa")

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
    if df.empty:
        st.warning("Dataset kosong. Tidak ada titik yang dapat dipetakan.")
    else:
        center_lat = df["latitude"].mean()
        center_lon = df["longitude"].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

        for _, row in df.iterrows():
            depth = row.get("depth", 0)
            mag = row.get("mag", 0)
            popup = f"Mag: {mag}, Depth: {depth} km"

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                popup=popup,
                color="red",
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

        st_folium(m, width=800, height=500)
