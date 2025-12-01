# Sistem Prediksi Kedalaman Gempa Bumi

Project ini berisi aplikasi Streamlit untuk:
- Memprediksi kelas kedalaman gempa (Shallow, Intermediate, Deep) menggunakan **XGBoost** dan **LSTM**
- Menampilkan visualisasi dataset gempa
- Menampilkan peta lokasi gempa
- Mengunduh dataset dalam format CSV

## Struktur Proyek

- `app.py` : halaman utama Streamlit (beranda)
- `pages/` : berisi halaman-halaman lain (Prediksi, Visualisasi, Peta, Unduh)
- `models/` : tempat menyimpan file model (`scaler.pkl`, `xgb_depth_class.pkl`, `lstm_depth_class.keras`)
- `dataset_gempa.csv` : dataset gempa dalam format CSV
- `requirements.txt` : daftar dependency Python

## Cara Menggunakan

1. Latih model di notebook terpisah dan simpan:
   - scaler → `models/scaler.pkl`
   - XGBoost → `models/xgb_depth_class.pkl`
   - LSTM → `models/lstm_depth_class.keras`

2. Simpan dataset ke file `dataset_gempa.csv` di root repository.

3. Deploy ke Streamlit Cloud dengan memilih:
   - Repository: repo ini
   - Branch: `main`
   - File utama: `app.py`