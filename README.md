

# 🚗 Prediksi Kecepatan Mobil Berdasarkan Spesifikasi Aerodinamika

## Deskripsi Project

Project ini bertujuan membangun model *machine learning regression* untuk memprediksi **kecepatan mobil Formula 1 (speed_kmh)** berdasarkan karakteristik aerodinamika kendaraan.

Variabel yang digunakan meliputi:

* `wing_angle_deg` → sudut sayap kendaraan
* `drs_active` → status aktivasi DRS
* `downforce_n` → gaya tekan ( *downforce* )
* `drag_n` → gaya hambat ( *drag* )
* `stability_index` → indeks stabilitas kendaraan

Beberapa algoritma regresi dibandingkan untuk memperoleh performa terbaik, yaitu:

* Random Forest Regressor
* XGBoost Regressor
* Support Vector Regression (SVR)

Project juga mencakup proses  *hyperparameter tuning* , evaluasi model, dan penyimpanan model terbaik untuk digunakan kembali.

---

## Anggota Tim

Kelompok A

1. Unaisyah Moulia Wardhani – [K1D024052]
2. Rif'at Fazle Mawla – [K1D024055]
3. Talenta Nusantara Wijaya – [K1D024057]
4. [Raichan Achmad Rabbani] – [K1D024064]
5. [Shadrina Izzati] – [K1D024066]

---

## Sumber Dataset

Dataset yang digunakan:

`actaruslab_f1_telemetry_2026.csv`

Karakteristik dataset:

* Jumlah observasi: **150.000 data**
* Jumlah variabel: **6 variabel**
* Tidak terdapat *missing value*
* Seluruh variabel bertipe numerik

Target prediksi:

* `speed_kmh`

---

## Cara Menjalankan Project

### 1. Clone repository

```bash
git clone <repository-url>
cd <nama-folder-project>
```

### 2. Install dependency

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib scipy
```

### 3. Siapkan dataset

Pastikan file dataset:

```text
actaruslab_f1_telemetry_2026.csv
```

berada pada direktori yang sesuai.

### 4. Jalankan notebook

Buka Jupyter Notebook:

```bash
jupyter notebook
```

Lalu buka file:

```text
Project_Machine_Learning_Kelompok_A_PREDIKSI_KECEPATAN_MOBIL_BERDASARKAN_SPESIFIKASI_AERODINAMIKA.ipynb
```

### 5. Jalankan seluruh sel

Gunakan:

```text
Run All
```

Model terbaik akan tersimpan sebagai:

```text
best_regression_model.joblib
```

---

## Ringkasan Hasil

Tahapan eksperimen meliputi:

* Eksplorasi Data (EDA)
* Pembagian data train–test (80:20)
* Pelatihan model
* Hyperparameter tuning
* Evaluasi performa

Hasil menunjukkan bahwa **Random Forest (Tuned)** menjadi model terbaik.

Performa utama:

* MAE ≈ **0,1298**
* RMSE ≈ **0,1813**
* R² ≈ **1,0000**

Model mampu memprediksi kecepatan kendaraan dengan tingkat galat yang sangat rendah berdasarkan spesifikasi aerodinamika yang diberikan.

---

## Output Project

* Model terbaik: `best_regression_model.joblib`
* Prediksi kecepatan kendaraan baru
* Visualisasi distribusi data dan evaluasi model

