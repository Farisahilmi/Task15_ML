# Student Burnout Prediction

Proyek Machine Learning untuk memprediksi tingkat burnout mahasiswa (Low, Medium, High) berdasarkan faktor akademik, gaya hidup, dan psikologis menggunakan algoritma **Random Forest Classifier**.

## Deskripsi Project
Burnout di kalangan mahasiswa adalah masalah serius yang memengaruhi kesehatan mental dan performa akademik. Proyek ini bertujuan untuk mengidentifikasi mahasiswa yang berisiko mengalami burnout berdasarkan data seperti jam belajar, kualitas tidur, tingkat stres, dan tekanan akademik. 
Sistem ini menggunakan pendekatan *Supervised Learning* (Klasifikasi) dengan *Random Forest* dan dideploy menggunakan antarmuka interaktif berbasis **Streamlit**.

## Dataset
Dataset yang digunakan berisi 150.000 data mahasiswa dengan 20 fitur. Kolom target untuk klasifikasi adalah `burnout_level` yang terdiri dari 3 kelas:
- Low
- Medium
- High

**Catatan**: Dataset CSV harus diletakkan di dalam folder `dataset/` sebelum menjalankan kode.

## Cara Install

1. Pastikan Anda telah menginstal Python versi 3.8 ke atas.
2. Clone atau unduh repositori ini.
3. Buka terminal/command prompt, arahkan ke folder proyek.
4. Instal semua library yang dibutuhkan dengan menjalankan perintah berikut:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan

1. **Jalankan Preprocessing dan Training Model (Opsional, karena `model.pkl` seharusnya sudah di-generate)**
   ```bash
   python train_model.py
   ```
   *Script ini akan memuat dataset, melakukan preprocessing, melatih model Random Forest, dan menyimpan model ke dalam file `model.pkl`.*

2. **Jalankan Aplikasi Web Streamlit**
   ```bash
   streamlit run app.py
   ```
   *Browser Anda akan otomatis terbuka dan menampilkan antarmuka aplikasi prediksi.*

3. **Membuka Notebook EDA**
   ```bash
   jupyter notebook notebook.ipynb
   ```
   *Buka file ini untuk melihat visualisasi dan proses Exploratory Data Analysis secara interaktif.*

## Struktur Folder
```
Student-Burnout-Prediction/
│
├── dataset/
│   └── student_mental_health_burnout.csv
├── notebook.ipynb        # Exploratory Data Analysis & Visualizations
├── preprocessing.py      # Script untuk membersihkan & memproses data
├── train_model.py        # Script untuk melatih model Random Forest
├── app.py                # Kode sumber untuk aplikasi Streamlit
├── model.pkl             # Model machine learning yang sudah dilatih (dihasilkan setelah training)
├── requirements.txt      # Daftar dependensi library Python
├── README.md             # Panduan dan deskripsi proyek
├── images/               # Folder untuk menyimpan screenshot deployment atau plot
└── laporan/              # Berisi laporan lengkap proyek dengan format skripsi (Laporan.md)
```

## Hasil Evaluasi
Model Random Forest memberikan hasil evaluasi yang sangat baik pada klasifikasi tingkat burnout, mengingat kemampuannya dalam menangani hubungan non-linear. Metrik evaluasi mencakup Accuracy, Precision, Recall, dan F1-Score yang dicetak langsung saat mengeksekusi `train_model.py`. 

Dari hasil *Feature Importance*, faktor dominan yang mempengaruhi burnout mahasiswa umumnya berkisar pada **Stress Level**, **Depression Score**, dan **Daily Sleep Hours**.

## Screenshot Deployment
*(Tambahkan screenshot dari aplikasi web Streamlit Anda ke dalam folder `images/` dan hubungkan di sini)*
![Streamlit App](images/screenshot.png)
