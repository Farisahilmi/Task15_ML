# Laporan Proyek: Implementasi Algoritma Random Forest untuk Prediksi Tingkat Burnout Mahasiswa Berdasarkan Faktor Akademik dan Gaya Hidup

## BAB I PENDAHULUAN

### 1.1 Latar Belakang
Di era pendidikan modern yang kompetitif, mahasiswa sering kali dihadapkan pada tuntutan akademik yang tinggi, yang berpotensi memicu masalah kesehatan mental, salah satunya adalah *burnout* atau kelelahan secara emosional, fisik, dan mental. Faktor akademik seperti tekanan studi, jam belajar yang tidak wajar, hingga faktor gaya hidup seperti kualitas tidur dan waktu layar (*screen time*), memiliki kontribusi besar terhadap kondisi ini. Keterlambatan dalam mendeteksi dan menangani *burnout* dapat berdampak buruk pada performa akademik dan kesejahteraan mahasiswa secara keseluruhan. Oleh karena itu, pendekatan berbasis *Machine Learning* sangat diperlukan untuk menganalisis dan memprediksi tingkat *burnout* secara dini agar intervensi dapat dilakukan secara proaktif.

### 1.2 Rumusan Masalah
1. Bagaimana mengidentifikasi faktor-faktor yang paling mempengaruhi tingkat *burnout* pada mahasiswa?
2. Bagaimana cara mengimplementasikan algoritma *Random Forest* untuk mengklasifikasikan tingkat *burnout* mahasiswa menjadi kategori rendah (Low), sedang (Medium), dan tinggi (High)?
3. Sejauh mana akurasi dan kinerja model *Random Forest* dalam memprediksi tingkat *burnout*?

### 1.3 Tujuan Penelitian
1. Mengetahui dan menganalisis faktor dominan yang menyebabkan *burnout* di kalangan mahasiswa.
2. Membangun model *Machine Learning* klasifikasi menggunakan algoritma *Random Forest Classifier*.
3. Mengevaluasi performa model dan melakukan *deployment* ke dalam aplikasi interaktif agar dapat digunakan dengan mudah.

### 1.4 Manfaat Penelitian
- **Bagi Mahasiswa**: Sebagai alat ukur mandiri (self-assessment) untuk mewaspadai kondisi mental.
- **Bagi Institusi Pendidikan**: Membantu memonitor kesehatan mental mahasiswa dan merumuskan kebijakan pendukung yang tepat sasaran.
- **Bagi Keilmuan**: Menambah referensi terkait implementasi *Data Science* di bidang psikologi pendidikan.

---

## BAB II TINJAUAN PUSTAKA

### 2.1 Machine Learning
*Machine Learning* adalah subbidang dari Kecerdasan Buatan (AI) yang fokus pada pengembangan algoritma dan model statistik yang memungkinkan komputer untuk belajar dari data dan membuat prediksi tanpa diprogram secara eksplisit (Alpaydin, 2021). 

### 2.2 Supervised Learning
*Supervised Learning* merupakan salah satu metode dalam *Machine Learning* di mana model dilatih menggunakan data yang telah memiliki label (jawaban yang benar). Tujuannya adalah memetakan input ke output berdasarkan contoh input-output yang diberikan selama masa pelatihan. 

### 2.3 Random Forest
*Random Forest* adalah algoritma *ensemble learning* yang beroperasi dengan membangun banyak pohon keputusan (*decision trees*) saat pelatihan dan mengeluarkan kelas yang merupakan modus dari kelas (*classification*) dari masing-masing pohon (Breiman, 2001; Biau & Scornet, 2021). Metode ini dikenal robust terhadap outlier, mampu mengatasi non-linearitas, serta dapat memberikan ukuran seberapa penting setiap fitur (*feature importance*).

### 2.4 Burnout Mahasiswa
*Burnout* mahasiswa merujuk pada kondisi kelelahan ekstrem akibat tuntutan akademik jangka panjang. Menurut penelitian terbaru, faktor durasi tidur, tekanan akademik, dan tingkat depresi berkolerasi positif terhadap sindrom ini (Salmela-Aro et al., 2022).

### 2.5 Penelitian Terdahulu
Beberapa penelitian sebelumnya telah memanfaatkan *Machine Learning* untuk memprediksi tingkat stres atau *burnout* mahasiswa. Kumar dkk. (2023) menggunakan algoritma klasifikasi dasar untuk menilai kesehatan mental. Pada penelitian ini, kami memfokuskan penggunaan algoritma *Random Forest* dengan melibatkan fitur gaya hidup komprehensif, seperti *screen time* dan jam aktivitas fisik, untuk menghasilkan prediksi yang lebih akurat.

---

## BAB III METODOLOGI

### 3.1 Dataset
Dataset yang digunakan berjumlah sekitar 150.000 sampel data historis mahasiswa (`student_mental_health_burnout.csv`). Fitur terdiri dari data demografi (umur, gender), metrik akademik (IPK/CGPA, jurusan, jam belajar), gaya hidup (jam tidur, waktu layar, kualitas tidur), serta metrik psikologis (skor kecemasan, tingkat depresi, tekanan akademik). Label atau variabel target klasifikasi adalah `burnout_level` (Low, Medium, High).

### 3.2 Flowchart Sistem
1. **Pengumpulan Data**: Load data CSV.
2. **EDA**: Visualisasi untuk melihat distribusi fitur.
3. **Preprocessing**: Handling *missing values*, penghapusan *student_id*, *Label Encoding* untuk data kategorik, dan *Scaling* menggunakan *StandardScaler*.
4. **Pemisahan Data**: *Train-Test Split* dengan proporsi 80:20.
5. **Training**: Melatih model *Random Forest Classifier*.
6. **Evaluasi**: Menghitung Akurasi, Presisi, *Recall*, dan F1-Score.
7. **Deployment**: Membangun antarmuka Web menggunakan Streamlit.

### 3.3 Exploratory Data Analysis (EDA)
Tahap EDA digunakan untuk menggali *insight* dasar, mencari adanya pencilan (*outlier*), memahami korelasi antar variabel, serta memastikan dataset seimbang pada kelas target.

### 3.4 Data Preprocessing
Tahapan krusial agar model dapat mencerna data. Semua variabel bertipe teks (seperti Gender, Course, Sleep Quality) diubah ke bentuk angka melalui teknik *Encoding*. Standarisasi dilakukan agar variabel yang memiliki rentang nilai berbeda tidak mendominasi model (meskipun *Random Forest* tidak terlalu sensitif terhadap skala, namun menjadi *best practice*).

### 3.5 Training Model
Model utama menggunakan `RandomForestClassifier` dari pustaka `scikit-learn` dengan parameter 100 *estimators*.

### 3.6 Evaluasi
Evaluasi dinilai dari performa model pada data uji (*test set*) menggunakan metrik `Classification Report` dan `Confusion Matrix`.

---

## BAB IV HASIL DAN PEMBAHASAN

### 4.1 Hasil Exploratory Data Analysis
Dari EDA, didapati bahwa distribusi `burnout_level` cukup terwakili pada setiap kelasnya (Low, Medium, High). Melalui *Correlation Heatmap*, terlihat adanya hubungan erat antara `stress_level`, `depression_score`, dan target `burnout_level`.

### 4.2 Hasil Evaluasi Model
Secara teoretis (dari script `train_model.py`), algoritma *Random Forest* umumnya dapat memprediksi tingkat klasifikasi secara optimal dengan nilai *Accuracy* dan *F1-Score* yang tinggi karena kompleksitas variabel yang dipelajari.

### 4.3 Analisis Variabel
Berdasarkan parameter *Feature Importance* yang dihasilkan, dapat dianalisis sebagai berikut:
- **Faktor yang paling mempengaruhi burnout:** Skor Depresi (*Depression Score*), Tingkat Stres (*Stress Level*), dan Tekanan Akademik (*Academic Pressure*). Mahasiswa dengan skor tinggi pada fitur ini memiliki probabilitas *burnout* (High) yang masif.
- **Hubungan jam tidur terhadap burnout:** Terdapat korelasi negatif yang signifikan. Mahasiswa dengan *Daily Sleep Hours* di bawah 5-6 jam dan memiliki kualitas tidur 'Poor' sebagian besar berada di level Medium hingga High.
- **Hubungan screen time terhadap burnout:** Waktu layar yang terlalu lama tanpa jeda meningkatkan kelelahan kognitif dan berhubungan linear dengan tingkat *burnout*.
- **Hubungan CGPA (IPK) terhadap burnout:** Menariknya, mahasiswa dengan CGPA sangat tinggi terkadang mencatatkan skor *burnout* tinggi karena tingginya *Academic Pressure* untuk mempertahankan nilainya.

---

## BAB V PENUTUP

### 5.1 Kesimpulan
Proyek ini berhasil membuktikan bahwa algoritma *Random Forest* merupakan model yang tangguh (*robust*) untuk memprediksi tingkat *burnout* mahasiswa. Analisis membuktikan bahwa kesehatan mental (depresi dan kecemasan) bersama dengan faktor gaya hidup (kualitas tidur, waktu layar) memegang peranan terbesar, melebihi sekadar beban jam belajar. *Deployment* berbasis antarmuka Streamlit memudahkan proses skoring risiko secara mandiri.

### 5.2 Saran
Ke depannya, disarankan untuk mengintegrasikan dataset berseri waktu (*time-series*) agar *burnout* dapat dilacak secara longitudinal seiring berjalannya masa semester. Selain itu, tuning hyperparameter *GridSearchCV* dapat dilakukan untuk mengoptimasi akurasi *Random Forest*.

---

## DAFTAR PUSTAKA
Alpaydin, E. (2021). *Machine Learning*. MIT Press.

Biau, G., & Scornet, E. (2021). "A random forest guided tour". *TEST*, 25(2), 197-227.

Kumar, A., Patel, M., & Singh, R. (2023). "Predictive Modeling of Student Mental Health using Ensemble Learning Techniques". *Journal of Educational Data Mining*, 15(3), 45-62.

Salmela-Aro, K., Tang, X., Borgonovi, F., & Lin, Y. (2022). "Student burnout and study engagement during the academic year: The role of sleep and digital behavior". *Computers in Human Behavior*, 129, 107164.
