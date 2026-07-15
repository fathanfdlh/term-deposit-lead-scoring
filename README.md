# JCDSAH-025_BETA_FINPRO
# Improving Conversion Rate through ML-Driven Lead Scoring for Term Deposits

### Ikhtisar Proyek

Proyek ini bertujuan untuk mengoptimalkan kampanye telemarketing bank untuk produk deposito berjangka dengan menerapkan model Machine Learning (ML) untuk penilaian prospek (lead scoring). Secara tradisional, bank mengandalkan telemarketing langsung, yang meskipun mencapai tingkat konversi rata-rata 11,3%, menghabiskan 88,7% waktu agen untuk prospek yang tidak membuka deposito. Dengan memanfaatkan ML, kami berupaya mengidentifikasi dan memprioritaskan prospek 'top-tier' yang paling mungkin untuk melakukan deposito, sehingga meningkatkan efisiensi dan mengurangi biaya operasional.

### Tujuan Bisnis

1.  **Membangun Model Propensity**: Mengembangkan model klasifikasi ML untuk memprediksi probabilitas nasabah membuka deposito berjangka.
2.  **Meningkatkan Efektivitas & Efisiensi Kampanye**: Memfokuskan upaya telemarketing pada prospek dengan probabilitas konversi tertinggi.
3.  **Mengembangkan Sistem Penilaian Prospek**: Membuat sistem penilaian berbasis ML yang menganalisis karakteristik dan riwayat interaksi nasabah.
4.  **Implementasi Prioritisasi Prospek**: Menyediakan daftar prospek yang diprioritaskan untuk tim telemarketing, mengurangi panggilan ke prospek dengan peluang rendah.

### Evaluasi Metrik

Kami memilih **F0.5-score** sebagai metrik evaluasi utama. Metrik ini memberikan bobot lebih besar pada *Precision* daripada *Recall*, yang sangat relevan untuk tujuan bisnis ini. Kami ingin meminimalkan *False Positives* (panggilan sia-sia ke nasabah yang tidak akan berkonversi) karena setiap panggilan telemarketing memerlukan waktu dan biaya operasional. Dengan demikian, model akan memprioritaskan akurasi dalam mengidentifikasi prospek yang benar-benar tertarik.

### Manfaat yang Diharapkan

Implementasi model ini diharapkan dapat menghasilkan:
-   **Penurunan Cost Per Acquisition (CPA)** secara signifikan.
-   **Peningkatan efisiensi operasional** tim telemarketing.
-   **Alokasi sumber daya** yang lebih cerdas dan terarah.
-   **Peningkatan kualitas interaksi** dengan nasabah yang lebih prospektif.

### Deskripsi Data

Dataset berisi 21 kolom dan 41188 entri, mencakup informasi demografi nasabah, pekerjaan, status pernikahan, pendidikan, riwayat kredit dan pinjaman, detail kontak kampanye, serta indikator ekonomi makro. Kolom target `y` (diubah menjadi `deposit`) menunjukkan apakah nasabah melakukan deposito berjangka (`yes`) atau tidak (`no`).

Fitur-fitur utama:
-   `age`: Usia nasabah.
-   `job`: Pekerjaan nasabah.
-   `marital`: Status pernikahan.
-   `education`: Tingkat pendidikan.
-   `default`, `housing`, `loan`: Status kredit, kepemilikan rumah, dan pinjaman pribadi.
-   `contact`, `month`, `day_of_week`: Detail kontak terakhir.
-   `duration`: Durasi kontak terakhir (dihapus untuk mencegah kebocoran data pada model prediksi).
-   `campaign`, `pdays`, `previous`, `poutcome`: Informasi terkait riwayat kampanye.
-   `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`: Indikator makroekonomi.

### Eksplorasi Data (EDA) & Pembersihan Data

-   **Pembersihan Data**: Duplikasi dihapus, format kolom dinormalisasi, dan nilai 'unknown' diidentifikasi.
-   **Fitur Rekayasa**: `age_group`, `pdays_bin`, `campaign_bin`, dan `duration_bin` dibuat untuk analisis yang lebih mendalam.
-   **Ketidakseimbangan Data**: Variabel target `deposit` sangat tidak seimbang (11.3% 'yes' vs 88.7% 'no'), menekankan perlunya teknik penanganan imbalance.
-   **Insight Kunci**: Durasi panggilan dan kontak pdays berkorelasi positif kuat dengan probabilitas deposit. Kondisi makroekonomi (`euribor3m`, `emp_var.rate`, `nr.employed`) berkorelasi negatif. Kelompok usia 'Seniors' dan 'Older Adults', serta pekerjaan 'Student' dan 'Retired', menunjukkan tingkat konversi lebih tinggi. Bulan-bulan tertentu (Maret, September, Oktober, Desember) menunjukkan efisiensi konversi yang lebih baik.

### Metodologi Machine Learning

1.  **Pemilihan Model**: Model *tree-based ensemble* (LightGBM, XGBoost, CatBoost, RandomForest, AdaBoost, DecisionTree) digunakan karena kemampuannya menangani data imbalance dan menghasilkan probabilitas yang baik. LightGBM menunjukkan performa terbaik pada *baseline*.
2.  **Penanganan Imbalance Data**: **TomekLinks Undersampling** dipilih karena memberikan keseimbangan optimal antara *Precision* dan *Recall* untuk F0.5-score, dibandingkan dengan SMOTE atau ENN yang cenderung mengorbankan *Precision*.
3.  **Pra-pemrosesan Data**: Pipeline (`ColumnTransformer`) digunakan untuk:
    -   `OneHotEncoder` untuk fitur nominal.
    -   `BinaryEncoder` untuk fitur kategorikal dengan kardinalitas tinggi.
    -   `SimpleImputer` dan `StandardScaler` untuk fitur numerik.
4.  **Penyesuaian Hiperparameter & Threshold**: `RandomizedSearchCV` digunakan untuk mengoptimalkan LightGBM dengan TomekLinks, menghasilkan peningkatan F0.5-score pada data latih. *Threshold tuning* dilakukan untuk mendapatkan *threshold* optimal sebesar **0.4925**, memaksimalkan F0.5-score dengan meminimalkan *False Positives*.

### Performa Model (Data Uji)

Model LightGBM yang telah dioptimasi dievaluasi pada data uji (unseen data) dengan *threshold* **0.4925**:
-   **Precision Test: 0.6204 (62.04%)**
-   **Recall Test: 0.3082 (30.82%)**
-   **F0.5 Test: 0.5159**

### Analisis Biaya-Manfaat (Cost-Benefit Analysis)

Simulasi menunjukkan dampak signifikan dari implementasi model ML:
-   **Penghematan Anggaran Pemasaran**: **€1,010,750** (**94.4%** lebih hemat).
-   **Pemangkasan Waktu Telemarketing**: **94.4%** waktu agen diselamatkan (hanya 461 dari 8236 nasabah yang perlu dihubungi).
-   **Penurunan Biaya Akuisisi (CAC)**: **81.8%** lebih murah per nasabah.

### Fitur Terpenting

Fitur-fitur dengan pengaruh terbesar terhadap prediksi model adalah:
1.  `num__euribor3m` (Suku bunga Euribor 3 bulan)
2.  `num__age` (Usia nasabah)
3.  `num__campaign` (Jumlah kontak dalam kampanye saat ini)
4.  `num__pdays` (Jumlah hari sejak kontak terakhir)
5.  `num__cons_price_idx` (Indeks harga konsumen)
6.  `num__cons_conf_idx` (Indeks kepercayaan konsumen)
7.  `num__nr_employed` (Jumlah karyawan)
8.  `onehot__contact_telephone` (Kontak Telepon)
9.  `onehot__poutcome_failure` (Hasil kampanye sebelumnya gagal)
10.  `onehot__day_of_week` (Hasil hari dalam 1 minggu)

### Kesimpulan

Model LightGBM, yang dikembangkan dengan teknik *undersampling* TomekLinks dan *threshold tuning*, terbukti **efektif dan efisien** dalam mencapai tujuan bisnis. Model ini tidak hanya mampu mengidentifikasi prospek 'top-tier' dengan akurasi tinggi dan meminimalkan pemborosan sumber daya, tetapi juga memberikan penghematan biaya yang substansial dan optimasi waktu operasional. Ini adalah solusi berbasis ML yang kuat untuk mengoptimalkan kampanye telemarketing.

### Rekomendasi Bisnis

1.  **Implementasi Sistem Lead Scoring**: Gunakan model ini untuk memprioritaskan prospek telemarketing, hanya menghubungi mereka yang memiliki probabilitas tinggi untuk berkonversi.
2.  **Optimalisasi Alokasi Sumber Daya**: Manfaatkan penghematan biaya dan waktu untuk investasi lain atau meningkatkan kualitas interaksi.
3.  **Strategi Kontak Berbasis Data**: Fokus pada nasabah yang baru saja dihubungi (*recency*), gunakan saluran seluler, dan pertimbangkan pola musiman untuk kampanye.
4.  **Penargetan Demografi**: Prioritaskan kelompok usia 'Seniors', 'Older Adults', pekerjaan 'Student' atau 'Retired', serta status 'Single'.
5.  **Pemantauan Makroekonomi**: Sesuaikan pesan dan penawaran kampanye berdasarkan kondisi ekonomi eksternal.

---
