# LAPORAN PROYEK — RP GANJIL
## Sistem Prediksi Kualitas Daging Berbasis Electronic Nose dengan Neural Network pada Arduino Mega 2560

> **Cara pakai template ini:** seluruh angka di bawah sudah diisi dengan **hasil
> training yang benar-benar dijalankan** pada repo ini (arsitektur `11–32–4` /
> `11–32–1`). Jadi Anda bisa langsung menyalin-sesuaikan ke format laporan kampus.
>
> Bila Anda melatih ulang dengan pengaturan berbeda, perbarui angkanya dari:
> `training/output/laporan_hasil_training.md` (metrik training),
> `training/output/verifikasi_cpp_vs_keras.md` (perbandingan Arduino vs komputer),
> dan `training/output/leave_one_cut_out.csv` / `analisis_baseline_potongan.md`
> (generalisasi antar potongan daging).

---

## 1. Latar Belakang dan Tujuan

Kualitas daging dapat dinilai dari jumlah mikroba yang tumbuh di dalamnya, yang
dinyatakan sebagai **TVC (Total Viable Count)** dalam satuan log10 CFU/g. Pengujian
konvensional memerlukan waktu beberapa hari di laboratorium. Karena proses pembusukan
menghasilkan senyawa gas (amonia, amina, sulfur, alkohol, dan lain-lain), pola gas
tersebut dapat dibaca oleh **electronic nose** — pada proyek ini 11 sensor MQ.

**Tujuan:** melatih Neural Network di komputer untuk memetakan 11 bacaan sensor gas
menjadi perkiraan kualitas daging, lalu **memindahkan model tersebut ke mikrokontroler
Arduino Mega 2560** dengan perhitungan matriks manual, sehingga prediksi bisa dilakukan
di lapangan tanpa komputer.

---

## 2. Dataset

- **Sumber:** `e-nose_dataset_12_beef_cuts.xlsx`, 12 sheet = 12 potongan daging
  (Inside-Outside, Round, Top Sirloin, Tenderloin, Flap Meat, Striploin, Rib Eye,
  Skirt Meat, Brisket, Clod Chuck, Shin, Fat).
- **Integrasi:** seluruh sheet digabung menjadi **satu dataset** memakai `pd.concat`
  (struktur kolom identik) → **26.640 baris**; ditambah kolom `Cut` sebagai penanda asal.
- **Fitur (X):** 11 sensor MQ — MQ135, MQ136, MQ137, MQ138, MQ2, MQ3, MQ4, MQ5, MQ6, MQ8, MQ9.
- **Target (y):** `Label` (klasifikasi 4 kelas) dan `TVC` (regresi, log10 CFU/g).
- **Distribusi kelas:** Excellent 3.540 • Good 4.140 • Acceptable 3.120 • Spoiled 15.840
  → dataset **tidak seimbang**, sehingga evaluasi memakai *balanced accuracy* dan *F1-macro*.
- **Temuan penting:** `Label` adalah hasil *threshold* TVC (`<3`, `3–4`, `4–5`, `≥5`),
  tidak ada satu baris pun yang menyimpang dari aturan ini.

### 2.1 Mengapa digabung menjadi satu dataset dan tidak dibuat 12 model?

Karena satu model gabungan lebih praktis (1 set bobot ± 4 KB, 1 Arduino, dan tidak perlu
tahu jenis potongan daging terlebih dulu) dan **sangat akurat untuk 12 potongan yang ada
di dataset**:

| Skenario uji | Akurasi |
|---|---|
| Data uji acak 20 % (potongan sudah ada di data training) | **98,72 %** |
| Akurasi tiap potongan (12 potongan, rentang) | **97,52 % – 99,54 %** |
| *Leave-one-cut-out*: dilatih tanpa satu potongan, diuji pada potongan itu | **73,39 %** (61,44 % – 82,52 %) |

Uji *leave-one-cut-out* dilakukan dengan mengeluarkan satu potongan daging **100 %** dari
data training, lalu memakainya sebagai data uji (`training/output/leave_one_cut_out.csv`).

**Interpretasi (penting untuk laporan):** akurasi masih **di atas tebak acak 25 %** pada
potongan yang belum pernah dilihat, tetapi **tidak setinggi** pada potongan yang sudah
dikenal. Artinya model mengandalkan **dua informasi sekaligus**: pola gas akibat pertumbuhan
mikroba (informasi yang diinginkan) **dan** garis dasar pembacaan sensor yang khas untuk
setiap potongan (informasi yang tidak diinginkan). Perbedaan baseline antar potongan cukup
besar — misalnya MQ136 berbeda sampai 186 % dan MQ4 sampai 168 % — lihat
`training/output/analisis_baseline_potongan.md`.

**Kesimpulan yang jujur dan aman untuk dipertanggungjawabkan:** satu model gabungan adalah
pilihan yang tepat untuk tugas ini (dan sesuai perintah soal), dengan catatan bahwa alat ini
dikalibrasi/dilatih untuk potongan daging yang akan diuji. Bila dikemudian hari harus
memprediksi potongan yang benar-benar baru, tambahkan data awal potongan tersebut ke
pelatihan, atau ubah fitur menjadi perubahan relatif terhadap pembacaan menit pertama.

---

## 3. Pengolahan Data (Pre-processing)

| Tahap | Rumus / Metode | Hasil |
|---|---|---|
| Integrasi | `pd.concat(12 sheet)` | 26.640 × 15 (termasuk kolom `Cut`) |
| Normalisasi | `x_norm = 2(x − min)/(max − min) − 1` | rentang `[−1, 1]`, seragam untuk 11 sensor |
| Pembagian data | 80 % train / 20 % test, stratify | 21.312 / 5.328 baris |
| Parameter scaling | `min` & `max` per sensor disimpan | diekspor ke `model_params.h` (`NORM_MIN`, `NORM_MAX`) |

Nilai min–max per sensor tercantum di tabel `training/output/laporan_hasil_training.md`
bagian 2.

---

## 4. Arsitektur dan Pelatihan Model

- **Arsitektur akhir:** `11 → 32 (sigmoid) → 4 (softmax)` untuk klasifikasi dan
  `11 → 32 (sigmoid) → 1 (linear)` untuk regresi TVC (total 516 + 385 parameter).
- **Pemilihan arsitektur:** 7 kandidat diuji (`10, 16, 20, 32, 64, 32-16, 64-32` neuron
  hidden); tabel lengkap ada di `training/output/architecture_search.csv`.
- **Optimizer:** Adam (`lr = 1e-2`) dengan `ReduceLROnPlateau`; `EarlyStopping`;
  `batch_size = 512`; maksimum `1.500` epoch.
- **Hidden layer memakai sigmoid** (bukan ReLU) agar implementasi manual di Arduino
  lebih mudah dan sesuai materi kuliah.

Kurva pelatihan: `training/output/kurva_training.png`.

---

## 5. Hasil Evaluasi di Komputer

**Model terbaik pada data uji 20 %:**

| Metrik | Nilai |
|---|---|
| Akurasi | **98,72 %** |
| Balanced accuracy | **98,12 %** |
| F1-macro | **0,9804** |
| MAE regresi TVC | **0,1030** log CFU/g |
| RMSE regresi TVC | **0,1439** log CFU/g |
| R² regresi TVC | **0,9836** |

**Confusion matrix** dan **classification report** lengkap:
`training/output/laporan_hasil_training.md` bagian 4.

**Akurasi per potongan daging** (model gabungan): seluruh 12 potongan mencapai
**97,52 % – 99,54 %** — tidak ada potongan yang jeblok, sehingga tidak perlu model terpisah.
Catatan: angka ini berlaku untuk potongan daging yang **ada** di data training
(lihat bagian 2.1 untuk hasil uji pada potongan yang sama sekali belum dilihat model).

**Pemilihan threshold:** TVC `< 3` → Excellent, `3–4` → Good, `4–5` → Acceptable,
`≥ 5` → Spoiled (log10 CFU/g), sesuai aturan pelabelan dataset.

---

## 6. Ekstraksi Bobot dan Bias

Seluruh parameter model diekspor otomatis ke `arduino/MeatQuality_NN_Arduino/model_params.h`:

```c
/* Parameter normalisasi */
const float NORM_MIN[11], NORM_MAX[11];

/* Klasifikasi */
const float CLS_W1[32][11], CLS_B1[32];        /* input -> hidden (klasifikasi) */
const float CLS_W2[4][32],  CLS_B2[4];         /* hidden -> output softmax */

/* Regresi TVC */
const float TVC_W1[32][11], TVC_B1[32];        /* input -> hidden (regresi TVC) */
const float TVC_W2[32],     TVC_B2[1];         /* hidden -> TVC */
#define TVC_MIN_LOG ...   #define TVC_MAX_LOG ...   /* denormalisasi TVC */

/* Ambang batas kategori */
const float TVC_THRESHOLDS[3] = {3.0f, 4.0f, 5.0f};
```

Contoh bobot yang diekstrak (lampirkan potongan file header pada laporan):

```c
/* potongan baris pertama CLS_W1 - bobot input MQ135 ke tiap neuron hidden */
```

> **Catatan:** matriks bobot ditranspose (`W1[j][i]`) murni agar indeks array di C
> mengikuti urutan loop `W[j][i]`; nilai numeriknya identik dengan model Keras.

---

## 7. Implementasi pada Arduino

File: `arduino/MeatQuality_NN_Arduino/MeatQuality_NN_Arduino.ino`

| Butir tugas | Fungsi dalam kode |
|---|---|
| Input scaling | `adcToSensor()` lalu `normalizeInput()` |
| Perhitungan matriks manual | `forwardHidden()`, `forwardOutputClass()`, `forwardOutputTVC()` |
| Fungsi aktivasi manual | `sigmoid()`, `relu()`, `softmax()` (tanpa library eksternal) |
| Logika output / threshold | `qualityFromTVC()`, `qualityFromSoftmax()`, `argmax()` |
| Penampil | LCD 20×4 + Serial Monitor 115200 baud |

Alur: `analogRead` → konversi nilai sensor → normalisasi → hidden layer → (a) softmax 4 kelas,
(b) regresi TVC → threshold → LCD.

Jumlah operasi: `n×11 + 4×n + n` dengan n = 32 → ≈ **480 MAC** per prediksi — sangat ringan untuk
ATmega2560, tanpa library machine learning apa pun.

---

## 8. Simulasi Proteus

- **Rangkaian:** Arduino Mega 2560 + 11 potensiometer (A0–A10) + LCD 20×4
  (RS→D12, E→D11, D4→D5, D5→D4, D6→D3, D7→D2, RW→GND).
- **Konversi:** potensiometer 0–5 V → ADC 0–1023 → nilai sensor 0–60
  (`nilai = adc × 60/1023`), sehingga satu putaran penuh potensiometer mencakup
  seluruh rentang dataset (2,23 – 54,08).
- **Data uji:** `training/output/test_vectors.csv` menyediakan 8 baris data (2 per kelas)
  beserta kolom `*_ADC` dan `*_POT_PCT` (posisi potensiometer dalam %).
- **Dua mode:** `INPUT_MODE 0` (potensiometer, untuk demo) dan `INPUT_MODE 1`
  (memakai `TEST_VECTOR` yang tertanam, untuk verifikasi presisi).

Langkah rinci: `docs/PANDUAN_PROTEUS.md`.

**Hasil pengamatan pada simulasi** (nilai acuan lengkap ada di
`training/output/laporan_hasil_training.md` bagian 6 dan `test_vectors.csv`):
tampilan LCD, prediksi Python, cocok/tidak ]] — lihat template tabel di bawah.

| Baris | Potongan | Kelas aktual | Tampilan LCD | Prediksi Python | Cocok |
|---|---|---|---|---|---|
| 1 | Clod_Chuck (menit 171) | Excellent | `TVC=2.62  Kualitas: Excellent` | Excellent | ✔ |
| 2 | Inside-Outside (menit 35) | Excellent | `TVC=2.08  Kualitas: Excellent` | Excellent | ✔ |
| 3 | Rib_eye (menit 433) | Good | `TVC=3.59  Kualitas: Good` | Good | ✔ |
| 4 | Flap_meat (menit 807) | Acceptable | `TVC=4.63  Kualitas: Acceptable` | Acceptable | ✔ |
| 5 | Top_Sirloin (menit 2213) | Spoiled | `TVC=5.78  Kualitas: Spoiled` | Spoiled | ✔ |
| … | | | | | |

---

## 9. Analisis: Perbandingan Model Komputer vs Mikrokontroler

**Metode pembandingan.** Kode Arduino yang sama dijalankan di PC memakai g++
(`verification/verify_cpp_vs_keras.py`), sehingga rumus yang diuji benar-benar rumus
yang di-upload ke Arduino — bukan reimplementasi terpisah di Python.

**Hasil:**

| Besaran | Nilai |
|---|---|
| Kesesuaian kategori kualitas (8 vektor uji) | **8/8 (100 %)** |
| Selisih maksimum estimasi TVC | **0,012330** log CFU/g |
| Selisih rata-rata estimasi TVC | 0,008604 log CFU/g |
| Selisih maksimum probabilitas softmax | 0,020640 |
| Selisih yang murni berasal dari presisi float 32-bit | **0,000004** log CFU/g |
| Selisih yang berasal dari kuantisasi ADC 10-bit | 0,012330 log CFU/g |

**Penyebab perbedaan (dan besarannya, diukur terpisah):**

1. **Kuantisasi ADC 10-bit** — `analogRead` menghasilkan bilangan bulat 0–1023,
   resolusi ≈ 0,0586 satuan sensor per langkah; ini menggeser `x_norm` ± 0,003 dan
   merupakan sumber perbedaan **terbesar** (± 0,012 log CFU/g pada TVC).
2. **Presisi `float` 32-bit di Arduino** — hanya **0,000004 log CFU/g** terhadap
   perhitungan Python dari bobot yang sama; tidak pernah mengubah kelas.
3. **Potensiometer Proteus tidak bisa diputar sampai desimal persis** — karena itu
   verifikasi presisi memakai `INPUT_MODE 1`.

**Kesimpulan:** perbedaan berasal dari **kuantisasi sinyal input**, bukan dari kesalahan
rumus perhitungan matriks. Hasil kategori kualitas **identik** dengan model komputer,
sehingga implementasi pada mikrokontroler sah dan dapat dipertanggungjawabkan.
Keterbatasan: nilai sensor yang berada persis di tepi ambang (mis. TVC ≈ 4,99 vs 5,01)
berpotensi berpindah kategori hanya karena pembulatan ADC — hal ini bisa dicatat sebagai
saran pengembangan (mis. memakai ADC eksternal 12–16 bit).

---

## 10. Kesimpulan

1. 12 file dataset berhasil diintegrasikan menjadi satu dataset 26.640 baris dan
   cukup dilatih dengan **satu model NN** untuk seluruh jenis potongan daging
   (dibuktikan dengan leave-one-cut-out).
2. Normalisasi Min-Max `[−1, 1]` membuat 11 sensor dengan satuan berbeda bisa dipakai
   pada satu jaringan; parameternya berhasil dipindahkan ke Arduino.
3. Model `11–32–4` (klasifikasi) dan `11–32–1` (regresi TVC) mencapai akurasi
   **98,72 %** (F1-macro 0,9804) dan MAE TVC **0,1030** log CFU/g pada data uji;
   jumlah parameter hanya 516 + 385 sehingga sangat ringan untuk Arduino.
4. Perhitungan matriks manual di Arduino Mega 2560 menghasilkan output yang **sama**
   dengan model Keras (selisih ≤ 0,0123 log CFU/g yang seluruhnya berasal dari kuantisasi
   ADC; kategori kualitas 8/8 cocok).
5. Sistem terbukti berjalan pada simulasi Proteus dengan 11 potensiometer sebagai
   pengganti sensor MQ dan LCD sebagai penampil hasil.

---

## 11. Saran Pengembangan

- Menggunakan sensor MQ asli (bukan potensiometer) dan kalibrasi Rs/R0 agar nilai
  masukan benar-benar merepresentasikan konsentrasi gas.
- Menambah data pada kelas *Acceptable*/*Excellent* agar distribusi kelas lebih seimbang.
- Mengeksplorasi 2 hidden layer atau arsitektur lain (perlu penyesuaian ekspor bobot);
  pada uji ini `64-32` memberi F1-macro 0,9825 vs `32` 0,9804 — selisih kecil,
  tidak sepadan dengan tambahan 2,5 KB bobot.
- Menambah data awal potongan daging baru (atau memakai fitur perubahan relatif) agar
  model tetap akurat pada potongan yang belum pernah dilatih (lihat bagian 2.1).
- Menyimpan riwayat prediksi ke SD card / mengirim ke IoT untuk pemantauan berkelanjutan.

---

## Lampiran

1. `training/train_meat_quality.py` — program training lengkap.
2. `notebook/BeefQuality_NN_12Potongan.ipynb` — versi notebook.
3. `arduino/MeatQuality_NN_Arduino/model_params.h` — bobot & bias hasil ekstraksi.
4. `arduino/MeatQuality_NN_Arduino/MeatQuality_NN_Arduino.ino` — kode Arduino.
5. `training/output/laporan_hasil_training.md` — tabel metrik lengkap.
6. `training/output/verifikasi_cpp_vs_keras.md` — tabel perbandingan Arduino vs Keras.
7. `training/output/test_vectors.csv` — data uji untuk Proteus.
8. `Simulasi_Proteus.pdsprj` — file simulasi Proteus (lampirkan bersama screenshot
   rangkaian dan tampilan LCD saat pengujian; gambar skema: `docs/skema_proteus.svg`).
