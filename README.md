# RP Ganjil — Sistem Prediksi Kualitas Daging Berbasis Electronic Nose

Implementasi Neural Network hasil training ke **Arduino Mega 2560** dengan
**perhitungan matriks manual** (tanpa library ML), lengkap dengan program training,
file bobot/bias siap pakai, dan sketsa untuk Proteus.

```
Dataset 12 potongan daging  ->  Training NN (Python)  ->  Ekstraksi bobot & bias
        (26.640 baris)              klasifikasi + TVC         ->  model_params.h
                                                                            |
                        LCD Proteus  <--  Arduino Mega  <--  .ino (matriks manual)
```

---

## 1. Jawaban singkat pertanyaan yang paling sering muncul

### ❓ "Gabungkan seluruh file dataset (12 jenis potongan daging) menjadi satu kesatuan data — maksudnya digabung jadi satu dataset besar?"

**Ya, digabung menjadi SATU tabel besar**, lalu **satu model NN** dilatih dari tabel itu —
**bukan** 12 model terpisah. Caranya:

```python
frames = [pd.read_excel(xlsx, sheet_name=s).assign(Cut=s) for s in xl.sheet_names]
df = pd.concat(frames, axis=0, ignore_index=True)     # 12 x 2.220 = 26.640 baris
```

Kolom setiap sheet identik (`Minute, TVC, Label, MQ135, MQ136, ..., MQ9`), jadi
penggabungannya *row-wise concat* (menumpuk baris ke bawah, bukan menggabung kolom).
Kolom tambahan `Cut` **hanya** penanda asal potongan untuk analisis — **tidak** dipakai
sebagai input NN (input tetap 11 sensor MQ).

### ❓ "Mending satu model gabungan, atau satu model untuk setiap jenis daging?"

**Untuk tugas ini: satu model gabungan (1 set bobot) untuk 12 potongan.** Alasannya
praktis *dan* terukur — tetapi ada satu catatan penting yang perlu Anda tulis di laporan:

| Skenario uji | Akurasi | Arti |
|---|---|---|
| Data uji acak 20 % (potongan ada di data training) | **98,72 %** | model gabungan sangat akurat untuk 12 potongan ini |
| Akurasi per potongan daging (98,7 % rata-rata, 97,5 – 99,5 %) | **97,5 – 99,5 %** | tidak ada potongan yang jeblok → tidak perlu model terpisah |
| **Leave-one-cut-out**: klasifikasi dilatih tanpa satu potongan, lalu diuji pada potongan itu | **73,4 %** (61 – 83 %) | model **masih memanfaatkan garis dasar (baseline) khas tiap potongan** |

Jadi menumpuk data saja **tidak langsung** membuat model buta terhadap jenis potongan:
tiap potongan punya tingkat pembacaan sensor sendiri (MQ136 berbeda sampai 186 % antar
potongan — lihat `training/output/analisis_baseline_potongan.md`), sehingga model
mengaitkan baseline tersebut dengan tingkat kebusukan.

**Rekomendasi praktis:**

- **Kerjakan seperti tugas (gabung jadi satu dataset, satu model).** Sesuai lembar tugas,
  sederhana, dan akurat 98,7 % untuk potongan yang ada di dataset — inilah pemakaian nyata
  alat e-nose: dikalibrasi untuk potongan yang akan diuji.
- **Kalau nanti harus memprediksi potongan yang benar-benar baru**, sertakan minimal
  beberapa menit pertama pengukuran potongan itu ke data training (`prediksi baseline`),
  atau ubah fiturnya menjadi **perubahan relatif** terhadap pembacaan awal
  (`MQ135(t) − MQ135(menit 1)`) sehingga pengaruh baseline hilang. (Catatan: jika fitur
  diubah, model harus dilatih & diekspor ulang.)
- **Jangan** melatih 12 model terpisah untuk tugas ini: 12× training, 12× set bobot
  (~12 KB float), dan Arduino harus tahu lebih dulu ini potongan apa — pertanyaan yang
  justru ingin dijawab sistem.

---

## 2. Isi repositori

| Path | Keterangan |
|---|---|
| `e-nose_dataset_12_beef_cuts.xlsx` | Dataset asli — 12 sheet, tiap sheet 1 potongan (2.220 baris) |
| `training/train_meat_quality.py` | **Program training utama** (Python/TensorFlow): merge data → normalisasi → cari arsitektur → training → evaluasi → ekspor bobot |
| `notebook/BeefQuality_NN_12Potongan.ipynb` | Versi notebook dari program training di atas (siap dijalankan di Colab/Jupyter) |
| `arduino/MeatQuality_NN_Arduino/model_params.h` | **Hasil ekstraksi bobot & bias** (dibuat otomatis) |
| `arduino/MeatQuality_NN_Arduino/MeatQuality_NN_Arduino.ino` | **Sketsa Arduino**: normalisasi, matriks manual, sigmoid/softmax manual, threshold |
| `verification/verify_cpp_vs_keras.py` | Membandingkan hasil kode Arduino dengan prediksi Keras (tanpa hardware) |
| `docs/skema_proteus.svg` | Skema rangkaian Proteus (11 potensiometer + LCD) |
| `tools/predict_manual.py` | Prediksi cepat dari 11 nilai sensor yang diketik manual |
| `tools/run_all.sh` | Menjalankan training + verifikasi sekali perintah |
| `docs/PANDUAN_PROTEUS.md` | Langkah demi langkah merangkai Proteus + tabel nilai potensiometer |
| `docs/LAPORAN_TEMPLATE.md` | Kerangka laporan proyek siap ditempel |
| `docs/ARSITEKTUR_DAN_RUMUS.md` | Rincian arsitektur + turunan rumus perhitungan manual |
| `BeefQualityClassification_Softmax.ipynb` | Notebook contoh dari dosen (referensi) |

---

## 3. Cara menjalankan (dari root repo)

```bash
# 0) cara cepat: jalankan semuanya
bash tools/run_all.sh

# atau manual:
# 1) siapkan library
pip install pandas numpy scikit-learn tensorflow openpyxl matplotlib

# 2) training + ekspor parameter Arduino  (hasil ke training/output/)
python training/train_meat_quality.py

# 3) opsional: buktikan kode Arduino == Keras (butuh g++, tanpa hardware)
python verification/verify_cpp_vs_keras.py

# 4) opsional: prediksi dari nilai sensor yang Anda ketik sendiri
python tools/predict_manual.py 17,85 15,58 7,95 21,23 16,52 24,07 12,29 9,02 14,26 40,29 14,4
```

Setelah langkah 2 selesai, file `arduino/MeatQuality_NN_Arduino/model_params.h`
berisi bobot & bias model Anda. **Salin folder `arduino/MeatQuality_NN_Arduino/`
ke `Documents/Arduino/`**, buka `.ino`-nya di Arduino IDE, lalu upload ke Arduino Mega.

Mau menjalankan di Google Colab? Pakai `notebook/BeefQuality_NN_12Potongan.ipynb`,
upload file xlsx-nya, lalu *Run all*.

> ⏱️ Sekali training penuh makan waktu ± 20–40 menit di CPU (7 kandidat arsitektur +
> 12 kali training leave-one-cut-out + 2 model final + 2 model deploy).
> Kalau hanya ingin cepat, kecilkan `search_archs`, `search_epochs`, dan
> `final_epochs` di bagian `CFG` (`training/train_meat_quality.py`).

---

## 4. Rangkaian pengolahan data (sesuai lembar tugas)

### Langkah 1 — Pengolahan dataset

| Butir tugas | Implementasi |
|---|---|
| Integrasi data 12 file | `load_and_merge()` → `pd.concat` 12 sheet → 26.640 baris, 1 DataFrame |
| Fitur 11 sensor MQ | `CFG["sensors"]` = MQ135, MQ136, MQ137, MQ138, MQ2, MQ3, MQ4, MQ5, MQ6, MQ8, MQ9 |
| Target | `Label` (4 kelas) untuk klasifikasi **dan** `TVC` (log10 CFU/g) untuk regresi |
| Normalisasi | Min-Max manual: `x_norm = 2(x - min)/(max - min) - 1` → rentang `[-1, 1]` |
| Simpan parameter scaling | `NORM_MIN[]`, `NORM_MAX[]` di `model_params.h` + kolom `*_ADC` di `test_vectors.csv` |

**Penting tentang Label:** pada dataset ini `Label` ternyata **hasil threshold TVC**
(terbukti dari rentang TVC tiap kelas, tidak ada satu pun baris yang menyimpang):

| Label | Kategori | Aturan TVC (log10 CFU/g) | Jumlah baris |
|---|---|---|---|
| 1 | Excellent | `< 3` | 3.540 |
| 2 | Good | `3 – 4` | 4.140 |
| 3 | Acceptable | `4 – 5` | 3.120 |
| 4 | Spoiled | `≥ 5` | 15.840 |

Karena itu repo ini menyediakan **dua cabang output** pada satu Hidden Layer bersama:
klasifikasi softmax (4 kelas) dan regresi TVC (1 nilai kontinu). Di Arduino keduanya
dihitung, lalu LCD menampilkan kategori dari hasil regresi TVC — sekaligus sebagai
pembanding kategori dari softmax.

### Langkah 2 — Pelatihan model

- Arsitektur: `11 input → Dense(n, sigmoid) → Dense(4, softmax)` dan `11 → Dense(n, sigmoid) → Dense(1, linear)`
- `architecture_search()` mencoba `[10], [16], [20], [32], [64], [32,16], [64,32]`
  dan memilih berdasarkan F1-macro validasi (hasil di `training/output/architecture_search.csv`)
- Split 80 % train / 20 % test (stratified), `batch_size` 512, Adam `lr = 1e-2`,
  `EarlyStopping` + `ReduceLROnPlateau`
- Aktivasi hidden **sigmoid** (bukan ReLU) supaya mudah ditulis manual di Arduino dan
  cocok dengan bahan kuliah

### Langkah 3 — Ekstraksi parameter

`export_model_params_header()` menulis `model_params.h` berisi:

```c
const float NORM_MIN[11], NORM_MAX[11];               // parameter scaling
const float CLS_W1[20][11], CLS_B1[20];               // input -> hidden (klasifikasi)
const float CLS_W2[4][20],  CLS_B2[4];                // hidden -> output softmax
const float TVC_W1[20][11], TVC_B1[20];               // input -> hidden (regresi)
const float TVC_W2[20],     TVC_B2[1];                // hidden -> TVC
const float TVC_THRESHOLDS[3] = {3.0, 4.0, 5.0};      // ambang batas kategori
```

Weight matrix sengaja **ditranspose** (`W1[j][i]` = bobot input ke-`i` untuk neuron
ke-`j`) supaya loop bersarang di Arduino langsung terbaca sebagai perkalian matriks.

### Langkah 4 — Implementasi di Arduino

Empat hal yang diminta lembar tugas, semuanya ada di `MeatQuality_NN_Arduino.ino`:

1. **Input scaling** → `adcToSensor()` + `normalizeInput()`
2. **Perhitungan manual** → `forwardHidden()`, `forwardOutputClass()`, `forwardOutputTVC()`
   (loop `sum += W[j][i] * x[i]` lalu `+ bias` — tidak ada library matriks)
3. **Fungsi aktivasi manual** → `sigmoid()`, `relu()`, `softmax()` (dengan trik
   `x - max` supaya `expf()` tidak overflow di Arduino)
4. **Logika output / threshold** → `qualityFromTVC()` dan `qualityFromSoftmax()`,
   hasil ditampilkan di LCD 20×4 + Serial Monitor

### Langkah 5 — Simulasi Proteus

![Skema rangkaian Proteus](docs/skema_proteus.svg)

Ikuti `docs/PANDUAN_PROTEUS.md`. Ringkas: Arduino Mega 2560 + 11 potensiometer
(A0–A10) + LCD 20×4 (D12, D11, D5, D4, D3, D2). Nilai potensiometer yang harus
diset untuk tiap baris data ada di `training/output/test_vectors.csv`
(kolom `MQ135_POT_PCT` = posisi potensiometer dalam %, `MQ135_ADC` = kode ADC).

Ada **dua mode pengujian** di dalam `.ino`:

```c
#define INPUT_MODE 0   // baca 11 potensiometer (presentasi / demo)
#define INPUT_MODE 1   // pakai TEST_VECTOR dari model_params.h (verifikasi presisi)
```

---

## 5. Hasil yang bisa dilaporkan

Hasil training yang sudah dijalankan (arsitektur `11 → 32 → 4` / `11 → 32 → 1`):

| Metrik | Nilai |
|---|---|
| Akurasi klasifikasi (data uji 20 %) | **98,72 %** (balanced accuracy 98,12 %, F1-macro 0,9804) |
| Regresi TVC | MAE **0,1030**, RMSE 0,1439, R² 0,9836 log CFU/g |
| Akurasi pada 8 vektor uji Proteus (Arduino vs Keras) | **8/8 kategori sama** |
| Selisih maksimum estimasi TVC | 0,0123 log CFU/g (dari kuantisasi ADC; pengaruh presisi float hanya 0,000004) |

Setelah training, folder `training/output/` berisi:

- `laporan_hasil_training.md` — tabel metrik lengkap (akurasi, F1, MAE TVC, confusion matrix, akurasi per potongan)
- `verifikasi_cpp_vs_keras.md` — tabel perbandingan kode Arduino vs Keras
- `test_vectors.csv` — 8 baris data uji + posisi potensiometer untuk Proteus
- `architecture_search.csv`, `leave_one_cut_out.csv` — bukti pemilihan arsitektur & generalisasi
- `analisis_baseline_potongan.md` — sidik jari tiap potongan (penjelas hasil leave-one-cut-out)
- `kurva_training.png` — kurva loss/accuracy
- `merged_dataset.csv` — dataset gabungan hasil integrasi 12 sheet
- `beef_quality_classifier.keras`, `tvc_regressor.keras` — model tersimpan

---

## 6. Catatan teknis penting

1. **Urutan sensor harus konsisten.** Urutan `MQ135 … MQ9` dipakai di dataset, di array
   bobot, dan di `SENSOR_PINS[]` (A0…A10). Jangan menukar urutan pin.
2. **Skala nilai sensor.** Di Proteus, potensiometer 0–5 V → `analogRead` 0–1023 →
   nilai sensor 0–60 (`NN_SENSOR_FULL_SCALE`). Rentang ini mewakili seluruh nilai pada
   dataset (min 2,23 – maks 54,08) sehingga satu putaran penuh potensiometer mencakup
   seluruh rentang data. Kalau nanti memakai sensor MQ asli, ganti `adcToSensor()`
   dengan persamaan kalibrasi Rs/R0 sensor tersebut.
3. **Presisi float.** Arduino memakai `float` 32-bit; selisih terhadap Keras ada di
   orde `1e-6` (sudah diukur oleh skrip verifikasi) sehingga tidak mengubah kelas hasil.
4. **Dataset tidak seimbang** (kelas *Spoiled* 59,5 %). Karena itu evaluasi memakai
   **balanced accuracy** dan **F1-macro**, bukan hanya akurasi.
5. **Model yang dipakai di Arduino** dilatih ulang memakai **100 % data**
   (setelah evaluasi 80/20 selesai). Ini praktik normal: evaluasi pakai data uji,
   model deploy pakai seluruh data. Parameter scaling yang diekspor juga berasal
   dari model deploy tersebut.

---

## 7. Pertanyaan / kendala umum

| Gejala | Penyebab & solusi |
|---|---|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| `FileNotFoundError: e-nose_dataset_12_beef_cuts.xlsx` | Jalankan perintah dari **root repo**, atau sesuaikan `CFG["dataset_xlsx"]` |
| Hasil LCD selalu "Spoiled" | Nilai potensiometer terlalu besar. Set posisi sesuai `*_POT_PCT` di `test_vectors.csv` |
| Hasil LCD berbeda dengan Python | Pastikan `NORM_MIN/NORM_MAX` di `model_params.h` berasal dari training yang sama dengan model yang dipakai memprediksi di Python, dan urutan pin benar |
| Model klasifikasi > 1 hidden layer | `model_params.h` hanya mencetak 2 matriks bobot; ubah `export_model_params_header()` bila ingin 2 hidden layer |
| LCD Proteus kosong/karakter aneh | Cek potensiometer kontras V0, dan `RW` harus ke GND |
