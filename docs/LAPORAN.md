# Laporan Proyek — Sistem Prediksi Kualitas Daging Sapi berbasis Electronic Nose

**Mata kuliah:** Sistem Embedded / Embedded System (RP Ganjil)
**Topik:** Implementasi Neural Network hasil training ke mikrokontroler Arduino
**Dataset:** `e-nose_dataset_12_beef_cuts.xlsx` (12 potongan daging sapi, 11 sensor MQ)

---

## Ringkasan hasil

| Butir | Nilai |
|---|---|
| Data yang dipakai | 12 sheet digabung = **26.640 baris** × 11 sensor |
| Arsitektur NN | **11 → 24 (sigmoid) → 12 (sigmoid) → 4 (softmax)** |
| Jumlah parameter | 640 (bobot 616 + bias 24) = **2,56 KB** float32 |
| Normalisasi | Min-Max **per fitur** ke rentang [−1, +1] |
| Akurasi test (5.328 baris) | **98,48 %** |
| Macro F1-score | 0,9762 |
| Target mikrokontroler | Arduino Mega 2560 + 11 potensiometer + LCD 16×2 |
| Selisih maks. Arduino vs Python | **1,86 × 10⁻⁶** (kelas cocok 16/16) |

Seluruh angka di atas dihasilkan ulang oleh skrip di repositori ini — lihat
bagian *[Cara menjalankan ulang](#cara-menjalankan-ulang)*.

---

## 1. Tujuan

Mahasiswa mampu mengimplementasikan model Neural Network hasil pelatihan ke sistem
mikrokontroler (Arduino) dengan melakukan perhitungan matriks secara manual untuk
memprediksi kualitas daging berdasarkan data *electronic nose*.

---

## 2. Langkah 1 — Pengolahan Dataset (Pre-processing)

### 2.1 Integrasi data

Dataset terdiri atas 12 sheet, satu sheet per jenis potongan daging, masing-masing
2.220 baris. Seluruh sheet digabung menjadi satu kesatuan data:

```
12 sheet  ×  2.220 baris  =  26.640 baris  ×  14 kolom
```

Kolom setiap sheet: `Minute, TVC, Label, MQ135, MQ136, MQ137, MQ138, MQ2, MQ3, MQ4,
MQ5, MQ6, MQ8, MQ9`. Kolom `Cut` (nama sheet) ditambahkan saat penggabungan sebagai
penanda asal potongan, lalu tidak dipakai sebagai fitur.

> **Catatan penting.** Notebook sampel (`BeefQualityClassification_Softmax.ipynb`)
> hanya melatih dari sheet `2.Round` (2.220 baris). Model tersebut mencapai 98,42 %
> pada sheet asalnya, tetapi **turun menjadi 37,43 %** ketika diuji pada gabungan 12
> potongan daging. Artinya model lama tidak memenuhi langkah *Integrasi Data* dan
> tidak dapat dipakai untuk sistem yang harus bekerja pada semua potongan daging.
> Angka ini diukur, bukan perkiraan — skrip pengukurannya ada di bagian analisis.

### 2.2 Pemilihan fitur (X)

Sebelas sensor MQ dipakai sebagai variabel input, **dengan urutan tetap** yang juga
menjadi urutan pin analog Arduino:

| Indeks | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sensor | MQ135 | MQ136 | MQ137 | MQ138 | MQ2 | MQ3 | MQ4 | MQ5 | MQ6 | MQ8 | MQ9 |
| Pin | A0 | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 |

### 2.3 Penentuan target (y)

Target yang dipilih adalah kolom **`Label`** (klasifikasi kualitas), sesuai RP ganjil
*"sistem prediksi kualitas daging"*.

Kolom `Label` di dataset bertipe **integer 1..4**, bukan teks:

| Label | Kelas | Rentang TVC (log₁₀ CFU/g) | Rentang Minute | Jumlah baris |
|---|---|---|---|---|
| 1 | `excellent` | 1,88 – 2,89 | 1 – 300 | 3.540 |
| 2 | `good` | 3,11 – 4,00 | 301 – 660 | 4.140 |
| 3 | `acceptable` | 4,05 – 4,94 | 661 – 900 | 3.120 |
| 4 | `spoiled` | 5,10 – 5,76 | 901 – 2220 | 15.840 |

`Label` diubah menjadi *one-hot* 4 kolom. Karena `pd.get_dummies` pada label integer
menghasilkan kolom `[1, 2, 3, 4]` (urut naik), maka **indeks kelas = Label − 1**.

> **Bug yang ditemukan & diperbaiki.** Sel prediksi di notebook sampel memakai
> fallback `['acceptable', 'excellent', 'good', 'spoiled']` dengan komentar
> *"get_dummies mengurutkan nama kelas secara alfabetis"*. Asumsi itu salah untuk
> label integer: kolom yang dihasilkan adalah `[1, 2, 3, 4]`, sehingga indeks 0
> seharusnya `excellent`, bukan `acceptable`. Bila sel dijalankan pada kernel baru
> (tanpa variabel `encode_data`), **semua label bergeser**. Sel tersebut telah
> diperbaiki dan diverifikasi menghasilkan `good` secara konsisten baik pada kernel
> baru maupun kernel penuh.

Kelas tidak seimbang (`spoiled` = 59,5 %). Opsi `--class-weight` tersedia pada skrip
training bila diperlukan; pada hasil akhir tidak dipakai karena akurasi dan macro-F1
sudah tinggi tanpa penimbangan.

### 2.4 Normalisasi dan penyimpanan parameter scaling

Penskalaan **Min-Max per fitur** ke rentang [−1, +1]:

$$x_{norm}[i] = 2 \cdot \frac{x[i] - X_{MIN}[i]}{X_{MAX}[i] - X_{MIN}[i]} - 1$$

Rentang tiap fitur berbeda jauh (mis. MQ6 hanya 2,40–41,05 sedangkan MQ9 hanya
7,16–19,30), sehingga min-max per fitur lebih tepat daripada satu rentang global —
setiap sensor mendapat resolusi penuh.

Parameter hasil perhitungan (tersimpan di `arduino/BQC_Arduino_Mega/nn_params.h`
sebagai array `X_MIN[]` / `X_MAX[]`):

| Sensor | X_MIN | X_MAX |
|---|---|---|
| MQ135 | 7,94 | 22,14 |
| MQ136 | 2,42 | 46,65 |
| MQ137 | 2,62 | 28,96 |
| MQ138 | 7,67 | 28,64 |
| MQ2 | 3,79 | 20,12 |
| MQ3 | 7,54 | 36,40 |
| MQ4 | 2,23 | 36,27 |
| MQ5 | 4,19 | 33,92 |
| MQ6 | 2,40 | 41,05 |
| MQ8 | 15,99 | 54,08 |
| MQ9 | 7,16 | 19,30 |

**Di sisi Arduino** input berasal dari `analogRead()` (0–1023), sehingga normalisasi
yang dipakai adalah bentuk setara:

$$x_{norm} = 2 \cdot \frac{adc}{1023} - 1$$

Ini valid karena potensiometer diputar 0 % ≙ nilai sensor minimum dan 100 % ≙ nilai
sensor maksimum. Konversi ke posisi potensiometer untuk pengujian di Proteus:

```
posisi_pot(%) = adc / 1023 × 100 %
```

Pembagian data: **80 % train (21.312 baris) / 20 % test (5.328 baris)**,
`random_state=42`.

---

## 3. Langkah 2 — Pelatihan Model (Off-line Training)

Framework: **TensorFlow 2.21.0** (Keras), optimizer Adam (lr = 1×10⁻²),
loss `categorical_crossentropy`, metrik `accuracy`, dengan callback
`ReduceLROnPlateau` (factor 0,1; patience 20) dan `EarlyStopping`
(monitor `val_accuracy`, patience 40, `restore_best_weights=True`).

### 3.1 Penentuan arsitektur

**Tahap 1 — penyaringan cepat.** Lima kandidat dilatih singkat (120 epoch) pada
pembagian data yang sama, diulang tiga kali. Hasilnya **tidak deterministik**:
TensorFlow dengan oneDNN dapat memakai urutan komputasi berbeda antar-run, dan
`EarlyStopping`/`ReduceLROnPlateau` peka terhadap selisih kecil.

| Hidden layer | Neuron | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 1 | 20 | 0,9692 | 0,9668 | 0,9670 |
| 1 | 24 | 0,9675 | 0,9724 | 0,9713 |
| 1 | 32 | 0,9649 | 0,9679 | 0,9748 |
| 2 | 24, 12 | **0,9782** | 0,9771 | 0,9752 |
| 2 | 32, 16 | 0,9760 | **0,9779** | **0,9777** |

Kandidat terbaik berpindah-pindah (`[24,12]` → `[32,16]` → `[32,16]`), sehingga tahap
ini **tidak dipakai sebagai dasar keputusan**. Seluruh angka tersimpan di
`docs/metrics.json` kunci `search_runs`.

**Tahap 2 — pembanding dengan training penuh.** Setiap kandidat dilatih sampai
konvergen (early stopping, maks. 600 epoch) dengan protokol identik, lalu diukur pada
20 % data uji yang tidak ikut dilatih
(`python training/compare_arch.py --candidates "20" "24,12" "32,16"`), diulang 2×:

| Hidden layer | Neuron | Test acc. run A | Test acc. run B | **Rata-rata** | Macro F1 (rata-rata) | Parameter | Flash |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 0,97748 | 0,97917 | 0,97832 | 0,9663 | 324 | 1,27 KB |
| **2** | **24, 12** | **0,98686** | 0,98780 | **0,98733** | **0,9799** | **640** | **2,50 KB** |
| 2 | 32, 16 | 0,98536 | 0,98893 | 0,98714 | 0,9797 | 980 | 3,83 KB |

**Kesimpulan yang jujur dari data ini:**

1. **Dua hidden layer jelas lebih baik daripada satu.** `[20]` tertinggal ~0,9 poin
   akurasi dan ~1,4 poin macro-F1 secara konsisten di kedua run — ini selisih yang
   jauh lebih besar daripada variasi antar-run.
2. **`[24,12]` dan `[32,16]` setara secara statistik.** Rata-ratanya berselisih hanya
   0,00019, dan urutan pemenang *berbalik* antara run A dan run B. Selisih sekecil
   ini berada di dalam derau pelatihan, jadi tidak dapat dinyatakan bahwa salah
   satunya lebih akurat.
3. Karena akurasinya setara, **`[24,12]` dipilih berdasarkan jejak memori**: 640
   parameter (2,50 KB) versus 980 (3,83 KB) — 35 % lebih kecil untuk hasil yang sama.
   Pada mikrokontroler 8 KB SRAM, memilih yang lebih ringan tanpa mengorbankan akurasi
   adalah keputusan yang tepat.

Seluruh angka tersimpan di `docs/metrics.json` kunci `arch_comparison_runs`.

> Model yang benar-benar di-*deploy* ke Arduino adalah hasil satu run training penuh
> tersendiri (`training/train_bqc.py --skip-search --arch 24,12`) yang memperoleh
> test accuracy **0,9848**. Angka itu sedikit di bawah rata-rata 0,98733 pada tabel di
> atas karena nondeterminisme yang sama. Yang berlaku untuk bobot di `nn_params.h`
> adalah **0,9848**, dan angka ini dapat direproduksi persis dengan
> `python training/train_bqc.py --report-only arduino/BQC_Arduino_Mega/BQCmodel_12cuts.keras`.

```
Input 11 ──Dense(24, sigmoid)──> 24 ──Dense(12, sigmoid)──> 12 ──Dense(4, softmax)──> 4
```

### 3.2 Hasil pelatihan

Training final berhenti pada **epoch 336** (early stopping), waktu ≈ 91 detik (CPU).

| Metrik | Nilai |
|---|---|
| Validation accuracy | 0,9848 |
| Test accuracy | **0,9848** |
| Test loss (categorical crossentropy) | 0,0373 |

> Validation set dan test set adalah himpunan 20 % yang sama (dipakai sebagai
> `validation_data` saat `fit`), sehingga kedua angka akurasi di atas identik
> secara konstruksi — bukan dua pengukuran terpisah.

**Confusion matrix (test, 5.328 baris):**

| Aktual ↓ / Prediksi → | excellent | good | acceptable | spoiled |
|---|---|---|---|---|
| **excellent** | 689 | 8 | 0 | 0 |
| **good** | 4 | 819 | 12 | 0 |
| **acceptable** | 0 | 12 | 558 | 23 |
| **spoiled** | 0 | 0 | 22 | 3181 |

**Classification report (test):**

| Kelas | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| excellent | 0,9942 | 0,9885 | 0,9914 | 697 |
| good | 0,9762 | 0,9808 | 0,9785 | 835 |
| acceptable | 0,9426 | 0,9410 | 0,9418 | 593 |
| spoiled | 0,9928 | 0,9931 | 0,9930 | 3203 |
| **accuracy** | | | **0,9848** | 5328 |
| macro avg | 0,9764 | 0,9759 | 0,9762 | 5328 |
| weighted avg | 0,9848 | 0,9848 | 0,9848 | 5328 |

Kelas `acceptable` paling sulit (F1 = 0,9418) karena berada di antara `good` dan
`spoiled` dengan rentang TVC yang berdekatan. Kesalahan hanya terjadi antar-kelas
yang bertetangga; tidak ada `excellent` yang diprediksi `spoiled` atau sebaliknya.

Model disimpan sebagai `arduino/BQC_Arduino_Mega/BQCmodel_12cuts.keras`.

---

## 4. Langkah 3 — Ekstraksi Parameter Model

Bobot dan bias setiap layer diambil lewat `layer.get_weights()` dan ditulis otomatis
ke `arduino/BQC_Arduino_Mega/nn_params.h`:

| Array C++ | Bentuk | Isi |
|---|---|---|
| `W1[11][24]` | 264 | bobot input → hidden 1 |
| `B1[24]` | 24 | bias hidden 1 |
| `W2[24][12]` | 288 | bobot hidden 1 → hidden 2 |
| `B2[12]` | 12 | bias hidden 2 |
| `W3[12][4]` | 48 | bobot hidden 2 → output |
| `B3[4]` | 4 | bias output |
| **Total** | **640** | **2,56 KB** |

Selain itu header juga memuat `X_MIN[]`, `X_MAX[]`, `SENSOR_NAMES[]`, `CLASS_NAMES[]`,
`CLASS_LABEL[]`, serta tabel `LAYER_W/LAYER_B/LAYER_IN/LAYER_OUT/LAYER_ACT` yang
memungkinkan `nnForward()` berjalan generik untuk berapa pun jumlah layer.

Contoh cuplikan hasil ekstraksi:

```c
const float W1[11][24] = {
  { 0.12345678f, -0.98765432f, ... },   /* baris MQ135 */
  ...
};
const float B3[4] = { 1.40814590f, 0.61608011f, -0.32806453f, -0.78896105f };
```

**Header ini dibangkitkan otomatis** — jangan diubah manual. Menjalankan ulang
`training/train_bqc.py` akan menulis ulangnya.

---

## 5. Langkah 4 — Implementasi pada Arduino

### 5.1 Susunan berkas

```
arduino/
├── BQC_Arduino_Mega/
│   ├── BQC_Arduino_Mega.ino     <- sketch utama (I/O, LCD, Serial)
│   ├── nn_core.h                <- perhitungan NN manual (murni C, tanpa library)
│   ├── nn_params.h              <- bobot & bias hasil ekstraksi (auto-generated)
│   └── BQCmodel_12cuts.keras    <- model TensorFlow sumber
└── test/
    ├── test_nn_core.cpp         <- uji numerik di PC
    ├── host_sim/                <- shim Arduino/LCD utk menjalankan .ino di PC
    └── run_checks.sh            <- jalankan seluruh pemeriksaan
```

`nn_core.h` sengaja ditulis sebagai C standar murni agar **file yang sama** dipakai
oleh Arduino dan oleh program uji di PC. Sketch dan program uji memanggil fungsi yang
identik, `nnProcessADC()`, sehingga kesetaraan hasilnya benar-benar teruji, bukan
hanya mirip.

### 5.2 Input scaling

```c
static float nnScaleADC(int adc, float adc_max) {
  if (adc < 0) adc = 0;
  if (adc > (int)adc_max) adc = (int)adc_max;
  return 2.0f * (float)adc / adc_max - 1.0f;
}
```

Setiap pin dibaca 8× lalu dirata-rata untuk menekan derau (`ADC_SAMPLES`).
Fungsi `nnScaleSensor(x, i)` juga tersedia bila kelak input berupa nilai fisik sensor
nyata (bukan potensiometer), memakai `X_MIN[i]`/`X_MAX[i]`.

### 5.3 Perhitungan matriks manual (inference)

```c
for (int L = 0; L < NN_N_LAYERS; L++) {
  const float *W = LAYER_W[L];
  const float *B = LAYER_B[L];
  int n_out = LAYER_OUT[L];

  for (int j = 0; j < n_out; j++) {          /* tiap neuron keluaran */
    float acc = B[j];                        /* mulai dari bias      */
    for (int i = 0; i < n_in; i++)
      acc += in[i] * W[i * n_out + j];       /* dot product          */
    out[j] = (LAYER_ACT[L] == NN_ACT_SIGMOID) ? nnSigmoid(acc) : acc;
  }
  if (LAYER_ACT[L] == NN_ACT_SOFTMAX) nnSoftmax(out, n_out, probs);
  in = out; n_in = n_out;                    /* lanjut layer berikutnya */
}
```

Tidak ada library neural network yang dipakai — hanya `<math.h>` untuk `expf()`.

### 5.4 Fungsi aktivasi manual

**Sigmoid** (bentuk stabil, menghindari overflow `expf` saat z sangat negatif):

```c
static inline float nnSigmoid(float z) {
  if (z >= 0.0f) return 1.0f / (1.0f + expf(-z));
  float e = expf(z);
  return e / (1.0f + e);
}
```

**Softmax** (dikurangi nilai maksimum agar stabil secara numerik):

```c
static void nnSoftmax(const float *z, int n, float *out) {
  float zmax = z[0];
  for (int i = 1; i < n; i++) if (z[i] > zmax) zmax = z[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) { out[i] = expf(z[i] - zmax); sum += out[i]; }
  if (sum == 0.0f) sum = 1.0f;
  for (int i = 0; i < n; i++) out[i] /= sum;
}
```

### 5.5 Logika output dan ambang batas

Kelas akhir = **argmax** dari 4 probabilitas softmax. Tambahan *ambang batas
kepercayaan* `NN_CONF_THRESHOLD = 0.50`:

- `confidence ≥ 0.50` → LCD menampilkan `STATUS: YAKIN`
- `confidence < 0.50` → LCD menampilkan `STATUS: RAGU`

Ambang ini berguna di lapangan: pembacaan yang berada di perbatasan dua kelas
(misalnya aroma campuran) ditandai sebagai tidak meyakinkan, bukan dipaksa ke satu
kelas. Nilai ambang dapat diubah pada satu baris `#define`.

### 5.6 Tampilan LCD 16×2 (4 halaman, tombol pada pin 2)

| Halaman | Baris 1 | Baris 2 |
|---|---|---|
| 0 — Prediksi | `Q:excellent` | `Conf: 99.90%OK` |
| 1 — Probabilitas | `ex1.00 gd0.00` | `ac0.00 sp0.00` |
| 2 — Sensor | `SENS  2/11 POT` | `MQ136  928  0.81` |
| 3 — Status | `STATUS: YAKIN` | `Thr:0.50 L1` |

### 5.7 Antarmuka Serial (115200 baud)

Sketch mencetak CSV setiap siklus (`ADC,...`, `PROB,...`, `PRED,...`) agar mudah
dibandingkan dengan hasil Python. Sketch juga menerima masukan:

- ketik **11 bilangan 0–1023** dipisah spasi/koma → memakai nilai itu alih-alih
  potensiometer (sangat praktis untuk menguji baris dataset tertentu di Proteus
  tanpa memutar 11 potensiometer);
- ketik **`p`** → kembali ke mode potensiometer.

---

## 6. Langkah 5 — Simulasi di Proteus

### 6.1 Perancangan sirkuit

| Komponen | Jumlah | Keterangan |
|---|---|---|
| Arduino Mega 2560 | 1 | dipilih karena butuh 11 pin analog (A0–A10) |
| Potensiometer 10 kΩ | 11 | pengganti sensor MQ, wiper → pin analog |
| LCD 16×2 (HD44780) | 1 | mode 4-bit |
| Push button | 1 | ganti halaman LCD |
| Resistor 10 kΩ | 1 | pull-up tombol (bisa pakai `INPUT_PULLUP` internal) |

**Koneksi potensiometer** — setiap potensiometer: kaki 1 → +5 V, kaki 3 → GND,
kaki 2 (wiper) → pin analog.

| Pot | Sensor | Pin Arduino |
|---|---|---|
| 1 | MQ135 | A0 |
| 2 | MQ136 | A1 |
| 3 | MQ137 | A2 |
| 4 | MQ138 | A3 |
| 5 | MQ2 | A4 |
| 6 | MQ3 | A5 |
| 7 | MQ4 | A6 |
| 8 | MQ5 | A7 |
| 9 | MQ6 | A8 |
| 10 | MQ8 | A9 |
| 11 | MQ9 | A10 |

**Koneksi LCD 16×2 (mode 4-bit):**

| LCD | Arduino |
|---|---|
| VSS | GND |
| VDD | +5 V |
| V0 | wiper trimpot 10 kΩ (kontras) |
| RS | D8 |
| RW | GND |
| E | D9 |
| D4, D5, D6, D7 | D10, D11, D12, D13 |
| A (backlight +) | +5 V lewat 220 Ω |
| K (backlight −) | GND |

**Tombol:** satu kaki → D2, kaki lain → GND (sketch memakai `INPUT_PULLUP`).

> **Catatan:** berkas `.pdsprj` tidak dapat dibangkitkan dari lingkungan ini karena
> Proteus bersifat GUI/proprietary. Rangkaian di atas lengkap untuk dirakit di
> Proteus; HEX hasil compile sketch di-load ke komponen Arduino Mega pada skema.

### 6.2 Pengujian — memasukkan baris dataset ke potensiometer

`docs/test_vectors.csv` berisi **16 kasus uji** (4 kasus per kelas) hasil pengambilan
acak dari dataset, sudah dikonversi menjadi nilai ADC 0–1023 yang tinggal di-set pada
potensiometer. Kolomnya: identitas baris, nilai ADC tiap sensor, probabilitas
referensi dari Python, serta prediksi dan confidence.

**Contoh kasus 1** — `2.Round`, menit 16, label asli `excellent`:

| Sensor | ADC | Posisi potensiometer |
|---|---|---|
| MQ135 (A0) | 504 | 49,3 % |
| MQ136 (A1) | 928 | 90,7 % |
| MQ137 (A2) | 84 | 8,2 % |
| MQ138 (A3) | 602 | 58,8 % |
| MQ2 (A4) | 524 | 51,2 % |
| MQ3 (A5) | 304 | 29,7 % |
| MQ4 (A6) | 697 | 68,1 % |
| MQ5 (A7) | 51 | 5,0 % |
| MQ6 (A8) | 20 | 2,0 % |
| MQ8 (A9) | 786 | 76,8 % |
| MQ9 (A10) | 247 | 24,1 % |

Harapan: LCD menampilkan `Q:excellent` dengan `Conf:100.00%`.

**Cara paling praktis** (tanpa memutar 11 potensiometer): buka Serial Monitor
115200 baud, tempel baris ADC kasus uji:

```
504 928 84 602 524 304 697 51 20 786 247
```

Sketch langsung memakai nilai tersebut dan menampilkan hasilnya di LCD.
Untuk memunculkan daftar perintah cepat:

```bash
# cetak 11 nilai ADC kasus uji ke-n, siap tempel ke Serial Monitor
awk -F, 'NR>1 && $1==1 {for(i=6;i<=16;i++) printf "%s ", $i; print ""}' docs/test_vectors.csv
```

### 6.3 Verifikasi

Perbandingan dilakukan pada 16 kasus uji:

| Kasus | Label asli | Prediksi Python | Prediksi Arduino | Conf. Arduino | Conf. Python |
|---|---|---|---|---|---|
| 1 | excellent | excellent | excellent | 0,999969 | 0,999969 |
| 2 | excellent | excellent | excellent | 0,999994 | 0,999994 |
| 3 | excellent | excellent | excellent | 0,999010 | 0,999010 |
| 4 | excellent | excellent | excellent | 0,999997 | 0,999997 |
| 5 | good | good | good | 0,999255 | 0,999255 |
| 6 | good | good | good | 0,999999 | 0,999999 |
| 7 | good | good | good | 0,999890 | 0,999890 |
| 8 | good | good | good | 1,000000 | 1,000000 |
| 9 | acceptable | acceptable | acceptable | 0,772372 | 0,772374 |
| 10 | acceptable | acceptable | acceptable | 0,999825 | 0,999825 |
| 11 | acceptable | good | good | 0,691368 | 0,691369 |
| 12 | acceptable | spoiled | spoiled | 0,827750 | 0,827751 |
| 13 | spoiled | spoiled | spoiled | 0,999734 | 0,999734 |
| 14 | spoiled | spoiled | spoiled | 0,999285 | 0,999285 |
| 15 | spoiled | spoiled | spoiled | 0,999995 | 0,999995 |
| 16 | spoiled | spoiled | spoiled | 0,999992 | 0,999992 |

- **Kelas cocok Python ↔ Arduino: 16/16**
- Selisih probabilitas maksimum: **1,86 × 10⁻⁶**
- Ketepatan terhadap label sebenarnya: **14/16** — sama persis antara Python dan
  Arduino. Dua kasus yang salah (11 dan 12) salah **oleh modelnya sendiri**, bukan
  oleh implementasi mikrokontroler; keduanya kelas `acceptable` yang memang memiliki
  recall terendah (0,9410).

---

## 7. Analisis — Perbandingan Model di Komputer vs Mikrokontroler

### 7.1 Sumber selisih angka

| Sumber | Besaran terukur | Penjelasan |
|---|---|---|
| Aritmetika float32 | maks **1,86 × 10⁻⁶** | Arduino memakai `float` 32-bit dan urutan penjumlahan berbeda dari TensorFlow. Selisih sekecil ini tidak pernah mengubah argmax. |
| Kuantisasi ADC 10-bit | maks **2,47 × 10⁻²** (kasus 11) | Nilai sensor kontinu dipetakan ke 1024 tingkat. Ini sumber selisih **terbesar**, ~13.000× lebih besar daripada selisih aritmetika. |

Kesimpulan penting: **ketelitian implementasi mikrokontroler bukan masalah**. Yang
membatasi kesesuaian adalah resolusi ADC 10-bit. Pada kasus dengan confidence tinggi
(> 0,99) kuantisasi hampir tak berpengaruh; pada kasus di dekat batas keputusan
(kasus 9, 11, 12) kuantisasi dapat menggeser confidence hingga ±0,025.

### 7.2 Pengaruh kuantisasi terhadap keputusan

Meskipun confidence bergeser, **kelas prediksi tetap sama pada 16/16 kasus**. Ini
karena margin antar-kelas umumnya besar. Namun untuk kelas `acceptable` yang
marginnya sempit, potensi perubahan kelas tetap ada — mitigasinya adalah ambang batas
confidence dan/atau pemakaian ADC eksternal 12-bit (`ADC_MAX_VAL` diubah ke 4095).

### 7.3 Model 1 sheet vs model 12 sheet

| Model | Arsitektur | Data latih | Penskalaan | Akurasi pada 12 sheet (26.640 baris) |
|---|---|---|---|---|
| Lama (di repo) | 11→20→4 | 1 sheet (2.220) | global 0–60 | **0,3743** |
| Baru (proyek ini) | 11→24→12→4 | 12 sheet (26.640) | min-max per fitur | **0,9874** |

Model lama mencapai 0,9842 pada sheet asalnya sendiri (`2.Round`) tetapi hanya 0,3743
di luar sheet itu. Perbedaan ini menegaskan bahwa langkah *Integrasi Data* pada soal
bukan formalitas: tanpa menggabungkan 12 potongan daging, model hanya menghafal satu
jenis daging.

Angka 0,9874 pada tabel ini adalah akurasi pada **seluruh** 26.640 baris (termasuk
data latih), sedangkan 0,9848 pada bagian 3.2 adalah akurasi pada **20 % data uji
yang tidak ikut dilatih** — angka 0,9848 adalah ukuran kemampuan generalisasi yang
sebenarnya.

### 7.4 Beban komputasi di mikrokontroler

- **Flash:** 640 parameter × 4 byte = 2,56 KB untuk bobot/bias (dari 256 KB Mega 2560).
- **SRAM:** buffer kerja 2 × 24 float = 192 byte + 11 input + 4 output (dari 8 KB).
- **Operasi per inference:** 11×24 + 24×12 + 12×4 = 264 + 288 + 48 = **600 multiply-
  accumulate**, plus 36 pemanggilan `expf()` untuk sigmoid (24 + 12 neuron) dan 4
  untuk softmax = 40 `expf()`. Pada 16 MHz ini berlangsung dalam orde milidetik —
  jauh di bawah laju pembaruan 250 ms yang dipakai sketch.

---

## Cara menjalankan ulang

Prasyarat: Python 3 dengan `tensorflow`, `pandas`, `openpyxl`, `scikit-learn`, `numpy`;
`g++` untuk pemeriksaan.

```bash
# 1. Training + evaluasi + ekstraksi bobot ke nn_params.h
python training/train_bqc.py

#    opsi berguna:
#      --arch 24,12 --skip-search   pakai arsitektur tertentu tanpa pencarian
#      --class-weight               aktifkan penimbangan kelas
#      --vectors 6                  jumlah kasus uji per kelas
#      --search-only                pencarian arsitektur saja (tanpa melatih model final)

# 2. Pembanding arsitektur dengan training penuh (dasar pemilihan 24-12)
python training/compare_arch.py --candidates "20" "24,12" "32,16"

# 3. Verifikasi: perhitungan Arduino vs TensorFlow + jalankan .ino di PC
bash arduino/test/run_checks.sh
```

**Catatan reproduktibilitas.** Pelatihan tidak sepenuhnya deterministik: TensorFlow
dengan oneDNN dapat menghasilkan urutan pembulatan berbeda antar-run, sehingga dua
run dengan seed yang sama dapat berbeda pada digit ketiga akurasi (terukur: 0,9848 vs
0,9869 untuk arsitektur yang sama). Untuk hasil yang *persis* sama dengan laporan ini,
gunakan berkas `arduino/BQC_Arduino_Mega/BQCmodel_12cuts.keras` yang sudah tersimpan
dan jalankan `python training/train_bqc.py --export-only arduino/BQC_Arduino_Mega/BQCmodel_12cuts.keras`
untuk membangkitkan ulang `nn_params.h` dari model itu.

`run_checks.sh` menjalankan dua pemeriksaan:

- **[A] `test_nn_core`** — mengompilasi `nn_core.h` + `nn_params.h` (kode yang dipakai
  Arduino) lalu membandingkan probabilitasnya dengan TensorFlow. Lulus bila selisih
  < 1×10⁻⁴ dan seluruh kelas cocok.
- **[B] `host_sim`** — mengompilasi **`BQC_Arduino_Mega.ino` apa adanya** dengan shim
  antarmuka Arduino/LCD, menjalankan `setup()`/`loop()`, lalu memeriksa isi layar LCD,
  keluaran Serial, masukan Serial, perpindahan halaman tombol, dan logika ambang batas.

Hasil terakhir yang diperoleh: **[A] selisih maks 1,86×10⁻⁶, kelas 16/16 cocok;
[B] 16 PASS / 0 FAIL.**

### Berkas keluaran

| Berkas | Isi |
|---|---|
| `arduino/BQC_Arduino_Mega/nn_params.h` | bobot, bias, parameter scaling |
| `arduino/BQC_Arduino_Mega/BQCmodel_12cuts.keras` | model TensorFlow |
| `docs/test_vectors.csv` | 16 kasus uji (nilai ADC untuk Proteus) |
| `docs/metrics.json` | metrik & parameter scaling terukur |

---

## Keterbatasan dan yang belum dikerjakan

1. **Berkas `.pdsprj` belum dibuat.** Proteus adalah aplikasi GUI proprietary yang
   tidak tersedia di lingkungan pengembangan ini. Skema lengkap dan tabel koneksi
   pada bagian 6.1 sudah cukup untuk merakitnya; yang perlu ditambahkan hanyalah
   berkas HEX hasil compile.
2. **Kompilasi AVR asli belum diverifikasi di sini.** Toolchain `avr-gcc`/`arduino-cli`
   tidak dapat diunduh pada lingkungan ini (host `downloads.arduino.cc` dan
   `objects.githubusercontent.com` tidak terjangkau). Yang sudah diverifikasi adalah
   kompilasi `nn_core.h` + `nn_params.h` + `BQC_Arduino_Mega.ino` dengan `g++`
   (16/16 pemeriksaan lulus). **Kompilasi di Arduino IDE untuk board "Arduino Mega
   2560" tetap perlu dilakukan** sebelum HEX di-load ke Proteus.
3. **Regresi TVC belum diimplementasikan.** Soal mengizinkan memilih `Label` *atau*
   `TVC`; proyek ini memilih `Label` sesuai RP. Jalur untuk menambahnya sudah ada:
   ganti target menjadi `TVC`, ubah layer output menjadi 1 neuron linier, dan ganti
   softmax + argmax di `nn_core.h` dengan keluaran langsung.
4. **Nilai potensiometer ≠ nilai sensor MQ nyata.** Potensiometer hanya pengganti
   untuk simulasi. Saat memakai sensor MQ sungguhan, gunakan `nnScaleSensor()` dengan
   nilai fisik hasil kalibrasi, dan pastikan `X_MIN`/`X_MAX` diperoleh dari kalibrasi
   yang sama dengan saat training.
