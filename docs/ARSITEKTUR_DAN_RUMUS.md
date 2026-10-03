# Arsitektur NN dan Perhitungan Manual di Arduino

Dokumen ini menjelaskan **apa yang terjadi di dalam** kode Arduino, baris per baris,
supaya mudah dipertanggungjawabkan saat presentasi/demo RP Ganjil.

---

## 1. Arsitektur jaringan

```
            INPUT (11)         HIDDEN (n)          OUTPUT
                                     │
   MQ135 ─┐                          │
   MQ136 ─┤                          ├──► Dense(4)  + softmax   → [P_excellent, P_good, P_acceptable, P_spoiled]
   MQ137 ─┤   normalisasi           │        cabang KLASIFIKASI
   MQ138 ─┤   x_norm ∈ [-1, 1]      │
   MQ2   ─┼──► Dense(n, sigmoid) ────┤
   MQ3   ─┤        W1[11×n] + b1     │
   MQ4   ─┤                          ├──► Dense(1)  + linear    → TVC (log10 CFU/g)
   MQ5   ─┤                          │        cabang REGRESI
   MQ6   ─┤                          │
   MQ8   ─┤                          │
   MQ9   ─┘                          │
```

* Satu **Hidden Layer bersama** (input → hidden) dipakai oleh kedua cabang.
* Cabang klasifikasi: 4 neuron keluaran + **softmax** → peluang tiap kelas kualitas.
* Cabang regresi: 1 neuron keluaran **linear** → nilai TVC kontinu.
* Jumlah neuron hidden `n` dipilih otomatis oleh `architecture_search()`.

---

## 2. Alur perhitungan di mikrokontroler

### Langkah 0 — Pembacaan ADC

```c
int adc = analogRead(SENSOR_PINS[i]);        // 0 .. 1023
```

Di Proteus, potensiometer 0–5 V mensimulasikan sensor dengan skala penuh 0–60,
jadi nilai sensor dikembalikan dengan perbandingan (rule of three):

```
nilai_sensor = adc × (60 / 1023)
```

Inilah satu-satunya langkah yang **hilang** saat model dilatih di komputer
(komputer langsung memakai angka desimal dari dataset, sedangkan Arduino hanya
punya ADC 10-bit → pembulatan ke bilangan bulat = sumber galat utama simulasi).

### Langkah 1 — Normalisasi (input scaling)

```c
x_norm = 2 * (nilai_sensor - NORM_MIN[k]) / (NORM_MAX[k] - NORM_MIN[k]) - 1;
```

Sama persis dengan `MinMaxScalerNN` di Python. `NORM_MIN[k]` dan `NORM_MAX[k]` adalah
nilai minimum/maksimum sensor ke-`k` **pada dataset training** yang sudah diekspor
ke `model_params.h`.

> Contoh: MQ136 = 45,12 pada dataset dengan min = 2,42 dan maks = 46,65:
> `x_norm = 2(45,12 − 2,42)/(46,65 − 2,42) − 1 = 0,9310`

### Langkah 2 — Hidden layer (perkalian matriks)

Rumus: `h_j = sigmoid( Σ_i W1[j][i] · x_i + b1[j] )` untuk `j = 0 … n−1`

```c
for (int j = 0; j < n_hidden; j++) {
    float sum = b1[j];                       // bias sebagai nilai awal
    for (int i = 0; i < 11; i++) {
        sum += W1[j][i] * x_norm[i];         // akumulasi w·x
    }
    h[j] = sigmoid(sum);
}
```

Kalau ditulis dalam notasi matriks:

```
H = f( W1 · X + b1 )       W1 : (n × 11)   X : (11 × 1)   b1 : (n × 1)
```

### Langkah 3a — Output klasifikasi (softmax)

```c
for (int k = 0; k < 4; k++) {
    z[k] = b2[k];
    for (int j = 0; j < n; j++) z[k] += W2[k][j] * h[j];
}
softmax(z, prob, 4);
```

Softmax:

```
prob_k = exp(z_k) / Σ_j exp(z_j)
```

Implementasi di `.ino` memakai trik numerik `expf(z_k − max(z))` supaya tidak
overflow, tapi **hasilnya identik** dengan rumus di atas.

### Langkah 3b — Output regresi TVC

```c
float tvc_norm = TVC_B2[0];
for (int j = 0; j < n; j++) tvc_norm += TVC_W2[j] * h_tvc[j];   // linear
tvc = (tvc_norm - TVC_OUT_LO)/(TVC_OUT_HI - TVC_OUT_LO)
      * (TVC_MAX_LOG - TVC_MIN_LOG) + TVC_MIN_LOG;              // denormalisasi
```

Model dilatih pada TVC yang sudah dinormalisasi `[-1, 1]`, sehingga hasilnya harus
dikembalikan ke skala asli log10 CFU/g dengan rumus *inverse* di atas.

### Langkah 4 — Threshold / logika keputusan

```c
int qualityFromTVC(float tvc) {
  if (tvc < 3.0f) return 0;   // Excellent
  if (tvc < 4.0f) return 1;   // Good
  if (tvc < 5.0f) return 2;   // Acceptable
  return 3;                   // Spoiled
}
```

Ambang 3, 4, dan 5 log10 CFU/g bukan angka sembarang — **itulah aturan pelabelan
yang dipakai dataset** (tiap baris dengan TVC < 3 selalu berlabel 1, dst.), sehingga
konsisten dengan kelas pada laporan.

---

## 3. Jumlah operasi (kenapa Arduino mampu)

| Bagian | Operasi perkalian + penjumlahan |
|---|---|
| Hidden (untuk 1 model) | n × 11 |
| Output softmax | 4 × n |
| Output TVC | n |
| **Total (n = 20)** | 20·11 + 4·20 + 20 = **320 MAC** |

Sekitar 320 perkalian–penjumlahan per prediksi: sangat ringan untuk ATmega2560
(16 MHz) — dieksekusi dalam orde ratusan mikrodetik. Tidak perlu library eksternal,
tidak perlu memori besar (bobot n=20 hanya ± 4 KB `float`), jauh dari batas 256 KB flash.

### 3.1 Catatan memori (penting untuk board selain Mega)

Dengan 32 neuron hidden, ukuran parameter yang ditanam di `model_params.h`:

| Bagian | Ukuran |
|---|---|
| Bobot & bias klasifikasi (`CLS_W1`, `CLS_B1`, `CLS_W2`, `CLS_B2`) | 2.064 byte |
| Bobot & bias regresi TVC (`TVC_W1`, `TVC_B1`, `TVC_W2`, `TVC_B2`) | 1.668 byte |
| Normalisasi, ambang TVC, vektor uji, daftar pin | 185 byte |
| Variabel kerja saat inferensi (`x`, `hidden`, `prob`) | 364 byte |
| **Total perkiraan** | **± 4,3 KB dari 8 KB SRAM Arduino Mega 2560 (52 %)** |

Pada Arduino Mega 2560 ini aman. Dua catatan:

- Kalau memakai board dengan SRAM kecil (Uno/Nano 2 KB), arsitektur 32 neuron **tidak akan
  masuk**. Pilih 10–16 neuron (mis. `10` neuron ≈ 1,6 KB) atau pindahkan bobot ke *flash*
  memori dengan `PROGMEM` + `pgm_read_float_near()` pada loop pembacaan bobot.
- Kode di repo ini sengaja **tidak** memakai `PROGMEM` agar tetap sederhana, mudah dibaca
  untuk laporan, dan bisa diuji di PC memakai g++.

---

## 4. Sisi Python: dari model ke header C++

```python
W1c, b1c = cls_model.layers[0].get_weights()     # (11, n) dan (n,)
W2c, b2c = cls_model.layers[1].get_weights()     # (n, 4) dan (4,)
W1c_t = W1c.T                                    # -> (n, 11) : W1[j][i]
W2c_t = W2c.T                                    # -> (4, n)  : W2[k][j]
```

Transpose dilakukan **hanya agar indeks array di C cocok** dengan urutan loop
`W[j][i]` (baris = neuron tujuan, kolom = neuron asal). Nilai numeriknya tidak berubah.

Model > 1 hidden layer belum didukung `model_params.h`; kalau ingin mencobanya,
tambahkan `export_model_params_header()` untuk layer ketiga, atau ubah arsitektur
menjadi 1 hidden layer (default repo ini).

---

## 5. Sumber perbedaan hasil Python vs Arduino

| Sumber | Besar pengaruh | Bisa dikurangi? |
|---|---|---|
| `analogRead` membulatkan ke bilangan bulat 10-bit (≈0,0586 satuan sensor per LSB) | dominan, mengubah `x_norm` ±0,003 | tidak (keterbatasan ADC) |
| `float` 32-bit vs `float32/float64` TensorFlow | ±1e-6 pada peluang | tidak perlu |
| Potensiometer Proteus tidak bisa diputar sampai desimal persis | bervariasi | pakai `INPUT_MODE 1` (vektor uji) untuk verifikasi presisi |

Kesimpulan yang bisa ditulis di laporan: **perbedaan hanya berasal dari kuantisasi
input ADC, bukan dari kesalahan rumus**, dan tidak pernah mengubah kategori kualitas
selama nilai sensor tidak berada persis di tepi ambang batas.

---

## 6. Istilah yang sering ditanyakan saat sidang

| Istilah | Arti singkat |
|---|---|
| TVC | *Total Viable Count* — jumlah mikroba hidup; di dataset sudah dalam log10 CFU/g |
| CFU | *Colony Forming Unit* |
| Electronic nose | Deret sensor gas (di sini 11 sensor MQ) yang meniru penciuman, membaca pola gas hasil metabolisme mikroba |
| Sigmoid | `1/(1+e^-x)`, memetakan nilai ke (0,1); dipakai di hidden layer |
| Softmax | Mengubah deret skor menjadi peluang yang totalnya 1 |
| MAC | *Multiply–Accumulate* — operasi `w·x + b` |
| Leave-one-cut-out | Uji generalisasi: satu potongan daging dikeluarkan dari training lalu dipakai sebagai data uji |
