# Hasil Training - Prediksi Kualitas Daging (Electronic Nose)

*Dibuat otomatis: 2026-10-03 14:10:00 | TensorFlow 2.21.0*

## 1. Dataset gabungan (12 potongan daging)

- Jumlah baris: **26640** (12 potongan daging x 2.220 menit pengukuran)
- Jumlah fitur input: **11** sensor MQ -> `MQ135, MQ136, MQ137, MQ138, MQ2, MQ3, MQ4, MQ5, MQ6, MQ8, MQ9`
- Target: `Label` (klasifikasi 4 kelas) dan `TVC` dalam log10 CFU/g (regresi)
- Data uji 20 % terpisah: 21312 training / 5328 uji

Distribusi kelas: Excellent = 3540, Good = 4140, Acceptable = 3120, Spoiled = 15840

## 2. Normalisasi Min-Max (parameter untuk Arduino)

| Sensor | min (training) | max (training) | min (deploy) | max (deploy) |
|---|---|---|---|---|
| MQ135 | 7.940 | 21.840 | 7.940 | 22.140 |
| MQ136 | 2.420 | 46.650 | 2.420 | 46.650 |
| MQ137 | 2.620 | 28.960 | 2.620 | 28.960 |
| MQ138 | 7.670 | 28.640 | 7.670 | 28.640 |
| MQ2 | 3.790 | 20.120 | 3.790 | 20.120 |
| MQ3 | 7.540 | 36.400 | 7.540 | 36.400 |
| MQ4 | 2.230 | 36.040 | 2.230 | 36.270 |
| MQ5 | 4.190 | 33.920 | 4.190 | 33.920 |
| MQ6 | 2.400 | 41.050 | 2.400 | 41.050 |
| MQ8 | 15.990 | 53.270 | 15.990 | 54.080 |
| MQ9 | 7.160 | 19.300 | 7.160 | 19.300 |

Rumus: `x_norm = 2*(x - min)/(max - min) - 1`  -> rentang [-1.0, 1.0]

## 3. Pemilihan arsitektur

| Hidden layers | Jumlah parameter | Akurasi validasi | F1-macro | Epoch | Waktu (s) |
|---|---|---|---|---|---|
| 64-32 | 2980 | 98.86 % | 0.9825 | 180 | 32.7 |
| 32 | 516 | 98.70 % | 0.9804 | 300 | 54.7 |
| 32-16 | 980 | 98.59 % | 0.9786 | 193 | 34.1 |
| 20 | 324 | 98.33 % | 0.9746 | 300 | 58.1 |
| 64 | 1028 | 98.22 % | 0.9737 | 240 | 42.3 |
| 16 | 260 | 98.12 % | 0.9710 | 300 | 50.1 |
| 10 | 164 | 97.45 % | 0.9588 | 300 | 49.4 |

**Arsitektur dipakai:** 11 -> 32 (sigmoid) -> 4 (softmax) dan 11 -> 32 (sigmoid) -> 1 (linear)

## 4. Performa model (data uji 20 %)

- Akurasi klasifikasi: **98.72 %**  (balanced accuracy 98.12 %, F1-macro 0.9804)
- Regresi TVC: MAE **0.1030** log CFU/g, RMSE 0.1439, R2 0.9836
- Akurasi kelas dari threshold TVC: 89.38 %
- Kesepakatan kelas softmax vs kelas-TVC: 89.40 %

### Classification report

```
              precision    recall  f1-score   support

   Excellent     0.9929    0.9887    0.9908       708
        Good     0.9795    0.9795    0.9795       828
  Acceptable     0.9509    0.9631    0.9570       624
     Spoiled     0.9953    0.9937    0.9945      3168

    accuracy                         0.9872      5328
   macro avg     0.9796    0.9812    0.9804      5328
weighted avg     0.9873    0.9872    0.9873      5328
```

### Akurasi per potongan daging (model gabungan)

| Potongan | Akurasi | Jumlah baris uji |
|---|---|---|
| Flap_meat | 97.52 % | 444 |
| Inside-Outside | 98.04 % | 459 |
| Brisket | 98.20 % | 445 |
| Striploin | 98.28 % | 464 |
| Top_Sirloin | 98.57 % | 420 |
| Shin | 98.93 % | 467 |
| Rib_eye | 99.06 % | 426 |
| Tenderloin | 99.07 % | 429 |
| Round | 99.09 % | 438 |
| Fat | 99.14 % | 463 |
| Clod_Chuck | 99.32 % | 441 |
| Skirt_meat | 99.54 % | 432 |

## 5. Model gabungan vs potongan yang belum pernah dilihat (leave-one-cut-out)

| Potongan yang diuji | Akurasi | F1-macro |
|---|---|---|
| Brisket | 76.80 % | 0.5623 |
| Clod_Chuck | 77.16 % | 0.5635 |
| Fat | 62.03 % | 0.3865 |
| Flap_meat | 64.68 % | 0.2998 |
| Inside-Outside | 70.81 % | 0.4742 |
| Rib_eye | 69.73 % | 0.4414 |
| Round | 77.21 % | 0.4575 |
| Shin | 82.52 % | 0.6389 |
| Skirt_meat | 61.44 % | 0.3262 |
| Striploin | 77.43 % | 0.4681 |
| Tenderloin | 79.50 % | 0.5633 |
| Top_Sirloin | 81.35 % | 0.7436 |

Rata-rata: **73.39 %** (min 61.44 %, max 82.52 %)

## 6. Vektor uji untuk Proteus

Nilai ADC dihitung dengan `ADC = nilai / 60 * 1023`; atur potensiometer
sampai tegangan atau kode ADC tersebut (lihat `PANDUAN_PROTEUS.md`).

| # | Potongan | Kelas aktual | Prediksi Python | TVC aktual | Prediksi TVC |
|---|---|---|---|---|---|
| 1 | Clod_Chuck | Excellent | Excellent | 2.4097 | 2.6156 |
| 2 | Inside-Outside | Excellent | Excellent | 2.0022 | 2.0765 |
| 3 | Clod_Chuck | Good | Good | 3.3164 | 3.3178 |
| 4 | Rib_eye | Good | Good | 3.9431 | 3.5877 |
| 5 | Flap_meat | Acceptable | Acceptable | 4.2575 | 4.6317 |
| 6 | Shin | Acceptable | Acceptable | 4.2575 | 4.6645 |
| 7 | Top_Sirloin | Spoiled | Spoiled | 5.7580 | 5.7777 |
| 8 | Striploin | Spoiled | Spoiled | 5.6111 | 5.4864 |

| Baris | Potongan | Kelas aktual | MQ135 | MQ136 | MQ137 | MQ138 | MQ2 | MQ3 | MQ4 | MQ5 | MQ6 | MQ8 | MQ9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| #1 | Clod_Chuck | Excellent | 14.58 <br> ADC 249 <br> 24.3 % | 9.90 <br> ADC 169 <br> 16.5 % | 11.86 <br> ADC 202 <br> 19.8 % | 14.70 <br> ADC 251 <br> 24.5 % | 7.92 <br> ADC 135 <br> 13.2 % | 19.18 <br> ADC 327 <br> 32.0 % | 6.30 <br> ADC 107 <br> 10.5 % | 11.64 <br> ADC 198 <br> 19.4 % | 18.00 <br> ADC 307 <br> 30.0 % | 36.78 <br> ADC 627 <br> 61.3 % | 16.63 <br> ADC 284 <br> 27.7 % |
| #2 | Inside-Outside | Excellent | 16.76 <br> ADC 286 <br> 27.9 % | 13.57 <br> ADC 231 <br> 22.6 % | 8.46 <br> ADC 144 <br> 14.1 % | 19.55 <br> ADC 333 <br> 32.6 % | 13.64 <br> ADC 233 <br> 22.7 % | 21.43 <br> ADC 365 <br> 35.7 % | 11.75 <br> ADC 200 <br> 19.6 % | 9.27 <br> ADC 158 <br> 15.4 % | 14.76 <br> ADC 252 <br> 24.6 % | 39.30 <br> ADC 670 <br> 65.5 % | 13.83 <br> ADC 236 <br> 23.1 % |
| #3 | Clod_Chuck | Good | 13.89 <br> ADC 237 <br> 23.1 % | 6.83 <br> ADC 116 <br> 11.4 % | 13.52 <br> ADC 231 <br> 22.5 % | 12.24 <br> ADC 209 <br> 20.4 % | 6.14 <br> ADC 105 <br> 10.2 % | 16.86 <br> ADC 287 <br> 28.1 % | 5.87 <br> ADC 100 <br> 9.8 % | 10.30 <br> ADC 176 <br> 17.2 % | 19.05 <br> ADC 325 <br> 31.8 % | 35.11 <br> ADC 599 <br> 58.5 % | 11.66 <br> ADC 199 <br> 19.4 % |
| #4 | Rib_eye | Good | 12.23 <br> ADC 209 <br> 20.4 % | 17.65 <br> ADC 301 <br> 29.4 % | 14.18 <br> ADC 242 <br> 23.6 % | 15.33 <br> ADC 261 <br> 25.6 % | 7.06 <br> ADC 120 <br> 11.8 % | 12.78 <br> ADC 218 <br> 21.3 % | 8.37 <br> ADC 143 <br> 13.9 % | 13.38 <br> ADC 228 <br> 22.3 % | 7.19 <br> ADC 123 <br> 12.0 % | 24.59 <br> ADC 419 <br> 41.0 % | 9.70 <br> ADC 165 <br> 16.2 % |
| #5 | Flap_meat | Acceptable | 8.46 <br> ADC 144 <br> 14.1 % | 5.25 <br> ADC 90 <br> 8.8 % | 17.07 <br> ADC 291 <br> 28.4 % | 8.49 <br> ADC 145 <br> 14.2 % | 5.32 <br> ADC 91 <br> 8.9 % | 14.71 <br> ADC 251 <br> 24.5 % | 4.66 <br> ADC 79 <br> 7.8 % | 15.89 <br> ADC 271 <br> 26.5 % | 17.75 <br> ADC 303 <br> 29.6 % | 37.44 <br> ADC 638 <br> 62.4 % | 14.88 <br> ADC 254 <br> 24.8 % |
| #6 | Shin | Acceptable | 11.66 <br> ADC 199 <br> 19.4 % | 6.50 <br> ADC 111 <br> 10.8 % | 17.14 <br> ADC 292 <br> 28.6 % | 9.47 <br> ADC 161 <br> 15.8 % | 6.32 <br> ADC 108 <br> 10.5 % | 14.01 <br> ADC 239 <br> 23.4 % | 5.09 <br> ADC 87 <br> 8.5 % | 15.63 <br> ADC 266 <br> 26.1 % | 15.49 <br> ADC 264 <br> 25.8 % | 42.11 <br> ADC 718 <br> 70.2 % | 11.34 <br> ADC 193 <br> 18.9 % |
| #7 | Top_Sirloin | Spoiled | 17.19 <br> ADC 293 <br> 28.6 % | 7.46 <br> ADC 127 <br> 12.4 % | 19.93 <br> ADC 340 <br> 33.2 % | 12.90 <br> ADC 220 <br> 21.5 % | 5.70 <br> ADC 97 <br> 9.5 % | 16.86 <br> ADC 287 <br> 28.1 % | 4.66 <br> ADC 79 <br> 7.8 % | 18.37 <br> ADC 313 <br> 30.6 % | 17.58 <br> ADC 300 <br> 29.3 % | 39.30 <br> ADC 670 <br> 65.5 % | 11.21 <br> ADC 191 <br> 18.7 % |
| #8 | Striploin | Spoiled | 13.03 <br> ADC 222 <br> 21.7 % | 5.45 <br> ADC 93 <br> 9.1 % | 18.99 <br> ADC 324 <br> 31.6 % | 8.79 <br> ADC 150 <br> 14.7 % | 5.48 <br> ADC 93 <br> 9.1 % | 13.19 <br> ADC 225 <br> 22.0 % | 4.11 <br> ADC 70 <br> 6.8 % | 16.64 <br> ADC 284 <br> 27.7 % | 15.98 <br> ADC 272 <br> 26.6 % | 31.74 <br> ADC 541 <br> 52.9 % | 9.04 <br> ADC 154 <br> 15.1 % |
