# Analisis Baseline Sensor per Potongan Daging

Rata-rata nilai sensor pada 300 menit pertama (kondisi masih segar):

| Potongan | MQ135 | MQ136 | MQ137 | MQ138 | MQ2 | MQ3 | MQ4 | MQ5 | MQ6 | MQ8 | MQ9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Brisket | 10.80 | 12.48 | 4.91 | 18.21 | 7.42 | 17.13 | 9.00 | 9.23 | 11.37 | 42.87 | 16.88 |
| Clod_Chuck | 14.54 | 10.26 | 11.47 | 14.51 | 7.52 | 18.01 | 6.19 | 10.26 | 17.55 | 36.44 | 16.49 |
| Fat | 17.69 | 11.90 | 11.60 | 19.01 | 11.06 | 29.92 | 8.58 | 10.01 | 16.22 | 43.35 | 15.04 |
| Flap_meat | 10.10 | 8.85 | 11.67 | 13.58 | 6.93 | 20.16 | 6.40 | 10.18 | 15.72 | 37.71 | 10.41 |
| Inside-Outside | 15.59 | 9.52 | 10.46 | 14.59 | 9.06 | 17.16 | 10.37 | 10.24 | 16.13 | 40.27 | 13.26 |
| Rib_eye | 15.84 | 24.58 | 7.59 | 17.07 | 10.35 | 14.71 | 12.15 | 5.83 | 4.44 | 25.16 | 12.58 |
| Round | 13.97 | 36.86 | 5.25 | 17.74 | 11.57 | 14.21 | 20.96 | 6.73 | 3.38 | 45.59 | 9.86 |
| Shin | 14.69 | 13.90 | 12.05 | 17.65 | 10.69 | 22.79 | 8.95 | 9.36 | 14.01 | 42.82 | 12.33 |
| Skirt_meat | 13.66 | 39.61 | 3.00 | 22.96 | 8.61 | 20.34 | 25.32 | 5.19 | 3.37 | 45.27 | 14.32 |
| Striploin | 16.15 | 9.22 | 9.94 | 13.45 | 9.69 | 18.92 | 8.52 | 9.57 | 14.28 | 35.16 | 10.26 |
| Tenderloin | 18.77 | 10.77 | 9.19 | 19.87 | 7.56 | 20.37 | 10.23 | 8.83 | 13.61 | 42.62 | 12.91 |
| Top_Sirloin | 19.20 | 10.83 | 12.87 | 17.02 | 10.03 | 23.41 | 9.67 | 9.84 | 15.01 | 42.75 | 13.55 |
| **Semua potongan** | 15.08 | 16.56 | 9.17 | 17.14 | 9.21 | 19.76 | 11.36 | 8.77 | 12.09 | 40.00 | 13.16 |

Seberapa berbeda baseline antar potongan (selisih maks-min, dalam % dari rata-rata):

| Sensor | Selisih antar potongan |
|---|---|
| MQ135 | 60.3 % |
| MQ136 | 185.7 % |
| MQ137 | 107.7 % |
| MQ138 | 55.4 % |
| MQ2 | 50.4 % |
| MQ3 | 79.5 % |
| MQ4 | 168.4 % |
| MQ5 | 57.7 % |
| MQ6 | 117.3 % |
| MQ8 | 51.1 % |
| MQ9 | 53.3 % |

Rata-rata perbedaan antar potongan: **89.7 %**

## Artinya

Setiap potongan daging menghasilkan tingkat pembacaan sensor (baseline) yang berbeda,
sebagian karena perbedaan kandungan lemak/kelembapan dan posisi sensor, bukan semata
karena tingkat kebusukan. Karena itu, model gabungan:

- **sangat akurat (± 98–99 %)** bila potongan daging yang diuji sudah ada di data training
  (pemakaian normal: alat ini dipakai untuk potongan yang sudah dikalibrasi),
- **turun menjadi ± 62–77 %** bila potongan tersebut belum pernah dilihat model
  (uji leave-one-cut-out) — masih jauh di atas tebak acak 25 %, tapi tidak sempurna.

## Saran pengembangan

1. Sertakan data potongan yang akan diuji saat melatih model (paling praktis).
2. Pakai fitur **perubahan relatif** terhadap pembacaan awal (`MQ135(t) - MQ135(menit 1)`),
   sehingga pengaruh baseline potongan hilang — tetapi fitur tidak lagi sekadar 11 sensor mentah.
3. Latih model terpisah per potongan bila memang harus memprediksi potongan yang belum dikalibrasi.
