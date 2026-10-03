# Verifikasi Kode Arduino vs Model Keras

Sumber binary: g++ / file .ino  
Vektor uji: `training/output/test_vectors.csv` (8 baris)  
Arsitektur: 11 - 32 - 4 (klasifikasi) | 11 - 32 - 1 (regresi TVC)

| # | Potongan | Kelas aktual | Kelas di LCD | Kelas Keras | Cocok | TVC Keras | TVC Arduino | Selisih total | Efek ADC | Efek float |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Clod_Chuck | Excellent | Excellent | Excellent | YA | 2.6156 | 2.6050 | 0.010620 | 0.010622 | 0.000002 |
| 2 | Inside-Outside | Excellent | Excellent | Excellent | YA | 2.0765 | 2.0770 | 0.000500 | 0.000498 | 0.000002 |
| 3 | Clod_Chuck | Good | Good | Good | YA | 3.3178 | 3.3280 | 0.010230 | 0.010234 | 0.000004 |
| 4 | Rib_eye | Good | Good | Good | YA | 3.5877 | 3.5780 | 0.009680 | 0.009677 | 0.000003 |
| 5 | Flap_meat | Acceptable | Acceptable | Acceptable | YA | 4.6317 | 4.6440 | 0.012330 | 0.012330 | 0.000000 |
| 6 | Shin | Acceptable | Acceptable | Acceptable | YA | 4.6645 | 4.6570 | 0.007540 | 0.007542 | 0.000002 |
| 7 | Top_Sirloin | Spoiled | Spoiled | Spoiled | YA | 5.7777 | 5.7880 | 0.010340 | 0.010338 | 0.000002 |
| 8 | Striploin | Spoiled | Spoiled | Spoiled | YA | 5.4864 | 5.4940 | 0.007590 | 0.007588 | 0.000002 |

- Kesesuaian kelas: **100.0 %**
- Selisih TVC maksimum: **0.012330 log CFU/g** (rata-rata 0.008604)
- Selisih probabilitas softmax maksimum: 0.020640

**Dari mana selisihnya?** Kolom *Efek ADC* dan *Efek float* memisahkan dua penyebabnya:

1. **Kuantisasi ADC 10-bit** (penyebab utama, sampai 0.012330 log CFU/g): `analogRead()` hanya menghasilkan bilangan bulat 0-1023, sehingga nilai sensor desimal (mis. 17,85) menjadi 17,83 saat dibaca ulang. Ini keterbatasan perangkat keras, bukan kesalahan rumus.
2. **Presisi `float` 32-bit** (hanya 0.000004 log CFU/g): perhitungan matriks di Arduino memakai `float` 32-bit sedangkan Python memakai float64/float32 dengan urutan operasi berbeda.

Karena jarak antar ambang batas kelas adalah 1,0 log CFU/g, selisih ≤ 0.0123 tidak mengubah kategori kualitas - seluruh 8 vektor uji menghasilkan kelas yang identik dengan prediksi Keras.
