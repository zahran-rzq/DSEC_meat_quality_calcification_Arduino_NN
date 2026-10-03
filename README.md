# Prediksi Kualitas Daging Sapi — Electronic Nose → Neural Network → Arduino

Implementasi lengkap tugas *implementasi model Neural Network hasil training ke
mikrokontroler*: data 12 potongan daging sapi dilatih menjadi neural network di
Python, bobotnya diekstrak ke header C++, lalu inference dijalankan **secara manual**
(perkalian matriks + fungsi aktivasi tulis tangan) pada Arduino Mega 2560.

```
Excel (12 sheet, 26.640 baris)
        │  pre-processing: gabung + min-max per fitur
        ▼
TensorFlow  11 → 24 (sigmoid) → 12 (sigmoid) → 4 (softmax)     akurasi test 98,48 %
        │  ekstraksi bobot & bias (640 parameter)
        ▼
nn_params.h  ──►  nn_core.h  ──►  BQC_Arduino_Mega.ino  ──►  LCD 16×2
                  (perkalian matriks manual, sigmoid & softmax manual)
```

## Isi repositori

| Berkas | Keterangan |
|---|---|
| `e-nose_dataset_12_beef_cuts.xlsx` | dataset asli, 12 sheet (12 potongan daging) |
| `BeefQualityClassification_Softmax.ipynb` | notebook sampel (1 sheet) — lihat catatan di bawah |
| `BQCmodelSoftmaxTF2.21.0.keras` | model lama hasil notebook sampel |
| `training/train_bqc.py` | pipeline lengkap: gabung data → latih → evaluasi → ekspor bobot |
| `training/compare_arch.py` | pembanding arsitektur dengan training penuh |
| `training/verify_arduino.py` | bandingkan perhitungan C++ Arduino dengan TensorFlow |
| `arduino/BQC_Arduino_Mega/BQC_Arduino_Mega.ino` | **sketch Arduino** (jawaban utama) |
| `arduino/BQC_Arduino_Mega/nn_core.h` | perhitungan NN manual, murni C tanpa library |
| `arduino/BQC_Arduino_Mega/nn_params.h` | bobot, bias, dan parameter scaling (auto-generated) |
| `arduino/test/` | pemeriksaan otomatis (numerik + host simulation `.ino`) |
| `docs/LAPORAN.md` | **laporan proyek lengkap** |
| `docs/test_vectors.csv` | 16 kasus uji, sudah berupa nilai ADC untuk Proteus |
| `docs/metrics.json` | metrik terukur dan parameter scaling |

## Mulai dari mana

1. Baca **`docs/LAPORAN.md`** — berisi seluruh langkah pengerjaan, arsitektur, tabel
   bobot, skema Proteus, dan analisis perbandingan.
2. Buka **`arduino/BQC_Arduino_Mega/BQC_Arduino_Mega.ino`** di Arduino IDE, pilih board
   *Arduino Mega 2560*, compile, lalu load HEX-nya ke Proteus.
3. Uji dengan nilai ADC dari `docs/test_vectors.csv` (lihat LAPORAN bagian 6.2).

## Menjalankan ulang pipeline

```bash
pip install tensorflow pandas openpyxl scikit-learn numpy

python training/train_bqc.py            # latih + evaluasi + tulis nn_params.h
bash   arduino/test/run_checks.sh       # verifikasi kode Arduino vs model Python
```

Perintah lain yang berguna:

```bash
python training/train_bqc.py --search-only          # pencarian arsitektur saja
python training/train_bqc.py --arch 24,12 --skip-search
python training/train_bqc.py --class-weight         # tangani ketidakseimbangan kelas
python training/train_bqc.py --export-only BQCmodel_12cuts.keras
```

## Hasil pemeriksaan terakhir

```
[A] selisih probabilitas maks Arduino vs Python : 1,86e-06
    kelas cocok Python vs Arduino               : 16/16
[B] host simulation BQC_Arduino_Mega.ino        : 16 PASS / 0 FAIL
```

## Catatan tentang notebook sampel

Notebook `BeefQualityClassification_Softmax.ipynb` hanya melatih dari sheet `2.Round`
(2.220 baris) dengan penskalaan global `0..60`. Model itu mencapai 98,42 % pada sheet
asalnya tetapi **hanya 37,43 %** pada gabungan 12 potongan daging — jadi tidak memenuhi
langkah *Integrasi Data* pada soal. Pipeline di `training/train_bqc.py` memperbaikinya.

Sel prediksi di notebook tersebut juga memuat asumsi keliru bahwa `pd.get_dummies`
mengurutkan kelas secara alfabetis; karena kolom `Label` bertipe integer 1..4, urutan
sebenarnya adalah `[1, 2, 3, 4]`. Sel itu sudah diperbaiki (lihat `docs/LAPORAN.md`
bagian 2.3).
