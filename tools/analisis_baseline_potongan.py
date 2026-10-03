"""
===============================================================================
 tools/analisis_baseline_potongan.py
 ---------------------------------------------------------------------------
 Menjawab pertanyaan: MENGAPA akurasi model gabungan turun saat diuji pada
 potongan daging yang belum pernah masuk data training (leave-one-cut-out)?

 Caranya: membandingkan "sidik jari" tiap potongan daging, yaitu rata-rata
 tiap sensor per potongan. Kalau antar potongan nilainya jauh berbeda, artinya
 tiap potongan punya garis dasar (baseline) sensor sendiri - dan model yang
 tidak pernah melihat potongan itu akan salah menafsirkan baseline tersebut.

 Output: training/output/analisis_baseline_potongan.md
 Pakai : python tools/analisis_baseline_potongan.py
===============================================================================
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, "e-nose_dataset_12_beef_cuts.xlsx")
OUT = os.path.join(ROOT, "training", "output", "analisis_baseline_potongan.md")
SENSORS = ["MQ135", "MQ136", "MQ137", "MQ138", "MQ2", "MQ3", "MQ4", "MQ5",
           "MQ6", "MQ8", "MQ9"]


def main():
    xl = pd.ExcelFile(XLSX)
    df = pd.concat([pd.read_excel(xl, sheet_name=s).assign(Cut=s.split(".", 1)[-1])
                    for s in xl.sheet_names], ignore_index=True)

    # rata-rata tiap sensor per potongan (saat daging masih segar, menit 1-300)
    fresh = df[df["Minute"] <= 300]
    mean_cut = fresh.groupby("Cut")[SENSORS].mean()
    # nilai sensor potongan yang diuji minus rata-rata potongan lain
    mean_all = fresh[SENSORS].mean()

    lines = ["# Analisis Baseline Sensor per Potongan Daging", "",
             "Rata-rata nilai sensor pada 300 menit pertama (kondisi masih segar):", ""]
    lines.append("| Potongan | " + " | ".join(SENSORS) + " |")
    lines.append("|---|" + "---|" * len(SENSORS))
    for cut, row in mean_cut.iterrows():
        lines.append(f"| {cut} | " + " | ".join(f"{row[s]:.2f}" for s in SENSORS) + " |")
    lines.append(f"| **Semua potongan** | " + " | ".join(f"{mean_all[s]:.2f}" for s in SENSORS) + " |")
    lines.append("")

    # seberapa jauh baseline antar potongan
    spread = (mean_cut.max() - mean_cut.min()) / mean_all * 100
    lines.append("Seberapa berbeda baseline antar potongan (selisih maks-min, dalam % dari rata-rata):")
    lines.append("")
    lines.append("| Sensor | Selisih antar potongan |")
    lines.append("|---|---|")
    for s in SENSORS:
        lines.append(f"| {s} | {spread[s]:.1f} % |")
    lines.append("")
    lines.append(f"Rata-rata perbedaan antar potongan: **{spread.mean():.1f} %**")
    lines.append("")
    lines.append("## Artinya")
    lines.append("")
    lines.append("Setiap potongan daging menghasilkan tingkat pembacaan sensor (baseline) yang berbeda,")
    lines.append("sebagian karena perbedaan kandungan lemak/kelembapan dan posisi sensor, bukan semata")
    lines.append("karena tingkat kebusukan. Karena itu, model gabungan:")
    lines.append("")
    lines.append("- **sangat akurat (± 98–99 %)** bila potongan daging yang diuji sudah ada di data training")
    lines.append("  (pemakaian normal: alat ini dipakai untuk potongan yang sudah dikalibrasi),")
    lines.append("- **turun menjadi ± 62–77 %** bila potongan tersebut belum pernah dilihat model")
    lines.append("  (uji leave-one-cut-out) — masih jauh di atas tebak acak 25 %, tapi tidak sempurna.")
    lines.append("")
    lines.append("## Saran pengembangan")
    lines.append("")
    lines.append("1. Sertakan data potongan yang akan diuji saat melatih model (paling praktis).")
    lines.append("2. Pakai fitur **perubahan relatif** terhadap pembacaan awal (`MQ135(t) - MQ135(menit 1)`),")
    lines.append("   sehingga pengaruh baseline potongan hilang — tetapi fitur tidak lagi sekadar 11 sensor mentah.")
    lines.append("3. Latih model terpisah per potongan bila memang harus memprediksi potongan yang belum dikalibrasi.")
    lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] {OUT}")
    print(mean_cut.round(2).to_string())
    print("\nSelisih baseline antar potongan (%):")
    print(spread.round(1).to_string())


if __name__ == "__main__":
    main()
