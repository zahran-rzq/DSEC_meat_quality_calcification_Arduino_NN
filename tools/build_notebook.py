"""
Konversi training/train_meat_quality.py  ->  notebook/BeefQuality_NN_12Potongan.ipynb

Skrip .py memakai penanda sel gaya jupytext (`# %%` dan `# %% [markdown]`),
sehingga bisa diubah menjadi notebook tanpa menjalankan ulang kode.

Pakai:
    python tools/build_notebook.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "training", "train_meat_quality.py")
DST_DIR = os.path.join(ROOT, "notebook")
DST = os.path.join(DST_DIR, "BeefQuality_NN_12Potongan.ipynb")

INTRO_MD = """# RP Ganjil - Prediksi Kualitas Daging (Electronic Nose 11 Sensor MQ)

Notebook ini adalah versi notebook dari `training/train_meat_quality.py`.

**Ringkasan alur:**

1. **Integrasi data** - 12 sheet (12 potongan daging) digabung menjadi satu dataset
   (26.640 baris) memakai `pd.concat`.
2. **Fitur** - 11 sensor MQ: MQ135, MQ136, MQ137, MQ138, MQ2, MQ3, MQ4, MQ5, MQ6, MQ8, MQ9.
3. **Target** - `Label` (4 kelas kualitas) *dan* `TVC` (regresi, log10 CFU/g).
   Pada dataset ini `Label` adalah hasil threshold TVC: `<3` Excellent, `3-4` Good,
   `4-5` Acceptable, `>=5` Spoiled.
4. **Normalisasi** - Min-Max ke rentang `[-1, 1]`; parameter `min`/`max` diekspor ke Arduino.
5. **Training** - pencarian arsitektur (jumlah hidden neuron) lalu training model final.
6. **Evaluasi** - akurasi, confusion matrix, F1, MAE/RMSE TVC, uji *leave-one-cut-out*.
7. **Ekstraksi** - bobot & bias ditulis ke `arduino/MeatQuality_NN_Arduino/model_params.h`
   dalam bentuk array `const float` siap tempel ke Arduino.
"""

SETUP_CELL = """# ============================================================================
# PERSIAPAN (jalankan sekali)
# ----------------------------------------------------------------------------
# 1) Install library. Di Google Colab baris ini perlu dijalankan; di komputer
#    lokal cukup sekali saja.
# 2) Pastikan file 'e-nose_dataset_12_beef_cuts.xlsx' ada di folder yang sama
#    dengan notebook ini. Kalau tidak ada, upload dulu (Colab: ikon folder di
#    kiri -> Upload).
# ============================================================================
import os

if not os.path.exists('e-nose_dataset_12_beef_cuts.xlsx'):
    try:
        from google.colab import files          # khusus Google Colab
        print('Upload file e-nose_dataset_12_beef_cuts.xlsx ...')
        files.upload()
    except ImportError:
        raise FileNotFoundError(
            "File 'e-nose_dataset_12_beef_cuts.xlsx' tidak ditemukan. "
            "Letakkan file dataset di folder yang sama dengan notebook ini.")

print('Dataset ditemukan:', os.path.getsize('e-nose_dataset_12_beef_cuts.xlsx'), 'byte')

# Di Colab/TensorFlow < 2.16, kalau ada galat import tensorflow jalankan:
# !pip install -q tensorflow pandas numpy scikit-learn matplotlib openpyxl
"""

COLAB_NOTE = """# Catatan Google Colab:
#   - tensorflow, pandas, numpy, matplotlib, sklearn, openpyxl sudah tersedia.
#   - Kalau muncul error 'No module named openpyxl':
#         !pip install -q openpyxl
"""


def md_from_comment(block_lines):
    out = []
    for ln in block_lines:
        ln = re.sub(r"^#\s?", "", ln.rstrip())
        out.append(ln)
    text = "\n".join(out).strip("\n")
    return text


def main():
    src = open(SRC, encoding="utf-8").read()

    # buang docstring modul (sudah digantikan INTRO_MD)
    src = re.sub(r'^\s*"""[\s\S]*?"""\s*', "", src, count=1, flags=re.M)

    parts = re.split(r"^# %%", src, flags=re.M)
    cells = []

    for part in parts[1:]:
        first_line, _, body = part.partition("\n")
        is_md = "markdown" in first_line
        body = body.rstrip("\n")
        if not body.strip():
            continue
        lines = (md_from_comment(body.splitlines()) if is_md else body).split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        if is_md:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": source})

    # sel pembuka
    intro = {"cell_type": "markdown", "metadata": {},
             "source": [ln + "\n" for ln in INTRO_MD.strip("\n").split("\n")]}
    intro["source"][-1] = intro["source"][-1].rstrip("\n")

    setup = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": [ln + "\n" for ln in SETUP_CELL.strip("\n").split("\n")]}
    setup["source"][-1] = setup["source"][-1].rstrip("\n")

    all_cells = [intro, setup] + cells

    # sel penutup: di notebook, blok `if __name__ == "__main__"` tidak berjalan,
    # jadi main() dipanggil eksplisit di sel terakhir.
    runner = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": ["# Jalankan seluruh alur: integrasi data -> training -> ekspor parameter Arduino\n"
                         "main()"]}
    all_cells = all_cells + [runner]

    nb = {
        "cells": all_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    os.makedirs(DST_DIR, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[OK] Notebook dibuat: {DST} ({len(all_cells)} sel)")

    # validasi
    with open(DST, encoding="utf-8") as f:
        check = json.load(f)
    n_md = sum(c["cell_type"] == "markdown" for c in check["cells"])
    print(f"     markdown: {n_md} sel, code: {len(check['cells'])-n_md} sel")


if __name__ == "__main__":
    main()
