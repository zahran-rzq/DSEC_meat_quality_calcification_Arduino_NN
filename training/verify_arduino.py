#!/usr/bin/env python3
"""
Verifikasi implementasi Arduino terhadap model Python.

Skrip ini mengompilasi kode C++ yang DIPAKAI ARDUINO (nn_core.h + nn_params.h,
dipanggil lewat nnProcessADC() -- fungsi yang sama dengan di .ino) dengan g++,
menjalankannya pada kasus uji, lalu membandingkan hasilnya dengan prediksi
TensorFlow.

Tiga hal yang diukur:
  A. Python(ADC) vs Arduino(ADC)   -> selisih murni aritmetika float32
                                      (harus sangat kecil, kelas harus sama)
  B. Python(float asli) vs Arduino(ADC) -> efek kuantisasi ADC 10-bit
  C. Kesesuaian kelas prediksi terhadap label sebenarnya

Pemakaian:
    python training/verify_arduino.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKETCH_DIR = os.path.join(ROOT, "arduino", "BQC_Arduino_Mega")
TEST_DIR = os.path.join(ROOT, "arduino", "test")
VECTORS = os.path.join(ROOT, "docs", "test_vectors.csv")
DATASET = os.path.join(ROOT, "e-nose_dataset_12_beef_cuts.xlsx")

SENSOR_NAMES = ["MQ135", "MQ136", "MQ137", "MQ138", "MQ2",
                "MQ3", "MQ4", "MQ5", "MQ6", "MQ8", "MQ9"]
CLASS_NAMES = ["excellent", "good", "acceptable", "spoiled"]
ADC_MAX = 1023.0


def compile_test() -> str:
    binary = os.path.join(TEST_DIR, "test_nn_core")
    src = os.path.join(TEST_DIR, "test_nn_core.cpp")
    cmd = ["g++", "-O2", "-std=c++11", "-Wall", "-Wextra",
           "-I", SKETCH_DIR, src, "-o", binary]
    print("[build] " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout)
    warns = [l for l in r.stderr.splitlines() if "warning" in l]
    errs = [l for l in r.stderr.splitlines() if "error" in l]
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("GAGAL mengompilasi kode uji Arduino")
    print(f"[build] sukses ({len(warns)} warning, {len(errs)} error)")
    for w in warns:
        print("        " + w.strip())
    return binary


def run_test(binary: str) -> dict:
    r = subprocess.run([binary, VECTORS], capture_output=True, text=True, cwd=TEST_DIR)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("Program uji gagal dijalankan")
    arch = thresh = None
    results = {}
    for line in r.stdout.strip().splitlines():
        p = line.split(",")
        if p[0] == "ARCH":
            arch = [int(v) for v in p[1:]]
        elif p[0] == "THRESHOLD":
            thresh = float(p[1])
        elif p[0] == "RESULT":
            results[int(p[1])] = {
                "pred": p[2], "conf": float(p[3]),
                "prob": np.array([float(v) for v in p[4:8]]),
                "verdict": p[9],
            }
    print(f"[run] arsitektur terkompilasi: {arch}, threshold={thresh}")
    return {"arch": arch, "threshold": thresh, "results": results}


def main() -> int:
    if not os.path.exists(VECTORS):
        print(f"ERROR: {VECTORS} tidak ada. Jalankan training/train_bqc.py dulu.")
        return 2

    binary = compile_test()
    cpp = run_test(binary)

    # ---------- model Python ----------
    model_path = os.path.join(SKETCH_DIR, "BQCmodel_12cuts.keras")
    model = tf.keras.models.load_model(model_path)
    py_arch = [int(l.units) for l in model.layers if hasattr(l, "units")]
    print(f"[py ] arsitektur model Python : {[model.input_shape[-1]] + py_arch}")
    if cpp["arch"] != [model.input_shape[-1]] + py_arch:
        print("!!! ARSITEKTUR TIDAK COCOK antara header C++ dan model Python")
        return 1

    vec = pd.read_csv(VECTORS)
    adc_cols = [f"adc_{s}" for s in SENSOR_NAMES]

    # parameter scaling yang sama dengan train_bqc.py (min-max per fitur)
    xl = pd.ExcelFile(DATASET)
    frames = []
    for s in xl.sheet_names:
        d = xl.parse(s)
        d.insert(0, "Cut", s)
        frames.append(d)
    allrows = pd.concat(frames, ignore_index=True)
    raw_all = allrows[SENSOR_NAMES].to_numpy(dtype=np.float64)
    xmin, xmax = raw_all.min(axis=0), raw_all.max(axis=0)
    span = np.where((xmax - xmin) == 0, 1.0, xmax - xmin)

    # --- A: Python dari nilai ADC (jalur yang sama persis dengan Arduino)
    adc = vec[adc_cols].to_numpy(dtype=np.float64)
    x_from_adc = 2.0 * adc / ADC_MAX - 1.0
    prob_adc = model.predict(x_from_adc, verbose=0)
    pred_adc = np.argmax(prob_adc, axis=1)

    # --- B: Python dari nilai sensor asli (float penuh, tanpa kuantisasi)
    #        cari kembali baris dataset berdasarkan Cut + Minute
    key = {(r.Cut, int(r.Minute)): i for i, r in enumerate(
        allrows[["Cut", "Minute"]].itertuples(index=False))}
    idx = [key[(r.cut, int(r.minute))] for r in vec.itertuples(index=False)]
    x_exact = 2.0 * (raw_all[idx] - xmin) / span - 1.0
    prob_exact = model.predict(x_exact, verbose=0)
    pred_exact = np.argmax(prob_exact, axis=1)

    # ---------- bandingkan ----------
    max_abs_diff = 0.0
    class_mismatch = []
    rows = []
    for i, case in enumerate(vec.case_id):
        c = cpp["results"][int(case)]
        d = float(np.max(np.abs(c["prob"] - prob_adc[i])))
        max_abs_diff = max(max_abs_diff, d)
        agree = (c["pred"] == CLASS_NAMES[pred_adc[i]])
        if not agree:
            class_mismatch.append(int(case))
        quant_diff = float(np.max(np.abs(prob_exact[i] - prob_adc[i])))
        rows.append({
            "case": int(case),
            "actual": vec.actual_class.iloc[i],
            "arduino": c["pred"],
            "py_adc": CLASS_NAMES[pred_adc[i]],
            "py_exact": CLASS_NAMES[pred_exact[i]],
            "conf_arduino": round(c["conf"], 6),
            "conf_py": round(float(prob_adc[i, pred_adc[i]]), 6),
            "max_abs_diff": f"{d:.3e}",
            "quant_diff": f"{quant_diff:.3e}",
        })

    df = pd.DataFrame(rows)
    print("\n================ HASIL VERIFIKASI ================")
    print(df.to_string(index=False))

    n = len(df)
    print("\n---------------- RINGKASAN ----------------")
    print(f"Jumlah kasus uji                     : {n}")
    print(f"Selisih probabilitas maks (A)        : {max_abs_diff:.3e}")
    print(f"Kelas tidak cocok Python vs Arduino  : {len(class_mismatch)} {class_mismatch}")
    print(f"Kelas cocok Python vs Arduino        : {n - len(class_mismatch)}/{n}")
    print(f"Arduino benar thd label sebenarnya   : "
          f"{(df.arduino == df.actual).sum()}/{n}")
    print(f"Python  benar thd label sebenarnya   : "
          f"{(df.py_exact == df.actual).sum()}/{n}")

    ok = (len(class_mismatch) == 0 and max_abs_diff < 1e-4)
    print("\n" + ("VERIFIKASI LULUS: perhitungan Arduino identik dengan model Python."
                  if ok else
                  "VERIFIKASI GAGAL: lihat tabel di atas."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
