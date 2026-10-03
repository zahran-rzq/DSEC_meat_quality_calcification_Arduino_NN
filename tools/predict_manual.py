"""
===============================================================================
 tools/predict_manual.py
 ---------------------------------------------------------------------------
 Memprediksi kualitas daging dari 11 nilai sensor yang Anda masukkan manual -
 memakai model hasil training (mirip sel terakhir notebook contoh dosen).

 Pakai:
    # 1) masukkan nilai sensor sebagai argumen
    python tools/predict_manual.py 17.85 15.58 7.95 21.23 16.52 24.07 12.29 9.02 14.26 40.29 14.40

    # 2) tanpa argumen -> memakai vektor uji TEST_VECTOR di model_params.h
    python tools/predict_manual.py

    # 3) koma desimal juga bisa
    python tools/predict_manual.py 17,85 15,58 7,95 21,23 16,52 24,07 12,29 9,02 14,26 40,29 14,4

    # 4) ambil baris tertentu dari dataset gabungan
    python tools/predict_manual.py --row 25000
===============================================================================
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "arduino", "MeatQuality_NN_Arduino", "model_params.h")
sys.path.insert(0, os.path.join(ROOT, "verification"))
from verify_cpp_vs_keras import parse_header, python_reference  # noqa: E402


def read_row_from_dataset(row_index):
    import pandas as pd
    xl = pd.ExcelFile(os.path.join(ROOT, "e-nose_dataset_12_beef_cuts.xlsx"))
    frames = [pd.read_excel(xl, sheet_name=s).assign(Cut=s.split(".", 1)[-1])
              for s in xl.sheet_names]
    df = pd.concat(frames, ignore_index=True)
    r = df.iloc[row_index]
    return r


def main():
    args = [a for a in sys.argv[1:]]
    meta = None
    if "--row" in args:
        i = args.index("--row")
        r = read_row_from_dataset(int(args[i + 1]))
        args = args[i + 2:]
        if not args:
            args = [f"{r[s]:.2f}" for s in parse_header(HEADER)["SENSOR_PINS_ORDER"]]
        meta = (r["Cut"], int(r["Minute"]), float(r["TVC"]), int(r["Label"]))

    param = parse_header(HEADER)
    sensors = param["SENSOR_PINS_ORDER"]
    classes = param["CLASS_NAMES"]

    if not args:
        vals = list(param["TEST_VECTOR"])
        print("(tidak ada argumen -> memakai TEST_VECTOR dari model_params.h)")
    else:
        vals = [float(v.replace(",", ".")) for v in args]

    if len(vals) != len(sensors):
        sys.exit(f"Butuh {len(sensors)} nilai sensor, diterima {len(vals)}.")

    adc, xn, prob, tvc = python_reference(param, vals)
    pred = classes[int(np.argmax(prob))]
    pred_tvc_class = classes[int(np.digitize(tvc, param["TVC_THRESHOLDS"]))]

    print("\nNilai sensor (dan posisi potensiometer untuk Proteus):")
    print(f"  {'sensor':8s} {'nilai':>8s} {'ADC':>6s} {'potensiometer':>14s}")
    for name, v, a in zip(sensors, vals, adc):
        print(f"  {name:8s} {v:8.2f} {a:6d} {a/1023*100:13.1f} %")

    if meta:
        cut, minute, tvc_act, lab = meta
        print(f"\nData asal     : potongan {cut}, menit ke-{minute}")
        print(f"TVC aktual    : {tvc_act:.4f} log CFU/g  (kelas {lab} = "
              f"{classes[lab-1]})")

    print("\nHasil prediksi model:")
    for k, name in enumerate(classes):
        bar = "#" * int(round(prob[k] * 40))
        print(f"  {name:11s} {prob[k]*100:6.2f} %  {bar}")
    print(f"\n  Kategori (softmax)      : {pred}")
    print(f"  Estimasi TVC            : {tvc:.4f} log10 CFU/g")
    print(f"  Kategori (threshold TVC): {pred_tvc_class}")
    print("\nNilai di atas adalah yang seharusnya tampil di LCD Proteus.")


if __name__ == "__main__":
    main()
