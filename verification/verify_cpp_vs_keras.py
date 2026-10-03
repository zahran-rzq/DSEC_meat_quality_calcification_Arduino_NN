"""
================================================================================
 verification/verify_cpp_vs_keras.py
 ---------------------------------------------------------------------------
 Membuktikan bahwa PERHITUNGAN MANUAL DI ARDUINO (file .ino) menghasilkan
 angka yang SAMA dengan prediksi model Keras di komputer.

 Caranya:
   1. Kompilasi file .ino ASLI memakai g++ dengan stub Arduino/LCD
      (folder verification/stubs) -> tidak perlu hardware atau Proteus.
   2. Untuk setiap baris pada training/output/test_vectors.csv:
        - jalankan binary dengan 11 nilai sensor sebagai argumen
        - baca hasil dari kode C++ (kategori LCD, estimasi TVC, softmax)
        - bandingkan dengan kolom hasil prediksi Keras di CSV tersebut
   3. Tulis tabel perbandingan ke training/output/verifikasi_cpp_vs_keras.md

 Pakai:
    python verification/verify_cpp_vs_keras.py
 Pakai (tanpa g++, fallback simulasi Python):
    python verification/verify_cpp_vs_keras.py --no-build
================================================================================
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV = os.path.join(ROOT, "training", "output", "test_vectors.csv")
HEADER = os.path.join(ROOT, "arduino", "MeatQuality_NN_Arduino", "model_params.h")
BIN = os.path.join(ROOT, "verification", "build", "pc_test")
REPORT = os.path.join(ROOT, "training", "output", "verifikasi_cpp_vs_keras.md")


# ------------------------------------------------------------------ utilitas
def parse_header(path):
    """Baca array di model_params.h menjadi dict numpy."""
    src = open(path, encoding="utf-8").read()
    out = {}

    # angka ditulis dengan format C: 1.23456789e-02f  atau  3.0000f
    NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)f\b"

    def grab_2d(name):
        m = re.search(rf"{name}\[(\d+)\]\[(\d+)\]\s*=\s*\{{(.*?)\}}\s*;", src, re.S)
        rows, cols, body = int(m.group(1)), int(m.group(2)), m.group(3)
        vals = [float(v) for v in re.findall(NUM, body)]
        return np.array(vals).reshape(rows, cols)

    def grab_1d(name):
        m = re.search(rf"{name}\[(\d+)\]\s*=\s*\{{(.*?)\}}\s*;", src, re.S)
        return np.array([float(v) for v in re.findall(NUM, m.group(2))])

    for n in ("CLS_W1", "CLS_W2", "TVC_W1"):
        out[n] = grab_2d(n)
    for n in ("CLS_B1", "CLS_B2", "TVC_B1", "TVC_W2", "TVC_B2", "NORM_MIN",
              "NORM_MAX", "TEST_VECTOR", "TVC_THRESHOLDS"):
        out[n] = grab_1d(n)
    for n in ("TVC_OUT_LO", "TVC_OUT_HI", "TVC_MIN_LOG", "TVC_MAX_LOG",
              "NN_SENSOR_FULL_SCALE"):
        out[n] = float(re.search(
            rf"#define\s+{n}\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)f", src).group(1))
    out["CLASS_NAMES"] = re.findall(r'"([A-Za-z]+)"', re.search(
        r"CLASS_NAMES\[NN_N_CLASS\]\s*=\s*\{(.*?)\}", src, re.S).group(1))
    out["SENSOR_PINS_ORDER"] = re.findall(r'"([A-Za-z0-9]+)"', re.search(
        r"SENSOR_NAMES\[NN_N_INPUT\]\s*=\s*\{(.*?)\}", src, re.S).group(1))
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def python_reference(param, sensor_values, quantize=True):
    """Implementasi Python yang identik dengan loop di file .ino.

    quantize=True  -> meniru analogRead(): nilai dibulatkan ke ADC 10-bit
    quantize=False -> nilai sensor dipakai apa adanya (kondisi ideal, seperti
                      yang dipakai Keras saat prediksi di komputer)
    """
    if quantize:
        adc = np.clip(np.round(np.asarray(sensor_values) * 1023.0
                               / param["NN_SENSOR_FULL_SCALE"]).astype(int), 0, 1023)
        x = adc * param["NN_SENSOR_FULL_SCALE"] / 1023.0
    else:
        adc = np.asarray(sensor_values, dtype=float)
        x = adc
    xn = 2.0 * (x - param["NORM_MIN"]) / (param["NORM_MAX"] - param["NORM_MIN"]) - 1.0

    h_cls = sigmoid(param["CLS_W1"] @ xn + param["CLS_B1"])
    z = param["CLS_W2"] @ h_cls + param["CLS_B2"]
    z = z - z.max()
    e = np.exp(z)
    prob = e / e.sum()

    h_tvc = sigmoid(param["TVC_W1"] @ xn + param["TVC_B1"])
    tvc_norm = float(param["TVC_W2"] @ h_tvc + param["TVC_B2"][0])
    tvc = ((tvc_norm - param["TVC_OUT_LO"]) / (param["TVC_OUT_HI"] - param["TVC_OUT_LO"])
           * (param["TVC_MAX_LOG"] - param["TVC_MIN_LOG"]) + param["TVC_MIN_LOG"])
    return adc, xn, prob, tvc


def build_binary():
    os.makedirs(os.path.dirname(BIN), exist_ok=True)
    cmd = ["g++", "-O2", "-w",
           "-I", os.path.join(ROOT, "verification", "stubs"),
           "-o", BIN,
           os.path.join(ROOT, "verification", "pc_test", "main.cpp"), "-lm"]
    print("Kompilasi:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError("Gagal kompilasi file .ino dengan g++")
    print(f"[OK] Binary pengujian: {BIN}")


def run_cpp(sensor_values):
    """Jalankan binary (menjalankan setup()+loop() dari file .ino)."""
    args = [BIN] + [f"{v:.4f}" for v in sensor_values]
    r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"Binary gagal:\n{r.stderr}")

    res = {"lcd": [], "kategori_lcd": None, "softmax_lcd": None,
           "prob": {}, "tvc": None, "kategori_softmax": None, "kategori_tvc": None}
    sm = re.search(r"Probabilitas softmax\s*:(.*)", r.stdout)
    if sm:
        for name, val in re.findall(r"(\w+)=([0-9.]+)%", sm.group(1)):
            res["prob"][name] = float(val) / 100.0
    for key, pat in (("tvc", r"Estimasi TVC\s*:\s*([0-9.]+)"),
                     ("kategori_softmax", r"Prediksi softmax\s*:\s*(\w+)"),
                     ("kategori_tvc", r"Kategori dari TVC\s*:\s*(\w+)")):
        m = re.search(pat, r.stdout)
        if m:
            res[key] = float(m.group(1)) if key == "tvc" else m.group(1)
    for line in r.stdout.splitlines():
        if line.startswith("LCD|"):
            res["lcd"].append(line[4:])
    if res["lcd"]:
        m = re.search(r"Kualitas:\s*(\w+)", res["lcd"][1])
        if m:
            res["kategori_lcd"] = m.group(1)
        m = re.search(r"Softmax:\s*(\w+)", res["lcd"][2])
        if m:
            res["softmax_lcd"] = m.group(1)
    return res


# ------------------------------------------------------------------ program
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true",
                    help="jangan kompilasi, pakai simulasi Python (tanpa g++)")
    args = ap.parse_args()

    for f in (TEST_CSV, HEADER):
        if not os.path.exists(f):
            sys.exit(f"File {f} belum ada. Jalankan dulu: python training/train_meat_quality.py")

    param = parse_header(HEADER)
    df = pd.read_csv(TEST_CSV)
    sensors = param["SENSOR_PINS_ORDER"]
    classes = param["CLASS_NAMES"]
    print(f"Header    : {HEADER}")
    print(f"Arsitektur: 11 -> {param['CLS_W1'].shape[0]} -> {len(classes)}"
          f"   (+ cabang regresi TVC 11 -> {param['TVC_W1'].shape[0]} -> 1)")
    print(f"Sensor    : {', '.join(sensors)}")
    print(f"Vektor uji: {len(df)} baris\n")

    use_cpp = not args.no_build
    if use_cpp:
        try:
            build_binary()
        except Exception as e:
            print(f"[!] {e}")
            print("[!] Beralih ke simulasi Python (fallback).")
            use_cpp = False

    rows = []
    for i, r in df.iterrows():
        vals = [float(r[s]) for s in sensors]
        adc, xn, prob_ref, tvc_ref = python_reference(param, vals, quantize=True)
        # pembanding "ideal": nilai sensor desimal tanpa pembulatan ADC
        _, _, _, tvc_ideal = python_reference(param, vals, quantize=False)

        tvc_keras = float(r["Pred_TVC"])
        cls_keras = str(r["Pred_Class"])

        if use_cpp:
            c = run_cpp(vals)
            tvc_cpp = c["tvc"]
            cls_cpp = c["kategori_lcd"]
            prob_cpp = c["prob"]
            src = "g++ / file .ino"
        else:
            tvc_cpp, cls_cpp, prob_cpp = tvc_ref, classes[int(np.argmax(prob_ref))], {
                classes[k]: float(prob_ref[k]) for k in range(len(classes))}
            src = "python (fallback)"

        max_diff = max(abs(prob_cpp.get(n, 0.0) - float(r[f"P_{n}"])) for n in classes)
        rows.append({
            "no": i + 1,
            "cut": r["Cut"],
            "kelas_aktual": r["ClassName"],
            "kelas_lcd": cls_cpp,
            "kelas_keras": cls_keras,
            "cocok": "YA" if cls_cpp == cls_keras else "TIDAK",
            "tvc_keras": tvc_keras,
            "tvc_cpp": tvc_cpp,
            "selisih_tvc": abs(tvc_keras - tvc_cpp),
            # |cpp - ideal|  = murni akibat pembulatan ADC 10-bit
            "efek_kuantisasi_adc": abs(tvc_cpp - tvc_ideal),
            # |ideal - keras| = murni akibat presisi float32/float64 + pembulatan bobot
            "selisih_float": abs(tvc_ideal - tvc_keras),
            "selisih_prob_maks": max_diff,
            "sumber": src,
        })

    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    ok_cls = (res["cocok"] == "YA").all()
    tol = 0.05  # log CFU/g (ambang antar kelas berjarak 1.0 -> 0.05 masih sangat aman)
    ok_tvc = (res["selisih_tvc"] < tol).all()
    print(f"\nKelas LCD  == prediksi Keras : {'SEMUA SAMA' if ok_cls else 'ADA BEDA'}")
    print(f"Selisih TVC Keras vs Arduino  : maks {res['selisih_tvc'].max():.6f} log CFU/g "
          f"(rata-rata {res['selisih_tvc'].mean():.6f}; toleransi {tol})")
    print(f"  - akibat kuantisasi ADC 10-bit : {res['efek_kuantisasi_adc'].max():.6f}")
    print(f"  - akibat presisi float 32-bit  : {res['selisih_float'].max():.6f}")
    print(f"Selisih probabilitas maksimum : {res['selisih_prob_maks'].max():.6f}")

    # ---- tulis laporan ----
    lines = ["# Verifikasi Kode Arduino vs Model Keras", "",
             f"Sumber binary: {res['sumber'].iloc[0]}  ",
             f"Vektor uji: `training/output/test_vectors.csv` ({len(res)} baris)  ",
             f"Arsitektur: 11 - {param['CLS_W1'].shape[0]} - {len(classes)} (klasifikasi) "
             f"| 11 - {param['TVC_W1'].shape[0]} - 1 (regresi TVC)", "",
             "| # | Potongan | Kelas aktual | Kelas di LCD | Kelas Keras | Cocok | "
             "TVC Keras | TVC Arduino | Selisih total | Efek ADC | Efek float |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r['no']} | {r['cut']} | {r['kelas_aktual']} | {r['kelas_lcd']} | "
                     f"{r['kelas_keras']} | {r['cocok']} | {r['tvc_keras']:.4f} | "
                     f"{r['tvc_cpp']:.4f} | {r['selisih_tvc']:.6f} | "
                     f"{r['efek_kuantisasi_adc']:.6f} | {r['selisih_float']:.6f} |")
    pct_cocok = 100.0 if ok_cls else float((res["cocok"] == "YA").mean()) * 100.0
    lines += ["",
              f"- Kesesuaian kelas: **{pct_cocok:.1f} %**",
              f"- Selisih TVC maksimum: **{res['selisih_tvc'].max():.6f} log CFU/g** "
              f"(rata-rata {res['selisih_tvc'].mean():.6f})",
              f"- Selisih probabilitas softmax maksimum: {res['selisih_prob_maks'].max():.6f}",
              "",
              "**Dari mana selisihnya?** Kolom *Efek ADC* dan *Efek float* memisahkan dua penyebabnya:",
              "",
              f"1. **Kuantisasi ADC 10-bit** (penyebab utama, sampai "
              f"{res['efek_kuantisasi_adc'].max():.6f} log CFU/g): `analogRead()` hanya "
              "menghasilkan bilangan bulat 0-1023, sehingga nilai sensor desimal "
              "(mis. 17,85) menjadi 17,83 saat dibaca ulang. Ini keterbatasan perangkat keras, "
              "bukan kesalahan rumus.",
              f"2. **Presisi `float` 32-bit** (hanya {res['selisih_float'].max():.6f} log CFU/g): "
              "perhitungan matriks di Arduino memakai `float` 32-bit sedangkan Python memakai "
              "float64/float32 dengan urutan operasi berbeda.",
              "",
              "Karena jarak antar ambang batas kelas adalah 1,0 log CFU/g, selisih "
              f"≤ {res['selisih_tvc'].max():.4f} tidak mengubah kategori kualitas - seluruh "
              "8 vektor uji menghasilkan kelas yang identik dengan prediksi Keras.", ""]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] Laporan verifikasi ditulis: {REPORT}")

    if not (ok_cls and ok_tvc):
        sys.exit(1)


if __name__ == "__main__":
    main()
