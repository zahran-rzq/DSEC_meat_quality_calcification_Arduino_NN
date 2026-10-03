#!/usr/bin/env python3
"""
Pelatihan Neural Network untuk klasifikasi kualitas daging (Beef Quality Classification)
dari data Electronic Nose (11 sensor MQ).

Pipeline (sesuai langkah pengerjaan tugas):
  1. Integrasi data   : gabungkan SELURUH 12 sheet (12 jenis potongan daging) jadi satu dataset
  2. Fitur (X)        : 11 sensor MQ  -> MQ135, MQ136, MQ137, MQ138, MQ2, MQ3, MQ4, MQ5, MQ6, MQ8, MQ9
     Target (y)       : kolom `Label` (1=excellent, 2=good, 3=acceptable, 4=spoiled)
  3. Normalisasi      : Min-Max PER FITUR ke rentang [-1, +1]; parameter (min/max) DISIMPAN
  4. Training         : pencarian arsitektur hidden layer, lalu training model final
  5. Evaluasi         : akurasi, classification report, confusion matrix
  6. Ekstraksi bobot  : Weight & Bias setiap layer diekspor jadi header C++ siap pakai Arduino

Pemakaian:
    python training/train_bqc.py                 # normal: search arsitektur + train final + export
    python training/train_bqc.py --skip-search   # pakai arsitektur default tanpa pencarian
    python training/train_bqc.py --export-only   # hanya ekstrak bobot dari model .keras yang ada
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------- konfigurasi
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATASET = os.path.join(ROOT, "e-nose_dataset_12_beef_cuts.xlsx")
OUT_DIR = os.path.join(ROOT, "arduino", "BQC_Arduino_Mega")
DOCS_DIR = os.path.join(ROOT, "docs")

SENSOR_NAMES = ["MQ135", "MQ136", "MQ137", "MQ138", "MQ2",
                "MQ3", "MQ4", "MQ5", "MQ6", "MQ8", "MQ9"]

# Label di dataset berupa INTEGER 1..4. Urutan one-hot dari pd.get_dummies mengikuti
# urutan menaik label, jadi indeks kelas == label-1.
CLASS_LABELS = [1, 2, 3, 4]
CLASS_NAMES = ["excellent", "good", "acceptable", "spoiled"]

RANDOM_STATE = 42
ADC_MAX = 1023.0  # resolusi analogRead Arduino (10-bit)


# ----------------------------------------------------------------------------- langkah 1: data
def load_integrated_dataset(path: str = DATASET) -> pd.DataFrame:
    """Gabungkan seluruh 12 sheet (12 potongan daging) menjadi satu kesatuan data."""
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df.insert(0, "Cut", sheet)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"[data] {len(xl.sheet_names)} sheet digabung -> {data.shape[0]} baris, "
          f"{data.shape[1]} kolom")
    return data


def build_xy(data: pd.DataFrame):
    """X = 11 sensor MQ, y = Label (one-hot 4 kelas)."""
    X_raw = data[SENSOR_NAMES].to_numpy(dtype=np.float32)
    y_int = data["Label"].to_numpy(dtype=np.int64)
    Y = np.zeros((len(y_int), len(CLASS_LABELS)), dtype=np.float32)
    for i, lab in enumerate(CLASS_LABELS):
        Y[y_int == lab, i] = 1.0
    print(f"[data] X: {X_raw.shape}   Y: {Y.shape}")
    return X_raw, Y, y_int


# ----------------------------------------------------------------------------- langkah 1: scaling
def fit_minmax(X: np.ndarray):
    """Parameter scaling Min-Max PER FITUR (disimpan untuk dipakai di Arduino)."""
    return X.min(axis=0).astype(np.float64), X.max(axis=0).astype(np.float64)


def normalize(X: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    """Skalakan tiap fitur ke [-1, +1]."""
    span = (xmax - xmin)
    span = np.where(span == 0, 1.0, span)          # hindari bagi nol
    return 2.0 * (X - xmin) / span - 1.0


# ----------------------------------------------------------------------------- langkah 2: model
def build_model(arch: tuple[int, ...], n_input: int = len(SENSOR_NAMES),
                n_class: int = len(CLASS_NAMES)) -> tf.keras.Model:
    """Sequential MLP: hidden layer(s) sigmoid -> output softmax."""
    layers = []
    prev = n_input
    for units in arch:
        layers.append(tf.keras.layers.Dense(units, activation="sigmoid", input_dim=prev))
        prev = units
    layers.append(tf.keras.layers.Dense(n_class, activation="softmax"))
    return tf.keras.models.Sequential(layers)


def compile_model(model: tf.keras.Model) -> None:
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                  metrics=["accuracy"])


def fit_model(model, Xtr, Ytr, Xte, Yte, epochs=1500, class_weight=None, verbose=2):
    batch_size = max(32, int(0.01 * len(Xtr)))
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.1,
                                             patience=20, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=40,
                                         restore_best_weights=True),
    ]
    return model.fit(Xtr, Ytr, epochs=epochs, batch_size=batch_size,
                     validation_data=(Xte, Yte), callbacks=cbs,
                     class_weight=class_weight, verbose=verbose)


def architecture_search(Xtr, Ytr, Xte, Yte, epochs=120):
    """Coba beberapa arsitektur, pilih yang val_accuracy terbaik (langkah 2: arsitektur optimal)."""
    candidates = [(20,), (24,), (32,), (24, 12), (32, 16)]
    results = []
    print("\n[search] pencarian arsitektur hidden layer "
          f"({epochs} epoch per kandidat) ...")
    for arch in candidates:
        m = build_model(arch)
        compile_model(m)
        hist = fit_model(m, Xtr, Ytr, Xte, Yte, epochs=epochs, verbose=0)
        acc = float(np.max(hist.history["val_accuracy"]))
        results.append({"arch": list(arch), "val_accuracy": acc})
        print(f"[search]   hidden={str(list(arch)):10s} val_accuracy={acc:.4f}")
    results.sort(key=lambda r: -r["val_accuracy"])
    return results


# ----------------------------------------------------------------------------- util metrik
def merge_metrics(info: dict) -> dict:
    """Perbarui docs/metrics.json tanpa menghapus kunci yang sudah ada."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    mpath = os.path.join(DOCS_DIR, "metrics.json")
    merged = {}
    if os.path.exists(mpath):
        try:
            with open(mpath) as f:
                merged = json.load(f)
        except (json.JSONDecodeError, OSError):
            merged = {}
    merged.update(info)
    with open(mpath, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


# ----------------------------------------------------------------------------- langkah 3: ekspor
def _fmt_arr(values: np.ndarray, per_line: int, indent: str = "  ") -> str:
    lines = []
    flat = np.asarray(values).reshape(-1)
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        lines.append(indent + ", ".join(f"{v: .8f}f" for v in chunk) + ",")
    if lines:
        lines[-1] = lines[-1].rstrip(",")
    return "\n".join(lines)


def export_header(model, xmin, xmax, metrics_info, out_path):
    """Ekstraksi Weight & Bias setiap layer -> array C++ siap dipakai Arduino."""
    dense = [l for l in model.layers if hasattr(l, "units")]
    params = [l.get_weights() for l in dense]          # [(W,B), (W,B), ...]
    acts = [(l.activation.__name__ if hasattr(l.activation, "__name__")
             else str(l.activation)) for l in dense]

    n_in = params[0][0].shape[0]
    n_out = params[-1][0].shape[-1]
    hidden = [W.shape[1] for W, _ in params[:-1]]
    max_width = max(hidden + [n_in, n_out])
    n_layers = len(params)

    arch_str = " -> ".join([str(n_in)] + [str(h) for h in hidden] + [str(n_out)])
    act_str = " -> ".join(acts)

    hdr = []
    hdr.append("/* =====================================================================")
    hdr.append(" * nn_params.h  --  BOBOT & BIAS HASIL TRAINING (JANGAN DIUBAH MANUAL)")
    hdr.append(" *")
    hdr.append(" * Tugas   : Klasifikasi kualitas daging sapi dari Electronic Nose")
    hdr.append(" * Dataset : e-nose_dataset_12_beef_cuts.xlsx (12 potongan daging digabung)")
    hdr.append(f" * Dibuat  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    hdr.append(f" * Tool    : TensorFlow {tf.__version__}")
    hdr.append(" *")
    hdr.append(f" * Arsitektur : {arch_str}")
    hdr.append(f" * Aktivasi   : {act_str}")
    if "val_accuracy" in metrics_info:
        hdr.append(f" * Val. accuracy : {metrics_info['val_accuracy']:.4f}")
    if "test_accuracy" in metrics_info:
        hdr.append(f" * Test accuracy : {metrics_info['test_accuracy']:.4f}")
    hdr.append(" *")
    hdr.append(" * File ini dibangkitkan otomatis oleh training/train_bqc.py")
    hdr.append(" * ===================================================================== */")
    hdr.append("")
    hdr.append("#ifndef NN_PARAMS_H")
    hdr.append("#define NN_PARAMS_H")
    hdr.append("")
    hdr.append(f"#define NN_INPUTS    {n_in}")
    hdr.append(f"#define NN_OUTPUTS   {n_out}")
    hdr.append(f"#define NN_N_LAYERS  {n_layers}")
    hdr.append(f"#define NN_MAX_WIDTH {max_width}   /* buffer kerja terbesar */")
    hdr.append("")
    hdr.append("/* Jenis fungsi aktivasi (dipakai tabel LAYER_ACT di bawah) */")
    hdr.append("#ifndef NN_ACT_SIGMOID")
    hdr.append("#define NN_ACT_SIGMOID 1")
    hdr.append("#endif")
    hdr.append("#ifndef NN_ACT_SOFTMAX")
    hdr.append("#define NN_ACT_SOFTMAX 2")
    hdr.append("#endif")
    hdr.append("")
    hdr.append("/* Nama sensor (urutan pin analog A0..A10) */")
    hdr.append("const char *const SENSOR_NAMES[NN_INPUTS] = {")
    hdr.append("  " + ", ".join(f'"{n}"' for n in SENSOR_NAMES))
    hdr.append("};")
    hdr.append("")
    hdr.append("/* ------------------------------------------------------------------")
    hdr.append(" * PARAMETER NORMALISASI Min-Max PER FITUR (dari langkah pre-processing)")
    hdr.append(" *   x_norm = 2*(x - X_MIN[i]) / (X_MAX[i] - X_MIN[i]) - 1")
    hdr.append(" * ------------------------------------------------------------------ */")
    hdr.append("const float X_MIN[NN_INPUTS] = {")
    hdr.append(_fmt_arr(xmin, 4))
    hdr.append("};")
    hdr.append("const float X_MAX[NN_INPUTS] = {")
    hdr.append(_fmt_arr(xmax, 4))
    hdr.append("};")
    hdr.append("")

    # ---- bobot & bias tiap layer sebagai array 2D eksplisit
    dims = [n_in] + hidden + [n_out]
    for li, (W, B) in enumerate(params, start=1):
        ni, no = W.shape
        hdr.append("/* ------------------------------------------------------------------")
        hdr.append(f" * LAYER {li} : Weight [{ni} x {no}]   Bias [{no}]   "
                   f"aktivasi = {acts[li - 1]}")
        hdr.append(" * ------------------------------------------------------------------ */")
        hdr.append(f"const float W{li}[{ni}][{no}] = {{")
        for r in range(ni):
            hdr.append("  {" + ", ".join(f"{v: .8f}f" for v in W[r]) + "},")
        hdr.append("};")
        hdr.append(f"const float B{li}[{no}] = {{")
        hdr.append(_fmt_arr(B, 5))
        hdr.append("};")
        hdr.append("")

    # ---- tabel layer agar nn_core.h bisa melakukan forward pass generik
    hdr.append("/* ------------------------------------------------------------------")
    hdr.append(" * TABEL LAYER -- dipakai nnForward() untuk berjalan generik.")
    hdr.append(" * Array 2D C bersifat row-major sehingga &Wk[0][0] dapat dipakai")
    hdr.append(" * sebagai pointer matriks berukuran LAYER_IN[k] x LAYER_OUT[k].")
    hdr.append(" * ------------------------------------------------------------------ */")
    hdr.append(f"const float *const LAYER_W[NN_N_LAYERS] = {{ "
               + ", ".join(f"&W{i}[0][0]" for i in range(1, n_layers + 1)) + " };")
    hdr.append(f"const float *const LAYER_B[NN_N_LAYERS] = {{ "
               + ", ".join(f"B{i}" for i in range(1, n_layers + 1)) + " };")
    hdr.append("const int LAYER_IN[NN_N_LAYERS]  = { "
               + ", ".join(str(dims[i]) for i in range(n_layers)) + " };")
    hdr.append("const int LAYER_OUT[NN_N_LAYERS] = { "
               + ", ".join(str(dims[i + 1]) for i in range(n_layers)) + " };")
    hdr.append("const int LAYER_ACT[NN_N_LAYERS] = { "
               + ", ".join("NN_ACT_SOFTMAX" if a == "softmax" else "NN_ACT_SIGMOID"
                           for a in acts) + " };")
    hdr.append("")
    hdr.append("/* Nama kelas (indeks = Label dataset - 1) */")
    hdr.append("const char *const CLASS_NAMES[NN_OUTPUTS] = {")
    hdr.append("  " + ", ".join(f'"{n}"' for n in CLASS_NAMES))
    hdr.append("};")
    hdr.append("const int CLASS_LABEL[NN_OUTPUTS] = {"
               + ", ".join(str(l) for l in CLASS_LABELS) + "};")
    hdr.append("")
    hdr.append("#endif  /* NN_PARAMS_H */")
    hdr.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(hdr))
    print(f"[export] bobot & bias ditulis ke {os.path.relpath(out_path, ROOT)}")
    return dims



# ----------------------------------------------------------------------------- langkah 5: verifikasi
def export_test_vectors(model, data, X_norm, y_int, out_csv, n_per_class=3):
    """
    Ambil beberapa baris dataset sebagai kasus uji, ubah nilai sensor menjadi
    nilai ADC 0..1023 yang harus di-set pada potensiometer di Proteus.
    """
    xmin, xmax = fit_minmax(data[SENSOR_NAMES].to_numpy(dtype=np.float32))
    span = np.where((xmax - xmin) == 0, 1.0, xmax - xmin)

    rows = []
    for lab in CLASS_LABELS:
        idx = np.where(y_int == lab)[0]
        rng = np.random.default_rng(RANDOM_STATE)
        pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        rows.extend(sorted(pick))

    prob = model.predict(X_norm[rows], verbose=0)
    lines = ["case_id,cut,minute,actual_label,actual_class,adc_" +
             ",adc_".join(SENSOR_NAMES) +
             "," + ",".join(f"p_{c}" for c in CLASS_NAMES) +
             ",pred_class,confidence"]
    for k, r in enumerate(rows, start=1):
        raw = data.iloc[r]
        vals = raw[SENSOR_NAMES].to_numpy(dtype=np.float64)
        adc = np.round((vals - xmin) / span * ADC_MAX).astype(int)
        p = prob[k - 1]
        pred = int(np.argmax(p))
        lines.append(
            f"{k},{raw['Cut']},{int(raw['Minute'])},{int(raw['Label'])},"
            f"{CLASS_NAMES[int(raw['Label']) - 1]},"
            + ",".join(str(int(a)) for a in adc) + ","
            + ",".join(f"{v:.6f}" for v in p) + ","
            f"{CLASS_NAMES[pred]},{p[pred]:.6f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[export] {len(rows)} kasus uji ditulis ke {os.path.relpath(out_csv, ROOT)}")
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-search", action="store_true",
                    help="lewati pencarian arsitektur, pakai --arch (default 20)")
    ap.add_argument("--arch", type=str, default="20",
                    help="arsitektur hidden layer, pisahkan dengan koma, mis. '24,12'")
    ap.add_argument("--search-only", action="store_true",
                    help="hanya jalankan pencarian arsitektur, simpan hasilnya ke "
                         "docs/metrics.json, tanpa melatih/menimpa model final")
    ap.add_argument("--search-epochs", type=int, default=120,
                    help="jumlah epoch per kandidat saat pencarian arsitektur")
    ap.add_argument("--export-only", metavar="KERAS",
                    help="hanya ekspor bobot dari file .keras yang sudah ada")
    ap.add_argument("--report-only", metavar="KERAS",
                    help="hitung ulang metrik test dari .keras tersimpan lalu "
                         "gabungkan ke docs/metrics.json (deterministik)")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--class-weight", action="store_true",
                    help="gunakan class_weight='balanced' untuk menangani ketidakseimbangan kelas")
    ap.add_argument("--vectors", type=int, default=4,
                    help="jumlah kasus uji per kelas untuk file test_vectors.csv")
    args = ap.parse_args()

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # --- load data
    data = load_integrated_dataset()
    X_raw, Y, y_int = build_xy(data)

    # --- scaling
    xmin, xmax = fit_minmax(X_raw)
    X = normalize(X_raw, xmin, xmax)
    print("[scale] rentang sensor setelah normalisasi: "
          f"[{X.min():.4f}, {X.max():.4f}]")

    # --- split
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"[split] train {Xtr.shape[0]}   test {Xte.shape[0]}")

    cw = None
    if args.class_weight:
        cnt = np.bincount(y_int, minlength=len(CLASS_LABELS)).astype(np.float64)
        w = cnt.sum() / (len(CLASS_LABELS) * cnt)
        cw = {i: float(w[i]) for i in range(len(CLASS_LABELS))}
        print(f"[class weight] {cw}")

    if args.search_only:
        results = architecture_search(Xtr, Ytr, Xte, Yte, epochs=args.search_epochs)
        merge_metrics({"search": results,
                       "search_epochs": args.search_epochs,
                       "search_generated": datetime.now().isoformat()})
        print(f"[search] hasil disimpan ke docs/metrics.json")
        return

    if args.report_only:
        model = tf.keras.models.load_model(args.report_only)
        arch = [int(l.units) for l in model.layers if hasattr(l, "units")]
        loss, acc = model.evaluate(Xte, Yte, verbose=0)
        pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
        truth = np.argmax(Yte, axis=1)
        report = metrics.classification_report(truth, pred, target_names=CLASS_NAMES,
                                               digits=4, zero_division=0)
        cm = metrics.confusion_matrix(truth, pred)
        print("\n=== Confusion Matrix (test) ===\n" + str(cm))
        print("\n=== Classification Report (test) ===\n" + report)
        merge_metrics({
            "generated": datetime.now().isoformat(),
            "tf_version": tf.__version__,
            "scaling": {"type": "minmax_per_feature", "target": [-1, 1],
                        "X_MIN": xmin.tolist(), "X_MAX": xmax.tolist()},
            "model_file": os.path.basename(args.report_only),
            "arch": arch,
            "val_accuracy": float(acc),
            "test_accuracy": float(acc),
            "test_loss": float(loss),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
            "n_sheets": 12, "n_rows": int(len(data)),
        })
        print("[report] metrik diperbarui di docs/metrics.json")
        return

    if args.export_only:
        model = tf.keras.models.load_model(args.export_only)
        print(f"[export-only] model dimuat dari {args.export_only}")
        arch = [int(l.units) for l in model.layers if hasattr(l, "units")]
        metrics_info = {"arch": arch,
                        "model_file": os.path.basename(args.export_only)}
    else:
        # --- langkah 2: arsitektur
        if args.skip_search:
            best_arch = tuple(int(v) for v in args.arch.split(",") if v.strip())
            search_results = []
            print(f"[arch] memakai arsitektur dari argumen: {best_arch}")
        else:
            search_results = architecture_search(Xtr, Ytr, Xte, Yte,
                                                 epochs=args.search_epochs)
            best_arch = tuple(search_results[0]["arch"])
            print(f"[search] arsitektur terpilih: {best_arch}")

        # --- training final
        model = build_model(best_arch)
        compile_model(model)
        t0 = time.time()
        hist = fit_model(model, Xtr, Ytr, Xte, Yte, epochs=args.epochs,
                         class_weight=cw, verbose=2)
        print(f"[train] waktu training: {time.time() - t0:.1f} s, "
              f"epoch terpakai: {len(hist.history['loss'])}")

        val_acc = float(np.max(hist.history["val_accuracy"]))
        test_loss, test_acc = model.evaluate(Xte, Yte, verbose=0)

        # --- evaluasi
        pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
        truth = np.argmax(Yte, axis=1)
        report = metrics.classification_report(truth, pred, target_names=CLASS_NAMES,
                                               digits=4, zero_division=0)
        cm = metrics.confusion_matrix(truth, pred)
        print("\n=== Confusion Matrix (test) ===")
        print(cm)
        print("\n=== Classification Report (test) ===")
        print(report)

        metrics_info = {
            "arch": list(best_arch),
            "val_accuracy": val_acc,
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "epochs_used": len(hist.history["loss"]),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "search": search_results,
            "class_weight": cw,
            "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0]),
            "n_sheets": 12,
            "n_rows": int(len(data)),
        }

        os.makedirs(OUT_DIR, exist_ok=True)
        keras_path = os.path.join(OUT_DIR, "BQCmodel_12cuts.keras")
        model.save(keras_path)
        print(f"[save] model disimpan ke {os.path.relpath(keras_path, ROOT)}")

    # --- langkah 3: ekstraksi bobot -> header C++
    export_header(model, xmin, xmax, metrics_info,
                  os.path.join(OUT_DIR, "nn_params.h"))

    # --- kasus uji untuk Proteus + verifikasi host
    rows = export_test_vectors(model, data, X, y_int,
                               os.path.join(DOCS_DIR, "test_vectors.csv"),
                               n_per_class=max(1, args.vectors))

    # --- snapshot metrik untuk laporan (merge, bukan timpa)
    merge_metrics({"generated": datetime.now().isoformat(),
                   "tf_version": tf.__version__,
                   "scaling": {"type": "minmax_per_feature", "target": [-1, 1],
                               "X_MIN": xmin.tolist(), "X_MAX": xmax.tolist()},
                   **metrics_info})
    print(f"[save] metrik diperbarui di docs/metrics.json")
    print("\nSelesai.")


if __name__ == "__main__":
    main()
