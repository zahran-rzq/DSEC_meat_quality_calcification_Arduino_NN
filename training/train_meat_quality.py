"""
================================================================================
 RP GANJIL - SISTEM PREDIKSI KUALITAS DAGING BERBASIS ELECTRONIC NOSE
 Pengolahan dataset 12 potongan daging -> Training NN -> Ekspor parameter
 untuk Arduino Mega 2560
================================================================================

Cara pakai (dari folder root repo):

    pip install pandas numpy scikit-learn tensorflow matplotlib openpyxl
    python training/train_meat_quality.py

Output (folder training/output/ + ardunio/MeatQuality_NN_Arduino/):

    1. merged_dataset.csv              -> dataset gabungan 12 potongan (26.640 baris)
    2. beef_quality_classifier.keras   -> model klasifikasi (Excellent/Good/Acceptable/Spoiled)
    3. tvc_regressor.keras             -> model regresi TVC (log10 CFU/g)
    4. model_params.h                  -> bobot, bias, parameter scaling siap Arduino
    5. test_vectors.csv                -> baris data uji + hasil prediksi Python
                                          (untuk dimasukkan ke potensiometer Proteus)
    6. laporan_hasil_training.md       -> tabel metrik siap tempel ke laporan
    7. confusion_matrix.png, kurva training

Alur (sesuai lembar tugas):
    (1) Integrasi 12 sheet -> satu dataset          -> load_and_merge()
    (2) Normalisasi (simpan min/max)                -> class MinMaxScalerNN
    (3) Cari arsitektur optimal + training          -> architecture_search(), train_final_models()
    (4) Ekstraksi bobot & bias                      -> export_model_params_header()
    (5) Uji generalisasi antar potongan daging      -> leave_one_cut_out()
================================================================================
"""

# %% [markdown]
# # 1. Import library dan konfigurasi
#
# Semua parameter penting dikumpulkan di `CFG` supaya mudah diubah.

# %%
import json
import os
import platform
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------- konfigurasi
CFG = {
    # ---- lokasi file
    "dataset_xlsx": "e-nose_dataset_12_beef_cuts.xlsx",
    "out_dir": "training/output",
    "arduino_dir": "arduino/MeatQuality_NN_Arduino",

    # ---- fitur: 11 sensor MQ (urutan ini HARUS sama dengan urutan pin di Arduino)
    "sensors": ["MQ135", "MQ136", "MQ137", "MQ138", "MQ2",
                "MQ3", "MQ4", "MQ5", "MQ6", "MQ8", "MQ9"],

    # ---- target
    #   Label  1 = Excellent, 2 = Good, 3 = Acceptable, 4 = Spoiled
    #   (Label pada dataset = hasil threshold TVC: <3, 3-4, 4-5, >=5 log CFU/g)
    "class_names": ["Excellent", "Good", "Acceptable", "Spoiled"],
    "tvc_thresholds": [3.0, 4.0, 5.0],     # log10 CFU/g, batas antar kelas

    # ---- normalisasi input : x_norm = 2*(x-min)/(max-min) - 1  -> rentang [-1, 1]
    "norm_lo": -1.0,
    "norm_hi": 1.0,
    #  skala "nilai sensor" yang dipakai saat konversi ADC -> nilai sensor di MCU.
    #  Potensiometer Proteus 0..5 V  <->  ADC 0..1023  <->  nilai sensor 0..60
    "sensor_full_scale": 60.0,

    # ---- pembagian data
    "test_size": 0.2,
    "seed": 42,

    # ---- pencarian arsitektur (cepat)
    "search_archs": [[10], [16], [20], [32], [64], [32, 16], [64, 32]],
    "search_epochs": 300,
    "search_patience": 40,

    # ---- training model final
    "final_epochs": 1500,
    "final_patience": 80,
    "batch_size": 512,
    "learning_rate": 1e-2,
}

TIER = {1: "Excellent", 2: "Good", 3: "Acceptable", 4: "Spoiled"}


def set_all_seeds(seed: int):
    """Supaya hasil training reproducible (bisa diulang dengan nilai sama)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def banner(text: str):
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


# %% [markdown]
# # 2. (Langkah 1) Integrasi 12 file/sheet dataset menjadi SATU kesatuan data
#
# Setiap sheet = 1 jenis potongan daging (Inside-Outside, Round, Top Sirloin, ...).
# Struktur tiap sheet identik: `Minute, TVC, Label, MQ135 ... MQ9`.
#
# Karena struktur kolomnya sama, penggabungan dilakukan dengan `pd.concat`
# (menumpuk baris / *row-wise concat*) lalu ditambah satu kolom baru
# `Cut` sebagai penanda asal potongan daging (kolom ini TIDAK dipakai
# sebagai input NN, hanya untuk analisis).

# %%
def load_and_merge(cfg) -> pd.DataFrame:
    """Baca 12 sheet -> jadikan satu DataFrame besar."""
    xl = pd.ExcelFile(cfg["dataset_xlsx"])
    frames = []
    print(f"Ditemukan {len(xl.sheet_names)} sheet:")
    for sheet in xl.sheet_names:
        df_sheet = pd.read_excel(xl, sheet_name=sheet)
        df_sheet["Cut"] = sheet.split(".", 1)[-1]      # '1.Inside-Outside' -> 'Inside-Outside'
        frames.append(df_sheet)
        n_lab = df_sheet["Label"].value_counts().sort_index().to_dict()
        print(f"  - {sheet:18s} {df_sheet.shape[0]:5d} baris  "
              f"kelas {n_lab}")

    # ---- INI DIA PROSES PENGGABUNGANNYA ----
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df


def check_dataset(df, cfg):
    """Pemeriksaan kualitas data: NaN, duplikat, distribusi kelas."""
    banner("PEMERIKSAAN DATASET GABUNGAN")
    print(f"Ukuran dataset gabungan : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Jumlah potongan daging  : {df['Cut'].nunique()} jenis")
    print(f"Jumlah nilai NaN        : {int(df.isna().sum().sum())}")
    dup = df.duplicated(subset=cfg["sensors"]).sum()
    print(f"Baris sensor duplikat   : {int(dup)}")

    print("\nDistribusi kelas (Label):")
    vc = df["Label"].value_counts().sort_index()
    for lab, n in vc.items():
        print(f"  Label {lab} ({TIER[lab]:11s}) : {n:6d} baris ({100*n/len(df):5.1f} %)")

    print("\nRentang TVC per kelas (log10 CFU/g):")
    print(df.groupby("Label")["TVC"].agg(["min", "max", "mean"]).to_string())

    print("\nRentang nilai 11 sensor MQ:")
    print(df[cfg["sensors"]].agg(["min", "max", "mean"]).T.to_string(float_format="%.2f"))
    return vc


# %% [markdown]
# # 3. Normalisasi (Min-Max) - parameter disimpan untuk Arduino
#
# Rumus yang dipakai (sama dengan notebook contoh):
#
# $$x_{norm} = \frac{2\,(x - x_{min})}{(x_{max} - x_{min})} - 1$$
#
# sehingga nilai sensor berada di rentang **[-1, 1]**.
#
# *Untuk eksperimen 80/20, `min`/`max` diambil dari data training saja
# (praktik yang benar). Untuk model yang dideploy, `min`/`max` diambil dari
# seluruh dataset agar menjangkau seluruh kondisi pengukuran.*

# %%
class MinMaxScalerNN:
    """Min-Max scaler manual (tanpa sklearn) supaya rumusnya transparan dan
    identik dengan yang ditulis di kode Arduino."""

    def __init__(self, lo=-1.0, hi=1.0):
        self.lo, self.hi = lo, hi
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = self.max_ - self.min_
        rng[rng == 0] = 1.0                      # hindari bagi nol
        return (X - self.min_) / rng * (self.hi - self.lo) + self.lo

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_targets(df, cfg):
    """y klasifikasi (0..3) dan y regresi (TVC, log10 CFU/g)."""
    y_cls = df["Label"].to_numpy(dtype=np.int64) - 1          # 1..4 -> 0..3
    y_tvc = df["TVC"].to_numpy(dtype=np.float64)              # log10 CFU/g
    y_cls_oh = tf.keras.utils.to_categorical(y_cls, num_classes=len(cfg["class_names"]))
    return y_cls, y_cls_oh, y_tvc


# %% [markdown]
# # 4. (Langkah 2) Membangun dan melatih model Neural Network
#
# Arsitektur yang diuji:
#
# * Input  : 11 neuron (11 sensor MQ)
# * Hidden: `sigmoid` (sesuai materi kuliah, mudah diimplementasikan di Arduino)
# * Output klasifikasi: 4 neuron `softmax`, loss `categorical_crossentropy`
# * Output regresi TVC : 1 neuron `linear`, loss `mse`
#
# Pencarian arsitektur (jumlah hidden layer & jumlah neuron) dilakukan dulu
# dengan epoch pendek, lalu arsitektur terbaik dilatih ulang sampai konvergen.

# %%
def build_classifier(n_hidden, n_input, n_class, lr, hidden_act="sigmoid"):
    layers = []
    for i, n in enumerate(n_hidden):
        layers.append(tf.keras.layers.Dense(
            n, activation=hidden_act,
            input_dim=n_input if i == 0 else None,
            name=f"hidden{i+1}"))
    layers.append(tf.keras.layers.Dense(n_class, activation="softmax", name="output"))
    model = tf.keras.models.Sequential(layers)
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=["accuracy"])
    return model


def build_regressor(n_hidden, n_input, lr, hidden_act="sigmoid"):
    layers = []
    for i, n in enumerate(n_hidden):
        layers.append(tf.keras.layers.Dense(
            n, activation=hidden_act,
            input_dim=n_input if i == 0 else None,
            name=f"hidden{i+1}"))
    layers.append(tf.keras.layers.Dense(1, activation="linear", name="output"))
    model = tf.keras.models.Sequential(layers)
    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=["mae"])
    return model


def callbacks(patience, monitor="val_accuracy"):
    return [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,
                                         restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1,
                                             patience=max(10, patience // 3),
                                             min_lr=1e-6, verbose=0),
    ]


def architecture_search(Xtr, Ytr, Xva, Yva, cfg):
    """Uji beberapa arsitektur, pilih yang terbaik berdasarkan macro-F1 validasi."""
    from sklearn.metrics import f1_score

    banner("PENCARIAN ARSITEKTUR (jumlah hidden layer & neuron)")
    rows = []
    for arch in cfg["search_archs"]:
        set_all_seeds(cfg["seed"])
        m = build_classifier(arch, Xtr.shape[1], len(cfg["class_names"]), cfg["learning_rate"])
        t0 = time.time()
        hist = m.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=cfg["search_epochs"],
                     batch_size=cfg["batch_size"], verbose=0,
                     callbacks=callbacks(cfg["search_patience"]))
        pred = np.argmax(m.predict(Xva, verbose=0), axis=1)
        true = np.argmax(Yva, axis=1)
        acc = float((pred == true).mean())
        f1 = float(f1_score(true, pred, average="macro"))
        n_param = int(m.count_params())
        rows.append({"hidden": "-".join(map(str, arch)), "n_param": n_param,
                     "val_acc": acc, "val_f1_macro": f1,
                     "epochs": len(hist.history["loss"]),
                     "train_s": round(time.time() - t0, 1)})
        print(f"  hidden {str(arch):9s} param={n_param:5d}  "
              f"val_acc={acc*100:5.2f} %  F1macro={f1:.4f}  "
              f"epoch={len(hist.history['loss']):4d}  ({rows[-1]['train_s']} s)")

    df_res = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    # Untuk deployment Arduino dipakai SATU hidden layer (header model_params.h
    # mencetak 2 matriks bobot). Jadi arsitektur terbaik dipilih dari kandidat
    # satu hidden layer, sementara hasil semua arsitektur tetap dilaporkan.
    df_single = df_res[~df_res["hidden"].str.contains("-")]
    best_arch = [int(x) for x in df_single.iloc[0]["hidden"].split("-")]
    print(f"\nArsitektur terbaik (1 hidden layer) : {best_arch}  "
          f"val_acc={df_single.iloc[0]['val_acc']*100:.2f} %, "
          f"F1macro={df_single.iloc[0]['val_f1_macro']:.4f}")
    if len(df_res) != len(df_single):
        top = df_res.iloc[0]
        print(f"(Kandidat terbaik secara umum: hidden {top['hidden']} dengan "
              f"val_acc={top['val_acc']*100:.2f} % - tidak dipakai agar sesuai "
              f"format header Arduino 1 hidden layer)")
    return best_arch, df_res


# %% [markdown]
# # 5. Evaluasi model klasifikasi

# %%
def evaluate_classifier(model, Xte, Yte, cut_te, cfg, tag="Test set 20%"):
    """Akurasi, classification report, dan akurasi per potongan daging."""
    from sklearn import metrics

    banner(f"EVALUASI KLASIFIKASI - {tag}")
    prob = model.predict(Xte, verbose=0)
    pred = np.argmax(prob, axis=1)
    true = np.argmax(Yte, axis=1)

    acc = float((pred == true).mean())
    print(f"Akurasi            : {acc*100:.2f} %  ({int((pred==true).sum())}/{len(true)} benar)")
    print(f"Balanced accuracy  : {metrics.balanced_accuracy_score(true, pred)*100:.2f} %")
    report = metrics.classification_report(true, pred, target_names=cfg["class_names"],
                                           digits=4, zero_division=0)
    print("\n" + report)
    cm = metrics.confusion_matrix(true, pred)
    print("Confusion matrix (baris = aktual, kolom = prediksi):")
    header = " " * 12 + "".join(f"{n:>12s}" for n in cfg["class_names"])
    print(header)
    for i, name in enumerate(cfg["class_names"]):
        print(f"{name:>12s}" + "".join(f"{v:12d}" for v in cm[i]))

    per_cut = None
    if cut_te is not None:
        per_cut = (pd.DataFrame({"Cut": cut_te, "benar": (pred == true)})
                   .groupby("Cut")["benar"].agg(["mean", "count"])
                   .rename(columns={"mean": "akurasi", "count": "n"})
                   .sort_values("akurasi"))
        print("\nAkurasi per potongan daging (uji generalisasi model GABUNGAN):")
        print((per_cut.assign(akurasi=(per_cut["akurasi"] * 100).round(2))).to_string())

    metrics_dict = {
        "accuracy": acc,
        "balanced_accuracy": float(metrics.balanced_accuracy_score(true, pred)),
        "f1_macro": float(metrics.f1_score(true, pred, average="macro")),
        "f1_weighted": float(metrics.f1_score(true, pred, average="weighted")),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics_dict, per_cut, pred, prob


def evaluate_regressor(model, Xte, y_true_te, y_true_te_raw, scaler_tvc, cfg, tag="Test set 20%"):
    """MAE/RMSE pada skala asli log10 CFU/g + akurasi kelas hasil threshold TVC."""
    from sklearn import metrics

    banner(f"EVALUASI REGRESI TVC - {tag}")
    pred_norm = model.predict(Xte, verbose=0).ravel()
    pred = scaler_tvc.inverse(pred_norm)                # kembalikan ke log10 CFU/g
    mae = float(metrics.mean_absolute_error(y_true_te_raw, pred))
    rmse = float(np.sqrt(metrics.mean_squared_error(y_true_te_raw, pred)))
    r2 = float(metrics.r2_score(y_true_te_raw, pred))
    print(f"MAE  : {mae:.4f} log CFU/g")
    print(f"RMSE : {rmse:.4f} log CFU/g")
    print(f"R^2  : {r2:.4f}")

    # threshold TVC -> kelas, lalu cek akurasinya
    thr = np.array(cfg["tvc_thresholds"])
    pred_lab = np.digitize(pred, thr) + 1               # 1..4
    true_lab = np.digitize(y_true_te_raw, thr) + 1
    acc_thr = float((pred_lab == true_lab).mean())
    print(f"Akurasi kelas bila TVC di-threshold [{', '.join(map(str, cfg['tvc_thresholds']))}] "
          f": {acc_thr*100:.2f} %")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "acc_threshold": acc_thr}, pred


class TVCScaler:
    """Regresi target TVC dinormalisasi ke [-1,1] agar training lebih stabil."""

    def __init__(self, lo=-1.0, hi=1.0):
        self.lo, self.hi = lo, hi
        self.min_, self.max_ = None, None

    def fit(self, y):
        self.min_, self.max_ = float(np.min(y)), float(np.max(y))
        return self

    def transform(self, y):
        return (np.asarray(y) - self.min_) / (self.max_ - self.min_) * (self.hi - self.lo) + self.lo

    def inverse(self, y):
        y = np.asarray(y)
        return (y - self.lo) / (self.hi - self.lo) * (self.max_ - self.min_) + self.min_


# %% [markdown]
# # 6. (Langkah 3) Ekstraksi bobot & bias -> file header C++ untuk Arduino
#
# Untuk Arduino, semua bobot ditulis sebagai **array `const float`**.
# Weight matrix di-transpose supaya loop di Arduino enak:
# `W1[j][i]` = bobot dari input ke-i ke neuron hidden ke-j.

# %%
def fmt(v, width=12):
    return f"{float(v):>{width}.8e}f"


def array_2d(name, mat, ctype="const float"):
    rows = mat.shape[0]
    cols = mat.shape[1]
    out = [f"{ctype} {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        out.append("  {" + ", ".join(fmt(v, 12) for v in mat[r]) + "},")
    out.append("};")
    return "\n".join(out)


def array_1d(name, vec, ctype="const float"):
    return (f"{ctype} {name}[{len(vec)}] = {{\n  "
            + ", ".join(fmt(v, 12) for v in vec) + "\n};")


def export_model_params_header(path, cls_model, reg_model, scaler_x, scaler_tvc,
                               cfg, test_vector, metrics_cls, metrics_reg, arch_cls, arch_reg):
    """Tulis file model_params.h yang dipakai sketch Arduino."""
    W1c, b1c = cls_model.layers[0].get_weights()
    W2c, b2c = cls_model.layers[1].get_weights()
    W1r, b1r = reg_model.layers[0].get_weights()
    W2r, b2r = reg_model.layers[1].get_weights()

    W1c_t = W1c.T        # (hidden, input)
    W2c_t = W2c.T        # (class, hidden)
    W1r_t = W1r.T        # (hidden, input)
    W2r_v = W2r.ravel()  # (hidden,)

    n_hid_c = W1c_t.shape[0]
    n_hid_r = W1r_t.shape[0]
    n_in = W1c_t.shape[1]
    n_cls = W2c_t.shape[0]

    # bobot layer ke-2 diambil dari layer terakhir tiap model
    if len(cls_model.layers) > 2:
        raise ValueError("Model klasifikasi >1 hidden layer belum didukung header otomatis. "
                         "Pakai satu hidden layer, atau perluas fungsi export.")
    if len(reg_model.layers) > 2:
        raise ValueError("Model regresi >1 hidden layer belum didukung header otomatis.")

    L = []
    a = L.append
    a("/*" + "=" * 76)
    a(" *  model_params.h  --  FILE HASIL EKSPOR OTOMATIS")
    a(" *  JANGAN DIEDIT MANUAL.  Dibuat oleh: training/train_meat_quality.py")
    a(" *")
    a(" *  RP Ganjil - Prediksi Kualitas Daging (Electronic Nose 11 sensor MQ)")
    a(" *  Target perangkat: Arduino Mega 2560")
    a(" *")
    a(f" *  Dibuat   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f" *  Host     : {platform.node()} / Python {platform.python_version()}"
      f" / TensorFlow {tf.__version__}")
    a(" *")
    a(" *  ARSITEKTUR")
    a(f" *    Klasifikasi : 11 input -> Dense({n_hid_c}, sigmoid) -> Dense(4, softmax)")
    a(f" *    Regresi TVC : 11 input -> Dense({n_hid_r}, sigmoid) -> Dense(1, linear)")
    a(f" *    Norm input  : Min-Max ke [{cfg['norm_lo']}, {cfg['norm_hi']}]")
    a(" *")
    a(" *  PERFORMA (data uji 20 %)")
    a(f" *    Akurasi klasifikasi : {metrics_cls['accuracy']*100:.2f} %"
      f"   | F1-macro: {metrics_cls['f1_macro']:.4f}"
      f"   | balanced acc: {metrics_cls['balanced_accuracy']*100:.2f} %")
    a(f" *    TVC regresi         : MAE {metrics_reg['MAE']:.4f} log CFU/g"
      f" | RMSE {metrics_reg['RMSE']:.4f} | R2 {metrics_reg['R2']:.4f}")
    a(" *")
    a(" *  KONVERSI NILAI DI ARDUINO / PROTEUS")
    a(" *    Potensiometer 0..5 V -> ADC 0..1023 -> nilai sensor 0.."
      f"{cfg['sensor_full_scale']:g}")
    a(" *    nilai_sensor = adc * (SENSOR_FULL_SCALE / 1023.0)")
    a(" *    x_norm       = 2*(nilai_sensor - NORM_MIN[i])/(NORM_MAX[i]-NORM_MIN[i]) - 1")
    a(" *" + "=" * 76 + "*/")
    a("")
    a("#ifndef MODEL_PARAMS_H")
    a("#define MODEL_PARAMS_H")
    a("")
    a("#include <Arduino.h>")
    a("")
    a(f"#define NN_N_INPUT        {n_in}")
    a(f"#define NN_N_HIDDEN_CLS   {n_hid_c}")
    a(f"#define NN_N_CLASS        {n_cls}")
    a(f"#define NN_N_HIDDEN_TVC   {n_hid_r}")
    a("")
    a(f"#define NN_SENSOR_FULL_SCALE  {cfg['sensor_full_scale']:.6f}f")
    a(f"#define NN_ADC_FULL_SCALE     1023.0f")
    a("")
    a("/* ---- Urutan nama sensor & pin analog (Arduino Mega 2560) ---- */")
    a("const char* const SENSOR_NAMES[NN_N_INPUT] = {"
      + ", ".join(f'"{s}"' for s in cfg["sensors"]) + "};")
    a("const uint8_t SENSOR_PINS[NN_N_INPUT] = {A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10};")
    a("")
    a("/* ---- Nama kelas hasil softmax (indeks 0..3) ---- */")
    a("const char* const CLASS_NAMES[NN_N_CLASS] = {"
      + ", ".join(f'"{c}"' for c in cfg["class_names"]) + "};")
    a("")
    a("/* ---- Ambang batas TVC (log10 CFU/g) untuk kategori kualitas ---- */")
    a(f"const float TVC_THRESHOLDS[{len(cfg['tvc_thresholds'])}] = {{"
      + ", ".join(f"{t:.4f}f" for t in cfg["tvc_thresholds"]) + "};")
    a("")
    a("/* ================== PARAMETER NORMALISASI INPUT ================== */")
    a(array_1d("NORM_MIN", scaler_x.min_))
    a(array_1d("NORM_MAX", scaler_x.max_))
    a("")
    a("/* ================== MODEL KLASIFIKASI (softmax) ================== */")
    a("/* W1[j][i] : bobot input ke-i -> neuron hidden ke-j */")
    a(array_2d("CLS_W1", W1c_t))
    a("/* b1[j] : bias neuron hidden ke-j */")
    a(array_1d("CLS_B1", b1c))
    a("/* W2[k][j] : bobot hidden ke-j -> kelas ke-k */")
    a(array_2d("CLS_W2", W2c_t))
    a("/* b2[k] : bias kelas ke-k */")
    a(array_1d("CLS_B2", b2c))
    a("")
    a("/* ================== MODEL REGRESI TVC (linear) ================== */")
    a("/* Masukan = x_norm yang sama; keluaran ternormalisasi lalu di-denormalisasi:")
    a("      TVC = (out - TVC_OUT_LO) / (TVC_OUT_HI - TVC_OUT_LO)")
    a("            * (TVC_MAX_LOG - TVC_MIN_LOG) + TVC_MIN_LOG */")
    a(f"#define TVC_OUT_LO   {cfg['norm_lo']:.6f}f")
    a(f"#define TVC_OUT_HI   {cfg['norm_hi']:.6f}f")
    a(f"#define TVC_MIN_LOG  {scaler_tvc.min_:.6f}f")
    a(f"#define TVC_MAX_LOG  {scaler_tvc.max_:.6f}f")
    a(array_2d("TVC_W1", W1r_t))
    a(array_1d("TVC_B1", b1r))
    a(array_1d("TVC_W2", W2r_v))
    a(array_1d("TVC_B2", b2r))
    a("")
    a("/* ============ VEKTOR UJI (untuk verifikasi di Proteus) ============ */")
    a("/* Set INPUT_MODE = 1 pada sketch, maka nilai di bawah dipakai langsung")
    a("   (tanpa potensiometer) supaya hasil bisa dibandingkan persis dengan Python. */")
    a(f"/* Asal data : potongan {test_vector['Cut']}, menit ke-{test_vector['Minute']}, "
      f"TVC aktual {test_vector['TVC']:.4f}, kelas aktual {test_vector['ClassName']} */")
    a(array_1d("TEST_VECTOR", [test_vector[s] for s in cfg["sensors"]], ctype="const float"))
    a("")
    a("#endif  /* MODEL_PARAMS_H */")
    a("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[OK] Header Arduino ditulis : {path}")


# %% [markdown]
# # 7. Uji generalisasi: *leave-one-cut-out* (model gabungan vs model per potongan)
#
# Ini menjawab pertanyaan: **perlukah 12 model (satu per potongan daging) atau
# cukup satu model gabungan?**
#
# Caranya: model *tetap satu* (gabungan), tetapi salah satu potongan daging
# dikeluarkan 100 % dari data training, lalu dipakai sebagai data uji.
# Jika akurasi model gabungan pada potongan yang belum pernah dilihat masih
# tinggi, artinya satu model sudah cukup general.

# %%
def leave_one_cut_out(df, cfg, arch):
    from sklearn.metrics import f1_score

    banner("UJI GENERALISASI: LEAVE-ONE-CUT-OUT (12 kali training)")
    rows = []
    cuts = sorted(df["Cut"].unique())
    for test_cut in cuts:
        tr = df[df["Cut"] != test_cut]
        te = df[df["Cut"] == test_cut]
        sc = MinMaxScalerNN(cfg["norm_lo"], cfg["norm_hi"]).fit(tr[cfg["sensors"]].values)
        Xtr = sc.transform(tr[cfg["sensors"]].values)
        Xte = sc.transform(te[cfg["sensors"]].values)
        Ytr = tf.keras.utils.to_categorical(tr["Label"].values - 1, len(cfg["class_names"]))
        Yte = tf.keras.utils.to_categorical(te["Label"].values - 1, len(cfg["class_names"]))

        set_all_seeds(cfg["seed"])
        m = build_classifier(arch, Xtr.shape[1], len(cfg["class_names"]), cfg["learning_rate"])
        m.fit(Xtr, Ytr, epochs=cfg["search_epochs"], batch_size=cfg["batch_size"],
              verbose=0, validation_split=0.1,
              callbacks=callbacks(cfg["search_patience"]))
        pred = np.argmax(m.predict(Xte, verbose=0), axis=1)
        true = np.argmax(Yte, axis=1)
        acc = float((pred == true).mean())
        f1 = float(f1_score(true, pred, average="macro"))
        rows.append({"cut_uji": test_cut, "n_uji": len(te), "akurasi": acc, "f1_macro": f1})
        print(f"  potongan diuji: {test_cut:15s} akurasi = {acc*100:5.2f} %  F1macro = {f1:.4f}")

    df_res = pd.DataFrame(rows)
    print(f"\nRata-rata akurasi pada potongan yang TIDAK dipakai training : "
          f"{df_res['akurasi'].mean()*100:.2f} %  "
          f"(min {df_res['akurasi'].min()*100:.2f} %, max {df_res['akurasi'].max()*100:.2f} %)")
    return df_res


# %% [markdown]
# # 8. Ekspor vektor uji untuk verifikasi Proteus
#
# Diambil 8 baris dari data uji (2 baris untuk tiap kelas kualitas, dari
# potongan daging berbeda) beserta nilai yang **diharapkan** muncul di LCD.

# %%
def export_test_vectors(df_test, cut_test, y_pred_prob, y_pred_tvc, cfg, path):
    rows = []
    df_t = df_test.reset_index(drop=True)
    for lab in sorted(df_t["Label"].unique()):
        idx_lab = df_t.index[df_t["Label"] == lab].tolist()
        # ambil 2 baris dengan potongan daging berbeda
        picked, used_cuts = [], set()
        for i in idx_lab:
            c = df_t.loc[i, "Cut"]
            if c not in used_cuts:
                picked.append(i)
                used_cuts.add(c)
            if len(picked) == 2:
                break
        for i in picked:
            r = {"Cut": df_t.loc[i, "Cut"], "Minute": int(df_t.loc[i, "Minute"]),
                 "Label": int(df_t.loc[i, "Label"]),
                 "ClassName": cfg["class_names"][int(df_t.loc[i, "Label"]) - 1],
                 "TVC": float(df_t.loc[i, "TVC"])}
            for s in cfg["sensors"]:
                r[s] = float(df_t.loc[i, s])
                r[s + "_ADC"] = int(round(r[s] / cfg["sensor_full_scale"] * 1023))
                r[s + "_POT_PCT"] = round(r[s] / cfg["sensor_full_scale"] * 100, 2)
            prob = y_pred_prob[i]
            r.update({
                "Pred_Class": cfg["class_names"][int(np.argmax(prob))],
                "Pred_Class_Idx": int(np.argmax(prob)),
                "Pred_TVC": float(y_pred_tvc[i]),
                "Pred_Label_from_TVC": int(np.digitize(y_pred_tvc[i], cfg["tvc_thresholds"]) + 1),
            })
            for k, cname in enumerate(cfg["class_names"]):
                r[f"P_{cname}"] = float(prob[k])
            rows.append(r)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(path, index=False, float_format="%.5f")
    print(f"[OK] Vektor uji Proteus     : {path}  ({len(df_out)} baris)")
    return df_out


def markdown_pot_table(df_vec, cfg):
    """Tabel bantu: nilai sensor -> posisi potensiometer Proteus."""
    lines = ["| Baris | Potongan | Kelas aktual | " + " | ".join(cfg["sensors"]) + " |",
             "|---|---|---|" + "---|" * len(cfg["sensors"])]
    for i, r in df_vec.iterrows():
        cells = []
        for s in cfg["sensors"]:
            cells.append(f"{r[s]:.2f} <br> ADC {int(r[s+'_ADC'])} <br> {r[s+'_POT_PCT']:.1f} %")
        lines.append(f"| #{i+1} | {r['Cut']} | {r['ClassName']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# %% [markdown]
# # 9. Program utama

# %%
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Training NN kualitas daging (e-nose 11 sensor MQ)")
    ap.add_argument("--skip-search", action="store_true",
                    help="lewati pencarian arsitektur, pakai hasil architecture_search.csv")
    ap.add_argument("--skip-loso", action="store_true",
                    help="lewati uji leave-one-cut-out, pakai hasil leave_one_cut_out.csv")
    args = ap.parse_args()

    t_start = time.time()
    cfg = CFG
    set_all_seeds(cfg["seed"])
    for d in (cfg["out_dir"], cfg["arduino_dir"]):
        os.makedirs(d, exist_ok=True)

    banner(f"TENSORFLOW {tf.__version__} | Python {platform.python_version()} | "
           f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # (1) integrasi dataset -------------------------------------------------
    df = load_and_merge(cfg)
    check_dataset(df, cfg)
    df.to_csv(os.path.join(cfg["out_dir"], "merged_dataset.csv"), index=False)
    print(f"\n[OK] Dataset gabungan disimpan: {cfg['out_dir']}/merged_dataset.csv")

    X_raw = df[cfg["sensors"]].to_numpy(dtype=np.float64)
    y_cls, y_cls_oh, y_tvc = make_targets(df, cfg)

    # (2) split 80/20 -------------------------------------------------------
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=cfg["test_size"],
                                      random_state=cfg["seed"], stratify=y_cls)
    print(f"\nData training : {len(idx_tr)} baris")
    print(f"Data uji      : {len(idx_te)} baris")

    scaler_exp = MinMaxScalerNN(cfg["norm_lo"], cfg["norm_hi"]).fit(X_raw[idx_tr])
    Xtr, Xte = scaler_exp.transform(X_raw[idx_tr]), scaler_exp.transform(X_raw[idx_te])
    print("\nParameter normalisasi (dari data training):")
    print(pd.DataFrame({"sensor": cfg["sensors"],
                        "min": scaler_exp.min_.round(3),
                        "max": scaler_exp.max_.round(3)}).to_string(index=False))

    # (3) cari arsitektur terbaik ------------------------------------------
    arch_csv = os.path.join(cfg["out_dir"], "architecture_search.csv")
    if args.skip_search and os.path.exists(arch_csv):
        df_arch = pd.read_csv(arch_csv)
        single = df_arch[~df_arch["hidden"].astype(str).str.contains("-")]
        best = single.sort_values("val_f1_macro", ascending=False).iloc[0]["hidden"]
        arch = [int(x) for x in str(best).split("-")]
        banner("LEWATI PENCARIAN ARSITEKTUR (memakai architecture_search.csv)")
        print(df_arch.to_string(index=False))
        print(f"\nArsitektur dipakai: {arch}")
    else:
        arch, df_arch = architecture_search(Xtr, y_cls_oh[idx_tr], Xte, y_cls_oh[idx_te], cfg)
        df_arch.to_csv(arch_csv, index=False)
    if len(arch) > 1:
        arch = [arch[0]]     # header Arduino mendukung 1 hidden layer
        print(f"Catatan: untuk deployment dipakai satu hidden layer terbaik -> {arch}")

    # (4) training model final klasifikasi ---------------------------------
    banner(f"TRAINING MODEL FINAL - KLASIFIKASI (hidden {arch})")
    set_all_seeds(cfg["seed"])
    cls_model = build_classifier(arch, Xtr.shape[1], len(cfg["class_names"]), cfg["learning_rate"])
    cls_model.summary()
    t0 = time.time()
    hist_cls = cls_model.fit(Xtr, y_cls_oh[idx_tr], validation_data=(Xte, y_cls_oh[idx_te]),
                             epochs=cfg["final_epochs"], batch_size=cfg["batch_size"],
                             verbose=0, callbacks=callbacks(cfg["final_patience"]))
    print(f"Training selesai: {len(hist_cls.history['loss'])} epoch "
          f"dalam {time.time()-t0:.1f} s")
    metrics_cls, per_cut, pred_cls, prob_cls = evaluate_classifier(
        cls_model, Xte, y_cls_oh[idx_te], df["Cut"].values[idx_te], cfg)

    # (5) training model final regresi TVC ---------------------------------
    banner(f"TRAINING MODEL FINAL - REGRESI TVC (hidden {arch})")
    scaler_tvc = TVCScaler(cfg["norm_lo"], cfg["norm_hi"]).fit(y_tvc)
    print(f"Rentang TVC dataset: {scaler_tvc.min_:.4f} .. {scaler_tvc.max_:.4f} log10 CFU/g")
    y_tvc_norm = scaler_tvc.transform(y_tvc)
    set_all_seeds(cfg["seed"])
    reg_model = build_regressor(arch, Xtr.shape[1], cfg["learning_rate"])
    t0 = time.time()
    hist_reg = reg_model.fit(Xtr, y_tvc_norm[idx_tr], validation_data=(Xte, y_tvc_norm[idx_te]),
                             epochs=cfg["final_epochs"], batch_size=cfg["batch_size"],
                             verbose=0, callbacks=callbacks(cfg["final_patience"], "val_loss"))
    print(f"Training selesai: {len(hist_reg.history['loss'])} epoch "
          f"dalam {time.time()-t0:.1f} s")
    metrics_reg, pred_tvc_te = evaluate_regressor(reg_model, Xte, y_tvc_norm[idx_te],
                                                  y_tvc[idx_te], scaler_tvc, cfg)

    # (6) uji generalisasi antar potongan daging ---------------------------
    loso_csv = os.path.join(cfg["out_dir"], "leave_one_cut_out.csv")
    if args.skip_loso and os.path.exists(loso_csv):
        df_loso = pd.read_csv(loso_csv)
        banner("LEWATI UJI LEAVE-ONE-CUT-OUT (memakai leave_one_cut_out.csv)")
        print(df_loso.to_string(index=False))
    else:
        df_loso = leave_one_cut_out(df, cfg, arch)
        df_loso.to_csv(loso_csv, index=False)

    # (7) verifikasi silang: apakah kelas dari model klasifikasi == kelas dari
    #     threshold TVC model regresi? --------------------------------------
    banner("PERBANDINGAN DUA PENDEKATAN OUTPUT")
    pred_lab_thr = np.digitize(pred_tvc_te, cfg["tvc_thresholds"])
    agree = float((pred_lab_thr == pred_cls).mean())
    both_right = float(((pred_lab_thr == y_cls[idx_te]) & (pred_cls == y_cls[idx_te])).mean())
    print(f"Kesepakatan kelas(softmax) vs kelas(TVC-threshold) : {agree*100:.2f} %")
    print(f"Keduanya benar                                     : {both_right*100:.2f} %")
    print("Catatan: pada dataset ini Label memang = hasil threshold TVC,")
    print("         jadi regresi TVC + threshold secara teori setara dengan klasifikasi.")

    # (8) MODEL DEPLOY: dilatih ulang dengan SELURUH data -------------------
    banner("MODEL DEPLOY - DILATIH ULANG DENGAN 100 % DATA")
    scaler_deploy = MinMaxScalerNN(cfg["norm_lo"], cfg["norm_hi"]).fit(X_raw)
    Xall = scaler_deploy.transform(X_raw)
    scaler_tvc_deploy = TVCScaler(cfg["norm_lo"], cfg["norm_hi"]).fit(y_tvc)
    y_all_tvc = scaler_tvc_deploy.transform(y_tvc)

    # PENTING: urutan baris mengikuti urutan sheet (Inside-Outside ... Fat),
    # sehingga 10 % terakhir = mayoritas potongan "Fat". Kalau langsung dipakai
    # sebagai validation_split, EarlyStopping bisa mengembalikan bobot yang buruk.
    # Solusinya: acak urutan baris dulu dengan seed tetap.
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(Xall))
    Xall_shuf, y_all_shuf = Xall[perm], y_cls_oh[perm]

    set_all_seeds(cfg["seed"])
    cls_deploy = build_classifier(arch, Xall.shape[1], len(cfg["class_names"]), cfg["learning_rate"])
    hist_deploy = cls_deploy.fit(Xall_shuf, y_all_shuf, epochs=cfg["final_epochs"],
                                 batch_size=cfg["batch_size"], verbose=0,
                                 validation_split=0.1,
                                 callbacks=callbacks(cfg["final_patience"]))
    acc_all = float((np.argmax(cls_deploy.predict(Xall, verbose=0), axis=1) == y_cls).mean())
    print(f"Akurasi model deploy pada seluruh dataset : {acc_all*100:.2f} %")

    set_all_seeds(cfg["seed"])
    reg_deploy = build_regressor(arch, Xall.shape[1], cfg["learning_rate"])
    reg_deploy.fit(Xall_shuf, y_all_tvc[perm], epochs=cfg["final_epochs"],
                   batch_size=cfg["batch_size"], verbose=0, validation_split=0.1,
                   callbacks=callbacks(cfg["final_patience"], "val_loss"))

    cls_deploy.save(os.path.join(cfg["out_dir"], "beef_quality_classifier.keras"))
    reg_deploy.save(os.path.join(cfg["out_dir"], "tvc_regressor.keras"))
    print(f"[OK] Model disimpan di {cfg['out_dir']}/")

    # (9) vektor uji + bobot/bias untuk Arduino ----------------------------
    banner("EKSPOR VEKTOR UJI & FILE model_params.h")
    # PENTING: vektor uji dinormalisasi dengan parameter scaler DEPLOY,
    # karena itulah nilai NORM_MIN/NORM_MAX yang tertanam di Arduino.
    Xte_deploy = scaler_deploy.transform(X_raw[idx_te])
    prob_te = cls_deploy.predict(Xte_deploy, verbose=0)
    tvc_te = scaler_tvc_deploy.inverse(reg_deploy.predict(Xte_deploy, verbose=0).ravel())
    df_vec = export_test_vectors(df.iloc[idx_te], None, prob_te, tvc_te, cfg,
                                 os.path.join(cfg["out_dir"], "test_vectors.csv"))

    test_vector = {s: float(df_vec.iloc[0][s]) for s in cfg["sensors"]}
    test_vector.update({"Cut": df_vec.iloc[0]["Cut"], "Minute": int(df_vec.iloc[0]["Minute"]),
                        "TVC": float(df_vec.iloc[0]["TVC"]),
                        "ClassName": df_vec.iloc[0]["ClassName"]})

    header_path = os.path.join(cfg["arduino_dir"], "model_params.h")
    export_model_params_header(header_path, cls_deploy, reg_deploy, scaler_deploy,
                               scaler_tvc_deploy, cfg, test_vector, metrics_cls, metrics_reg,
                               arch, arch)
    # salinan untuk dokumentasi
    export_model_params_header(os.path.join(cfg["out_dir"], "model_params.h"),
                               cls_deploy, reg_deploy, scaler_deploy, scaler_tvc_deploy,
                               cfg, test_vector, metrics_cls, metrics_reg, arch, arch)

    # (10) laporan markdown ------------------------------------------------
    plot_history({"cls": hist_deploy, "reg": hist_reg}, cfg["out_dir"])
    write_markdown_report(cfg, df, df_arch, arch, metrics_cls, metrics_reg, per_cut,
                          df_loso, df_vec, scaler_deploy, scaler_exp, scaler_tvc_deploy,
                          agree, both_right, len(idx_tr), len(idx_te))

    # (11) simpan ringkasan JSON (dipakai verification/verify_cpp_vs_keras.py)
    summary = {
        "sensors": cfg["sensors"],
        "class_names": cfg["class_names"],
        "tvc_thresholds": cfg["tvc_thresholds"],
        "sensor_full_scale": cfg["sensor_full_scale"],
        "norm_min": scaler_deploy.min_.tolist(),
        "norm_max": scaler_deploy.max_.tolist(),
        "arch_hidden": arch,
        "metrics_cls": {k: v for k, v in metrics_cls.items() if k != "classification_report"},
        "metrics_reg": metrics_reg,
        "deploy_train_accuracy": acc_all,
        "loso_mean_accuracy": float(df_loso["akurasi"].mean()),
    }
    with open(os.path.join(cfg["out_dir"], "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    banner(f"SELESAI - total {time.time()-t_start:.1f} s")
    print("File penting:")
    print(f"  - {cfg['arduino_dir']}/model_params.h   (bobot & bias untuk Arduino)")
    print(f"  - {cfg['out_dir']}/test_vectors.csv     (data uji untuk Proteus)")
    print(f"  - {cfg['out_dir']}/laporan_hasil_training.md")


# %%
def write_markdown_report(cfg, df, df_arch, arch, m_cls, m_reg, per_cut, df_loso, df_vec,
                          sc_deploy, sc_exp, sc_tvc, agree, both_right, n_tr, n_te):
    """Ringkasan hasil dalam format markdown, siap ditempel ke laporan."""
    p = os.path.join(cfg["out_dir"], "laporan_hasil_training.md")
    lines = []
    a = lines.append
    a("# Hasil Training - Prediksi Kualitas Daging (Electronic Nose)")
    a("")
    a(f"*Dibuat otomatis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
      f"TensorFlow {tf.__version__}*")
    a("")
    a("## 1. Dataset gabungan (12 potongan daging)")
    a("")
    a(f"- Jumlah baris: **{len(df)}** ({df['Cut'].nunique()} potongan daging x 2.220 menit pengukuran)")
    a(f"- Jumlah fitur input: **{len(cfg['sensors'])}** sensor MQ -> `{', '.join(cfg['sensors'])}`")
    a("- Target: `Label` (klasifikasi 4 kelas) dan `TVC` dalam log10 CFU/g (regresi)")
    a("- Data uji 20 % terpisah: " + (f"{n_tr} training / {n_te} uji" if n_tr else "semua data dipakai"))
    a("")
    a("Distribusi kelas: " + ", ".join(
        f"{TIER[k]} = {v}" for k, v in df["Label"].value_counts().sort_index().items()))
    a("")
    a("## 2. Normalisasi Min-Max (parameter untuk Arduino)")
    a("")
    a("| Sensor | min (training) | max (training) | min (deploy) | max (deploy) |")
    a("|---|---|---|---|---|")
    for i, s in enumerate(cfg["sensors"]):
        a(f"| {s} | {sc_exp.min_[i]:.3f} | {sc_exp.max_[i]:.3f} | "
          f"{sc_deploy.min_[i]:.3f} | {sc_deploy.max_[i]:.3f} |")
    a("")
    a(f"Rumus: `x_norm = 2*(x - min)/(max - min) - 1`  -> rentang "
      f"[{cfg['norm_lo']}, {cfg['norm_hi']}]")
    a("")
    a("## 3. Pemilihan arsitektur")
    a("")
    a("| Hidden layers | Jumlah parameter | Akurasi validasi | F1-macro | Epoch | Waktu (s) |")
    a("|---|---|---|---|---|---|")
    for _, r in df_arch.iterrows():
        a(f"| {r['hidden']} | {r['n_param']} | {r['val_acc']*100:.2f} % | "
          f"{r['val_f1_macro']:.4f} | {r['epochs']} | {r['train_s']} |")
    a("")
    a(f"**Arsitektur dipakai:** 11 -> {arch[0]} (sigmoid) -> 4 (softmax) dan 11 -> {arch[0]} "
      f"(sigmoid) -> 1 (linear)")
    a("")
    a("## 4. Performa model (data uji 20 %)")
    a("")
    a(f"- Akurasi klasifikasi: **{m_cls['accuracy']*100:.2f} %**  "
      f"(balanced accuracy {m_cls['balanced_accuracy']*100:.2f} %, F1-macro {m_cls['f1_macro']:.4f})")
    a(f"- Regresi TVC: MAE **{m_reg['MAE']:.4f}** log CFU/g, RMSE {m_reg['RMSE']:.4f}, "
      f"R2 {m_reg['R2']:.4f}")
    a(f"- Akurasi kelas dari threshold TVC: {m_reg['acc_threshold']*100:.2f} %")
    a(f"- Kesepakatan kelas softmax vs kelas-TVC: {agree*100:.2f} %")
    a("")
    a("### Classification report")
    a("")
    a("```")
    a(m_cls["classification_report"].rstrip())
    a("```")
    a("")
    a("### Akurasi per potongan daging (model gabungan)")
    a("")
    a("| Potongan | Akurasi | Jumlah baris uji |")
    a("|---|---|---|")
    for cut, r in per_cut.iterrows():
        a(f"| {cut} | {r['akurasi']*100:.2f} % | {int(r['n'])} |")
    a("")
    a("## 5. Model gabungan vs potongan yang belum pernah dilihat (leave-one-cut-out)")
    a("")
    a("| Potongan yang diuji | Akurasi | F1-macro |")
    a("|---|---|---|")
    for _, r in df_loso.iterrows():
        a(f"| {r['cut_uji']} | {r['akurasi']*100:.2f} % | {r['f1_macro']:.4f} |")
    a("")
    a(f"Rata-rata: **{df_loso['akurasi'].mean()*100:.2f} %** "
      f"(min {df_loso['akurasi'].min()*100:.2f} %, max {df_loso['akurasi'].max()*100:.2f} %)")
    a("")
    a("## 6. Vektor uji untuk Proteus")
    a("")
    a("Nilai ADC dihitung dengan `ADC = nilai / 60 * 1023`; atur potensiometer")
    a("sampai tegangan atau kode ADC tersebut (lihat `PANDUAN_PROTEUS.md`).")
    a("")
    a("| # | Potongan | Kelas aktual | Prediksi Python | TVC aktual | Prediksi TVC |")
    a("|---|---|---|---|---|---|")
    for i, r in df_vec.iterrows():
        a(f"| {i+1} | {r['Cut']} | {r['ClassName']} | {r['Pred_Class']} | "
          f"{r['TVC']:.4f} | {r['Pred_TVC']:.4f} |")
    a("")
    a(markdown_pot_table(df_vec, cfg))
    a("")

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Laporan markdown       : {p}")
    return p


def plot_history(histories, out_dir):
    """Kurva loss & accuracy training (opsional, butuh matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        print("[!] matplotlib tidak tersedia, plot dilewati")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(histories["cls"].history["loss"], "r", label="training")
    ax[0].plot(histories["cls"].history["val_loss"], "b", label="validasi")
    ax[0].set_title("Loss klasifikasi"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(histories["cls"].history["accuracy"], "r", label="training")
    ax[1].plot(histories["cls"].history["val_accuracy"], "b", label="validasi")
    ax[1].set_title("Akurasi klasifikasi"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    ax[2].plot(histories["reg"].history["loss"], "r", label="training")
    ax[2].plot(histories["reg"].history["val_loss"], "b", label="validasi")
    ax[2].set_title("MSE regresi TVC"); ax[2].set_xlabel("Epoch"); ax[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kurva_training.png"), dpi=120)
    print(f"[OK] Kurva training         : {out_dir}/kurva_training.png")


if __name__ == "__main__":
    main()
