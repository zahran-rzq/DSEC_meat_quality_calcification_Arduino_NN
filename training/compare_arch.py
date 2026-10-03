#!/usr/bin/env python3
"""
Perbandingan arsitektur dengan TRAINING PENUH.

Pencarian cepat (architecture_search, 120 epoch) ternyata TIDAK deterministik:
TensorFlow dengan oneDNN dapat menghasilkan urutan pembulatan yang berbeda antar-run,
dan karena EarlyStopping/ReduceLROnPlateau peka terhadap selisih kecil, peringkat
kandidat bisa berubah dari satu run ke run berikutnya.

Skrip ini membandingkan kandidat secara lebih adil: setiap arsitektur dilatih
sampai konvergen (early stopping) dengan protokol yang sama persis, lalu diukur
pada 20% data uji yang tidak ikut dilatih.

Pemakaian:
    python training/compare_arch.py --candidates "20" "24,12" "32,16"
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
from datetime import datetime

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_bqc as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", default=["20", "24,12", "32,16"])
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()

    np.random.seed(T.RANDOM_STATE)
    tf.random.set_seed(T.RANDOM_STATE)

    data = T.load_integrated_dataset()
    X_raw, Y, _ = T.build_xy(data)
    xmin, xmax = T.fit_minmax(X_raw)
    X = T.normalize(X_raw, xmin, xmax)
    Xtr, Xte, Ytr, Yte = train_test_split(
        X, Y, test_size=0.2, random_state=T.RANDOM_STATE)

    results = []
    for spec in args.candidates:
        arch = tuple(int(v) for v in spec.split(","))
        # seed di-set ulang sebelum tiap kandidat agar kondisi awalnya sama
        tf.random.set_seed(T.RANDOM_STATE)
        np.random.seed(T.RANDOM_STATE)

        model = T.build_model(arch)
        T.compile_model(model)
        t0 = time.time()
        hist = T.fit_model(model, Xtr, Ytr, Xte, Yte, epochs=args.epochs, verbose=0)
        dt = time.time() - t0

        loss, acc = model.evaluate(Xte, Yte, verbose=0)
        pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
        truth = np.argmax(Yte, axis=1)
        f1 = metrics.f1_score(truth, pred, average="macro")
        n_par = int(np.sum([np.prod(w.shape) for w in model.trainable_variables]))

        results.append({
            "arch": list(arch),
            "test_accuracy": float(acc),
            "test_loss": float(loss),
            "macro_f1": float(f1),
            "val_accuracy_best": float(np.max(hist.history["val_accuracy"])),
            "epochs_used": int(len(hist.history["loss"])),
            "n_params": n_par,
            "flash_kb": round(n_par * 4 / 1024.0, 2),
            "train_seconds": round(dt, 1),
        })
        r = results[-1]
        print(f"[compare] hidden={str(list(arch)):10s} test_acc={acc:.5f} "
              f"macro_f1={f1:.4f} params={n_par:5d} ({r['flash_kb']} KB) "
              f"epochs={r['epochs_used']} time={dt:.0f}s")

    results.sort(key=lambda r: -r["test_accuracy"])
    print(f"\n[compare] terbaik (training penuh) pada run ini: {results[0]['arch']} "
          f"dengan test_accuracy={results[0]['test_accuracy']:.5f}")
    print("[compare] CATATAN: pelatihan tidak deterministik; selisih < ~0.002 antar "
          "arsitektur lebar belum tentu berarti. Bandingkan beberapa run.")

    stamp = datetime.now().isoformat()
    mpath = os.path.join(T.DOCS_DIR, "metrics.json")
    info = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            info = json.load(f)
    info["arch_comparison_full_training"] = results      # run terbaru
    info["arch_comparison_generated"] = stamp
    runs = info.get("arch_comparison_runs", [])
    runs.append({"generated": stamp, "results": results})
    info["arch_comparison_runs"] = runs                  # seluruh run terakumulasi
    os.makedirs(T.DOCS_DIR, exist_ok=True)
    with open(mpath, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[compare] hasil disimpan ke docs/metrics.json "
          f"(total {len(runs)} run tercatat)")


if __name__ == "__main__":
    main()
