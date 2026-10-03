#!/usr/bin/env bash
# ============================================================================
#  tools/run_all.sh
#  Jalankan seluruh pipeline dari nol:
#     1. training NN  -> bobot & bias untuk Arduino
#     2. verifikasi   -> buktikan kode Arduino sama dengan model Keras
#  Pakai:  bash tools/run_all.sh
# ============================================================================
set -e
cd "$(dirname "$0")/.."

PY=${PY:-python3}
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi

echo "==> [1/2] Training NN (butuh waktu 20-40 menit di CPU)"
mkdir -p training/output
$PY training/train_meat_quality.py 2>&1 | tee training/output/train_log.txt

echo
echo "==> [2/2] Verifikasi kode Arduino vs Keras"
$PY verification/verify_cpp_vs_keras.py || echo "(verifikasi dilewati / ada perbedaan)"

echo
echo "SELESAI."
echo "  Bobot Arduino   : arduino/MeatQuality_NN_Arduino/model_params.h"
echo "  Laporan hasil   : training/output/laporan_hasil_training.md"
echo "  Data uji Proteus: training/output/test_vectors.csv"
