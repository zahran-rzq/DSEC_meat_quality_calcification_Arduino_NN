#!/usr/bin/env bash
# =====================================================================
# run_checks.sh -- memeriksa implementasi Arduino di PC
#
#   [A] test_nn_core : mengompilasi nn_core.h + nn_params.h (kode yang
#       dipakai Arduino) dan membandingkan hasilnya dengan TensorFlow.
#   [B] host_sim     : mengompilasi BQC_Arduino_Mega.ino apa adanya
#       dengan shim antarmuka Arduino/LCD, lalu menjalankan setup()/loop().
#
# Pemakaian:  bash arduino/test/run_checks.sh
# =====================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
SKETCH="$HERE/../BQC_Arduino_Mega"
SIM="$HERE/host_sim"
VECTORS="$HERE/../../docs/test_vectors.csv"
PY="${PYTHON:-/home/user/.venv/bin/python}"

fail=0

echo "############################################################"
echo "# [A] Verifikasi perhitungan NN  (C++ vs TensorFlow)"
echo "############################################################"
if [ -x "$PY" ]; then
  ( cd "$HERE/../.." && "$PY" training/verify_arduino.py ) || fail=1
else
  echo "Python tidak ditemukan di $PY -- lewati bagian A"
fi

echo
echo "############################################################"
echo "# [B] Host simulation sketch BQC_Arduino_Mega.ino"
echo "############################################################"
if ! g++ -O2 -std=gnu++11 -Wall -Wextra -Wno-unused-function \
        -I "$SIM" -I "$SKETCH" "$SIM/main_host.cpp" -o "$SIM/host_sim"; then
  echo "GAGAL mengompilasi sketch"
  exit 1
fi
"$SIM/host_sim" "$VECTORS" || fail=1

echo
if [ "$fail" -eq 0 ]; then
  echo ">>> SEMUA PEMERIKSAAN LULUS"
else
  echo ">>> ADA PEMERIKSAAN YANG GAGAL"
fi
exit "$fail"
