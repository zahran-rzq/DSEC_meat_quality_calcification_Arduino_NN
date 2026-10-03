/* =====================================================================
 * nn_core.h  --  INTI PERHITUNGAN NEURAL NETWORK (manual, tanpa library)
 *
 * File ini SENGAJA dibuat murni C/C++ standar (tanpa dependensi Arduino
 * maupun TensorFlow) supaya:
 *   - bisa di-#include langsung oleh sketch .ino, DAN
 *   - bisa dikompilasi di PC (g++) untuk memverifikasi bahwa hasil
 *     perhitungan di mikrokontroler IDENTIK dengan model di Python.
 *
 * Alur perhitungan (per layer):
 *     z = X . W + B          <- perkalian matriks manual + bias
 *     a = aktivasi(z)        <- sigmoid (hidden) / softmax (output)
 *
 * Semua bobot & bias diambil dari nn_params.h (hasil ekstraksi training).
 * ===================================================================== */

#ifndef NN_CORE_H
#define NN_CORE_H

#include <math.h>      /* expf() */
#include <stdbool.h>   /* bool   */

#include "nn_params.h"

/* Jenis fungsi aktivasi (juga didefinisikan di nn_params.h sebagai cadangan) */
#ifndef NN_ACT_SIGMOID
#define NN_ACT_SIGMOID 1
#endif
#ifndef NN_ACT_SOFTMAX
#define NN_ACT_SOFTMAX 2
#endif

/* ---------------------------------------------------------------------
 * FUNGSI AKTIVASI 1 -- SIGMOID :  f(z) = 1 / (1 + exp(-z))
 * Ditulis manual tanpa library eksternal.
 * ------------------------------------------------------------------ */
static inline float nnSigmoid(float z) {
  if (z >= 0.0f) {
    return 1.0f / (1.0f + expf(-z));
  }
  /* Bentuk setara ini dipakai saat z negatif agar expf() tidak meluap */
  float e = expf(z);
  return e / (1.0f + e);
}

/* ---------------------------------------------------------------------
 * FUNGSI AKTIVASI 2 -- SOFTMAX (stabil secara numerik, dikurangi maks.)
 *     p_i = exp(z_i - max) / SUM_j exp(z_j - max)
 * ------------------------------------------------------------------ */
static void nnSoftmax(const float *z, int n, float *out) {
  float zmax = z[0];
  for (int i = 1; i < n; i++) {
    if (z[i] > zmax) zmax = z[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    out[i] = expf(z[i] - zmax);
    sum += out[i];
  }
  if (sum == 0.0f) sum = 1.0f;               /* jaga-jaga bagi nol */
  for (int i = 0; i < n; i++) out[i] /= sum;
}

/* ---------------------------------------------------------------------
 * NORMALISASI INPUT
 * (a) dari nilai fisik sensor:  x_norm = 2*(x-X_MIN)/(X_MAX-X_MIN) - 1
 *     X_MIN/X_MAX = parameter scaling hasil pre-processing di Python.
 * ------------------------------------------------------------------ */
static float nnScaleSensor(float x, int i) {
  float span = X_MAX[i] - X_MIN[i];
  if (span == 0.0f) span = 1.0f;             /* hindari bagi nol */
  return 2.0f * (x - X_MIN[i]) / span - 1.0f;
}

/* ---------------------------------------------------------------------
 * (b) dari nilai analogRead (0..ADC_MAX, default 1023):
 *     x_norm = 2*adc/ADC_MAX - 1
 *     Potensiometer 0 = nilai sensor minimum, penuh = nilai maksimum,
 *     sehingga pemetaan ADC ke [-1,+1] setara dengan pemetaan sensor.
 * ------------------------------------------------------------------ */
static float nnScaleADC(int adc, float adc_max) {
  if (adc < 0) adc = 0;
  if (adc > (int)adc_max) adc = (int)adc_max;
  return 2.0f * (float)adc / adc_max - 1.0f;
}

/* ---------------------------------------------------------------------
 * INFERENCE / FORWARD PASS -- perkalian matriks manual, per layer
 *
 *   x     : input ternormalisasi   [NN_INPUTS]
 *   probs : probabilitas tiap kelas [NN_OUTPUTS]  (hasil softmax)
 *   return: indeks kelas dengan probabilitas terbesar (argmax)
 *
 * Buffer bufA/bufB dipakai bergantian sebagai penampung aktivasi layer.
 * ------------------------------------------------------------------ */
static int nnForward(const float *x, float *probs) {
  static float bufA[NN_MAX_WIDTH];
  static float bufB[NN_MAX_WIDTH];

  const float *in = x;          /* penunjuk data masukan layer saat ini */
  float *out = bufA;
  int n_in = NN_INPUTS;
  bool wrote_probs = false;     /* sudah adakah layer ber-softmax? */

  for (int L = 0; L < NN_N_LAYERS; L++) {
    const float *W = LAYER_W[L];      /* matriks bobot, row-major n_in x n_out */
    const float *B = LAYER_B[L];      /* vektor bias            n_out          */
    int n_out = LAYER_OUT[L];

    /* ---- perkalian matriks + bias :  z[j] = SUM_i in[i]*W[i][j] + B[j] ---- */
    for (int j = 0; j < n_out; j++) {
      float acc = B[j];
      for (int i = 0; i < n_in; i++) {
        acc += in[i] * W[i * n_out + j];
      }
      /* ---- fungsi aktivasi ---- */
      if (LAYER_ACT[L] == NN_ACT_SIGMOID) {
        out[j] = nnSigmoid(acc);
      } else {                        /* layer terakhir : softmax ditunda */
        out[j] = acc;
      }
    }

    /* softmax hanya pada layer output */
    if (LAYER_ACT[L] == NN_ACT_SOFTMAX) {
      nnSoftmax(out, n_out, probs);
      wrote_probs = true;
    }

    /* siapkan iterasi berikutnya (tukar buffer) */
    in = out;
    n_in = n_out;
    out = (out == bufA) ? bufB : bufA;
  }

  /* Bila tidak ada layer ber-softmax (mis. keluaran linier untuk regresi),
     salin keluaran layer terakhir apa adanya agar argmax tetap berfungsi. */
  if (!wrote_probs) {
    for (int k = 0; k < NN_OUTPUTS && k < n_in; k++) probs[k] = in[k];
  }

  /* ---- argmax : kelas dengan probabilitas terbesar ---- */
  int best = 0;
  for (int k = 1; k < NN_OUTPUTS; k++) {
    if (probs[k] > probs[best]) best = k;
  }
  return best;
}

/* ---------------------------------------------------------------------
 * RANGKAIAN LENGKAP : ADC -> normalisasi -> forward pass -> argmax
 *
 * Fungsi inilah yang dipanggil oleh sketch Arduino (BQC_Arduino_Mega.ino)
 * DAN oleh program uji di PC (arduino/test/test_nn_core.cpp), sehingga
 * keduanya dijamin menempuh jalur perhitungan yang IDENTIK.
 *
 *   adc   : nilai analogRead tiap sensor  [NN_INPUTS]
 *   norm  : (keluaran) nilai ternormalisasi [-1..+1]
 *   probs : (keluaran) probabilitas tiap kelas
 *   return: indeks kelas hasil prediksi (argmax)
 * ------------------------------------------------------------------ */
static int nnProcessADC(const int *adc, int n, float adc_max,
                        float *norm, float *probs) {
  for (int i = 0; i < n && i < NN_INPUTS; i++) {
    norm[i] = nnScaleADC(adc[i], adc_max);
  }
  return nnForward(norm, probs);
}

/* ---------------------------------------------------------------------
 * AMBANG BATAS (THRESHOLD) KEPERCAYAAN
 * Bila probabilitas kelas terbaik < NN_CONF_THRESHOLD maka hasil dianggap
 * TIDAK YAKIN (aroma campuran / pembacaan sensor belum stabil).
 * ------------------------------------------------------------------ */
#ifndef NN_CONF_THRESHOLD
#define NN_CONF_THRESHOLD 0.50f
#endif

static bool nnIsConfident(float confidence) {
  return confidence >= NN_CONF_THRESHOLD;
}

#endif  /* NN_CORE_H */
