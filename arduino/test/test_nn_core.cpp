/* =====================================================================
 * test_nn_core.cpp
 *
 * Program uji yang dikompilasi di PC (g++) untuk MEMVERIFIKASI bahwa
 * perhitungan Neural Network di Arduino menghasilkan angka yang sama
 * dengan model TensorFlow di Python.
 *
 * Program ini meng-#include nn_core.h + nn_params.h -- yaitu file yang
 * PERSIS sama dengan yang dipakai sketch BQC_Arduino_Mega.ino -- dan
 * memanggil nnProcessADC(), fungsi yang juga dipanggil sketch tersebut.
 *
 * Cara pakai:
 *   g++ -O2 -std=c++11 -I ../BQC_Arduino_Mega test_nn_core.cpp -o test_nn_core
 *   ./test_nn_core ../../docs/test_vectors.csv
 *
 * Keluaran (stdout, format CSV):
 *   RESULT,<case_id>,<pred_class>,<confidence>,<p0>,<p1>,<p2>,<p3>
 * ===================================================================== */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "nn_core.h"

/* Resolusi ADC yang sama dengan di sketch Arduino */
#define TEST_ADC_MAX 1023.0f

static std::vector<std::string> splitCSV(const std::string &line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ',') { out.push_back(cur); cur.clear(); }
    else if (c != '\r' && c != '\n') cur.push_back(c);
  }
  out.push_back(cur);
  return out;
}

int main(int argc, char **argv) {
  const char *path = (argc > 1) ? argv[1] : "../../docs/test_vectors.csv";
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "ERROR: tidak dapat membuka %s\n", path);
    return 2;
  }

  /* --- tampilkan arsitektur yang terkompilasi --- */
  std::printf("ARCH,%d", NN_INPUTS);
  for (int L = 0; L < NN_N_LAYERS; L++) std::printf(",%d", LAYER_OUT[L]);
  std::printf("\n");
  std::printf("THRESHOLD,%.4f\n", (float)NN_CONF_THRESHOLD);

  std::string header;
  if (!std::getline(in, header)) {
    std::fprintf(stderr, "ERROR: file CSV kosong\n");
    return 2;
  }
  std::vector<std::string> cols = splitCSV(header);

  /* cari indeks kolom yang dibutuhkan */
  int i_case = -1, i_actual = -1;
  std::vector<int> i_adc(NN_INPUTS, -1);
  for (size_t c = 0; c < cols.size(); c++) {
    if (cols[c] == "case_id") i_case = (int)c;
    else if (cols[c] == "actual_class") i_actual = (int)c;
    else {
      for (int s = 0; s < NN_INPUTS; s++) {
        std::string want = std::string("adc_") + SENSOR_NAMES[s];
        if (cols[c] == want) i_adc[s] = (int)c;
      }
    }
  }
  for (int s = 0; s < NN_INPUTS; s++) {
    if (i_adc[s] < 0) {
      std::fprintf(stderr, "ERROR: kolom adc_%s tidak ditemukan\n", SENSOR_NAMES[s]);
      return 2;
    }
  }

  int n = 0, n_correct = 0, n_confident = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> f = splitCSV(line);

    int adc[NN_INPUTS];
    for (int s = 0; s < NN_INPUTS; s++) adc[s] = atoi(f[i_adc[s]].c_str());

    float norm[NN_INPUTS];
    float probs[NN_OUTPUTS];

    /* >>> jalur perhitungan yang sama dengan sketch Arduino <<< */
    int pred = nnProcessADC(adc, NN_INPUTS, TEST_ADC_MAX, norm, probs);
    float conf = probs[pred];

    std::printf("RESULT,%s,%s,%.6f", f[i_case].c_str(), CLASS_NAMES[pred], conf);
    for (int k = 0; k < NN_OUTPUTS; k++) std::printf(",%.6f", probs[k]);
    std::printf(",%s,%s\n",
                nnIsConfident(conf) ? "CONFIDENT" : "UNCERTAIN",
                (i_actual >= 0 && f[i_actual] == CLASS_NAMES[pred]) ? "OK" : "WRONG");
    std::fflush(stdout);

    n++;
    if (i_actual >= 0 && f[i_actual] == CLASS_NAMES[pred]) n_correct++;
    if (nnIsConfident(conf)) n_confident++;
  }

  std::printf("SUMMARY,%d,%d,%d\n", n, n_correct, n_confident);
  return 0;
}
