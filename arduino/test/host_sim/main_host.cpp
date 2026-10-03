/* =====================================================================
 * main_host.cpp -- menjalankan BQC_Arduino_Mega.ino di PC (host simulation)
 *
 * Program ini MENGOMPILASI FILE .ino YANG SEBENARNYA (bukan salinan logika)
 * dengan shim antarmuka Arduino/LCD, lalu:
 *   1. menyuapkan nilai ADC kasus uji ke "potensiometer" virtual,
 *   2. menjalankan setup()/loop() seperti di mikrokontroler,
 *   3. memeriksa isi layar LCD dan keluaran Serial.
 *
 * Cara pakai (lihat arduino/test/run_checks.sh):
 *   g++ -O2 -std=gnu++11 -I . -I ../../BQC_Arduino_Mega main_host.cpp -o host_sim
 *   ./host_sim ../../../docs/test_vectors.csv
 * ===================================================================== */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Arduino.h"

/* definisi global yang dideklarasikan extern oleh shim Arduino.h */
unsigned long host_millis = 0;
int host_adc[128] = {0};
int host_button = HIGH;
HardwareSerialShim Serial;

/* >>> sketch Arduino yang diuji, di-include apa adanya <<< */
#include "BQC_Arduino_Mega.ino"

/* ------------------------------------------------------------- helper */
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

static int g_fail = 0;
static int g_pass = 0;

static void check(bool cond, const std::string &what) {
  if (cond) { g_pass++; std::printf("  [PASS] %s\n", what.c_str()); }
  else      { g_fail++; std::printf("  [FAIL] %s\n", what.c_str()); }
}

/* jalankan loop() sampai satu siklus pembaruan (gate 250 ms) selesai */
static void pumpOnce() {
  host_millis += 260;
  loop();
}

/* baca isi baris LCD dari buffer LiquidCrystal */
static std::string lcdLine(int row) {
  std::string s = lcd.screen();
  size_t start = 0;
  for (int r = 0; r <= row; r++) {
    size_t a = s.find('|', start);
    size_t b = s.find('|', a + 1);
    if (a == std::string::npos || b == std::string::npos) return "";
    std::string line = s.substr(a + 1, b - a - 1);
    if (r == row) {
      size_t e = line.find_last_not_of(' ');
      return (e == std::string::npos) ? "" : line.substr(0, e + 1);
    }
    start = b + 1;
  }
  return "";
}

/* =============================================================== main */
int main(int argc, char **argv) {
  const char *csv = (argc > 1) ? argv[1] : "../../../docs/test_vectors.csv";

  std::ifstream in(csv);
  if (!in) { std::fprintf(stderr, "tidak dapat membuka %s\n", csv); return 2; }

  std::string header;
  std::getline(in, header);
  std::vector<std::string> cols = splitCSV(header);
  int i_case = -1, i_pred = -1, i_conf = -1;
  std::vector<int> i_adc(NN_INPUTS, -1);
  for (size_t c = 0; c < cols.size(); c++) {
    if (cols[c] == "case_id") i_case = (int)c;
    else if (cols[c] == "pred_class") i_pred = (int)c;
    else if (cols[c] == "confidence") i_conf = (int)c;
    else for (int s = 0; s < NN_INPUTS; s++)
      if (cols[c] == std::string("adc_") + SENSOR_NAMES[s]) i_adc[s] = (int)c;
  }

  /* ------------------------------------------------- 1. mode potensiometer */
  std::printf("\n=== [1] setup() + mode potensiometer ===\n");
  setup();
  /* splash ditulis ke LCD lalu dihapus lcd.clear() di akhir setup(),
     jadi yang dapat diperiksa setelah setup() adalah banner Serial. */
  check(Serial.out.find("Beef Quality Classification") != std::string::npos,
        "setup() mencetak banner ke Serial");
  check(Serial.out.find("11 -> 24 -> 12 -> 4") != std::string::npos,
        "setup() melaporkan arsitektur NN dari nn_params.h");
  check(lcdLine(0).empty() && lcdLine(1).empty(),
        "LCD bersih setelah splash setup() selesai");

  struct Case { int id; int adc[NN_INPUTS]; std::string pred; float conf; };
  std::vector<Case> cases;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> f = splitCSV(line);
    Case c;
    c.id = atoi(f[i_case].c_str());
    for (int s = 0; s < NN_INPUTS; s++) c.adc[s] = atoi(f[i_adc[s]].c_str());
    c.pred = f[i_pred];
    c.conf = (float)atof(f[i_conf].c_str());
    cases.push_back(c);
  }
  in.close();
  std::printf("  %d kasus uji dimuat dari %s\n", (int)cases.size(), csv);

  int shown_ok = 0;
  for (size_t k = 0; k < cases.size(); k++) {
    const Case &c = cases[k];
    for (int s = 0; s < NN_INPUTS; s++) host_adc[A0 + s] = c.adc[s];
    Serial.out.clear();
    pumpOnce();

    /* prediksi ulang lewat jalur yang sama, sebagai pembanding LCD */
    float nrm[NN_INPUTS], prb[NN_OUTPUTS];
    int expect = nnProcessADC(c.adc, NN_INPUTS, ADC_MAX_VAL, nrm, prb);

    std::string l0 = lcdLine(0);
    bool ok = l0.find(CLASS_NAMES[expect]) != std::string::npos;
    if (ok) shown_ok++;
    if (k < 4 || !ok) {
      std::printf("  kasus %2d  ADC=%4d %4d %4d ...  LCD: [%s / %s]  harap=%s %s\n",
                  c.id, c.adc[0], c.adc[1], c.adc[2],
                  l0.c_str(), lcdLine(1).c_str(), CLASS_NAMES[expect],
                  ok ? "" : "  <-- TIDAK COCOK");
    }
  }
  check(shown_ok == (int)cases.size(),
        "LCD menampilkan kelas yang benar untuk semua kasus");

  /* ------------------------------------------- 2. konsistensi dgn Python */
  std::printf("\n=== [2] prediksi .ino vs prediksi Python (kolom pred_class CSV) ===\n");
  int match = 0;
  for (size_t k = 0; k < cases.size(); k++) {
    for (int s = 0; s < NN_INPUTS; s++) host_adc[A0 + s] = cases[k].adc[s];
    pumpOnce();
    if (lcdLine(0).find(cases[k].pred) != std::string::npos) match++;
  }
  std::printf("  kelas sama seperti hasil Python : %d/%d\n", match, (int)cases.size());
  check(match == (int)cases.size(), "semua kelas cocok dengan Python");

  /* ------------------------------------------------- 3. input via Serial */
  std::printf("\n=== [3] input manual lewat Serial Monitor ===\n");
  {
    const Case &c = cases[8];              /* kasus acceptable */
    std::string cmd;
    for (int s = 0; s < NN_INPUTS; s++) {
      cmd += std::to_string(c.adc[s]);
      cmd += (s == NN_INPUTS - 1) ? "\n" : " ";
    }
    Serial.inbox = cmd;
    Serial.pos = 0;
    Serial.out.clear();
    pumpOnce();
    check(Serial.out.find("input serial diterima") != std::string::npos,
          "sketch menerima 11 nilai ADC dari Serial");

    float nrm[NN_INPUTS], prb[NN_OUTPUTS];
    int expect = nnProcessADC(c.adc, NN_INPUTS, ADC_MAX_VAL, nrm, prb);
    Serial.out.clear();
    pumpOnce();
    check(lcdLine(0).find(CLASS_NAMES[expect]) != std::string::npos,
          "LCD menampilkan hasil dari input serial");
    check(Serial.out.find("PROB,") != std::string::npos,
          "sketch mencetak baris CSV PROB ke Serial");

    /* mode serial tetap bertahan walau potensiometer diubah */
    for (int s = 0; s < NN_INPUTS; s++) host_adc[A0 + s] = 0;
    pumpOnce();
    check(lcdLine(0).find(CLASS_NAMES[expect]) != std::string::npos,
          "mode serial tidak terpengaruh perubahan potensiometer");

    /* kembali ke potensiometer */
    Serial.inbox = "p\n";
    Serial.pos = 0;
    Serial.out.clear();
    pumpOnce();
    check(Serial.out.find("potensiometer") != std::string::npos,
          "perintah 'p' mengembalikan mode potensiometer");
  }

  /* -------------------------------------------------- 4. halaman & tombol */
  std::printf("\n=== [4] tombol ganti halaman LCD ===\n");
  {
    std::vector<std::string> seen;
    for (int p = 0; p < PAGE_COUNT; p++) {
      host_button = LOW;                 /* tombol ditekan */
      pumpOnce();
      host_button = HIGH;
      pumpOnce();
      seen.push_back(lcdLine(0));
      std::printf("  halaman %d : [%s / %s]\n", p, lcdLine(0).c_str(), lcdLine(1).c_str());
    }
    bool distinct = true;
    for (size_t a = 0; a < seen.size(); a++)
      for (size_t b = a + 1; b < seen.size(); b++)
        if (seen[a] == seen[b]) distinct = false;
    check(distinct, "tiap halaman menampilkan konten berbeda");
  }

  /* ------------------------------------------------------ 5. ambang batas */
  std::printf("\n=== [5] logika ambang batas (threshold) ===\n");
  {
    int adc_hi[NN_INPUTS], adc_lo[NN_INPUTS];
    for (int s = 0; s < NN_INPUTS; s++) { adc_hi[s] = 1023; adc_lo[s] = 0; }

    float nrm[NN_INPUTS], prb[NN_OUTPUTS];
    int p1 = nnProcessADC(adc_hi, NN_INPUTS, ADC_MAX_VAL, nrm, prb);
    bool conf_hi = nnIsConfident(prb[p1]);
    std::printf("  semua pot 1023 -> %s conf=%.4f -> %s\n",
                CLASS_NAMES[p1], prb[p1], conf_hi ? "YAKIN" : "RAGU");
    check(prb[p1] >= 0.0f && prb[p1] <= 1.0f, "confidence berada pada rentang 0..1");

    int p0 = nnProcessADC(adc_lo, NN_INPUTS, ADC_MAX_VAL, nrm, prb);
    std::printf("  semua pot    0 -> %s conf=%.4f -> %s\n",
                CLASS_NAMES[p0], prb[p0], nnIsConfident(prb[p0]) ? "YAKIN" : "RAGU");
    check(prb[p0] >= 0.0f && prb[p0] <= 1.0f,
          "confidence tetap valid pada input ekstrem 0");

    float sum = 0.0f;
    for (int k = 0; k < NN_OUTPUTS; k++) sum += prb[k];
    std::printf("  jumlah probabilitas softmax = %.6f\n", sum);
    check(sum > 0.999f && sum < 1.001f, "keluaran softmax berjumlah 1");

    check(nnIsConfident(0.75f) == true,  "conf 0.75 >= threshold 0.50 -> yakin");
    check(nnIsConfident(0.25f) == false, "conf 0.25 <  threshold 0.50 -> ragu");
  }

  /* ------------------------------------------------------------ ringkasan */
  std::printf("\n================ RINGKASAN HOST SIM ================\n");
  std::printf("  PASS : %d\n  FAIL : %d\n", g_pass, g_fail);
  std::printf("====================================================\n");
  return g_fail == 0 ? 0 : 1;
}
