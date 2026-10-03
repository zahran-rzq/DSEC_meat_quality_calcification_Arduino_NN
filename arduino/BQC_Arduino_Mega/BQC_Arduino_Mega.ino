/* =====================================================================
 *  BQC_Arduino_Mega.ino
 *  Sistem Prediksi Kualitas Daging Sapi berbasis Electronic Nose
 *  Implementasi inference Neural Network MANUAL pada Arduino Mega 2560
 * ---------------------------------------------------------------------
 *  Arsitektur NN  : 11 input -> 24 (sigmoid) -> 12 (sigmoid) -> 4 (softmax)
 *  Bobot & bias   : nn_params.h  (hasil ekstraksi training TensorFlow)
 *  Perhitungan    : nn_core.h    (perkalian matriks manual, tanpa library NN)
 *
 *  Perangkat keras (lihat docs/LAPORAN.md untuk skema Proteus):
 *    - Arduino Mega 2560
 *    - 11 potensiometer 10k  -> pin analog A0..A10 (pengganti sensor MQ)
 *    - LCD 16x2 (mode 4-bit) -> RS 8, EN 9, D4..D7 = 10,11,12,13
 *    - 1 push button         -> pin 2  (ganti halaman tampilan)
 *
 *  Urutan pin sensor A0..A10 : MQ135 MQ136 MQ137 MQ138 MQ2 MQ3 MQ4 MQ5 MQ6 MQ8 MQ9
 * ===================================================================== */

#include <Arduino.h>
#include <LiquidCrystal.h>

/* Inti neural network + seluruh bobot/bias hasil training */
#include "nn_core.h"

/* ---------------------------------------------------------------- pin */
#define PIN_SENSOR_START  A0        /* A0..A10 = 11 potensiometer       */
#define PIN_BUTTON        2         /* tombol ganti halaman LCD         */

/* Resolusi analogRead Arduino (10-bit). Bila memakai ADC eksternal 12-bit
   ubah menjadi 4095.0f dan sesuaikan parameter scaling-nya.            */
#define ADC_MAX_VAL       1023.0f

/* Jumlah pembacaan ADC yang dirata-rata agar nilai stabil */
#define ADC_SAMPLES       8

/* Aktifkan input manual 11 nilai ADC lewat Serial Monitor (untuk verifikasi) */
#define ENABLE_SERIAL_INPUT 1

/* ---------------------------------------------------------------- LCD */
LiquidCrystal lcd(8, 9, 10, 11, 12, 13);   /* RS, EN, D4, D5, D6, D7 */
#define LCD_COLS 16
#define LCD_ROWS 2

/* Halaman tampilan: 0=prediksi, 1=probabilitas, 2=sensor, 3=status */
#define PAGE_COUNT 4

/* ------------------------------------------------------------------ */
static float  g_norm[NN_INPUTS];     /* input ternormalisasi [-1..1]  */
static int    g_adc[NN_INPUTS];      /* nilai ADC mentah 0..1023      */
static float  g_prob[NN_OUTPUTS];    /* probabilitas tiap kelas       */
static int    g_pred = 0;            /* indeks kelas hasil prediksi   */
static float  g_conf = 0.0f;         /* probabilitas tertinggi        */
static int    g_page = 0;
static int    g_sensorCursor = 0;
static bool   g_useSerial = false;   /* true jika input dari Serial   */
static unsigned long g_lastUpdate = 0;
static unsigned long g_lastDebounce = 0;
static bool   g_lastBtn = HIGH;

/* =====================================================================
 * 1. PEMBACAAN SENSOR  (potensiometer pengganti MQ)
 *    Rata-rata beberapa sampel untuk menekan derau.
 * =================================================================== */
static int readSensorADC(int idx) {
  long sum = 0;
  for (int s = 0; s < ADC_SAMPLES; s++) {
    sum += analogRead(PIN_SENSOR_START + idx);
  }
  return (int)(sum / ADC_SAMPLES);
}

static void readAllSensors() {
  if (!g_useSerial) {
    for (int i = 0; i < NN_INPUTS; i++) g_adc[i] = readSensorADC(i);
  }
}

/* =====================================================================
 * 2. INFERENCE -- normalisasi + perkalian matriks manual (lihat nn_core.h)
 *    nnProcessADC() adalah fungsi yang sama persis dengan yang dipakai
 *    program uji di PC, sehingga hasilnya dapat dibandingkan 1:1.
 * =================================================================== */
static void runInference() {
  g_pred = nnProcessADC(g_adc, NN_INPUTS, ADC_MAX_VAL, g_norm, g_prob);
  g_conf = g_prob[g_pred];
}

/* =====================================================================
 * 3. TAMPILAN LCD
 * =================================================================== */
static void lcdPad(const char *text, int width) {
  lcd.print(text);
  int n = strlen(text);
  for (int i = n; i < width; i++) lcd.print(' ');
}

static void showPagePredict() {
  lcd.setCursor(0, 0);
  char buf[24];
  snprintf(buf, sizeof(buf), "Q:%s", CLASS_NAMES[g_pred]);
  lcdPad(buf, LCD_COLS);
  lcd.setCursor(0, 1);
  snprintf(buf, sizeof(buf), "Conf:%6.2f%%%s",
           g_conf * 100.0f, nnIsConfident(g_conf) ? "OK" : "? ");
  lcdPad(buf, LCD_COLS);
}

static void showPageProb() {
  char buf[24];
  lcd.setCursor(0, 0);
  snprintf(buf, sizeof(buf), "ex%.2f gd%.2f", g_prob[0], g_prob[1]);
  lcdPad(buf, LCD_COLS);
  lcd.setCursor(0, 1);
  snprintf(buf, sizeof(buf), "ac%.2f sp%.2f", g_prob[2], g_prob[3]);
  lcdPad(buf, LCD_COLS);
}

static void showPageSensor() {
  char buf[24];
  lcd.setCursor(0, 0);
  snprintf(buf, sizeof(buf), "SENS %2d/%d %s",
           g_sensorCursor + 1, NN_INPUTS, g_useSerial ? "SER" : "POT");
  lcdPad(buf, LCD_COLS);
  lcd.setCursor(0, 1);
  snprintf(buf, sizeof(buf), "%-5s %4d %.2f",
           SENSOR_NAMES[g_sensorCursor], g_adc[g_sensorCursor],
           g_norm[g_sensorCursor]);
  lcdPad(buf, LCD_COLS);
}

static void showPageStatus() {
  char buf[24];
  lcd.setCursor(0, 0);
  /* LANGKAH 4.4 : logika ambang batas (threshold) pada keluaran */
  lcdPad(nnIsConfident(g_conf) ? "STATUS: YAKIN   " : "STATUS: RAGU    ", LCD_COLS);
  lcd.setCursor(0, 1);
  snprintf(buf, sizeof(buf), "Thr:%.2f L%d", (float)NN_CONF_THRESHOLD,
           CLASS_LABEL[g_pred]);
  lcdPad(buf, LCD_COLS);
}

static void updateDisplay() {
  switch (g_page) {
    case 0: showPagePredict(); break;
    case 1: showPageProb();    break;
    case 2: showPageSensor();  break;
    case 3: showPageStatus();  break;
  }
}

/* =====================================================================
 * 4. TOMBOL : ganti halaman (dengan debounce)
 * =================================================================== */
static void handleButton() {
  bool now = digitalRead(PIN_BUTTON);
  if (g_lastBtn == HIGH && now == LOW &&
      (millis() - g_lastDebounce) > 200UL) {
    g_lastDebounce = millis();
    g_page = (g_page + 1) % PAGE_COUNT;
    if (g_page == 2) g_sensorCursor = 0;
    lcd.clear();
    updateDisplay();
  }
  g_lastBtn = now;
}

/* =====================================================================
 * 5. SERIAL
 *    - Keluaran CSV : memudahkan verifikasi terhadap hasil Python
 *    - Masukan      : ketik 11 bilangan (0..1023) dipisah spasi/koma
 *                     untuk menguji baris data tertentu dari dataset
 *      ketik "p"    : kembali ke mode potensiometer
 * =================================================================== */
static void printCSV() {
  Serial.print("ADC,");
  for (int i = 0; i < NN_INPUTS; i++) {
    Serial.print(g_adc[i]);
    Serial.print(i == NN_INPUTS - 1 ? '\n' : ',');
  }
  Serial.print("PROB,");
  for (int k = 0; k < NN_OUTPUTS; k++) {
    Serial.print(g_prob[k], 6);
    Serial.print(k == NN_OUTPUTS - 1 ? '\n' : ',');
  }
  Serial.print("PRED,");
  Serial.print(CLASS_NAMES[g_pred]);
  Serial.print(',');
  Serial.print(g_conf, 6);
  Serial.print(',');
  Serial.println(nnIsConfident(g_conf) ? "CONFIDENT" : "UNCERTAIN");
}

#if ENABLE_SERIAL_INPUT
static void handleSerial() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line.equalsIgnoreCase("p")) {
    g_useSerial = false;
    Serial.println("MODE: potensiometer (A0..A10)");
    return;
  }

  line.replace(',', ' ');

  /* salin ke buffer lokal agar strtok() dapat menulis pemisah '\0' */
  static char buf[128];
  strncpy(buf, line.c_str(), sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  int vals[NN_INPUTS];
  int n = 0;
  char *save = NULL;
  char *tok = strtok_r(buf, " \t", &save);
  while (tok != NULL && n < NN_INPUTS) {
    vals[n++] = atoi(tok);
    tok = strtok_r(NULL, " \t", &save);
  }
  if (n != NN_INPUTS) {
    Serial.print("ERROR: butuh ");
    Serial.print(NN_INPUTS);
    Serial.println(" nilai ADC (0..1023). Contoh: 512 300 ...");
    return;
  }
  for (int i = 0; i < NN_INPUTS; i++) g_adc[i] = vals[i];
  g_useSerial = true;
  Serial.println("MODE: input serial diterima");
}
#endif

/* ===================================================================== */
void setup() {
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  Serial.begin(115200);
  lcd.begin(LCD_COLS, LCD_ROWS);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("BEEF QUALITY NN ");
  lcd.setCursor(0, 1);
  lcd.print("E-Nose 11 sensor");
  delay(1500);
  lcd.clear();

  Serial.println();
  Serial.println("=== Beef Quality Classification - Arduino Mega ===");
  Serial.print("Arsitektur : ");
  Serial.print(NN_INPUTS);
  for (int L = 0; L < NN_N_LAYERS; L++) {
    Serial.print(" -> ");
    Serial.print(LAYER_OUT[L]);
  }
  Serial.println();
  Serial.print("Threshold  : ");
  Serial.println((float)NN_CONF_THRESHOLD, 3);
  Serial.println("Ketik 11 nilai ADC (0..1023) dipisah spasi untuk uji manual,");
  Serial.println("atau ketik 'p' untuk kembali ke potensiometer.");
}

void loop() {
  handleButton();
#if ENABLE_SERIAL_INPUT
  handleSerial();
#endif

  /* perbarui tiap 250 ms */
  if (millis() - g_lastUpdate < 250UL) return;
  g_lastUpdate = millis();

  readAllSensors();
  runInference();
  updateDisplay();

  if (g_page == 2) {
    g_sensorCursor = (g_sensorCursor + 1) % NN_INPUTS;
  }
  printCSV();
}
