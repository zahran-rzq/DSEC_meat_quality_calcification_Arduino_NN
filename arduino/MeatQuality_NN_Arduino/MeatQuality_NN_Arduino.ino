/* ============================================================================
 *  MeatQuality_NN.ino
 *  ---------------------------------------------------------------------------
 *  RP Ganjil - SISTEM PREDIKSI KUALITAS DAGING BERBASIS ELECTRONIC NOSE
 *
 *  Implementasi Neural Network hasil training (Python/TensorFlow) ke
 *  Arduino Mega 2560 DENGAN PERHITUNGAN MATRIKS MANUAL - tanpa library
 *  machine learning apa pun (hanya LiquidCrystal.h bawaan Arduino IDE).
 *
 *  ARSITEKTUR (hasil training, lihat model_params.h):
 *      Input      : 11 neuron  (11 sensor MQ)
 *      Hidden     : 20 neuron, aktivasi Sigmoid
 *      Output #1  : 4 neuron,  aktivasi Softmax   -> Excellent/Good/Acceptable/Spoiled
 *      Output #2  : 1 neuron,  aktivasi Linear    -> TVC (log10 CFU/g)
 *
 *  ALUR:
 *      1. analogRead() 11 pin  -> nilai ADC 0..1023
 *      2. konversi      -> nilai sensor (0..60), sama seperti dataset
 *      3. normalisasi   -> x_norm = 2*(x-min)/(max-min) - 1   (rentang -1..1)
 *      4. hidden  : h = sigmoid( W1 . x_norm + b1 )           (matriks 20x11)
 *      5. kelas   : o = softmax( W2 . h + b2 )                (matriks 4x20)
 *      6. TVC     : t = W2t . h + b2t  (linear, 1x20) lalu di-denormalisasi
 *      7. threshold: TVC<3 Excellent, 3-4 Good, 4-5 Acceptable, >=5 Spoiled
 *      8. tampilkan ke LCD 20x4 + Serial Monitor
 *
 *  PENGUJIAN DI PROTEUS:
 *      - INPUT_MODE 0 : baca 11 potensiometer pada pin A0..A10
 *                        (potensiometer = pengganti sensor MQ)
 *      - INPUT_MODE 1 : pakai nilai TEST_VECTOR dari model_params.h
 *                        (untuk membandingkan hasil persis dengan Python)
 *
 *  LCD 20x4 (mode 4-bit):
 *      RS -> D12, E -> D11, D4 -> D5, D5 -> D4, D6 -> D3, D7 -> D2
 *      RW -> GND, VSS -> GND, VDD -> +5V, V0 -> potensiometer kontras
 *
 *  Penulis sketch  : (isi nama Anda) - dari template RP Ganjil
 *  Sumber parameter: model_params.h (hasil ekspor otomatis dari Python)
 * ==========================================================================*/

#include <LiquidCrystal.h>
#include "model_params.h"

/* ---------------------------------------------------------------------------
 *  KONFIGURASI
 * -------------------------------------------------------------------------*/
#define INPUT_MODE        0      // 0 = potensiometer (Proteus), 1 = TEST_VECTOR
#define USE_BOTH_MODELS   1      // 1 = tampilkan kelas DAN estimasi TVC
#define SERIAL_BAUD       115200
#define SENSOR_SETTLE_MS  300    // jeda setelah ganti mode

/* LCD: RS, E, D4, D5, D6, D7 */
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

/* ---------------------------------------------------------------------------
 *  VARIABEL GLOBAL
 * -------------------------------------------------------------------------*/
float x_input[NN_N_INPUT];          // nilai sensor mentah (0..60)
float x_norm[NN_N_INPUT];           // hasil normalisasi (-1..1)

float hidden_cls[NN_N_HIDDEN_CLS];  // aktivasi hidden layer model klasifikasi
float prob_cls[NN_N_CLASS];         // probabilitas tiap kelas (softmax)

float hidden_tvc[NN_N_HIDDEN_TVC];  // aktivasi hidden layer model regresi
float tvc_value = 0.0f;             // hasil estimasi TVC (log10 CFU/g)

int   pred_class = 0;               // indeks kelas prediksi (0..3)
unsigned long last_print = 0;

/* ===========================================================================
 *  1. FUNGSI AKTIVASI (ditulis manual, tanpa library)
 * ==========================================================================*/

/* Sigmoid:  s(x) = 1 / (1 + e^-x) */
float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

/* ReLU:  r(x) = max(0, x)   -- disediakan bila ingin ganti aktivasi hidden */
float relu(float x) {
  return (x > 0.0f) ? x : 0.0f;
}

/* Softmax untuk array berisi n nilai:  y_i = e^(x_i) / sum(e^(x_j))
 * Trik numerik: kurangi nilai maksimum supaya expf() tidak overflow
 * (di Arduino, expf(>88) = inf). */
void softmax(const float *x, float *y, int n) {
  float xmax = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > xmax) xmax = x[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    y[i] = expf(x[i] - xmax);
    sum += y[i];
  }
  if (sum <= 0.0f) sum = 1.0f;              // jaga-jaga
  for (int i = 0; i < n; i++) {
    y[i] = y[i] / sum;
  }
}

/* ===========================================================================
 *  2. INPUT SCALING
 *     ADC (0..1023) -> nilai sensor (0..60) -> normalisasi (-1..1)
 * ==========================================================================*/
float adcToSensor(int adc) {
  /* Potensiometer Proteus 0..5 V identik dengan pembacaan sensor 0..60.
     Kalau nanti dipakai sensor MQ asli, ganti baris ini dengan persamaan
     kalibrasi Rs/R0 milik sensor tersebut. */
  return (float)adc * (NN_SENSOR_FULL_SCALE / NN_ADC_FULL_SCALE);
}

float normalizeInput(float value, int i) {
  float range = NORM_MAX[i] - NORM_MIN[i];
  if (range == 0.0f) range = 1.0f;
  return 2.0f * (value - NORM_MIN[i]) / range - 1.0f;
}

void readSensors() {
  for (int i = 0; i < NN_N_INPUT; i++) {
    if (INPUT_MODE == 1) {
      x_input[i] = TEST_VECTOR[i];                    // data uji dari Python
    } else {
      int adc = analogRead(SENSOR_PINS[i]);           // baca potensiometer
      x_input[i] = adcToSensor(adc);
    }
    x_norm[i] = normalizeInput(x_input[i], i);
  }
}

/* ===========================================================================
 *  3. INFERENCE - PERKALIAN MATRIKS MANUAL
 *
 *     Untuk setiap neuron: output = activation( sum(w_i * x_i) + bias )
 *     Loop bersarang di bawah ini PERSIS operasi matriks  o = f(W.x + b).
 * ==========================================================================*/

/* Hidden layer: input 11 -> neuron sebanyak n_hidden, aktivasi sigmoid */
void forwardHidden(const float *in, int n_in, const float W[][NN_N_INPUT],
                   const float *b, int n_hidden, float *out) {
  for (int j = 0; j < n_hidden; j++) {
    float sum = b[j];                          // mulai dari bias
    for (int i = 0; i < n_in; i++) {
      sum += W[j][i] * in[i];                  // akumulasi w * x
    }
    out[j] = sigmoid(sum);                     // fungsi aktivasi manual
  }
}

/* Output layer klasifikasi: n_hidden -> n_class, aktivasi softmax */
void forwardOutputClass(const float *in, int n_in, const float W[][NN_N_HIDDEN_CLS],
                        const float *b, int n_class, float *prob) {
  float z[NN_N_CLASS];
  for (int k = 0; k < n_class; k++) {
    float sum = b[k];
    for (int j = 0; j < n_in; j++) {
      sum += W[k][j] * in[j];
    }
    z[k] = sum;
  }
  softmax(z, prob, n_class);
}

/* Output layer regresi TVC: n_hidden -> 1, aktivasi linear (identitas) */
float forwardOutputTVC(const float *in, int n_in, const float *W,
                       const float *b) {
  float sum = b[0];
  for (int j = 0; j < n_in; j++) {
    sum += W[j] * in[j];
  }
  return sum;                                  // linear: y = z
}

/* ===========================================================================
 *  4. LOGIKA OUTPUT / THRESHOLD
 * ==========================================================================*/
int argmax(const float *v, int n) {
  int idx = 0;
  for (int i = 1; i < n; i++) {
    if (v[i] > v[idx]) idx = i;
  }
  return idx;
}

/* Kategori kualitas dari nilai TVC (log10 CFU/g).
 * Ambang batas ini identik dengan aturan pelabelan pada dataset:
 *   TVC < 3          -> 0 Excellent   (masih segar)
 *   3 <= TVC < 4     -> 1 Good
 *   4 <= TVC < 5     -> 2 Acceptable
 *   TVC >= 5         -> 3 Spoiled     (tidak layak konsumsi) */
int qualityFromTVC(float tvc) {
  if (tvc < TVC_THRESHOLDS[0]) return 0;
  if (tvc < TVC_THRESHOLDS[1]) return 1;
  if (tvc < TVC_THRESHOLDS[2]) return 2;
  return 3;
}

/* Kategori kualitas dari output softmax model klasifikasi */
int qualityFromSoftmax(const float *prob) {
  return argmax(prob, NN_N_CLASS);
}

/* ===========================================================================
 *  5. SETUP & LOOP
 * ==========================================================================*/
void setup() {
  Serial.begin(SERIAL_BAUD);
  lcd.begin(20, 4);

  lcd.setCursor(0, 0); lcd.print("Prediksi Kualitas");
  lcd.setCursor(0, 1); lcd.print("Daging - E-Nose");
  lcd.setCursor(0, 2); lcd.print("NN 11-");
  lcd.print(NN_N_HIDDEN_CLS);
  lcd.print("-");
  lcd.print(NN_N_CLASS);
  Serial.println(F("=========================================="));
  Serial.println(F(" Prediksi Kualitas Daging - Electronic Nose"));
  Serial.print(F(" Arsitektur NN: 11 - "));
  Serial.print(NN_N_HIDDEN_CLS);
  Serial.print(F(" (sigmoid) - "));
  Serial.print(NN_N_CLASS);
  Serial.println(F(" (softmax) / 1 (linear TVC)"));
  Serial.print(F(" Mode input: "));
  Serial.println(INPUT_MODE == 1 ? F("TEST_VECTOR (data uji)") : F("POTENSIOMETER"));
  Serial.println(F("=========================================="));
  delay(1500);
  lcd.clear();
}

void loop() {
  /* --- (a) baca 11 input & normalisasi --- */
  readSensors();

  /* --- (b) hidden layer (berbagi input yang sama untuk dua model) --- */
  forwardHidden(x_norm, NN_N_INPUT, CLS_W1, CLS_B1, NN_N_HIDDEN_CLS, hidden_cls);
  forwardHidden(x_norm, NN_N_INPUT, TVC_W1, TVC_B1, NN_N_HIDDEN_TVC, hidden_tvc);

  /* --- (c) output layer --- */
  forwardOutputClass(hidden_cls, NN_N_HIDDEN_CLS, CLS_W2, CLS_B2, NN_N_CLASS, prob_cls);

  float tvc_norm = forwardOutputTVC(hidden_tvc, NN_N_HIDDEN_TVC, TVC_W2, TVC_B2);
  /* denormalisasi: (0..?) -> rentang TVC sebenarnya */
  tvc_value = (tvc_norm - TVC_OUT_LO) / (TVC_OUT_HI - TVC_OUT_LO)
              * (TVC_MAX_LOG - TVC_MIN_LOG) + TVC_MIN_LOG;

  /* --- (d) logika output --- */
  pred_class = qualityFromSoftmax(prob_cls);            // dari model klasifikasi
  int class_by_tvc = qualityFromTVC(tvc_value);         // dari model regresi

#if USE_BOTH_MODELS
  int shown_class = class_by_tvc;                       // kategori final
#else
  int shown_class = pred_class;
#endif

  /* --- (e) tampilkan ke LCD --- */
  char buf[21];
  lcd.setCursor(0, 0);
  dtostrf(tvc_value, 5, 2, buf);
  lcd.print("TVC="); lcd.print(buf); lcd.print(" logCFU/g");
  lcd.setCursor(0, 1);
  lcd.print("Kualitas: ");
  lcd.print(CLASS_NAMES[shown_class]);
  lcd.print("      ");
  lcd.setCursor(0, 2);
  lcd.print("Softmax: ");
  lcd.print(CLASS_NAMES[pred_class]);
  lcd.print("      ");
  lcd.setCursor(0, 3);
  for (int k = 0; k < NN_N_CLASS; k++) {
    lcd.print((int)(prob_cls[k] * 100));
    lcd.print("% ");
  }
  lcd.print("   ");

  /* --- (f) kirim detail ke Serial Monitor --- */
  if (millis() - last_print > 1000) {          // tiap 1 detik
    last_print = millis();
    Serial.println(F("------------------------------------------"));
    Serial.print(F("Input (sensor -> ADC -> normalisasi):"));
    for (int i = 0; i < NN_N_INPUT; i++) {
      Serial.print(F("\n  "));
      Serial.print(SENSOR_NAMES[i]);
      Serial.print(F(" = "));
      Serial.print(x_input[i], 2);
      Serial.print(F("  -> "));
      Serial.print(x_norm[i], 4);
    }
    Serial.println();
    Serial.print(F("Probabilitas softmax :"));
    for (int k = 0; k < NN_N_CLASS; k++) {
      Serial.print(F("  "));
      Serial.print(CLASS_NAMES[k]);
      Serial.print(F("="));
      Serial.print(prob_cls[k] * 100.0f, 2);
      Serial.print(F("%"));
    }
    Serial.println();
    Serial.print(F("Prediksi softmax     : "));
    Serial.println(CLASS_NAMES[pred_class]);
    Serial.print(F("Estimasi TVC         : "));
    Serial.print(tvc_value, 3);
    Serial.println(F(" log10 CFU/g"));
    Serial.print(F("Kategori dari TVC    : "));
    Serial.println(CLASS_NAMES[class_by_tvc]);
    Serial.print(F("Keputusan final LCD  : "));
    Serial.println(CLASS_NAMES[shown_class]);
    if (shown_class == 3) {
      Serial.println(F(">> Daging TIDAK LAYAK dikonsumsi (TVC >= 5 log CFU/g)"));
    }
  }

  delay(200);
}
