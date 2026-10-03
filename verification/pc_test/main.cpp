/* ============================================================================
 *  verification/pc_test/main.cpp
 *  ---------------------------------------------------------------------------
 *  Menjalankan KODE ARDUINO YANG SAMA (file .ino) di PC memakai g++ supaya
 *  perhitungan matriks Neural Network bisa diverifikasi terhadap Keras
 *  TANPA hardware dan TANPA Proteus.
 *
 *  Kompilasi + jalankan otomatis:  python verification/verify_cpp_vs_keras.py
 *  Kompilasi manual:
 *      g++ -O2 -I verification/stubs -o /tmp/pc_test \
 *          verification/pc_test/main.cpp -lm
 *
 *  Cara pakai binary:
 *      ./pc_test 17.85 15.58 7.95 21.23 16.52 24.07 12.29 9.02 14.26 40.29 14.40
 *  (11 nilai sensor; dikonversi ke ADC seperti pada Proteus, lalu loop() dipanggil)
 * ==========================================================================*/

#include <cstdio>
#include <cstdlib>

#include "Arduino.h"
#include "LiquidCrystal.h"

/* ---- definisi variabel stub (dideklarasikan 'extern' di Arduino.h) ---- */
int g_pot_values[70] = {0};
unsigned long g_millis = 0;
HardwareSerialStub Serial;

/* ---- panggil file .ino ASLI (tanpa duplikasi kode) ---- */
#include "../../arduino/MeatQuality_NN_Arduino/MeatQuality_NN_Arduino.ino"

int main(int argc, char **argv) {
  const int n_sensor = NN_N_INPUT;
  if (argc < 1 + n_sensor) {
    fprintf(stderr, "Butuh %d nilai sensor sebagai argumen.\n", n_sensor);
    fprintf(stderr, "Contoh: ./pc_test 17.85 15.58 7.95 21.23 16.52 24.07 12.29 9.02 14.26 40.29 14.40\n");
    return 1;
  }

  /* set potensiometer "Proteus": nilai sensor -> ADC (0..1023) */
  for (int i = 0; i < n_sensor; i++) {
    float v = atof(argv[1 + i]);
    int adc = (int)(v * NN_ADC_FULL_SCALE / NN_SENSOR_FULL_SCALE + 0.5f);
    if (adc < 0) adc = 0;
    if (adc > 1023) adc = 1023;
    g_pot_values[SENSOR_PINS[i]] = adc;
  }

  setup();
  printf("\n");
  for (int k = 0; k < 3; k++) loop();     /* 3 kali supaya nilai stabil */
  lcd.dump();
  return 0;
}
