/* =====================================================================
 * Arduino.h -- SHIM untuk mengompilasi sketch di PC (host simulation)
 *
 * TUJUAN : memungkinkan BQC_Arduino_Mega.ino dikompilasi & dijalankan di
 *          PC (g++) sehingga logika sketch dapat diuji tanpa perangkat
 *          keras / simulator Proteus.
 *
 * CATATAN: ini HANYA tiruan antarmuka (API) Arduino. Perhitungan neural
 *          network TIDAK ditiru -- ia diambil apa adanya dari nn_core.h
 *          dan nn_params.h yang sama dengan yang dipakai di mikrokontroler.
 * ===================================================================== */
#ifndef ARDUINO_SHIM_H
#define ARDUINO_SHIM_H

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>

/* -------------------------------------------------- konstanta Arduino */
#define HIGH 1
#define LOW  0
#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define LSBFIRST 0
#define MSBFIRST 1

/* Pin analog Mega 2560: A0..A15 = 54..69 */
#define A0 54
#define A1 55
#define A2 56
#define A3 57
#define A4 58
#define A5 59
#define A6 60
#define A7 61
#define A8 62
#define A9 63
#define A10 64
#define A11 65
#define A12 66
#define A13 67
#define A14 68
#define A15 69

/* --------------------------------------------------- kelas String mini */
class String {
 public:
  String() : s_() {}
  String(const char *p) : s_(p ? p : "") {}
  String(const std::string &x) : s_(x) {}

  unsigned int length() const { return (unsigned int)s_.size(); }

  void trim() {
    size_t a = s_.find_first_not_of(" \t\r\n");
    size_t b = s_.find_last_not_of(" \t\r\n");
    s_ = (a == std::string::npos) ? "" : s_.substr(a, b - a + 1);
  }

  bool equalsIgnoreCase(const char *other) const {
    if (!other) return false;
    if (s_.size() != strlen(other)) return false;
    for (size_t i = 0; i < s_.size(); i++) {
      if (tolower((unsigned char)s_[i]) != tolower((unsigned char)other[i])) return false;
    }
    return true;
  }

  void replace(char from, char to) {
    for (size_t i = 0; i < s_.size(); i++) if (s_[i] == from) s_[i] = to;
  }

  /* buffer dapat ditulis, meniru perilaku String Arduino (dipakai strtok_r) */
  char *c_str() { return s_.empty() ? &nul_ : &s_[0]; }
  const char *c_str() const { return s_.c_str(); }

  const std::string &std_str() const { return s_; }

 private:
  std::string s_;
  char nul_ = '\0';
};

/* ------------------------------------------------------- Serial (mini) */
class HardwareSerialShim {
 public:
  std::string out;        /* semua yang dicetak sketch terkumpul di sini */
  std::string inbox;      /* masukan yang "diketik" ke Serial Monitor    */
  size_t pos = 0;

  void begin(long) {}
  void end() {}

  int available() { return (int)(inbox.size() - pos); }

  String readStringUntil(char term) {
    std::string r;
    while (pos < inbox.size()) {
      char c = inbox[pos++];
      if (c == term) break;
      r.push_back(c);
    }
    return String(r);
  }

  void print(const char *s) { out += (s ? s : ""); }
  void print(char c) { out.push_back(c); }
  void print(int v) { out += std::to_string(v); }
  void print(long v) { out += std::to_string(v); }
  void print(unsigned long v) { out += std::to_string(v); }
  void print(float v) { char b[32]; snprintf(b, sizeof b, "%.2f", v); out += b; }
  void print(double v) { print((float)v); }
  void print(float v, int d) { char b[32]; snprintf(b, sizeof b, "%.*f", d, v); out += b; }

  void println() { out.push_back('\n'); }
  void println(const char *s) { print(s); println(); }
  void println(char c) { print(c); println(); }
  void println(int v) { print(v); println(); }
  void println(long v) { print(v); println(); }
  void println(float v) { print(v); println(); }
  void println(float v, int d) { print(v, d); println(); }
};

/* ------------------------------------------------- state simulasi host */
extern unsigned long host_millis;
extern int host_adc[128];
extern int host_button;
extern HardwareSerialShim Serial;

inline unsigned long millis() { return host_millis; }
inline unsigned long micros() { return host_millis * 1000UL; }
inline void delay(unsigned long ms) { host_millis += ms; }
inline void delayMicroseconds(unsigned int) {}

inline int analogRead(int pin) { return host_adc[pin & 127]; }
inline void analogWrite(int, int) {}
inline int digitalRead(int pin) { (void)pin; return host_button; }
inline void digitalWrite(int, int) {}
inline void pinMode(int, int) {}

#endif /* ARDUINO_SHIM_H */
