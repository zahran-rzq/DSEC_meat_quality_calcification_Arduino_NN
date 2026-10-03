/* ============================================================================
 *  verification/stubs/Arduino.h
 *  Stub minimal supaya file .ino (kode Arduino) bisa dikompilasi dengan g++
 *  di PC dan diuji TANPA hardware / Proteus.
 *  Hanya untuk keperluan pengujian - tidak dipakai saat upload ke Arduino.
 * ==========================================================================*/
#ifndef ARDUINO_STUB_H
#define ARDUINO_STUB_H

#include <math.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#define PROGMEM
#define F(str) (str)
#define HIGH 1
#define LOW 0
#define OUTPUT 1
#define INPUT 0

/* --- pin analog (Arduino Mega 2560: A0..A15) --- */
enum {
  A0 = 54, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15
};

/* --- nilai potensiometer yang "dibaca" analogRead() --- */
extern int   g_pot_values[70];
extern unsigned long g_millis;

inline void pinMode(int, int) {}
inline void digitalWrite(int, int) {}
inline int  analogRead(int pin) { return g_pot_values[pin]; }
inline unsigned long millis() { return g_millis; }
inline void delay(unsigned long ms) { g_millis += (ms < 1000 ? ms : 1000); }
inline long map(long x, long a, long b, long c, long d) { return (x - a) * (d - c) / (b - a) + c; }

inline char *dtostrf(double value, int width, int prec, char *buf) {
  char fmt[16];
  snprintf(fmt, sizeof(fmt), "%%%d.%df", width, prec);
  sprintf(buf, fmt, value);
  return buf;
}

/* --- Serial Monitor (cetak ke stdout) --- */
class HardwareSerialStub {
public:
  void begin(long) {}
  void print(const char *s)             { printf("%s", s); }
  void print(char *s)                   { printf("%s", s); }
  void print(char c)                    { printf("%c", c); }
  void print(int v)                     { printf("%d", v); }
  void print(unsigned int v)            { printf("%u", v); }
  void print(long v)                    { printf("%ld", v); }
  void print(unsigned long v)           { printf("%lu", v); }
  void print(double v)                  { printf("%.2f", v); }
  void print(double v, int digits)      { printf("%.*f", digits, v); }
  void println()                        { printf("\n"); }
  void println(const char *s)           { printf("%s\n", s); }
  void println(char *s)                 { printf("%s\n", s); }
  void println(char c)                  { printf("%c\n", c); }
  void println(int v)                   { printf("%d\n", v); }
  void println(long v)                  { printf("%ld\n", v); }
  void println(double v, int digits = 2){ printf("%.*f\n", digits, v); }
  operator bool() const { return true; }
};
extern HardwareSerialStub Serial;

#endif
