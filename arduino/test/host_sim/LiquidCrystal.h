/* =====================================================================
 * LiquidCrystal.h -- SHIM LCD karakter untuk simulasi host.
 * Menyimpan isi layar ke buffer teks sehingga dapat dicetak & diperiksa.
 * ===================================================================== */
#ifndef LIQUIDCRYSTAL_SHIM_H
#define LIQUIDCRYSTAL_SHIM_H

#include <cstring>
#include <string>

class LiquidCrystal {
 public:
  static const int MAX_COLS = 20;
  static const int MAX_ROWS = 4;

  LiquidCrystal(int, int, int, int, int, int) { reset(); }

  void begin(int cols, int rows) {
    cols_ = (cols > MAX_COLS) ? MAX_COLS : cols;
    rows_ = (rows > MAX_ROWS) ? MAX_ROWS : rows;
    reset();
  }
  void clear() { reset(); }
  void home() { cx_ = 0; cy_ = 0; }
  void setCursor(int x, int y) { cx_ = x; cy_ = y; }
  void noDisplay() {}
  void display() {}
  void noCursor() {}
  void cursor() {}
  void noBlink() {}
  void blink() {}
  void createChar(uint8_t, uint8_t[]) {}

  void print(const char *s) {
    if (!s) return;
    for (const char *p = s; *p; ++p) write_(*p);
  }
  void print(char c) { write_(c); }
  void print(int v) { char b[16]; snprintf(b, sizeof b, "%d", v); print(b); }
  void print(float v) { char b[16]; snprintf(b, sizeof b, "%.2f", v); print(b); }
  void println(const char *s = "") { print(s); }

  /* render isi layar sebagai string (dipakai program uji) */
  std::string screen() const {
    std::string r;
    for (int y = 0; y < rows_; y++) {
      r += "|";
      r += std::string(buf_[y]);
      r += "|\n";
    }
    return r;
  }

 private:
  void reset() {
    for (int y = 0; y < MAX_ROWS; y++) memset(buf_[y], ' ', MAX_COLS);
    for (int y = 0; y < MAX_ROWS; y++) buf_[y][cols_ > 0 ? cols_ : MAX_COLS] = '\0';
    cx_ = 0; cy_ = 0;
  }

  void write_(char c) {
    if (cy_ < 0 || cy_ >= rows_) return;
    if (cx_ >= cols_) { cx_ = 0; cy_++; }
    if (cy_ >= rows_) return;
    buf_[cy_][cx_++] = c;
    buf_[cy_][cols_] = '\0';
  }

  char buf_[MAX_ROWS][MAX_COLS + 1] = {};
  int cols_ = 16, rows_ = 2, cx_ = 0, cy_ = 0;
};

#endif /* LIQUIDCRYSTAL_SHIM_H */
