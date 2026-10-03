/* ============================================================================
 *  verification/stubs/LiquidCrystal.h
 *  Stub LCD 20x4: bukan menampilkan di layar, tapi menyimpan isi layar ke
 *  buffer supaya bisa diperiksa / dicetak (untuk pengujian tanpa Proteus).
 * ==========================================================================*/
#ifndef LIQUIDCRYSTAL_STUB_H
#define LIQUIDCRYSTAL_STUB_H

#include <cstdio>
#include <cstring>

class LiquidCrystal {
public:
  LiquidCrystal(int, int, int, int, int, int) { clear(); }
  void begin(int, int) { clear(); }
  void clear() {
    memset(grid_, ' ', sizeof(grid_));
    col_ = 0; row_ = 0;
  }
  void setCursor(int col, int row) { col_ = col; row_ = row; }
  void home() { col_ = 0; row_ = 0; }

  void print(const char *s) { while (*s) putChar(*s++); }
  void print(char *s) { while (*s) putChar(*s++); }
  void print(char c) { putChar(c); }
  void print(int v) { char b[24]; snprintf(b, sizeof(b), "%d", v); print(b); }
  void print(unsigned long v) { char b[24]; snprintf(b, sizeof(b), "%lu", v); print(b); }
  void print(double v) { char b[24]; snprintf(b, sizeof(b), "%.2f", v); print(b); }
  void print(double v, int d) { char b[32]; snprintf(b, sizeof(b), "%.*f", d, v); print(b); }

  /* cetak isi LCD ke stdout dengan penanda "LCD|" */
  void dump() {
    for (int r = 0; r < 4; r++) {
      char line[21];
      memcpy(line, &grid_[r][0], 20);
      line[20] = '\0';
      printf("LCD|%s\n", line);
    }
  }

private:
  char grid_[4][20];
  int col_, row_;
  void putChar(char c) {
    if (row_ >= 0 && row_ < 4 && col_ >= 0 && col_ < 20) grid_[row_][col_] = c;
    col_++;
  }
};

#endif
