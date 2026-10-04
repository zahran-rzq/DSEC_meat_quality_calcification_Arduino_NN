"""
nilai_sensor_ke_tegangan.py
Konversi NILAI SENSOR (0..60, sama seperti di dataset) menjadi TEGANGAN (V),
kode ADC, dan posisi potensiometer (%) untuk simulasi Proteus.

Rumus (skala penuh 60 -> 5 V, sama dengan kode Arduino):

    tegangan = nilai / 12          karena 5/60 = 1/12
    ADC      = nilai * 1023 / 60   (ADC 10-bit Arduino Mega = 0..1023)
    posisi   = ADC / 1023 * 100 %  (kira-kira, untuk memutar potensiometer)

Contoh pemakaian:
    python nilai_sensor_ke_tegangan.py 14.58 9.90 11.86 14.70 7.92 19.18 6.30 11.64 18.00 36.78 16.63
    python nilai_sensor_ke_tegangan.py        (lalu tempel angkanya saat diminta)
"""
import sys

VCC        = 5.0      # tegangan referensi Arduino
NILAI_MAKS = 60.0     # nilai sensor maksimum pada dataset
ADC_MAKS   = 1023.0   # resolusi ADC 10-bit

# Urutan pin A0..A10
SENSOR = ["MQ135", "MQ136", "MQ137", "MQ138", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ8", "MQ9"]


def nilai_ke_tegangan(nilai):
    """Nilai sensor (0..60) -> tegangan (V).  Rumus inti: nilai / 12."""
    return float(nilai) * VCC / NILAI_MAKS


def nilai_ke_adc(nilai):
    """Nilai sensor (0..60) -> kode ADC 0..1023 (dibulatkan)."""
    return int(round(float(nilai) * ADC_MAKS / NILAI_MAKS))


def baca_input(teks):
    """Ubah '17,11 11,63 7,14' atau '17.11,11.63' atau '17.11 11.63' jadi list angka."""
    teks = teks.replace(";", " ").replace("\t", " ").strip()

    def coba(pemisah):
        try:
            return [float(x) for x in teks.replace(",", pemisah).split()]
        except ValueError:
            return []

    satu = coba(".")                  # tafsir 1: koma = tanda desimal
    if len(satu) == 11:
        return satu
    dua = coba(" ")                   # tafsir 2: koma = pemisah angka
    if len(dua) == 11:
        return dua
    return satu or dua                # biar pesan errornya menyebut jumlah yang jelas


def main():
    # --- ambil 11 angka dari argumen, atau tanya lewat keyboard ---
    if len(sys.argv) > 1:
        nilai = baca_input(" ".join(sys.argv[1:]))
    else:
        print("Masukkan 11 nilai sensor (dipisah spasi/koma), lalu Enter.")
        print("Contoh: 14.58 9.90 11.86 14.70 7.92 19.18 6.30 11.64 18.00 36.78 16.63")
        nilai = baca_input(input("> "))

    if len(nilai) != 11:
        print(f"ERROR: butuh 11 nilai, yang dimasukkan {len(nilai)}.")
        return 1

    # --- tulis tabel ---
    print()
    print("=" * 78)
    print("  NILAI SENSOR  ->  TEGANGAN  (untuk menyetel potensiometer Proteus)")
    print("=" * 78)
    print(f"{'pin':>4} {'sensor':>8} {'nilai':>8} {'tegangan':>11} {'ADC':>6} {'posisi pot':>11}")
    print("-" * 78)
    for i, (nama, v) in enumerate(zip(SENSOR, nilai)):
        volt = nilai_ke_tegangan(v)
        adc = nilai_ke_adc(v)
        print(f"{'A' + str(i):>4} {nama:>8} {v:8.2f} {volt:9.3f} V {adc:6d} "
              f"{adc / ADC_MAKS * 100:10.1f} %")
    print("-" * 78)
    print(f"  Cara cepat di kertas: tegangan = nilai / 12     (contoh: 60 -> 5,000 V)")
    print(f"  Resolusi 1 langkah ADC = {VCC / (ADC_MAKS + 1) * 1000:.3f} mV "
          f"= {NILAI_MAKS / (ADC_MAKS + 1):.4f} satuan nilai sensor")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())