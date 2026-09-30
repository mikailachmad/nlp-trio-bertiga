# Formulasi Target Variabel — Prediksi Kurs USD/IDR (JISDOR)

Dua formulasi target dibangun sekaligus dari sumber yang sama (`data/processed/jisdor_clean.csv`), lewat `src/jisdor/target_formulation.py`, karena keduanya valid menurut spesifikasi tugas (klasifikasi atau regresi) dan tim ingin membandingkan performa arah dengan magnitude pada pergerakan kurs.

## 1. Klasifikasi 3-class (kolom `target`)

Label arah pergerakan kurs besok: **NAIK** / **STABIL** / **TURUN**.

- **Kenapa 3-class, bukan biner NAIK/TURUN**: divalidasi empiris dari 1201 hari data historis, tanpa kelas STABIL, 30.1% hari harus didrop karena `pct_change` terlalu dekat nol untuk dilabeli NAIK/TURUN secara wajar. Kelas STABIL menghindari pembuangan data sebanyak itu.
- **Threshold ±0.1%** pada `pct_change` harian memisahkan STABIL dari NAIK/TURUN. Nilai ini divalidasi dari statistik deskriptif kurs (standar deviasi `pct_change` harian ~0.33%, rata-rata 0.019%), jadi threshold 0.1% cukup ketat untuk menangkap pergerakan yang benar-benar mendatar.

## 2. Regresi (kolom `kurs_besok`)

Nilai mentah `kurs_jisdor` hari berikutnya

- **Kenapa nilai mentah, bukan `pct_change`**: spesifikasi tugas secara eksplisit mencontohkan regresi sebagai Nilai Penutupan Hari Berikutnya, jadi target regresi mengikuti definisi tersebut. `pct_change` tetap tersedia sebagai kolom fitur/turunan untuk eksperimen lanjutan.
- **Baseline Persistence (Random Walk)**: prediksi `kurs_besok` hari t = `kurs_jisdor` hari t. Ini baseline standar untuk data FX harian. Literatur keuangan menunjukkan kurs harian mendekati random walk, sehingga nilai hari ini adalah prediktor terbaik tanpa informasi tambahan. Model apapun (ARIMA, XGBoost, model gabungan dengan fitur NLP) harus mengalahkan RMSE/MAE baseline ini untuk dianggap punya nilai tambah.

## Kesamaan mekanisme

Kedua target dihitung dengan `shift(-1)` di fungsi yang sama (`build_target()`): fitur hari t dipasangkan ke label/nilai hari t+1, bukan hari t sendiri. Baris terakhir didrop untuk keduanya sekaligus, karena NaN di baris yang sama.

Karena `src/jisdor/split_dataset.py` men-split berdasarkan tanggal (bukan isi kolom target), satu kali split kronologis 80/10/10 menghasilkan `train.csv` / `val.csv` / `test.csv` yang membawa kolom `target` dan `kurs_besok` sekaligus, tidak perlu split terpisah untuk tiap formulasi. Lihat `docs/temporal_alignment_strategy.md` untuk aturan penyelarasan tanggal, dan docstring `src/jisdor/split_dataset.py` untuk alasan embargo satu baris di titik potong Train/Val (mencegah boundary leakage dari `shift(-1)`).

## Syarat validitas perbandingan dengan Model Gabungan (kurs + NLP)

Lower bound yaitu Macro-F1 Test > 0.3637, RMSE Test < 57.4694 dihitung pada Test period baseline ini secara spesifik: **2026-02-23 s.d. 2026-08-31**, memakai seluruh 5 tahun riwayat kurs (2021-09-01 dst.) untuk Train.

Ketika Model Baseline Time-Series dibandingkan dengan Model Gabungan berita+kurs , klaim mengalahkan baseline hanya valid kalau **kedua model dievaluasi pada Test set yang sama** (tanggal sama, jumlah baris sama). Kalau pipeline gabungan memakai rasio atau rentang tanggal berbeda maka salah satu dari ini perlu dilakukan sebelum klaim mengalahkan baseline dianggap sah:

- pipeline gabungan memakai `chronological_split()` dari `src/jisdor/split_dataset.py` (rasio & titik potong sama), atau
- baseline ini dihitung ulang khusus pada rentang tanggal yang dipakai pipeline gabungan, dan lower boundnya diperbarui sesuai itu.
