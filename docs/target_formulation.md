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
