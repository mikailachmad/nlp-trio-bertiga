# Prediksi Peristiwa Geopolitik Global: Dampak Terhadap Nilai Tukar Dolar

Proyek mata kuliah Pemrosesan Bahasa Alami: membangun pipeline NLP & analisis end-to-end untuk menguji hipotesis: **berita geopolitik global punya pengaruh signifikan terhadap fluktuasi nilai tukar USD/IDR**.

## Tim

| Nama                                                 | Peran                                           |
| ---------------------------------------------------- | ----------------------------------------------- |
| Mikail Achmad - 24/542370/PA/23026 (Project Manager) | Financial Data Acquisition & Temporal Alignment |
| Widad Muhammad Rafi - 24/545635/PA/23190             | News Scraping (TBA)                             |
| PRATAMA NANINDRA AJI - 24/533677/PA/22604            | Text Cleaning & Filtering Strategy              |

## Struktur Repositori

```
.
├── data/
│   ├── raw/                        # data mentah (sebelum dibersihkan)
│   ├── processed/                  # data yang sudah dibersihkan & diselaraskan
│   │   ├── jisdor_clean.csv        # kurs JISDOR bersih
│   │   └── jisdor_target.csv       # kurs + target 3-class (NAIK/STABIL/TURUN) + kurs_besok (regresi)
│   └── split/                      # data yang sudah dibagi kronologis
│       ├── train.csv               # 80% data awal (2021-09 → 2025-08)
│       ├── val.csv                 # 10% berikutnya (2025-08 → 2026-02)
│       └── test.csv                # 10% terakhir (2026-02 → 2026-08)
├── src/
│   ├── jisdor/
│   │   ├── preprocess_jisdor.py    # cleaning data kurs JISDOR
│   │   ├── target_formulation.py   # formulasi target 3-class + regresi (kurs_besok)
│   │   ├── split_dataset.py        # split kronologis Train/Val/Test
│   │   ├── temporal_alignment.py   # fungsi penyelarasan berita <-> trading day
│   │   └── visualize_jisdor.py     # visualisasi tren kurs
│   ├── models/
│   │   ├── naive_baseline.py             # Naive Baseline classification (Persistence & Majority-Class)
│   │   └── naive_baseline_regression.py  # Naive Baseline regresi (Persistence / Random Walk)
│   ├── scrapping/
│   │   └── news_scraper.py         # [TODO] scraping berita GDELT (kerja aktual di Scrappings.ipynb)
│   ├── cleaning/
│   │   └── text_cleaning.py        # cleaning teks berita
│   ├── filtering/
│   │   └── text_filtering.py       # filtering teks berita (TF-IDF + K-Means relevansi)
│   ├── preprocessing/
│   │   └── text_preprocessing.py   # preprocessing teks berita
│   └── main.py                     # CLI entrypoint (--stage cleaning/filtering/preprocessing/all)
├── notebook/
│   ├── naive_baseline_experiment.ipynb             # EDA & eksperimen Naive Baseline classification
│   └── naive_baseline_regression_experiment.ipynb  # EDA & eksperimen Naive Baseline regresi
├── results/
│   ├── naive_baseline_results.json             # hasil evaluasi Naive Baseline classification
│   └── naive_baseline_regression_results.json  # hasil evaluasi Naive Baseline regresi
├── docs/
│   ├── temporal_alignment_strategy.md   # penjelasan aturan penyelarasan
│   └── target_formulation.md            # alasan formulasi target classification & regresi
├── report/                        # laporan PDF (outline sudah dibuat)
└── README.md
```

## Alur Kerja

### Data Scrapping

Data berita diperoleh dari **CNBC** sebagai sumber utama. **Google News RSS** digunakan sebagai *article discovery* untuk menemukan artikel CNBC secara otomatis berdasarkan kata kunci terkait geopolitik, ekonomi, nilai tukar, serta rentang waktu penelitian. Hasil discovery kemudian digabungkan dan dilakukan deduplikasi untuk memperoleh daftar URL artikel.

Setiap URL artikel selanjutnya diakses langsung dari CNBC menggunakan **HTTP request** dengan library `requests`. Halaman HTML diproses menggunakan **BeautifulSoup** untuk mengekstraksi isi berita, sedangkan metadata seperti judul dan tanggal publikasi diperoleh dari struktur **JSON-LD** (`<script type="application/ld+json">`). Data hasil scraping disimpan dalam dataset terstruktur yang mencakup URL, judul, tanggal publikasi, penulis, dan isi berita.

**Pipeline:**
`Google News RSS → Article Discovery → Deduplication → URL CNBC → HTTP Request → HTML/JSON-LD Parsing → Ekstraksi Data → Dataset Berita Mentah`

### Data Cleaning

1. Ekstraksi Komponen Utama:
   - Judul/Headline: Mengambil nilai kunci "headline" pada tag `<script type="application/ld+json">`.
   - Konten Berita: Mengekstrak teks dari elemen `<p>` dan `<h2>`.
2. Pembersihan Tag HTML: Mengisolasi teks agar hanya menghasilkan paragraf berita murni.
3. Perbaikan Karakter: Memperbaiki karakter yang korup atau tidak valid.
4. Normalisasi Teks & Spasi: Menyelaraskan karakter ke standar Unicode serta mengonversi baris baru, tab, dan indentasi menjadi spasi tunggal.
5. Strukturisasi Data: Menyusun hasil ke format structured data dengan atribut `title`, `heading`, `body`, `author`, dan `publish_date`

> Catatan: Jika dataset awal sudah berupa paragraf mentah yang telah dibersihkan, tahap cleaning dapat dilewati dan langsung masuk ke proses filtering dan preprocessing.

### Data Filtering

1. Penetapan keyword berdasarkan topik pembahasan:
   - Geopolitik:
     > konflik, perang, ancaman, perdamaian, diplomasi, geopolitik, global, dunia, internasional, Amerika, United States, US, AS, China, Rusia, Ukraina, Timur Tengah, Iran, Palestina, Gaza, Rafah, Israel, persaingan, klaim, presiden, perdana menteri, ekspor, impor, perdagangan, pemerintah.
   - Nilai tukar rupiah:
     > rupiah, dolar AS, nilai tukar, kurs, ekonomi, perdagangan, keuangan, peningkatan, inflasi, mata uang, global, penguatan, menguat, pelemahan, melemah, suku bunga, pasar, ekspor, impor, devisa, valuta, bank, dana.
2. Perhitungan nilai probabilitas keterkaitan tiap dokumen berita terhadap seluruh kosa kata dari keyword pada masing-masing topik menggunakan TF-IDF
3. Pemetaan skor probabilitas ke dalam ruang dua dimensi untuk mengukur tingkat kedekatan dengan kedua topik pembahasan.
4. Pengelompokan (clustering) dokumen ke dalam 4 kategori menggunakan metode K-Means:
   - Kategori 1: Keterkaitan geopolitik tinggi dan nilai tukar tinggi.
   - Kategori 2: Keterkaitan geopolitik tinggi dan nilai tukar rendah.
   - Kategori 3: Keterkaitan geopolitik rendah dan nilai tukar tinggi.
   - Kategori 4: Keterkaitan geopolitik rendah dan nilai tukar rendah.
5. Pemilihan dokumen berita yang tergolong dalam kluster berisikan tingkat geopolitik dan nilai tukar tinggi hasil tahap (c).

### Data Preprocessing

1. Deduplikasi Paragraf: Menghapus duplikasi teks baik secara persis (exact duplication) maupun menyerupai (near duplication) pada tingkat blok.
2. Penandaan Entitas Khusus: Mengubah URL dan email menjadi placeholder "URL" dan "EMAIL".
3. Penyeragaman Huruf Kecil (Case Folding): Mengonversi seluruh huruf besar pada teks menjadi huruf kecil (lowercase).
4. Retensi Entitas & Stopwords: Mempertahankan nama, angka, dan kata henti (stopword) dengan menyimpan nama entitas ke teks asli.

## Status Progress — Tugas 1

| Komponen                                           | Status                                                     | Keterangan                                                               |
| -------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| Data kurs USD (BI JISDOR, 2021–2026)               | ✅ Selesai                                                 | `data/raw/Informasi_Kurs_Jisdor.xlsx`, `data/processed/jisdor_clean.csv` |
| Preprocessing data kurs                            | ✅ Selesai                                                 | `src/jisdor/preprocess_jisdor.py`                                               |
| Strategi & fungsi temporal alignment               | ✅ Selesai (siap dijalankan, support 3 timezone)        | `src/jisdor/temporal_alignment.py`, `docs/temporal_alignment_strategy.md`       |
| Visualisasi tren kurs                              | ✅ Selesai (output PNG & HTML plotly)                      | `src/jisdor/visualize_jisdor.py`                                                |
| Scraping data berita                               | 🟡 sebagian selesai                                           | `src/scrapping/Scrappings.ipynb`                                                                       |
| Cleaning, filtering, dan preprocessing teks berita | ✅ Selesai (revisi terbaru dari tim)                           | `src/cleaning/text_cleaning.py`, `src/filtering/text_filtering.py`, `src/preprocessing/text_preprocessing.py`, `src/main.py`                                                                         |
| Laporan PDF                                        | 🟡 Outline dibuat, konten belum diisi                      | `report/`                                                                |

## Status Progress — Tugas 2

| Komponen                                                        | Status       | Keterangan                                                                                                       |
| --------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------- |
| Formulasi target variabel — classification (3-class: NAIK/STABIL/TURUN) | ✅ Selesai   | `src/jisdor/target_formulation.py`, output: `data/processed/jisdor_target.csv`                                   |
| Formulasi target variabel — regresi (`kurs_besok`)              | ✅ Selesai   | `src/jisdor/target_formulation.py`, `docs/target_formulation.md`                                                 |
| Desain train/val/test split kronologis (80/10/10)                | ✅ Selesai   | `src/jisdor/split_dataset.py`, output: `data/split/{train,val,test}.csv` (membawa kedua target sekaligus)         |
| Naive Baseline classification (Persistence of Direction & Majority-Class) | ✅ Selesai   | `src/models/naive_baseline.py`, notebook: `notebook/naive_baseline_experiment.ipynb`                              |
| Naive Baseline regresi (Persistence / Random Walk)               | ✅ Selesai   | `src/models/naive_baseline_regression.py`, notebook: `notebook/naive_baseline_regression_experiment.ipynb`       |
| Strategi ekstraksi fitur NLP (sentimen Loughran-McDonald, TF-IDF) | 🔲 Belum     | Butir 1a–1c tugas 2                                                                                              |
| Model baseline time-series (ARIMA / XGBoost)                    | 🔲 Belum     | Butir 2b tugas 2                                                                                                 |
| Model gabungan awal (time-series + fitur NLP)                   | 🔲 Belum     | Butir 2c tugas 2                                                                                                 |
| Metrik evaluasi & analisis perbandingan                         | 🟡 Sebagian  | Naive baseline (classification & regresi) sudah dievaluasi; model lain belum                                    |
| Laporan PDF Tugas 2                                             | 🔲 Belum     |                                                                                                                  |

### Detail Formulasi Target Variabel

Dua formulasi target dibangun sekaligus dari sumber yang sama — lihat `docs/target_formulation.md` untuk alasan lengkap.

**1. Classification (kolom `target`)**
- **Keputusan**: Klasifikasi **3-class** (NAIK / STABIL / TURUN), bukan biner.
- **Threshold**: ±0.1% pada `pct_change` harian.
  - `pct_change > +0.1%` → NAIK
  - `pct_change < -0.1%` → TURUN
  - Sisanya → STABIL
- **Alasan**: Tanpa kelas STABIL, ~30% hari trading akan di-drop karena perubahan kurs terlalu kecil (mendekati nol). Threshold 0.1% tervalidasi empiris dari statistik deskriptif data (std ≈ 0.33%, mean ≈ 0.019%).

**2. Regresi (kolom `kurs_besok`)**
- **Keputusan**: Nilai mentah `kurs_jisdor` hari berikutnya ("Nilai Penutupan Hari Berikutnya" sesuai spesifikasi tugas).
- **Alasan kedua formulasi dipertahankan**: classification menangkap arah, regresi menangkap magnitude — tim ingin membandingkan keduanya, dan spesifikasi tugas mengizinkan salah satu atau keduanya.

Keduanya pakai mekanisme sama: fitur hari t dipasangkan ke label/nilai hari t+1 (**shift(-1)**, prediksi *besok*), bukan hari t sendiri.

### Detail Split Kronologis

| Split | Baris | Rentang Tanggal           | NAIK  | STABIL | TURUN | `kurs_besok` (min–mean–max) |
|-------|-------|---------------------------|-------|--------|-------|------------------------------|
| Train | 959   | 2021-09-01 → 2025-08-26   | 380   | 283    | 296   | 14080 – 15394 – 16943        |
| Val   | 119   | 2025-08-28 → 2026-02-19   | 43    | 47     | 29    | 16348 – 16686 – 16981        |
| Test  | 121   | 2026-02-23 → 2026-08-31   | 54    | 31     | 36    | 16758 – 17569 – 18171        |

- Embargo: 1 baris di-drop di ujung Train dan Val untuk mencegah *boundary leakage* akibat shift target (berlaku untuk kedua target sekaligus, karena NaN di baris yang sama).
- Cross-validation: TimeSeriesSplit 10-fold (expanding window) dengan gap=1, hanya di dalam Train.

### Hasil Naive Baseline — Classification

| Strategi                 | Split | Accuracy | Macro F1 |
|--------------------------|-------|----------|----------|
| Persistence of Direction | Val   | 37.82%   | 35.39%   |
| Persistence of Direction | Test  | 38.02%   | 36.37%   |
| Majority-Class (NAIK)    | Val   | 36.13%   | 17.70%   |
| Majority-Class (NAIK)    | Test  | 44.63%   | 20.57%   |

**Lower bound**: Model selanjutnya (ARIMA, XGBoost, model gabungan NLP) harus menghasilkan **Macro F1 > 36.37%** agar dianggap memberikan nilai tambah di atas naive baseline.

### Hasil Naive Baseline — Regresi

| Strategi                  | Split | RMSE    | MAE     |
|----------------------------|-------|---------|---------|
| Persistence (Random Walk) | Val   | 37.51   | 28.53   |
| Persistence (Random Walk) | Test  | 57.47   | 43.60   |

**Lower bound**: Model selanjutnya harus menghasilkan **RMSE < 57.47** (Test) agar dianggap memberikan nilai tambah di atas random-walk baseline.

**Legenda:** ✅ selesai · 🟡 sedang berjalan / sebagian · 🔲 belum dimulai

## Sumber Data

- **Berita geopolitik:** TBA
- **Kurs USD/IDR:** [Bank Indonesia — JISDOR](https://www.bi.go.id/), rentang 1 September 2021 – 1 September 2026.

## Cara Menjalankan

1. Buat dan jalankan virtual environment python dengan menjalankan perintah berikut
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies dengan menjalankan perintah berikut
```bash
pip install -r requirements.txt
```
3. Jalankan program utama dengan perintah berikut:
```bash
python src/main.py --stage <cleaning, filtering, preprocessing, all> --input <input_path> --output <output_path>.json
```
4. Untuk preprocessing data JISDOR:
```bash
python -m src.jisdor.preprocess_jisdor
```
5. Untuk formulasi target variabel:
```bash
python -m src.jisdor.target_formulation
```
6. Untuk split dataset kronologis:
```bash
python -m src.jisdor.split_dataset
```
7. Untuk evaluasi Naive Baseline classification:
```bash
python -m src.models.naive_baseline
```
8. Untuk evaluasi Naive Baseline regresi:
```bash
python -m src.models.naive_baseline_regression
```
9. Untuk visualisasi tren JISDOR:
```bash
python -m src.jisdor.visualize_jisdor
```

> Catatan: script scraping & cleaning berita masih dalam pengerjaan tim,
> akan ditambahkan cara menjalankannya begitu tersedia.

## Catatan

README ini masih versi awal, akan terus diupdate seiring progres dari seluruh anggota tim, terutama begitu bagian scraping & cleaning berita selesai.
