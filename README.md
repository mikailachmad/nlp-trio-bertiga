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
│   │   └── jisdor_target.csv       # kurs + target 3-class (NAIK/STABIL/TURUN)
│   └── split/                      # data yang sudah dibagi kronologis
│       ├── train.csv               # 80% data awal (2021-09 → 2025-08)
│       ├── val.csv                 # 10% berikutnya (2025-08 → 2026-02)
│       └── test.csv                # 10% terakhir (2026-02 → 2026-08)
├── src/
│   ├── jisdor/
│   │   ├── preprocess_jisdor.py    # cleaning data kurs JISDOR
│   │   ├── target_formulation.py   # formulasi target 3-class
│   │   ├── split_dataset.py        # split kronologis Train/Val/Test
│   │   ├── temporal_alignment.py   # fungsi penyelarasan berita <-> trading day
│   │   └── visualize_jisdor.py     # visualisasi tren kurs
│   ├── models/
│   │   └── naive_baseline.py       # Naive Baseline (Persistence & Majority-Class)
│   ├── scrapping/
│   │   └── news_scraper.py         # [TODO] scraping berita GDELT
│   ├── cleaning/
│   │   └── text_cleaning.py        # [TODO] cleaning teks berita
│   ├── filtering/
│   │   └── text_filtering.py       # [TODO] filtering teks berita
│   ├── preprocessing/
│   │   └── text_preprocessing.py   # [TODO] preprocessing teks berita
│   └── main.py                     # [TODO] inisisasi sistem
├── notebook/
│   └── naive_baseline_experiment.ipynb  # EDA & eksperimen Naive Baseline
├── results/
│   └── naive_baseline_results.json # hasil evaluasi Naive Baseline
├── docs/
│   └── temporal_alignment_strategy.md   # penjelasan aturan penyelarasan
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
| Cleaning, filtering, dan preprocessing teks berita | 🟡 Kode program selesai, diperlukan testing lebih lanjut                                           | `src/cleaning/text_cleaning.py`, `src/filtering/text_filtering.py`, `src/preprocessing/text_preprocessing.py`, `src/main.py`                                                                         |
| Laporan PDF                                        | 🟡 Outline dibuat, konten belum diisi                      | `report/`                                                                |

## Status Progress — Tugas 2

| Komponen                                                        | Status       | Keterangan                                                                                                       |
| --------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------- |
| Formulasi target variabel (3-class: NAIK/STABIL/TURUN)          | ✅ Selesai   | `src/jisdor/target_formulation.py`, output: `data/processed/jisdor_target.csv`                                   |
| Desain train/val/test split kronologis (80/10/10)                | ✅ Selesai   | `src/jisdor/split_dataset.py`, output: `data/split/{train,val,test}.csv`                                         |
| Naive Baseline (Persistence of Direction & Majority-Class)       | ✅ Selesai   | `src/models/naive_baseline.py`, notebook: `notebook/naive_baseline_experiment.ipynb`                              |
| Strategi ekstraksi fitur NLP (sentimen, TF-IDF)                 | 🔲 Belum     | Butir 1a–1c tugas 2                                                                                              |
| Model baseline time-series (ARIMA / XGBoost)                    | 🔲 Belum     | Butir 2b tugas 2                                                                                                 |
| Model gabungan awal (time-series + fitur NLP)                   | 🔲 Belum     | Butir 2c tugas 2                                                                                                 |
| Metrik evaluasi & analisis perbandingan                         | 🟡 Sebagian  | Naive baseline sudah dievaluasi; model lain belum                                                                |
| Laporan PDF Tugas 2                                             | 🔲 Belum     |                                                                                                                  |

### Detail Formulasi Target Variabel

- **Keputusan**: Klasifikasi **3-class** (NAIK / STABIL / TURUN), bukan biner.
- **Threshold**: ±0.1% pada `pct_change` harian.
  - `pct_change > +0.1%` → NAIK
  - `pct_change < -0.1%` → TURUN
  - Sisanya → STABIL
- **Alasan**: Tanpa kelas STABIL, ~30% hari trading akan di-drop karena perubahan kurs terlalu kecil (mendekati nol). Threshold 0.1% tervalidasi empiris dari statistik deskriptif data (std ≈ 0.33%, mean ≈ 0.019%).
- **Target shift**: Fitur hari t dipasangkan ke label hari t+1 (prediksi *besok*), bukan label hari t sendiri.

### Detail Split Kronologis

| Split | Baris | Rentang Tanggal           | NAIK  | STABIL | TURUN |
|-------|-------|---------------------------|-------|--------|-------|
| Train | 959   | 2021-09-01 → 2025-08-26   | 380   | 283    | 296   |
| Val   | 119   | 2025-08-28 → 2026-02-19   | 43    | 47     | 29    |
| Test  | 121   | 2026-02-23 → 2026-08-31   | 54    | 31     | 36    |

- Embargo: 1 baris di-drop di ujung Train dan Val untuk mencegah *boundary leakage* akibat shift target.
- Cross-validation: TimeSeriesSplit 10-fold (expanding window) dengan gap=1, hanya di dalam Train.

### Hasil Naive Baseline

| Strategi                 | Split | Accuracy | Macro F1 |
|--------------------------|-------|----------|----------|
| Persistence of Direction | Val   | 37.82%   | 35.39%   |
| Persistence of Direction | Test  | 38.02%   | 36.37%   |
| Majority-Class (NAIK)    | Val   | 36.13%   | 17.70%   |
| Majority-Class (NAIK)    | Test  | 44.63%   | 20.57%   |

**Lower bound**: Model selanjutnya (ARIMA, XGBoost, model gabungan NLP) harus menghasilkan **Macro F1 > 36.37%** agar dianggap memberikan nilai tambah di atas naive baseline.

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
7. Untuk evaluasi Naive Baseline:
```bash
python -m src.models.naive_baseline
```
8. Untuk visualisasi tren JISDOR:
```bash
python -m src.jisdor.visualize_jisdor
```

> Catatan: script scraping & cleaning berita masih dalam pengerjaan tim,
> akan ditambahkan cara menjalankannya begitu tersedia.

## Catatan

README ini masih versi awal, akan terus diupdate seiring progres dari seluruh anggota tim, terutama begitu bagian scraping & cleaning berita selesai.
