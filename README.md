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
│   │   └── Cleaning_article.ipynb  # varian pipeline notebook (output title_clean/body_clean + title_bow/body_bow)
│   ├── filtering/
│   │   └── text_filtering.py       # filtering revisi: field wajib + keyword geopolitik (filter_news)
│   ├── preprocessing/
│   │   └── text_preprocessing.py   # preprocessing revisi: light & classical (preprocess_documents)
│   └── main.py                     # CLI legacy: cleaning → filtering → preprocessing berita
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

Data berita diperoleh dari **CNBC** sebagai sumber utama. **Google News RSS** digunakan sebagai _article discovery_ untuk menemukan artikel CNBC secara otomatis berdasarkan kata kunci terkait geopolitik, ekonomi, nilai tukar, serta rentang waktu penelitian. Hasil discovery kemudian digabungkan dan dilakukan deduplikasi untuk memperoleh daftar URL artikel.

Setiap URL artikel selanjutnya diakses langsung dari CNBC menggunakan **HTTP request** dengan library `requests`. Halaman HTML diproses menggunakan **BeautifulSoup** untuk mengekstraksi isi berita, sedangkan metadata seperti judul dan tanggal publikasi diperoleh dari struktur **JSON-LD** (`<script type="application/ld+json">`). Data hasil scraping disimpan dalam dataset terstruktur yang mencakup URL, judul, tanggal publikasi, penulis, dan isi berita.

**Pipeline:**
`Google News RSS → Article Discovery → Deduplication → URL CNBC → HTTP Request → HTML/JSON-LD Parsing → Ekstraksi Data → Dataset Berita Mentah`

### Data Cleaning, Filtering & Preprocessing

Alur revisi (acuan: `src/main.ipynb` + modul `src/filtering/` & `src/preprocessing/`):
`CSV mentah → scrape HTML → cleaning → filtering → preprocessing → data/processed/news_preprocessed.csv`.
`src/main.py` dipertahankan sebagai **CLI legacy** untuk pipeline ini.

### Data Cleaning

Membersihkan hasil scraping HTML menjadi kolom siap pakai (`src/main.ipynb` sel 2, fungsi `cleaning.clean_text`):

1. Ekstraksi dari HTML dengan **BeautifulSoup**:
   - Tanggal terbit dari tag `<time datetime>`, penulis dari `a.Author-authorName`, judul dari `<title>`.
   - Buang `script/style/noscript/nav/footer/header`, ambil paragraf `<p>` di dalam `[class*='ArticleBody']` / `<article>` (fallback: seluruh teks konten).
2. Normalisasi teks (`clean_text`): gabung paragraf, rapikan spasi/karakter.
3. Kolom hasil (`combine_first` dengan data discovery agar tidak ada yang hilang):
   `author`, `title`, `body`, `published_at`.

### Data Filtering

Seleksi relevansi via `filter_news(df)` (`src/filtering/text_filtering.py`) — dua lapis:

1. **Validasi field wajib** (`filter_required_fields`): baris hanya lolos jika
   `author, title, published_at, body, discovery_title, url` semuanya terisi (tidak null/kosong).
2. **Filter relevansi geopolitik** (`filter_geopolitical_relevance`, recall tinggi berbasis keyword):
   - 5 kelompok kosakata: `conflict` (war, invasion, missile, …), `sanction` (sanction, embargo, export control, …), `diplomacy` (summit, treaty, negotiation, …), `trade` (tariff, trade war, export ban, …), `political_risk` (election, coup, protest, …).
   - Pencocokan case-insensitive dengan batas kata, digabung dari `title + body` namun **asal tiap temuan tetap dicatat** (`find_geopolitical_keywords` → kolom `geopolitical_matches` berisi daftar temuan per `title`/`body`/`discovery_title`).
   - Baris lolos jika ada ≥ 1 temuan di salah satu dari ketiganya.
3. **Rencana lanjutan** (tertulis di docstring modul, belum diimplementasikan): anotasi manual
   terstratifikasi tahun & sumber lalu latih classifier dengan label
   `0 = tidak relevan`, `1 = relevan geopolitik dampak ekonomi rendah`,
   `2 = relevan geopolitik berpotensi memengaruhi pasar`
   (kandidat: Logistic Regression / Linear SVM + TF-IDF, transformer multilingual, zero-shot sebagai alat bantu).

### Data Preprocessing

Preprocessing via `preprocess_documents` (`src/preprocessing/text_preprocessing.py`,
mendukung input DataFrame/Series/list/str/dict) — menghasilkan **dua versi teks** per artikel:

**A. Light (`text_clean_light`)** — untuk transformer, sentiment model, NER, embedding:
1. Normalisasi Unicode NFC, unescape + hapus tag HTML, hapus karakter kontrol.
2. Masking entitas khusus: URL → `URL`, email → `EMAIL` (email dulu agar domainnya tidak terpotong regex URL).
3. Deduplikasi paragraf: exact (case-insensitive) + near-duplicate via Jaccard 3-shingle ≥ `0.8`.
4. Normalisasi spasi. Kapitalisasi, tanda baca, negasi, nama, angka, dan urutan kata **dipertahankan**.

**B. Classical (`text_clean_classical` + `tokens_classical`/`unigrams`/`bigrams`)** — untuk TF-IDF, LDA, BoW:
1. Berangkat dari hasil light, lalu case folding (lowercase).
2. Tokenisasi kata (menjaga kata & angka), bersihkan posesif (`china's` → `china`).
3. Lematisasi WordNet **POS-aware** (bukan stemming; ada fallback aturan dasar bila NLTK tak tersedia).
4. Stopword removal dengan **kata negasi dijamin tidak terhapus**
   (`no/not/never/tidak/bukan/jangan/…`, termasuk kontraksi `don't/won't/…`).
5. Filter token < `min_token_len` (3, kecuali kata negasi) + bentuk **unigram & bigram**
   (`suku_bunga`).

Kolom output: `text_clean_light`, `text_clean_classical`, `tokens_classical`,
`unigrams`, `bigrams`, `text_final` (= light, untuk kompatibilitas), `preserved_entities`
(kandidat kapital & angka). Pipeline notebook terpisah (`src/cleaning/Cleaning_article.ipynb`)
menyediakan varian alternatif dengan output `title_clean`/`body_clean` (VADER) dan
`title_bow`/`body_bow` (BoW/Loughran-McDonald).

## Status Progress — Tugas 1

| Komponen                                           | Status                                           | Keterangan                                                                                                                   |
| -------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Data kurs USD (BI JISDOR, 2021–2026)               | ✅ Selesai                                       | `data/raw/Informasi_Kurs_Jisdor.xlsx`, `data/processed/jisdor_clean.csv`                                                     |
| Preprocessing data kurs                            | ✅ Selesai                                       | `src/jisdor/preprocess_jisdor.py`                                                                                            |
| Strategi & fungsi temporal alignment               | ✅ Selesai (siap dijalankan, support 3 timezone) | `src/jisdor/temporal_alignment.py`, `docs/temporal_alignment_strategy.md`                                                    |
| Visualisasi tren kurs                              | ✅ Selesai (output PNG & HTML plotly)            | `src/jisdor/visualize_jisdor.py`                                                                                             |
| Scraping data berita                               | 🟡 sebagian selesai                              | `src/scrapping/Scrappings.ipynb`                                                                                             |
| Cleaning, filtering, dan preprocessing teks berita | ✅ Selesai (revisi terbaru dari tim)             | `src/main.ipynb` + `src/filtering/text_filtering.py`, `src/preprocessing/text_preprocessing.py` → `data/processed/news_preprocessed.csv` |
| Laporan PDF                                        | 🟡 Outline dibuat, konten belum diisi            | `report/`                                                                                                                    |

## Status Progress — Tugas 2

| Komponen                                                                  | Status      | Keterangan                                                                                                 |
| ------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------- |
| Formulasi target variabel — classification (3-class: NAIK/STABIL/TURUN)   | ✅ Selesai  | `src/jisdor/target_formulation.py`, output: `data/processed/jisdor_target.csv`                             |
| Formulasi target variabel — regresi (`kurs_besok`)                        | ✅ Selesai  | `src/jisdor/target_formulation.py`, `docs/target_formulation.md`                                           |
| Desain train/val/test split kronologis (80/10/10)                         | ✅ Selesai  | `src/jisdor/split_dataset.py`, output: `data/split/{train,val,test}.csv` (membawa kedua target sekaligus)  |
| Naive Baseline classification (Persistence of Direction & Majority-Class) | ✅ Selesai  | `src/models/naive_baseline.py`, notebook: `notebook/naive_baseline_experiment.ipynb`                       |
| Naive Baseline regresi (Persistence / Random Walk)                        | ✅ Selesai  | `src/models/naive_baseline_regression.py`, notebook: `notebook/naive_baseline_regression_experiment.ipynb` |
| Strategi ekstraksi fitur NLP (sentimen Loughran-McDonald, TF-IDF)         | 🔲 Belum    | Butir 1a–1c tugas 2                                                                                        |
| Model baseline time-series (ARIMA / XGBoost)                              | 🔲 Belum    | Butir 2b tugas 2                                                                                           |
| Model gabungan awal (time-series + fitur NLP)                             | 🔲 Belum    | Butir 2c tugas 2                                                                                           |
| Metrik evaluasi & analisis perbandingan                                   | 🟡 Sebagian | Naive baseline (classification & regresi) sudah dievaluasi; model lain belum                               |
| Laporan PDF Tugas 2                                                       | 🔲 Belum    |                                                                                                            |

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

Keduanya pakai mekanisme sama: fitur hari t dipasangkan ke label/nilai hari t+1 (**shift(-1)**, prediksi _besok_), bukan hari t sendiri.

### Detail Split Kronologis

| Split | Baris | Rentang Tanggal         | NAIK | STABIL | TURUN | `kurs_besok` (min–mean–max) |
| ----- | ----- | ----------------------- | ---- | ------ | ----- | --------------------------- |
| Train | 959   | 2021-09-01 → 2025-08-26 | 380  | 283    | 296   | 14080 – 15394 – 16943       |
| Val   | 119   | 2025-08-28 → 2026-02-19 | 43   | 47     | 29    | 16348 – 16686 – 16981       |
| Test  | 121   | 2026-02-23 → 2026-08-31 | 54   | 31     | 36    | 16758 – 17569 – 18171       |

- Embargo: 1 baris di-drop di ujung Train dan Val untuk mencegah _boundary leakage_ akibat shift target (berlaku untuk kedua target sekaligus, karena NaN di baris yang sama).
- Cross-validation: TimeSeriesSplit 10-fold (expanding window) dengan gap=1, hanya di dalam Train.

### Hasil Naive Baseline — Classification

| Strategi                 | Split | Accuracy | Macro F1 |
| ------------------------ | ----- | -------- | -------- |
| Persistence of Direction | Val   | 37.82%   | 35.39%   |
| Persistence of Direction | Test  | 38.02%   | 36.37%   |
| Majority-Class (NAIK)    | Val   | 36.13%   | 17.70%   |
| Majority-Class (NAIK)    | Test  | 44.63%   | 20.57%   |

**Lower bound**: Model selanjutnya (ARIMA, XGBoost, model gabungan NLP) harus menghasilkan **Macro F1 > 36.37%** agar dianggap memberikan nilai tambah di atas naive baseline.

### Hasil Naive Baseline — Regresi

| Strategi                  | Split | RMSE  | MAE   |
| ------------------------- | ----- | ----- | ----- |
| Persistence (Random Walk) | Val   | 37.51 | 28.53 |
| Persistence (Random Walk) | Test  | 57.47 | 43.60 |

**Lower bound**: Model selanjutnya harus menghasilkan **RMSE < 57.47** (Test) agar dianggap memberikan nilai tambah di atas random-walk baseline.

> **Syarat perbandingan dengan Model Gabungan (kurs + NLP)**: kedua lower bound di atas berlaku spesifik untuk Test period baseline ini (2026-02-23 → 2026-08-31). Klaim "mengalahkan baseline" hanya valid kalau Model Gabungan dievaluasi pada Test set yang sama persis — lihat `docs/target_formulation.md` bagian "Syarat validitas perbandingan dengan Model Gabungan" untuk detail.

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

3. Jalankan pipeline berita revisi via `src/main.ipynb`
   (scrape HTML → cleaning → `filter_news` → `preprocess_documents` → `data/processed/news_preprocessed.csv`),
   atau via CLI legacy:

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

> Catatan: `src/main.py` dipertahankan sebagai CLI legacy untuk pipeline berita
> (tahap cleaning/filtering/preprocessing/all). Alur kerja aktual ada di `src/main.ipynb`.

## Catatan

README ini masih versi awal, akan terus diupdate seiring progres dari seluruh anggota tim, terutama begitu bagian scraping & cleaning berita selesai.
