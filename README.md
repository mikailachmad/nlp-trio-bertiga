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
│   └── processed/                  # data yang sudah dibersihkan & diselaraskan
├── src/
│   ├── jisdor/
│   │   ├── preprocess_jisdor.py    # cleaning data kurs JISDOR
│   │   ├── temporal_alignment.py   # fungsi penyelarasan berita <-> trading day
│   │   └── visualize_jisdor.py     # visualisasi tren kurs
│   ├── scrapping/
│   │   └── news_scraper.py         # [TODO] scraping berita GDELT
│   ├── cleaning/
│   │   └── text_cleaning.py        # [TODO] cleaning teks berita
│   ├── filtering/
│   │   └── text_filtering.py       # [TODO] filtering teks berita
│   ├── preprocessing/
│   │   └── text_preprocessing.py   # [TODO] preprocessing teks berita
│   └── main.py                     # [TODO] inisisasi sistem
├── docs/
│   └── temporal_alignment_strategy.md   # penjelasan aturan penyelarasan
├── report/                        # laporan PDF (outline sudah dibuat)
└── README.md
```

## Alur Kerja

### Data Scrapping

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
| Preprocessing data kurs                            | ✅ Selesai                                                 | `src/preprocess_jisdor.py`                                               |
| Strategi & fungsi temporal alignment               | 🟡 Desain selesai, implementasi penuh menunggu data berita | `src/temporal_alignment.py`, `docs/temporal_alignment_strategy.md`       |
| Visualisasi tren kurs                              | ✅ Selesai                                                 | `src/visualize_jisdor.py`                                                |
| Scraping data berita                               | 🔲 Belum dimulai                                           | —                                                                        |
| Cleaning, filtering, dan preprocessing teks berita | 🟡 Kode program selesai, diperlukan testing lebih lanjut                                           | `src/cleaning/text_cleaning.py`, `src/filtering/text_filtering.py`, `src/preprocessing/text_preprocessing.py`, `src/main.py`                                                                         |
| Laporan PDF                                        | 🟡 Outline dibuat, konten belum diisi                      | `report/`                                                                |

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

> Catatan: script scraping & cleaning berita masih dalam pengerjaan tim,
> akan ditambahkan cara menjalankannya begitu tersedia.

## Catatan

README ini masih versi awal, akan terus diupdate seiring progres dari seluruh anggota tim, terutama begitu bagian scraping & cleaning berita selesai.
