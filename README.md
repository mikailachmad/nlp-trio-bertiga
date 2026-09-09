# Prediksi Peristiwa Geopolitik Global: Dampak Terhadap Nilai Tukar Dolar

Proyek mata kuliah Pemrosesan Bahasa Alami: membangun pipeline NLP & analisis end-to-end untuk menguji hipotesis: **berita geopolitik global punya pengaruh signifikan terhadap fluktuasi nilai tukar USD/IDR**.

## Tim

| Nama | Peran |
|---|---|
| Mikail Achmad - 24/542370/PA/23026 (Project Manager) | Financial Data Acquisition & Temporal Alignment |
| Widad Muhammad Rafi - 24/545635/PA/23190  | News Scraping (TBA) |
| PRATAMA NANINDRA AJI - 24/533677/PA/22604 | Text Cleaning & Filtering Strategy |

## Struktur Repo

```
.
├── data/
│   ├── raw/            # data mentah (sebelum dibersihkan)
│   └── processed/      # data yang sudah dibersihkan & diselaraskan
├── src/
│   ├── preprocess_jisdor.py       # cleaning data kurs JISDOR
│   ├── temporal_alignment.py      # fungsi penyelarasan berita <-> trading day
│   ├── visualize_jisdor.py        # visualisasi tren kurs
│   ├── news_scraper.py            # [TODO] scraping berita GDELT
│   └── text_cleaning.py           # [TODO] cleaning & filtering teks berita
├── docs/
│   └── temporal_alignment_strategy.md   # penjelasan aturan penyelarasan
├── report/                        # laporan PDF (outline sudah dibuat)
└── README.md
```

## Status Progress — Tugas 1

| Komponen | Status | Keterangan |
|---|---|---|
| Data kurs USD (BI JISDOR, 2021–2026) | ✅ Selesai | `data/raw/Informasi_Kurs_Jisdor.xlsx`, `data/processed/jisdor_clean.csv` |
| Preprocessing data kurs | ✅ Selesai | `src/preprocess_jisdor.py` |
| Strategi & fungsi temporal alignment | 🟡 Desain selesai, implementasi penuh menunggu data berita | `src/temporal_alignment.py`, `docs/temporal_alignment_strategy.md` |
| Visualisasi tren kurs | ✅ Selesai | `src/visualize_jisdor.py` |
| Scraping data berita | 🔲 Belum dimulai | — |
| Cleaning & filtering teks berita | 🔲 Belum dimulai | — |
| Laporan PDF | 🟡 Outline dibuat, konten belum diisi | `report/` |

**Legenda:** ✅ selesai · 🟡 sedang berjalan / sebagian · 🔲 belum dimulai

## Sumber Data

- **Berita geopolitik:** TBA
- **Kurs USD/IDR:** [Bank Indonesia — JISDOR](https://www.bi.go.id/), rentang 1 September 2021 – 1 September 2026.

## Cara Menjalankan

COMING SOON!

> Catatan: script scraping & cleaning berita masih dalam pengerjaan tim,
> akan ditambahkan cara menjalankannya begitu tersedia.

## Catatan

README ini masih versi awal, akan terus diupdate seiring progres dari seluruh anggota tim, terutama begitu bagian scraping & cleaning berita selesai.
