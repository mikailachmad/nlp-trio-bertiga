"""
Preprocessing data kurs USD/IDR dari BI JISDOR.

Alur:
  1. Baca ``data/processed/jisdor_clean.csv`` (kolom: tanggal, kurs_jisdor).
  2. Standarisasi format tanggal ke ``YYYY-MM-DD`` (agar kompatibel merge_asof).
  3. Validasi: deteksi nilai kosong, kurs negatif/nol, dan baris duplikat.
  4. Urutkan kronologis ascending.
  5. Simpan kembali ke path output (default: file yang sama).

Cara pakai:
  python -m src.jisdor.preprocess_jisdor
  python -m src.jisdor.preprocess_jisdor --input data/processed/jisdor_clean.csv \\
                                          --output data/processed/jisdor_clean.csv

Catatan: script ini sengaja *tidak* memproses file .xlsx mentah karena data
sudah pernah diekspor ke CSV dengan benar. Tugasnya hanya memastikan skema
dan format kolom konsisten sebelum dipakai oleh temporal_alignment.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

__all__ = [
    "load_jisdor_csv",
    "standardize_date_column",
    "validate_jisdor",
    "sort_chronological",
    "preprocess_jisdor",
]

# Konstanta path default (relatif terhadap root repo)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = _REPO_ROOT / "data" / "processed" / "jisdor_clean.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "processed" / "jisdor_clean.csv"

# Nama kolom yang diharapkan di CSV sumber
COL_DATE = "tanggal"
COL_RATE = "kurs_jisdor"

# 1. Pembacaan CSV
def load_jisdor_csv(path: str | Path) -> pd.DataFrame:
    """
    Baca CSV JISDOR dan kembalikan DataFrame mentah (belum divalidasi).

    Parameters
    ----------
    path : str | Path
        Lokasi file CSV (misal ``data/processed/jisdor_clean.csv``).

    Returns
    -------
    pd.DataFrame
        DataFrame dengan setidaknya kolom ``tanggal`` dan ``kurs_jisdor``.

    Raises
    ------
    FileNotFoundError
        Jika file tidak ditemukan.
    ValueError
        Jika kolom wajib tidak ada.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path, dtype=str)  # baca semua sebagai str 

    # Periksa kolom 
    missing_cols = {COL_DATE, COL_RATE} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di CSV: {missing_cols}. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    return df

# 2. Standarisasi kolom tanggal
def standardize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konversi kolom ``tanggal`` ke format ``datetime64[ns]`` (YYYY-MM-DD).

    Format asal yang didukung: ``M/D/YYYY``, ``MM/DD/YYYY``, ISO 8601, dsb.
    Baris yang gagal di-parse akan di-drop dengan peringatan.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom ``tanggal`` bertipe string.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan ``tanggal`` bertipe ``datetime64[ns]``.
    """
    df = df.copy()

    # pandas coba parse otomatis; baris tak terbaca -> NaT
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=False)

    jumlah_nat = df[COL_DATE].isna().sum()
    if jumlah_nat > 0:
        print(
            f"[preprocess] Peringatan: {jumlah_nat} baris gagal di-parse sebagai "
            f"tanggal dan akan di-drop."
        )
        df = df.dropna(subset=[COL_DATE])

    return df

# 3. Validasi
def validate_jisdor(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Periksa kualitas data dan kembalikan DataFrame bersih beserta log peringatan.

    Pemeriksaan yang dilakukan:
    - Nilai kosong / NaN pada kolom kurs.
    - Kurs non-positif (nol atau negatif) yang tidak masuk akal.
    - Baris duplikat pada kolom tanggal (ambil rata-rata jika ada).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom ``tanggal`` (datetime) dan ``kurs_jisdor`` (str/num).

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame yang sudah divalidasi dan dibersihkan.
    warnings : list[str]
        Daftar pesan peringatan yang ditemukan selama validasi.
    """
    df = df.copy()
    warnings: List[str] = []

    # Konversi kurs ke numerik
    df[COL_RATE] = pd.to_numeric(df[COL_RATE], errors="coerce")

    # Periksa NaN pada kurs
    jumlah_nan = df[COL_RATE].isna().sum()
    if jumlah_nan > 0:
        warnings.append(
            f"{jumlah_nan} baris memiliki kurs kosong/tidak valid, di-drop."
        )
        df = df.dropna(subset=[COL_RATE])

    # Periksa kurs non-positif
    non_positif = (df[COL_RATE] <= 0).sum()
    if non_positif > 0:
        warnings.append(
            f"{non_positif} baris memiliki kurs <= 0 (tidak wajar), di-drop."
        )
        df = df[df[COL_RATE] > 0]

    # Periksa duplikat tanggal
    duplikat = df.duplicated(subset=[COL_DATE], keep=False)
    jumlah_dup = duplikat.sum()
    if jumlah_dup > 0:
        tanggal_dup = df.loc[duplikat, COL_DATE].dt.strftime("%Y-%m-%d").unique().tolist()
        warnings.append(
            f"{jumlah_dup} baris duplikat ditemukan pada {len(tanggal_dup)} tanggal "
            f"({tanggal_dup[:5]}{'...' if len(tanggal_dup) > 5 else ''}). "
            f"Kurs di-rata-ratakan per tanggal."
        )
        df = (
            df.groupby(COL_DATE, as_index=False)[COL_RATE]
            .mean()
            .round({COL_RATE: 0})
        )
        df[COL_RATE] = df[COL_RATE].astype(int)

    return df, warnings

# 4. Pengurutan kronologis
def sort_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """
    Urutkan DataFrame berdasarkan ``tanggal`` secara ascending dan reset index.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    return df.sort_values(COL_DATE).reset_index(drop=True)

# 5. Pipeline utama
def preprocess_jisdor(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Jalankan seluruh pipeline preprocessing JISDOR.

    Tahap: baca CSV -> standarisasi tanggal -> validasi -> urutkan -> simpan.

    Parameters
    ----------
    input_path : str | Path
        Path ke file CSV sumber (default: ``data/processed/jisdor_clean.csv``).
    output_path : str | Path
        Path tujuan penyimpanan hasil (default: sama dengan input).
    verbose : bool
        Jika True, cetak ringkasan ke stdout.

    Returns
    -------
    pd.DataFrame
        DataFrame JISDOR yang sudah bersih dengan kolom:
        - ``tanggal``: datetime64[ns], format YYYY-MM-DD
        - ``kurs_jisdor``: int64, kurs USD/IDR
    """
    if verbose:
        print(f"[preprocess] Membaca: {input_path}")

    df = load_jisdor_csv(input_path)

    if verbose:
        print(f"[preprocess] {len(df)} baris dimuat.")

    df = standardize_date_column(df)
    df, warnings = validate_jisdor(df)

    for w in warnings:
        print(f"[preprocess] Peringatan: {w}")

    df = sort_chronological(df)

    if verbose:
        print(
            f"[preprocess] Setelah validasi: {len(df)} baris, "
            f"rentang {df[COL_DATE].min().date()} s.d. {df[COL_DATE].max().date()}."
        )

    # Simpan output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    if verbose:
        print(f"[preprocess] Tersimpan: {output_path}")

    return df

# CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocessing data kurs JISDOR (standarisasi & validasi CSV)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Path CSV sumber (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Path CSV output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Sembunyikan output log.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        preprocess_jisdor(
            input_path=args.input,
            output_path=args.output,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[preprocess] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
