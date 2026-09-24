"""
Pembagian dataset kronologis (Train / Validation / Test) untuk data kurs JISDOR.

Keputusan desain:
  - Split kronologis 80/10/10 berdasarkan urutan tanggal, tanpa shuffle.
  - Embargo: baris terakhir Train dan Val di-drop. Target dibuat dengan
    shift(-1) di target_formulation.py, sehingga label baris terakhir tiap
    split dihitung dari harga hari pertama split berikutnya (boundary leakage).
  - Cross-validation hanya di dalam Train: TimeSeriesSplit 10-fold
    (expanding window) dengan gap=1 untuk alasan leakage yang sama.
    Val dipakai untuk pemilihan model akhir, Test hanya disentuh sekali.

Cara pakai:
  from src.jisdor.split_dataset import chronological_split, get_cv_splitter
  train, val, test = chronological_split(df_target)
  for train_idx, val_idx in get_cv_splitter().split(train):
      ...

CLI:
  python -m src.jisdor.split_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

__all__ = [
    "TRAIN_RATIO",
    "VAL_RATIO",
    "EMBARGO",
    "N_SPLITS",
    "chronological_split",
    "get_cv_splitter",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = _REPO_ROOT / "data" / "processed" / "jisdor_target.csv"
DEFAULT_OUTDIR = _REPO_ROOT / "data" / "split"

COL_DATE = "tanggal"
COL_TARGET = "target"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Jumlah baris yang di-drop di ujung Train dan Val; sama dengan horizon shift target.
EMBARGO = 1
N_SPLITS = 10


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    embargo: int = EMBARGO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bagi DataFrame secara kronologis jadi Train, Val, dan Test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame hasil :func:`src.jisdor.target_formulation.build_target`,
        dengan kolom ``tanggal`` (datetime64).
    train_ratio : float
        Proporsi Train dari total baris (default 0.8).
    val_ratio : float
        Proporsi Val dari total baris (default 0.1). Sisanya jadi Test.
    embargo : int
        Jumlah baris yang di-drop di ujung Train dan Val (default 1).

    Returns
    -------
    train, val, test : pd.DataFrame
        Tiga DataFrame terurut kronologis dengan index di-reset.
    """
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            f"Rasio tidak valid: train={train_ratio}, val={val_ratio}. "
            f"Keduanya harus > 0 dan jumlahnya < 1."
        )

    df = df.sort_values(COL_DATE).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[: train_end - embargo]
    val = df.iloc[train_end : val_end - embargo]
    test = df.iloc[val_end:]

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def get_cv_splitter(n_splits: int = N_SPLITS, gap: int = EMBARGO) -> TimeSeriesSplit:
    """
    Buat splitter walk-forward (expanding window) untuk CV di dalam Train.

    Parameters
    ----------
    n_splits : int
        Jumlah fold (default 10).
    gap : int
        Jumlah baris yang dilewati antara data latih dan validasi tiap fold
        (default 1, sama dengan embargo).

    Returns
    -------
    TimeSeriesSplit
        Pakai ``.split(train_df)`` untuk mendapatkan pasangan index tiap fold.
    """
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def _summarize(name: str, df: pd.DataFrame) -> str:
    dist = (df[COL_TARGET].value_counts(normalize=True) * 100).round(1).to_dict()
    return (
        f"{name:<5} {len(df):>4} baris | "
        f"{df[COL_DATE].min().date()} -> {df[COL_DATE].max().date()} | {dist}"
    )


# CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split kronologis Train/Val/Test untuk data target JISDOR."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Path CSV hasil target_formulation (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help=f"Direktori output train/val/test.csv (default: {DEFAULT_OUTDIR}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"Proporsi Train (default: {TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Proporsi Val (default: {VAL_RATIO}).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[split_dataset] ERROR: File tidak ditemukan: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])

    try:
        train, val, test = chronological_split(df, args.train_ratio, args.val_ratio)
    except ValueError as exc:
        print(f"[split_dataset] ERROR: {exc}", file=sys.stderr)
        return 1

    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"[split_dataset] {_summarize(name, part)}")

    print(f"[split_dataset] Walk-forward CV di dalam Train ({N_SPLITS} fold, gap={EMBARGO}):")
    for k, (tr_idx, va_idx) in enumerate(get_cv_splitter().split(train), 1):
        print(
            f"[split_dataset]   Fold {k:>2}: latih {len(tr_idx):>3} baris "
            f"(s.d. {train[COL_DATE].iloc[tr_idx[-1]].date()}) | "
            f"validasi {len(va_idx)} baris "
            f"({train[COL_DATE].iloc[va_idx[0]].date()} -> "
            f"{train[COL_DATE].iloc[va_idx[-1]].date()})"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False, date_format="%Y-%m-%d")
        print(f"[split_dataset] Tersimpan: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
