"""
Formulasi target variabel untuk prediksi kurs USD/IDR (JISDOR).

Dua formulasi target dibangun sekaligus dari sumber yang sama (lihat
docs/target_formulation.md untuk alasan lengkap):
  - Klasifikasi 3-class (kolom ``target``): NAIK / STABIL / TURUN (bukan
    biner). Threshold +-0.1% pada pct_change harian, tervalidasi empiris
    dari data/processed/jisdor_clean.csv (1201 hari): tanpa kelas STABIL,
    30.1% hari akan ke-drop karena pct_change terlalu dekat nol.
  - Regresi (kolom ``kurs_besok``): nilai kurs mentah hari berikutnya,
    mengikuti definisi "Nilai Penutupan Hari Berikutnya" di spesifikasi
    tugas. Dipakai untuk baseline regresi terpisah dari classification.

Keduanya diprediksi untuk *besok* (shift -1): fitur hari t dipasangkan ke
label/nilai hari t+1, bukan hari t sendiri.

Cara pakai:
  from src.jisdor.target_formulation import build_target
  df = build_target(load_jisdor_csv("data/processed/jisdor_clean.csv"))

CLI:
  python -m src.jisdor.target_formulation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

__all__ = [
    "THRESHOLD_PCT",
    "label_direction",
    "build_target",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = _REPO_ROOT / "data" / "processed" / "jisdor_clean.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "processed" / "jisdor_target.csv"

COL_DATE = "tanggal"
COL_RATE = "kurs_jisdor"
COL_PCT_CHANGE = "pct_change"
COL_TARGET = "target"
COL_TARGET_REGRESI = "kurs_besok"

# Threshold pct_change (%) yang memisahkan NAIK/TURUN dari STABIL.
# Tervalidasi empiris: std. deviasi pct_change harian ~0.33%, rata-rata 0.019%.
THRESHOLD_PCT = 0.1


def label_direction(pct_change: float, threshold: float = THRESHOLD_PCT) -> Optional[str]:
    """
    Label arah pergerakan kurs jadi NAIK/TURUN/STABIL.

    Parameters
    ----------
    pct_change : float
        Perubahan kurs harian dalam persen.
    threshold : float
        Ambang batas (%) pemisah STABIL dari NAIK/TURUN (default 0.1).

    Returns
    -------
    str | None
        ``"NAIK"``, ``"TURUN"``, ``"STABIL"``, atau ``None`` jika NaN.
    """
    if pd.isna(pct_change):
        return None
    if pct_change > threshold:
        return "NAIK"
    if pct_change < -threshold:
        return "TURUN"
    return "STABIL"


def build_target(df: pd.DataFrame, threshold: float = THRESHOLD_PCT) -> pd.DataFrame:
    """
    Hitung ``pct_change``, label 3-class hari ini, lalu shift jadi target
    prediksi "besok" (fitur hari t -> label/nilai hari t+1). Sekaligus
    menyiapkan target regresi ``kurs_besok`` dari kolom mentah yang sama.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame kurs JISDOR terurut kronologis ascending, kolom
        ``tanggal`` (datetime64) dan ``kurs_jisdor`` (numerik). Lihat
        :func:`src.jisdor.preprocess_jisdor.preprocess_jisdor`.
    threshold : float
        Ambang batas (%) pemisah STABIL dari NAIK/TURUN.

    Returns
    -------
    pd.DataFrame
        DataFrame input ditambah kolom:
        - ``pct_change``: perubahan kurs harian (%) hari itu.
        - ``target``: label 3-class untuk prediksi *besok*
          (``df["target"]`` hari t = arah pergerakan kurs hari t+1).
        - ``kurs_besok``: nilai ``kurs_jisdor`` hari t+1 (target regresi).
        Baris terakhir (tidak punya "besok") di-drop karena kedua target
        NaN di baris yang sama.
    """
    df = df.copy()

    df[COL_PCT_CHANGE] = df[COL_RATE].pct_change() * 100
    label_besok = df[COL_PCT_CHANGE].apply(lambda x: label_direction(x, threshold)).shift(-1)
    df[COL_TARGET] = label_besok
    df[COL_TARGET_REGRESI] = df[COL_RATE].shift(-1)
    df = df.dropna(subset=[COL_TARGET, COL_TARGET_REGRESI]).reset_index(drop=True)

    return df


# CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formulasi target 3-class (NAIK/STABIL/TURUN) untuk prediksi kurs JISDOR."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Path CSV kurs bersih (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Path CSV output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_PCT,
        help=f"Ambang batas (%%) pemisah STABIL (default: {THRESHOLD_PCT}).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[target_formulation] ERROR: File tidak ditemukan: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.sort_values(COL_DATE).reset_index(drop=True)

    df_target = build_target(df, threshold=args.threshold)

    print(f"[target_formulation] {len(df_target)} baris dengan target (threshold={args.threshold}%).")
    print(f"[target_formulation] Distribusi target (classification):\n{df_target[COL_TARGET].value_counts()}")
    print(
        f"[target_formulation] Target regresi ({COL_TARGET_REGRESI}): "
        f"min={df_target[COL_TARGET_REGRESI].min():.0f}, "
        f"mean={df_target[COL_TARGET_REGRESI].mean():.0f}, "
        f"max={df_target[COL_TARGET_REGRESI].max():.0f}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_target.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    print(f"[target_formulation] Tersimpan: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
