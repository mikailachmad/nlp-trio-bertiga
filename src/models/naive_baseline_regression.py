"""
Naive Baseline regresi untuk nilai kurs USD/IDR (JISDOR) hari berikutnya.

Satu strategi baseline diimplementasikan:
  - Persistence (Random Walk) — prediksi kurs_besok hari t = kurs_jisdor
    hari t. Asumsi: untuk data FX harian, random walk adalah baseline
    standar karena kurs hari ini adalah prediktor terbaik untuk kurs besok
    tanpa informasi tambahan.

Baseline ini jadi lower-bound performa; model apapun yang tidak mengalahkan
RMSE/MAE-nya dianggap tidak layak.

Referensi tugas:
  - Tugas 2, butir 2a: formulasi target regresi ("Nilai Penutupan Hari
    Berikutnya").
  - Tugas 2, butir 2b: "Latih Model Baseline Time-Series ... (misal: ARIMA,
    XGBoost, atau Naive Baseline)."
  - Tugas 2, butir 3b: metrik evaluasi regresi (RMSE, MAE).

Cara pakai:
  from src.models.naive_baseline_regression import (
      predict_persistence_regression, evaluate_regression_baseline
  )
  y_pred = predict_persistence_regression(df["kurs_jisdor"])
  report = evaluate_regression_baseline(y_true, y_pred, strategy_name="Persistence")

CLI:
  python -m src.models.naive_baseline_regression
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

__all__ = [
    "predict_persistence_regression",
    "evaluate_regression_baseline",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPLIT_DIR = _REPO_ROOT / "data" / "split"
DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"

COL_DATE = "tanggal"
COL_RATE = "kurs_jisdor"
COL_TARGET_REGRESI = "kurs_besok"


# Strategi Prediksi
def predict_persistence_regression(kurs: pd.Series) -> np.ndarray:
    """
    Persistence (Random Walk): prediksi kurs_besok hari t = kurs_jisdor hari t.

    Parameters
    ----------
    kurs : pd.Series
        Kolom ``kurs_jisdor`` kronologis dari split yang sama dengan target.

    Returns
    -------
    np.ndarray
        Array prediksi ``kurs_besok`` sepanjang ``kurs``.
    """
    return kurs.values.copy()


# Evaluasi
def evaluate_regression_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_name: str,
) -> Dict:
    """
    Hitung metrik evaluasi regresi untuk satu strategi baseline.

    Parameters
    ----------
    y_true : array-like
        Nilai ``kurs_besok`` aktual.
    y_pred : array-like
        Prediksi model.
    strategy_name : str
        Nama strategi (untuk display).

    Returns
    -------
    dict
        Dictionary berisi rmse, mae, dan strategy_name.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {
        "strategy": strategy_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }


def print_evaluation(result: Dict, split_name: str = "Test") -> None:
    """Pretty-print hasil evaluasi ke stdout."""
    print(f"\n{'='*60}")
    print(f"  Naive Baseline Regresi: {result['strategy']}")
    print(f"  Split: {split_name}")
    print(f"{'='*60}")
    print(f"  RMSE : {result['rmse']:.4f}")
    print(f"  MAE  : {result['mae']:.4f}")
    print(f"{'='*60}\n")


# CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Naive Baseline regresi: evaluasi Persistence pada data kurs JISDOR."
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=str(DEFAULT_SPLIT_DIR),
        help=f"Direktori berisi train.csv, val.csv, test.csv (default: {DEFAULT_SPLIT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Direktori output hasil evaluasi JSON (default: {DEFAULT_RESULTS_DIR}).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    split_dir = Path(args.split_dir)
    for name in ("train.csv", "val.csv", "test.csv"):
        if not (split_dir / name).exists():
            print(
                f"[naive_baseline_regression] ERROR: {split_dir / name} tidak ditemukan. "
                f"Jalankan `python -m src.jisdor.split_dataset` terlebih dahulu.",
                file=sys.stderr,
            )
            return 1

    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")

    print(f"[naive_baseline_regression] Val: {len(val)} | Test: {len(test)}")

    all_results = {}

    # Evaluasi pada Val dan Test
    for split_name, split_df in [("val", val), ("test", test)]:
        y_true = split_df[COL_TARGET_REGRESI].values

        y_pred = predict_persistence_regression(split_df[COL_RATE])
        res = evaluate_regression_baseline(y_true, y_pred, "Persistence (Random Walk)")
        print_evaluation(res, split_name.capitalize())

        all_results[split_name] = {"persistence": res}

    # Simpan hasil ke JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "naive_baseline_regression_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[naive_baseline_regression] Hasil tersimpan: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
