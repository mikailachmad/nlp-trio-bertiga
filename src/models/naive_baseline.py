"""
Naive Baseline untuk klasifikasi arah kurs USD/IDR (JISDOR).

Dua strategi baseline diimplementasikan:
  1. Persistence of Direction — prediksi arah besok = arah hari ini.
     Asumsi: tren pasar cenderung lanjut (momentum jangka pendek).
  2. Majority-Class — selalu prediksi kelas mayoritas di Train.
     Asumsi: tanpa informasi apapun, tebakan terbaik = kelas terbanyak.

Kedua strategi ini menjadi lower-bound performa; model apapun yang tidak
mengalahkan keduanya dianggap tidak layak.

Referensi tugas:
  - Tugas 2, butir 2b: "Latih Model Baseline Time-Series menggunakan data
    historis nilai tukar (misal: ARIMA, XGBoost, atau Naive Baseline)."

Cara pakai:
  from src.models.naive_baseline import (
      predict_persistence, predict_majority, evaluate_baseline
  )
  y_pred = predict_persistence(y_true_series)
  report = evaluate_baseline(y_true, y_pred, strategy_name="Persistence")

CLI:
  python -m src.models.naive_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

__all__ = [
    "predict_persistence",
    "predict_majority",
    "evaluate_baseline",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPLIT_DIR = _REPO_ROOT / "data" / "split"
DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"

COL_DATE = "tanggal"
COL_TARGET = "target"
LABEL_ORDER = ["NAIK", "STABIL", "TURUN"]


# Strategi Prediksi
def predict_persistence(targets: pd.Series) -> np.ndarray:
    """
    Persistence of Direction: prediksi arah besok = arah hari ini.

    Baris pertama tidak punya "kemarin", jadi diisi dengan kelas mayoritas
    keseluruhan sebagai fallback.

    Parameters
    ----------
    targets : pd.Series
        Label target kronologis (NAIK/STABIL/TURUN).

    Returns
    -------
    np.ndarray
        Array prediksi sepanjang ``targets``.
    """
    preds = targets.shift(1).values.copy()
    # Baris pertama: fallback ke kelas mayoritas
    majority = targets.mode().iloc[0]
    preds[0] = majority
    return preds


def predict_majority(targets_train: pd.Series, n: int) -> np.ndarray:
    """
    Majority-Class: selalu prediksi kelas mayoritas di data Train.

    Parameters
    ----------
    targets_train : pd.Series
        Label target dari split Train (untuk menentukan kelas mayoritas).
    n : int
        Jumlah prediksi yang dihasilkan.

    Returns
    -------
    np.ndarray
        Array prediksi berisi satu kelas saja, sepanjang ``n``.
    """
    majority = targets_train.mode().iloc[0]
    return np.full(n, majority)

# Evaluasi
def evaluate_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_name: str,
    labels: Optional[List[str]] = None,
) -> Dict:
    """
    Hitung semua metrik evaluasi untuk satu strategi baseline.

    Parameters
    ----------
    y_true : array-like
        Label ground truth.
    y_pred : array-like
        Prediksi model.
    strategy_name : str
        Nama strategi (untuk display).
    labels : list[str], optional
        Urutan label (default: NAIK, STABIL, TURUN).

    Returns
    -------
    dict
        Dictionary berisi accuracy, macro_f1, classification_report (dict),
        confusion_matrix (list of lists), dan strategy_name.
    """
    if labels is None:
        labels = LABEL_ORDER

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "strategy": strategy_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def print_evaluation(result: Dict, split_name: str = "Test") -> None:
    """Pretty-print hasil evaluasi ke stdout."""
    print(f"\n{'='*60}")
    print(f"  Naive Baseline: {result['strategy']}")
    print(f"  Split: {split_name}")
    print(f"{'='*60}")
    print(f"  Accuracy           : {result['accuracy']:.4f}")
    print(f"  Macro F1-Score     : {result['macro_f1']:.4f}")
    print(f"{'-'*60}")
    print("  Classification Report:")
    report = result["classification_report"]
    print(f"  {'Kelas':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label in LABEL_ORDER:
        if label in report:
            r = report[label]
            print(
                f"  {label:<10} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                f"{r['f1-score']:>10.4f} {r['support']:>10.0f}"
            )
    print(f"{'-'*60}")
    print("  Confusion Matrix (baris=aktual, kolom=prediksi):")
    print(f"  {'':>10}", end="")
    for label in LABEL_ORDER:
        print(f" {label:>8}", end="")
    print()
    cm = result["confusion_matrix"]
    for i, label in enumerate(LABEL_ORDER):
        print(f"  {label:>10}", end="")
        for val in cm[i]:
            print(f" {val:>8}", end="")
        print()
    print(f"{'='*60}\n")

# CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Naive Baseline: evaluasi Persistence & Majority-Class pada data kurs JISDOR."
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
                f"[naive_baseline] ERROR: {split_dir / name} tidak ditemukan. "
                f"Jalankan `python -m src.jisdor.split_dataset` terlebih dahulu.",
                file=sys.stderr,
            )
            return 1

    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")

    print(f"[naive_baseline] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    all_results = {}

    # Evaluasi pada Val dan Test
    for split_name, split_df in [("val", val), ("test", test)]:
        y_true = split_df[COL_TARGET].values

        # Strategi 1: Persistence of Direction
        y_pred_persist = predict_persistence(split_df[COL_TARGET])
        res_persist = evaluate_baseline(y_true, y_pred_persist, "Persistence of Direction")
        print_evaluation(res_persist, split_name.capitalize())

        # Strategi 2: Majority-Class
        y_pred_majority = predict_majority(train[COL_TARGET], len(split_df))
        res_majority = evaluate_baseline(y_true, y_pred_majority, "Majority-Class")
        print_evaluation(res_majority, split_name.capitalize())

        all_results[split_name] = {
            "persistence": res_persist,
            "majority_class": res_majority,
        }

    # Simpan hasil ke JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "naive_baseline_results.json"

    # Pastikan semua nilai bisa di-serialize ke JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[naive_baseline] Hasil tersimpan: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
