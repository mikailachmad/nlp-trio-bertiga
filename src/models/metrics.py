"""
Metrik evaluasi terpadu untuk prediksi kurs USD/IDR (JISDOR).

Dua track paralel (lihat docs/target_formulation.md):

  Klasifikasi 3-class (kolom ``target``: NAIK/STABIL/TURUN)
    - Primer : Macro-F1. Kelas bergeser antar split (Train 380/283/296 vs
      Test 54/31/36); Majority-Class dapat Accuracy 44.6% tapi Macro-F1
      hanya ~0.21, jadi Accuracy saja menipu.
    - Sekunder: accuracy, balanced_accuracy.
    - Diagnostik: per-class precision/recall/f1 + confusion matrix
      (kelas TURUN paling lemah di baseline, F1 ~0.31).
    - Tahap akhir: Cohen's Kappa (kesepakatan di atas kebetulan).
      Pelengkap perbandingan semua model; dihitung hanya bila
      ``include_kappa=True`` / flag ``--with-kappa``.

  Regresi (kolom ``kurs_besok``, satuan rupiah)
    - Primer : RMSE (penalti error besar; konsisten dgn baseline random-walk).
    - Sekunder: MAE (robust outlier; RMSE >> MAE = sinyal lompatan kurs).
    - Nilai tambah: skill score ``1 - RMSE_model / RMSE_naive``.
      Literatur FX: random-walk sulit dikalahkan; skill > 0 = bukti sinyal.
    - Jembatan antar track: directional hit-rate dari prediksi regresi.
    - Tahap akhir: MAPE (%) — mean kurs bergeser antar split
      (Train ~15394 vs Test ~17569) sehingga % comparable lintas rezim.
      Pelengkap perbandingan semua model; dihitung hanya bila
      ``include_mape=True`` / flag ``--with-mape``.

Lower bound dari naive baseline (jangan diubah manual, lihat results/):
  - Klasifikasi: Macro-F1 Test 0.3637 (Persistence of Direction).
  - Regresi : RMSE Test 57.4694 (Persistence / Random Walk).

Aturan evaluasi: kronologis ketat (tuning di Train + TimeSeriesSplit,
pilih di Val, sentuh Test sekali — tanpa shuffle). Modul ini hanya
menghitung metrik; pembagian data tetap di src/jisdor/split_dataset.py.

Cara pakai (fase sekarang — ramping):
  from src.models.metrics import evaluate_classifier, evaluate_regression
  clf = evaluate_classifier(y_true, y_pred)
  reg = evaluate_regression(y_true, y_pred, rmse_naive=57.4694)

Tahap akhir (perbandingan semua model — aktifkan pelengkap):
  clf = evaluate_classifier(y_true, y_pred, include_kappa=True)
  reg = evaluate_regression(y_true, y_pred, rmse_naive=57.4694, include_mape=True)

CLI (evaluasi ulang naive baseline dari data split):
  python -m src.models.metrics --split-dir data/split --output-dir results
  python -m src.models.metrics --with-kappa --with-mape   # tahap akhir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

__all__ = [
    "LABEL_ORDER",
    "evaluate_classifier",
    "evaluate_regression",
    "directional_hit_from_regression",
    "print_classifier",
    "print_regressor",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPLIT_DIR = _REPO_ROOT / "data" / "split"
DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"

LABEL_ORDER = ["NAIK", "STABIL", "TURUN"]

COL_TARGET = "target"
COL_RATE = "kurs_jisdor"
COL_TARGET_REG = "kurs_besok"


# Klasifikasi
def evaluate_classifier(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Optional[List[str]] = None,
    include_kappa: bool = False,
) -> Dict:
    """Hitung metrik klasifikasi 3-class.

    Parameters
    ----------
    y_true, y_pred : array-like
        Label aktual dan prediksi (NAIK/STABIL/TURUN).
    labels : list[str], optional
        Urutan label (default: NAIK, STABIL, TURUN).
    include_kappa : bool
        Jika True, hitung juga Cohen's Kappa (pelengkap tahap akhir
        perbandingan semua model; default False).

    Returns
    -------
    dict
        accuracy, balanced_accuracy, macro_f1 (primer),
        classification_report (dict), confusion_matrix (list),
        dan opsional kappa bila ``include_kappa=True``.
    """
    if labels is None:
        labels = LABEL_ORDER
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, y_pred)), 4
        ),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", labels=labels)), 4
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }
    if include_kappa:
        out["kappa"] = round(float(cohen_kappa_score(y_true, y_pred)), 4)
    return out


# Regresi
def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    rmse_naive: Optional[float] = None,
    include_mape: bool = False,
) -> Dict:
    """Hitung metrik regresi kurs_besok (satuan rupiah).

    Parameters
    ----------
    y_true, y_pred : array-like
        Nilai aktual dan prediksi ``kurs_besok``.
    rmse_naive : float, optional
        RMSE baseline random-walk pada split yang sama. Jika diisi,
        dihitung juga ``skill_vs_naive = 1 - rmse/rmse_naive``.
    include_mape : bool
        Jika True, hitung juga MAPE (%) — pelengkap tahap akhir
        perbandingan semua model (default False).

    Returns
    -------
    dict
        rmse (primer), mae, opsional skill_vs_naive dan mape
        (bila masing-masing diaktifkan).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    out: Dict = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }
    if include_mape:
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        out["mape"] = round(float(mape), 4)
    if rmse_naive:
        out["skill_vs_naive"] = round(1 - rmse / float(rmse_naive), 4)
    return out


def directional_hit_from_regression(
    kurs_today: Sequence[float],
    y_true_next: Sequence[float],
    y_pred_next: Sequence[float],
) -> float:
    """Hitung directional hit-rate dari prediksi regresi.

    Proporsi hari di mana tanda ``(pred - kurs_hari_ini)`` sama dengan
    tanda ``(aktual - kurs_hari_ini)``. Menjembatani track regresi dan
    klasifikasi: regresi yang baik seharusnya juga benar arahnya.

    Nol-change (selisih tepat 0) dihitung sebagai benar hanya jika
    kedua sisi nol.
    """
    today = np.asarray(kurs_today, dtype=float)
    true = np.asarray(y_true_next, dtype=float)
    pred = np.asarray(y_pred_next, dtype=float)
    return round(float(np.mean(np.sign(pred - today) == np.sign(true - today))), 4)


# Pretty-print
def print_classifier(result: Dict, name: str = "Model", split: str = "Test") -> None:
    print(f"\n{'=' * 60}")
    print(f"  Klasifikasi: {name} | Split: {split}")
    print(f"{'=' * 60}")
    print(f"  Accuracy           : {result['accuracy']:.4f}")
    print(f"  Balanced Accuracy  : {result['balanced_accuracy']:.4f}")
    print(f"  Macro F1 (primer)  : {result['macro_f1']:.4f}")
    if "kappa" in result:
        print(f"  Cohen Kappa (pelengkap): {result['kappa']:.4f}")
    print(f"{'-' * 60}")
    report = result["classification_report"]
    print(f"  {'Kelas':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label in LABEL_ORDER:
        if label in report:
            r = report[label]
            print(
                f"  {label:<10} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                f"{r['f1-score']:>10.4f} {r['support']:>10.0f}"
            )
    print(f"{'=' * 60}\n")


def print_regressor(result: Dict, name: str = "Model", split: str = "Test") -> None:
    print(f"\n{'=' * 60}")
    print(f"  Regresi: {name} | Split: {split}")
    print(f"{'=' * 60}")
    print(f"  RMSE (primer) : {result['rmse']:.4f}")
    print(f"  MAE           : {result['mae']:.4f}")
    if "mape" in result:
        print(f"  MAPE (%) (pelengkap): {result['mape']:.4f}")
    if "skill_vs_naive" in result:
        print(f"  Skill vs naive: {result['skill_vs_naive']:+.4f}")
    if "directional_hit" in result:
        print(f"  Directional hit: {result['directional_hit']:.4f}")
    print(f"{'=' * 60}\n")


# CLI: evaluasi ulang naive baseline memakai metrik terpadu
def _persistence_direction(targets: pd.Series) -> np.ndarray:
    preds = targets.shift(1).values.copy()
    preds[0] = targets.mode().iloc[0]
    return preds


def _majority(targets_train: pd.Series, n: int) -> np.ndarray:
    return np.full(n, targets_train.mode().iloc[0])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluasi metrik terpadu (klasifikasi + regresi) "
        "pada naive baseline dari data split."
    )
    p.add_argument("--split-dir", type=str, default=str(DEFAULT_SPLIT_DIR))
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    p.add_argument(
        "--with-kappa",
        action="store_true",
        help="Hitung juga Cohen's Kappa (pelengkap tahap akhir).",
    )
    p.add_argument(
        "--with-mape",
        action="store_true",
        help="Hitung juga MAPE %% (pelengkap tahap akhir).",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    split_dir = Path(args.split_dir)
    for name in ("train.csv", "val.csv", "test.csv"):
        if not (split_dir / name).exists():
            print(
                f"[metrics] ERROR: {split_dir / name} tidak ditemukan. "
                "Jalankan `python -m src.jisdor.split_dataset` dulu.",
                file=sys.stderr,
            )
            return 1
    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")

    out: Dict = {"classification": {}, "regression": {}}

    for split_name, df in (("val", val), ("test", test)):
        y_true = df[COL_TARGET].values

        y_persist = _persistence_direction(df[COL_TARGET])
        r_persist = evaluate_classifier(
            y_true, y_persist, include_kappa=args.with_kappa
        )
        print_classifier(r_persist, "Persistence of Direction", split_name.capitalize())

        y_maj = _majority(train[COL_TARGET], len(df))
        r_maj = evaluate_classifier(y_true, y_maj, include_kappa=args.with_kappa)
        print_classifier(r_maj, "Majority-Class", split_name.capitalize())

        out["classification"][split_name] = {
            "persistence": r_persist,
            "majority_class": r_maj,
        }

        # Regresi: persistence (random walk) + skill (diri sendiri = 0) + hit-rate
        y_reg_true = df[COL_TARGET_REG].values
        y_reg_pred = df[COL_RATE].values.copy()
        r_reg = evaluate_regression(
            y_reg_true, y_reg_pred, include_mape=args.with_mape
        )
        r_reg["directional_hit"] = directional_hit_from_regression(
            df[COL_RATE].values, y_reg_true, y_reg_pred
        )
        # skill vs diri sendiri selalu 0.0 — penanda format untuk model berikutnya
        r_reg["skill_vs_naive"] = 0.0
        print_regressor(r_reg, "Persistence (Random Walk)", split_name.capitalize())
        out["regression"][split_name] = {"persistence": r_reg}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "unified_metrics_baseline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[metrics] Tersimpan: {path}")
    print("[metrics] Lower bound klasifikasi: Macro-F1 Test > 0.3637")
    print("[metrics] Lower bound regresi: RMSE Test < 57.4694 (skill > 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
