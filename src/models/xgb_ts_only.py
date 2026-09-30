"""XGBoost TS-only baseline (butir 2b): fitur kurs historis saja, tanpa NLP.

Ablasi langsung dari src/04_modeling.ipynb (XGBoost Gabungan TS+NLP):
- Fitur: FEAT_TS = [ret_lag1, ret_lag2, ret_lag3, ret_std5] saja.
- Target ganda sama: `target` 3-class + `kurs_besok` via `delta_besok`.
- Hiperparameter identik 04 untuk perbandingan murni efek fitur.
- Evaluasi via src/models/metrics.py (modul yang sama).
- Naive dihitung ulang same-rows via src/models/naive_baseline.py.
- Gabungan dimuat dari results/04_gabungan_results.json untuk tabel 3-arah.

CLI:
  python -m src.models.xgb_ts_only
  python -m src.models.xgb_ts_only --output-prefix 05_ts_only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.jisdor.target_formulation import THRESHOLD_PCT, label_direction  # noqa: E402
from src.models.metrics import (  # noqa: E402
    LABEL_ORDER,
    directional_hit_from_regression,
    evaluate_classifier,
    evaluate_regression,
)
from src.models.naive_baseline import predict_majority, predict_persistence  # noqa: E402

FEATURE_DIR = _REPO_ROOT / "data" / "feature_extracted"
RESULTS_DIR = _REPO_ROOT / "results"

FEAT_TS = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_std5"]
LABEL2ID = {lab: i for i, lab in enumerate(LABEL_ORDER)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}


def make_xgb_classifier() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABEL_ORDER),
        eval_metric="mlogloss",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )


def make_xgb_regressor() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="XGBoost TS-only baseline (tanpa NLP).")
    p.add_argument("--output-prefix", default="05_ts_only",
                   help="Prefix file output di results/ (default: 05_ts_only).")
    args = p.parse_args(argv)

    df = pd.read_csv(FEATURE_DIR / "model_dataset.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    if "target" not in df.columns:
        print("PERINGATAN: kolom `target` dihitung ulang dari kurs_besok.")
        pct_besok = (df["kurs_besok"] / df["kurs_jisdor"] - 1) * 100
        df["target"] = pct_besok.apply(lambda v: label_direction(v, THRESHOLD_PCT))
    df["delta_besok"] = df["kurs_besok"] - df["kurs_jisdor"]

    missing = [c for c in FEAT_TS + ["kurs_jisdor", "kurs_besok", "target", "split"]
               if c not in df.columns]
    if missing:
        print(f"[ts_only] ERROR: kolom hilang: {missing}", file=sys.stderr)
        return 1

    # Integritas kronologis
    bounds = df.groupby("split")["date"].agg(["min", "max", "size"])
    print(bounds.to_string())
    assert bounds.loc["train", "max"] < bounds.loc["val", "min"] < bounds.loc["test", "min"], \
        "Split tidak kronologis"

    splits = {s: df[df["split"] == s].sort_values("date").copy()
              for s in ["train", "val", "test"]}
    tr, va, te = splits["train"], splits["val"], splits["test"]
    X = {s: d[FEAT_TS] for s, d in splits.items()}
    y_clf = {s: d["target"].map(LABEL2ID).astype(int) for s, d in splits.items()}
    y_reg = {s: d["kurs_besok"].astype(float) for s, d in splits.items()}
    y_delta = {s: d["delta_besok"].astype(float) for s, d in splits.items()}
    kurs_today = {s: d["kurs_jisdor"].astype(float).to_numpy() for s, d in splits.items()}
    print({s: X[s].shape for s in X})

    all_results: list[dict] = []

    def add_result(model, split, task, metrics):
        all_results.append({"model": model, "split": split, "task": task, **{
            k: v for k, v in metrics.items()
            if k not in ("classification_report", "confusion_matrix")}})

    # 1) Naive same-rows
    rmse_naive: dict[str, float] = {}
    for s in ["val", "test"]:
        d = splits[s]
        r_p = evaluate_classifier(d["target"].to_numpy(),
                                  predict_persistence(d["target"]), include_kappa=True)
        r_m = evaluate_classifier(d["target"].to_numpy(),
                                  predict_majority(tr["target"], len(d)), include_kappa=True)
        add_result("Naive Persistence of Direction", s, "clf", r_p)
        add_result("Naive Majority-Class", s, "clf", r_m)
        r_r = evaluate_regression(y_reg[s], kurs_today[s], include_mape=True)
        rmse_naive[s] = r_r["rmse"]
        r_r["skill_vs_naive"] = 0.0
        add_result("Naive Persistence (Random Walk)", s, "reg", r_r)
        print(f"[{s}] Persist-F1={r_p['macro_f1']:.4f} Majority-F1={r_m['macro_f1']:.4f} "
              f"RW-RMSE={r_r['rmse']:.2f}")

    # 2) XGBoost TS-only klasifikasi
    w_train = compute_sample_weight(class_weight="balanced", y=y_clf["train"])
    xgb_clf = make_xgb_classifier()
    xgb_clf.fit(X["train"], y_clf["train"], sample_weight=w_train,
                eval_set=[(X["val"], y_clf["val"])], verbose=False)
    print("Best iteration (clf TS-only):", xgb_clf.best_iteration)
    pred_clf = {s: pd.Series(xgb_clf.predict(X[s])).map(ID2LABEL).to_numpy()
                for s in ["val", "test"]}
    clf_metrics = {}
    for s in ["val", "test"]:
        m = evaluate_classifier(splits[s]["target"].to_numpy(), pred_clf[s],
                                include_kappa=True)
        clf_metrics[s] = m
        print(f"[TS-only clf {s}] acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"kappa={m.get('kappa')}")
        add_result("XGBoost TS-only", s, "clf", m)

    # 3) XGBoost TS-only regresi via delta
    xgb_reg = make_xgb_regressor()
    xgb_reg.fit(X["train"], y_delta["train"],
                eval_set=[(X["val"], y_delta["val"])], verbose=False)
    print("Best iteration (reg TS-only):", xgb_reg.best_iteration)
    pred_reg = {s: kurs_today[s] + xgb_reg.predict(X[s]) for s in ["val", "test"]}
    reg_metrics = {}
    for s in ["val", "test"]:
        m = evaluate_regression(y_reg[s], pred_reg[s], rmse_naive=rmse_naive[s],
                                include_mape=True)
        m["directional_hit"] = directional_hit_from_regression(
            kurs_today[s], y_reg[s], pred_reg[s])
        reg_metrics[s] = m
        print(f"[TS-only reg {s}] RMSE={m['rmse']:.2f} skill={m['skill_vs_naive']:+.4f} "
              f"hit={m['directional_hit']:.4f}")
        add_result("XGBoost TS-only", s, "reg", m)

    # 4) Muat gabungan untuk tabel 3-arah
    gab_path = RESULTS_DIR / "04_gabungan_results.json"
    gab = None
    if gab_path.exists():
        gab = json.loads(gab_path.read_text(encoding="utf-8"))
        for s in ["val", "test"]:
            if "classification" in gab and s in gab["classification"]:
                g = {k: v for k, v in gab["classification"][s].items()
                     if k not in ("classification_report", "confusion_matrix")}
                add_result("XGBoost Gabungan (TS+NLP)", s, "clf", g)
            if "regression" in gab and s in gab["regression"]:
                add_result("XGBoost Gabungan (TS+NLP)", s, "reg", gab["regression"][s])
        print("[ts_only] Gabungan dimuat dari", gab_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(all_results)
    prefix = args.output_prefix
    res.round(6).to_csv(RESULTS_DIR / f"{prefix}_comparison.csv", index=False)

    out = {
        "model": "XGBoost TS-only (tanpa NLP)",
        "features": {"ts": FEAT_TS},
        "target": {"classification": f"target (NAIK/STABIL/TURUN, threshold {THRESHOLD_PCT:.2f}%)",
                   "regression": "kurs_besok (dilatih pada selisih, dikonversi ke level)"},
        "best_iteration": {"clf": int(xgb_clf.best_iteration),
                           "reg": int(xgb_reg.best_iteration)},
        "classification": {s: {k: v for k, v in m.items()} for s, m in clf_metrics.items()},
        "regression": {s: m for s, m in reg_metrics.items()},
        "rmse_naive_same_rows": rmse_naive,
    }
    with open(RESULTS_DIR / f"{prefix}_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Plot ringkas: confusion matrix test + prediksi vs aktual test
    cm = np.array(clf_metrics["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticks(range(3))
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("XGBoost TS-only: Confusion Matrix (Test)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    d_te = splits["test"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    ax[0].plot(d_te["date"], y_reg["test"], label="Aktual kurs_besok", lw=1.6)
    ax[0].plot(d_te["date"], kurs_today["test"], label="Naive Persistence",
               ls="--", alpha=0.8)
    ax[0].plot(d_te["date"], pred_reg["test"], label="XGBoost TS-only", lw=1.2)
    ax[0].set_title("Kurs JISDOR: aktual vs prediksi (Test, TS-only)")
    ax[0].set_ylabel("Rupiah per USD")
    ax[0].legend()
    ax[0].tick_params(axis="x", rotation=30)
    actual_delta = y_delta["test"].to_numpy()
    pred_delta = pred_reg["test"] - kurs_today["test"]
    lim = float(max(np.abs(actual_delta).max(), np.abs(pred_delta).max()))
    ax[1].scatter(actual_delta, pred_delta, s=14, alpha=0.7)
    ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].axvline(0, color="grey", lw=0.5)
    ax[1].set_xlabel("Selisih aktual (rupiah)")
    ax[1].set_ylabel("Selisih prediksi (rupiah)")
    ax[1].set_title("Selisih kurs besok: aktual vs prediksi (Test, TS-only)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{prefix}_predicted_vs_actual.png", dpi=150)
    plt.close()

    print(f"[ts_only] Tersimpan: {prefix}_comparison.csv, {prefix}_results.json, "
          f"{prefix}_confusion_matrix.png, {prefix}_predicted_vs_actual.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
