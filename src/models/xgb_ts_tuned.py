"""Model A+ (Tugas 3): XGBoost TS-only yang DIPERKAYA + hyperparameter tuning.

Kontribusi Tugas 3 di atas Model A baseline (src/models/xgb_ts_only.py, 4 fitur):

1. Feature engineering indikator teknikal (tanpa lookahead; fitur hari t hanya
   memakai data <= t untuk memprediksi t+1):
     - Moving averages return : ret_sma{5,10,20}, ret_ema{5,10} + momentum kurs
       terhadap moving average (mom_vs_sma{10,20}).
     - Volatilitas multi-window: ret_std{10,20} + rasio vol pendek/panjang
       (vol_ratio_5_20) sebagai sinyal rezim.
   Digabung dengan FEAT_TS baseline (ret_lag1-3, ret_std5) -> 14 fitur.

2. Hyperparameter tuning (sebelumnya TIDAK ADA di repo): RandomizedSearchCV
   dengan TimeSeriesSplit (kronologis, tanpa shuffle) HANYA pada data train,
   skor macro-F1. Param terbaik lalu dipakai untuk refit final + early stopping
   di val + class balancing (sample_weight) seperti baseline. Test disentuh sekali.

Target ganda, metrik, dan format output identik dengan xgb_ts_only.py agar bisa
disandingkan langsung di matriks ablasi (Model A vs Model A+ vs B/C/C+) dan di
pipeline analisis error/korelasi (save_predictions format seragam).

CLI:
  python -m src.models.xgb_ts_tuned
  python -m src.models.xgb_ts_tuned --n-iter 60 --output-prefix 05b_ts_tuned
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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.jisdor.target_formulation import THRESHOLD_PCT, label_direction  # noqa: E402
from src.models.metrics import (  # noqa: E402
    LABEL_ORDER,
    directional_hit_from_regression,
    evaluate_classifier,
    evaluate_regression,
    save_predictions,
)
from src.models.naive_baseline import predict_majority, predict_persistence  # noqa: E402

FEATURE_DIR = _REPO_ROOT / "data" / "feature_extracted"
RESULTS_DIR = _REPO_ROOT / "results"

FEAT_TS_BASE = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_std5"]
# Fitur hasil feature-engineering (dihitung di add_ts_features).
FEAT_TS_ENGINEERED = [
    "ret_sma5", "ret_sma10", "ret_sma20",
    "ret_ema5", "ret_ema10",
    "mom_vs_sma10", "mom_vs_sma20",
    "ret_std10", "ret_std20",
    "vol_ratio_5_20",
]
FEAT_TS = FEAT_TS_BASE + FEAT_TS_ENGINEERED

LABEL2ID = {lab: i for i, lab in enumerate(LABEL_ORDER)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}


def add_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turunkan indikator teknikal dari kurs/return pada SELURUH seri terurut.

    Semua rolling memakai jendela yang berakhir di baris t (data <= t), jadi
    nilainya sudah diketahui di akhir hari t saat memprediksi t+1 -> tanpa
    lookahead. Dihitung sebelum split agar jendela awal val/test memakai ekor
    train (informasi yang memang tersedia di waktu prediksi, bukan kebocoran).
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["kurs_jisdor"]).diff()
    r = df["log_return"]
    kurs = df["kurs_jisdor"]

    # Moving averages return + momentum kurs terhadap MA.
    df["ret_sma5"] = r.rolling(5).mean()
    df["ret_sma10"] = r.rolling(10).mean()
    df["ret_sma20"] = r.rolling(20).mean()
    df["ret_ema5"] = r.ewm(span=5, adjust=False).mean()
    df["ret_ema10"] = r.ewm(span=10, adjust=False).mean()
    df["mom_vs_sma10"] = kurs / kurs.rolling(10).mean() - 1.0
    df["mom_vs_sma20"] = kurs / kurs.rolling(20).mean() - 1.0

    # Volatilitas multi-window + rasio rezim pendek/panjang.
    df["ret_std10"] = r.rolling(10).std()
    df["ret_std20"] = r.rolling(20).std()
    df["vol_ratio_5_20"] = df["ret_std5"] / df["ret_std20"].replace(0.0, np.nan)
    return df


def tune_classifier(X_tr, y_tr, n_iter: int, cv_splits: int, seed: int):
    """RandomizedSearchCV kronologis (TimeSeriesSplit) pada train, skor macro-F1.

    Tanpa sample_weight di dalam CV (menghindari bug slicing bobot per-fold);
    class balancing diterapkan saat refit final di main().
    """
    base = xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(LABEL_ORDER),
        eval_metric="mlogloss", n_estimators=300,
        random_state=seed, n_jobs=-1,
    )
    param_dist = {
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "gamma": [0.0, 0.1, 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=n_iter,
        scoring="f1_macro", cv=TimeSeriesSplit(n_splits=cv_splits),
        random_state=seed, n_jobs=-1, verbose=0, refit=False,
    )
    search.fit(X_tr, y_tr)
    return search.best_params_, float(search.best_score_)


def tune_regressor(X_tr, y_tr, n_iter: int, cv_splits: int, seed: int):
    """RandomizedSearchCV kronologis pada train, skor RMSE (delta kurs)."""
    base = xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse",
        n_estimators=300, random_state=seed, n_jobs=-1,
    )
    param_dist = {
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "gamma": [0.0, 0.1, 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=cv_splits),
        random_state=seed, n_jobs=-1, verbose=0, refit=False,
    )
    search.fit(X_tr, y_tr)
    return search.best_params_, float(-search.best_score_)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Model A+ : XGBoost TS diperkaya + hyperparameter tuning.")
    p.add_argument("--output-prefix", default="05b_ts_tuned",
                   help="Prefix output di results/ (default: 05b_ts_tuned).")
    p.add_argument("--n-iter", type=int, default=50,
                   help="Jumlah kandidat RandomizedSearchCV (default: 50).")
    p.add_argument("--cv-splits", type=int, default=5,
                   help="Lipatan TimeSeriesSplit untuk tuning (default: 5).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    df = pd.read_csv(FEATURE_DIR / "model_dataset.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if "target" not in df.columns:
        pct_besok = (df["kurs_besok"] / df["kurs_jisdor"] - 1) * 100
        df["target"] = pct_besok.apply(lambda v: label_direction(v, THRESHOLD_PCT))
    df["delta_besok"] = df["kurs_besok"] - df["kurs_jisdor"]

    # Feature engineering (seluruh seri dulu, baru split).
    df = add_ts_features(df)

    missing = [c for c in FEAT_TS + ["kurs_jisdor", "kurs_besok", "target", "split"]
               if c not in df.columns]
    if missing:
        print(f"[ts_tuned] ERROR: kolom hilang: {missing}", file=sys.stderr)
        return 1

    bounds = df.groupby("split")["date"].agg(["min", "max", "size"])
    print(bounds.to_string())
    assert bounds.loc["train", "max"] < bounds.loc["val", "min"] < bounds.loc["test", "min"], \
        "Split tidak kronologis"

    splits = {s: df[df["split"] == s].sort_values("date").copy()
              for s in ["train", "val", "test"]}
    tr = splits["train"]
    X = {s: d[FEAT_TS] for s, d in splits.items()}
    y_clf = {s: d["target"].map(LABEL2ID).astype(int) for s, d in splits.items()}
    y_reg = {s: d["kurs_besok"].astype(float) for s, d in splits.items()}
    y_delta = {s: d["delta_besok"].astype(float) for s, d in splits.items()}
    kurs_today = {s: d["kurs_jisdor"].astype(float).to_numpy() for s, d in splits.items()}
    print(f"[ts_tuned] {len(FEAT_TS)} fitur ({len(FEAT_TS_BASE)} baseline + "
          f"{len(FEAT_TS_ENGINEERED)} engineered) | train={len(tr)}")

    all_results: list[dict] = []

    def add_result(model, split, task, metrics):
        all_results.append({"model": model, "split": split, "task": task, **{
            k: v for k, v in metrics.items()
            if k not in ("classification_report", "confusion_matrix")}})

    # 1) Naive same-rows (acuan identik dgn skrip lain)
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

    # 2) Tuning klasifikasi -> refit final (class weight + early stopping di val)
    print(f"[ts_tuned] Tuning classifier: RandomizedSearchCV n_iter={args.n_iter} "
          f"TimeSeriesSplit({args.cv_splits}) ...")
    best_clf_params, best_clf_cv = tune_classifier(
        X["train"], y_clf["train"], args.n_iter, args.cv_splits, args.seed)
    print(f"[ts_tuned] best clf macro-F1 (CV)={best_clf_cv:.4f} | params={best_clf_params}")

    w_train = compute_sample_weight(class_weight="balanced", y=y_clf["train"])
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(LABEL_ORDER),
        eval_metric="mlogloss", n_estimators=500, random_state=args.seed,
        n_jobs=-1, early_stopping_rounds=30, **best_clf_params)
    xgb_clf.fit(X["train"], y_clf["train"], sample_weight=w_train,
                eval_set=[(X["val"], y_clf["val"])], verbose=False)
    print("Best iteration (clf):", xgb_clf.best_iteration)
    pred_clf = {s: pd.Series(xgb_clf.predict(X[s])).map(ID2LABEL).to_numpy()
                for s in ["val", "test"]}
    clf_metrics = {}
    for s in ["val", "test"]:
        m = evaluate_classifier(splits[s]["target"].to_numpy(), pred_clf[s],
                                include_kappa=True)
        clf_metrics[s] = m
        print(f"[ts_tuned clf {s}] acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"kappa={m.get('kappa')}")
        add_result("XGBoost TS+ (tuned)", s, "clf", m)

    # 3) Tuning regresi (delta) -> refit final + early stopping
    print(f"[ts_tuned] Tuning regressor ...")
    best_reg_params, best_reg_cv = tune_regressor(
        X["train"], y_delta["train"], max(args.n_iter // 2, 20),
        args.cv_splits, args.seed)
    print(f"[ts_tuned] best reg RMSE delta (CV)={best_reg_cv:.4f} | params={best_reg_params}")

    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse", n_estimators=500,
        random_state=args.seed, n_jobs=-1, early_stopping_rounds=30,
        **best_reg_params)
    xgb_reg.fit(X["train"], y_delta["train"],
                eval_set=[(X["val"], y_delta["val"])], verbose=False)
    print("Best iteration (reg):", xgb_reg.best_iteration)
    pred_reg = {s: kurs_today[s] + xgb_reg.predict(X[s]) for s in ["val", "test"]}
    reg_metrics = {}
    for s in ["val", "test"]:
        m = evaluate_regression(y_reg[s], pred_reg[s], rmse_naive=rmse_naive[s],
                                include_mape=True)
        m["directional_hit"] = directional_hit_from_regression(
            kurs_today[s], y_reg[s], pred_reg[s])
        reg_metrics[s] = m
        print(f"[ts_tuned reg {s}] RMSE={m['rmse']:.2f} skill={m['skill_vs_naive']:+.4f} "
              f"hit={m['directional_hit']:.4f}")
        add_result("XGBoost TS+ (tuned)", s, "reg", m)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    pd.DataFrame(all_results).round(6).to_csv(
        RESULTS_DIR / f"{prefix}_comparison.csv", index=False)

    pred_frames = pd.concat([splits["val"], splits["test"]])
    save_predictions(
        RESULTS_DIR / f"{prefix}_predictions.csv",
        model="XGBoost TS+ (tuned)",
        date=pred_frames["date"], split=pred_frames["split"],
        y_true_clf=np.concatenate([splits["val"]["target"].to_numpy(),
                                   splits["test"]["target"].to_numpy()]),
        y_pred_clf=np.concatenate([pred_clf["val"], pred_clf["test"]]),
        kurs_today=np.concatenate([kurs_today["val"], kurs_today["test"]]),
        y_true_reg=np.concatenate([y_reg["val"].to_numpy(), y_reg["test"].to_numpy()]),
        y_pred_reg=np.concatenate([pred_reg["val"], pred_reg["test"]]),
    )

    out = {
        "model": "XGBoost TS+ (enriched + tuned)",
        "features": {"ts_base": FEAT_TS_BASE, "ts_engineered": FEAT_TS_ENGINEERED,
                     "n_features": len(FEAT_TS)},
        "tuning": {"method": "RandomizedSearchCV + TimeSeriesSplit",
                   "n_iter": args.n_iter, "cv_splits": args.cv_splits,
                   "best_clf_params": best_clf_params, "best_clf_cv_macro_f1": round(best_clf_cv, 4),
                   "best_reg_params": best_reg_params, "best_reg_cv_rmse": round(best_reg_cv, 4)},
        "target": {"classification": f"target (NAIK/STABIL/TURUN, threshold {THRESHOLD_PCT:.2f}%)",
                   "regression": "kurs_besok (dilatih pada selisih, dikonversi ke level)"},
        "best_iteration": {"clf": int(xgb_clf.best_iteration),
                           "reg": int(xgb_reg.best_iteration)},
        "classification": clf_metrics,
        "regression": reg_metrics,
        "rmse_naive_same_rows": rmse_naive,
    }
    with open(RESULTS_DIR / f"{prefix}_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Confusion matrix test
    cm = np.array(clf_metrics["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticks(range(3)); ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title("XGBoost TS+ (tuned): Confusion Matrix (Test)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (gain) — untuk bagian interpretabilitas laporan
    try:
        imp = xgb_clf.feature_importances_
        order = np.argsort(imp)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.barh([FEAT_TS[i] for i in order][::-1], imp[order][::-1], color="#2563eb")
        ax.set_title("Model A+ : Feature Importance (gain, classifier)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{prefix}_feature_importance.png", dpi=150)
        plt.close()
    except Exception as e:
        print("[ts_tuned] lewati feature importance:", e)

    print(f"[ts_tuned] Tersimpan: {prefix}_results.json, {prefix}_comparison.csv, "
          f"{prefix}_predictions.csv, {prefix}_confusion_matrix.png, "
          f"{prefix}_feature_importance.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
