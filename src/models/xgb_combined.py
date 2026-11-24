"""XGBoost gabungan: fitur kurs historis + fitur berita.

Fitur menggabungkan FEAT_TS, FEAT_LEXICON, dan seluruh tfidf_svd_* dari
model_dataset.csv. Klasifikasi dan regresi memakai konfigurasi yang sama
dengan model TS-only dan NLP-only agar hasilnya dapat dibandingkan langsung.

CLI:
  python -m src.models.xgb_combined
  python -m src.models.xgb_combined --output-prefix 04_gabungan
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
    save_predictions,
)
from src.models.naive_baseline import predict_majority, predict_persistence  # noqa: E402

FEATURE_DIR = _REPO_ROOT / "data" / "feature_extracted"
RESULTS_DIR = _REPO_ROOT / "results"

FEAT_TS = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_std5"]
FEAT_LEXICON = [
    "vader_title_mean", "vader_body_mean", "vader_body_min", "vader_neg_share",
    "lm_negative_mean", "lm_positive_mean", "lm_uncertainty_mean",
    "lm_litigious_mean", "lm_constraining_mean", "lm_net_mean", "no_news",
]
LABEL2ID = {lab: i for i, lab in enumerate(LABEL_ORDER)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}


def make_xgb_classifier() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(LABEL_ORDER),
        eval_metric="mlogloss", n_estimators=500, max_depth=3,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, early_stopping_rounds=30,
    )


def make_xgb_regressor() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse",
        n_estimators=500, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, early_stopping_rounds=30,
    )


def _tfidf_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in df.columns if c.startswith("tfidf_svd_")),
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="XGBoost Gabungan (TS+NLP).")
    parser.add_argument(
        "--output-prefix", default="04_gabungan",
        help="Prefix file output di results/ (default: 04_gabungan).",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(FEATURE_DIR / "model_dataset.csv")
    required = ["date", "kurs_jisdor", "kurs_besok", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[xgb_combined] ERROR: kolom hilang: {missing}", file=sys.stderr)
        return 1

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    if "target" not in df.columns:
        pct_besok = (df["kurs_besok"] / df["kurs_jisdor"] - 1) * 100
        df["target"] = pct_besok.apply(
            lambda value: label_direction(value, THRESHOLD_PCT)
        )
    df["delta_besok"] = df["kurs_besok"] - df["kurs_jisdor"]

    tfidf_cols = _tfidf_cols(df)
    features = FEAT_TS + FEAT_LEXICON + tfidf_cols
    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        print(f"[xgb_combined] ERROR: kolom fitur hilang: {missing_features}",
              file=sys.stderr)
        return 1
    print(f"[xgb_combined] Fitur: TS={len(FEAT_TS)} + "
          f"lexicon={len(FEAT_LEXICON)} + tfidf_svd={len(tfidf_cols)} "
          f"= {len(features)}")

    bounds = df.groupby("split")["date"].agg(["min", "max", "size"])
    expected_splits = ["train", "val", "test"]
    absent_splits = [s for s in expected_splits if s not in bounds.index]
    if absent_splits or (bounds.loc[expected_splits, "size"] == 0).any():
        print(f"[xgb_combined] ERROR: split kosong atau tidak tersedia: "
              f"{absent_splits}", file=sys.stderr)
        return 1
    print(bounds.to_string())
    if not (bounds.loc["train", "max"] < bounds.loc["val", "min"] < bounds.loc["test", "min"]):
        print("[xgb_combined] ERROR: split tidak kronologis.", file=sys.stderr)
        return 1

    splits = {
        s: df[df["split"] == s].sort_values("date").copy()
        for s in expected_splits
    }
    train = splits["train"]
    X = {s: part[features] for s, part in splits.items()}
    y_clf = {s: part["target"].map(LABEL2ID).astype(int)
             for s, part in splits.items()}
    y_reg = {s: part["kurs_besok"].astype(float)
             for s, part in splits.items()}
    y_delta = {s: part["delta_besok"].astype(float)
               for s, part in splits.items()}
    kurs_today = {
        s: part["kurs_jisdor"].astype(float).to_numpy()
        for s, part in splits.items()
    }
    print({s: X[s].shape for s in X})

    all_results: list[dict] = []

    def add_result(model, split, task, metrics):
        all_results.append({
            "model": model,
            "split": split,
            "task": task,
            **{
                key: value for key, value in metrics.items()
                if key not in ("classification_report", "confusion_matrix")
            },
        })

    rmse_naive: dict[str, float] = {}
    for split in ("val", "test"):
        part = splits[split]
        persistence = evaluate_classifier(
            part["target"].to_numpy(),
            predict_persistence(part["target"]),
            include_kappa=True,
        )
        majority = evaluate_classifier(
            part["target"].to_numpy(),
            predict_majority(train["target"], len(part)),
            include_kappa=True,
        )
        add_result("Naive Persistence of Direction", split, "clf", persistence)
        add_result("Naive Majority-Class", split, "clf", majority)
        random_walk = evaluate_regression(
            y_reg[split], kurs_today[split], include_mape=True
        )
        rmse_naive[split] = random_walk["rmse"]
        random_walk["skill_vs_naive"] = 0.0
        add_result("Naive Persistence (Random Walk)", split, "reg", random_walk)
        print(f"[{split}] Persist-F1={persistence['macro_f1']:.4f} "
              f"Majority-F1={majority['macro_f1']:.4f} "
              f"RW-RMSE={random_walk['rmse']:.2f}")

    model_name = "XGBoost Gabungan (TS+NLP)"
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_clf["train"])
    classifier = make_xgb_classifier()
    classifier.fit(
        X["train"], y_clf["train"], sample_weight=sample_weight,
        eval_set=[(X["val"], y_clf["val"])], verbose=False,
    )
    print("Best iteration (clf Gabungan):", classifier.best_iteration)
    pred_clf = {}
    clf_metrics = {}
    for split in ("val", "test"):
        pred_clf[split] = (
            pd.Series(classifier.predict(X[split])).map(ID2LABEL).to_numpy()
        )
        metrics = evaluate_classifier(
            splits[split]["target"].to_numpy(), pred_clf[split], include_kappa=True
        )
        clf_metrics[split] = metrics
        add_result(model_name, split, "clf", metrics)
        print(f"[Gabungan clf {split}] acc={metrics['accuracy']:.4f} "
              f"macroF1={metrics['macro_f1']:.4f} kappa={metrics.get('kappa')}")

    regressor = make_xgb_regressor()
    regressor.fit(
        X["train"], y_delta["train"],
        eval_set=[(X["val"], y_delta["val"])], verbose=False,
    )
    print("Best iteration (reg Gabungan):", regressor.best_iteration)
    pred_reg = {}
    reg_metrics = {}
    for split in ("val", "test"):
        pred_reg[split] = kurs_today[split] + regressor.predict(X[split])
        metrics = evaluate_regression(
            y_reg[split], pred_reg[split], rmse_naive=rmse_naive[split],
            include_mape=True,
        )
        metrics["directional_hit"] = directional_hit_from_regression(
            kurs_today[split], y_reg[split], pred_reg[split]
        )
        reg_metrics[split] = metrics
        add_result(model_name, split, "reg", metrics)
        print(f"[Gabungan reg {split}] RMSE={metrics['rmse']:.2f} "
              f"skill={metrics['skill_vs_naive']:+.4f} "
              f"hit={metrics['directional_hit']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    pd.DataFrame(all_results).round(6).to_csv(
        RESULTS_DIR / f"{prefix}_comparison.csv", index=False
    )

    prediction_rows = pd.concat([splits["val"], splits["test"]])
    save_predictions(
        RESULTS_DIR / f"{prefix}_predictions.csv",
        model=model_name,
        date=prediction_rows["date"],
        split=prediction_rows["split"],
        y_true_clf=np.concatenate([
            splits["val"]["target"].to_numpy(),
            splits["test"]["target"].to_numpy(),
        ]),
        y_pred_clf=np.concatenate([pred_clf["val"], pred_clf["test"]]),
        kurs_today=np.concatenate([kurs_today["val"], kurs_today["test"]]),
        y_true_reg=np.concatenate([
            y_reg["val"].to_numpy(), y_reg["test"].to_numpy(),
        ]),
        y_pred_reg=np.concatenate([pred_reg["val"], pred_reg["test"]]),
    )

    output = {
        "model": model_name,
        "features": {
            "ts": FEAT_TS,
            "lexicon": FEAT_LEXICON,
            "tfidf_svd": len(tfidf_cols),
            "n_features": len(features),
        },
        "target": {
            "classification": (
                f"target (NAIK/STABIL/TURUN, threshold {THRESHOLD_PCT:.2f}%)"
            ),
            "regression": (
                "kurs_besok (dilatih pada selisih, dikonversi ke level)"
            ),
        },
        "best_iteration": {
            "clf": int(classifier.best_iteration),
            "reg": int(regressor.best_iteration),
        },
        "classification": clf_metrics,
        "regression": reg_metrics,
        "rmse_naive_same_rows": rmse_naive,
    }
    with open(RESULTS_DIR / f"{prefix}_results.json", "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    cm = np.asarray(clf_metrics["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("XGBoost Gabungan: Confusion Matrix (Test)")
    for row in range(len(LABEL_ORDER)):
        for col in range(len(LABEL_ORDER)):
            ax.text(
                col, row, cm[row, col], ha="center", va="center",
                color="white" if cm[row, col] > cm.max() / 2 else "black",
            )
    fig.colorbar(image, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    test = splits["test"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes[0].plot(test["date"], y_reg["test"], label="Aktual kurs_besok", lw=1.6)
    axes[0].plot(
        test["date"], kurs_today["test"], label="Naive Persistence",
        ls="--", alpha=0.8,
    )
    axes[0].plot(test["date"], pred_reg["test"], label=model_name, lw=1.2)
    axes[0].set_title("Kurs JISDOR: aktual vs prediksi (Test, Gabungan)")
    axes[0].set_ylabel("Rupiah per USD")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    actual_delta = y_delta["test"].to_numpy()
    predicted_delta = pred_reg["test"] - kurs_today["test"]
    limit = float(max(np.abs(actual_delta).max(), np.abs(predicted_delta).max(), 1))
    axes[1].scatter(actual_delta, predicted_delta, s=14, alpha=0.7)
    axes[1].plot([-limit, limit], [-limit, limit], "k--", lw=1)
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].axvline(0, color="grey", lw=0.5)
    axes[1].set_xlabel("Selisih aktual (rupiah)")
    axes[1].set_ylabel("Selisih prediksi (rupiah)")
    axes[1].set_title("Selisih kurs besok: aktual vs prediksi (Test)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    importance = pd.Series(
        classifier.feature_importances_, index=features
    ).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    importance.plot.barh(ax=ax)
    ax.set_xlabel("Feature importance")
    ax.set_title("XGBoost Gabungan: 20 fitur terpenting (klasifikasi)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_feature_importance.png", dpi=150)
    plt.close(fig)

    print(f"[xgb_combined] Tersimpan: {prefix}_comparison.csv, "
          f"{prefix}_results.json, {prefix}_predictions.csv, "
          f"{prefix}_confusion_matrix.png, {prefix}_predicted_vs_actual.png, "
          f"{prefix}_feature_importance.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())