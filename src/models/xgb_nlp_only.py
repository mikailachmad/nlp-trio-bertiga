"""XGBoost NLP-only ablasi: fitur berita saja, tanpa kurs historis.

Ablasi langsung dari src/04_modeling.ipynb (Gabungan) dan
src/models/xgb_ts_only.py (TS-only):
- Fitur: FEAT_LEXICON (11, sama persis dengan 04) + tfidf_svd_* (50) = 61.
  Tanpa ret_lag1-3 / ret_std5. Tanpa scaling (pohon).
- Target ganda sama: `target` 3-class + `kurs_besok` via `delta_besok`.
- Hiperparameter identik 04/05 untuk perbandingan murni efek fitur.
- Evaluasi via src/models/metrics.py. Naive same-rows via naive_baseline.py.
- TS-only (05) + Gabungan (04) dimuat untuk tabel 4-arah.

CLI:
  python -m src.models.xgb_nlp_only
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="XGBoost NLP-only (tanpa kurs).")
    p.add_argument("--output-prefix", default="06_nlp_only")
    args = p.parse_args(argv)

    df = pd.read_csv(FEATURE_DIR / "model_dataset.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    if "target" not in df.columns:
        pct_besok = (df["kurs_besok"] / df["kurs_jisdor"] - 1) * 100
        df["target"] = pct_besok.apply(lambda v: label_direction(v, THRESHOLD_PCT))
    df["delta_besok"] = df["kurs_besok"] - df["kurs_jisdor"]

    tfidf_cols = sorted([c for c in df.columns if c.startswith("tfidf_svd_")],
                        key=lambda v: int(v.split("_")[-1]))
    FEATS = FEAT_LEXICON + tfidf_cols
    missing = [c for c in FEATS + ["kurs_jisdor", "kurs_besok", "target", "split"]
               if c not in df.columns]
    if missing:
        print(f"[nlp_only] ERROR: kolom hilang: {missing}", file=sys.stderr)
        return 1
    print(f"[nlp_only] Fitur NLP: lexicon={len(FEAT_LEXICON)} + svd={len(tfidf_cols)} "
          f"= {len(FEATS)}")

    splits = {s: df[df["split"] == s].sort_values("date").copy()
              for s in ["train", "val", "test"]}
    tr = splits["train"]
    X = {s: d[FEATS] for s, d in splits.items()}
    y_clf = {s: d["target"].map(LABEL2ID).astype(int) for s, d in splits.items()}
    y_reg = {s: d["kurs_besok"].astype(float) for s, d in splits.items()}
    y_delta = {s: d["delta_besok"].astype(float) for s, d in splits.items()}
    kurs_today = {s: d["kurs_jisdor"].astype(float).to_numpy() for s, d in splits.items()}

    all_results: list[dict] = []

    def add_result(model, split, task, metrics):
        all_results.append({"model": model, "split": split, "task": task, **{
            k: v for k, v in metrics.items()
            if k not in ("classification_report", "confusion_matrix")}})

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
        print(f"[{s}] Persist-F1={r_p['macro_f1']:.4f} RW-RMSE={r_r['rmse']:.2f}")

    w_train = compute_sample_weight(class_weight="balanced", y=y_clf["train"])
    xgb_clf = make_xgb_classifier()
    xgb_clf.fit(X["train"], y_clf["train"], sample_weight=w_train,
                eval_set=[(X["val"], y_clf["val"])], verbose=False)
    print("Best iteration (clf NLP-only):", xgb_clf.best_iteration)
    pred_clf = {s: pd.Series(xgb_clf.predict(X[s])).map(ID2LABEL).to_numpy()
                for s in ["val", "test"]}
    clf_metrics = {}
    for s in ["val", "test"]:
        m = evaluate_classifier(splits[s]["target"].to_numpy(), pred_clf[s],
                                include_kappa=True)
        clf_metrics[s] = m
        print(f"[NLP-only clf {s}] acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
        add_result("XGBoost NLP-only", s, "clf", m)

    xgb_reg = make_xgb_regressor()
    xgb_reg.fit(X["train"], y_delta["train"],
                eval_set=[(X["val"], y_delta["val"])], verbose=False)
    print("Best iteration (reg NLP-only):", xgb_reg.best_iteration)
    pred_reg = {s: kurs_today[s] + xgb_reg.predict(X[s]) for s in ["val", "test"]}
    reg_metrics = {}
    for s in ["val", "test"]:
        m = evaluate_regression(y_reg[s], pred_reg[s], rmse_naive=rmse_naive[s],
                                include_mape=True)
        m["directional_hit"] = directional_hit_from_regression(
            kurs_today[s], y_reg[s], pred_reg[s])
        reg_metrics[s] = m
        print(f"[NLP-only reg {s}] RMSE={m['rmse']:.2f} skill={m['skill_vs_naive']:+.4f} "
              f"hit={m['directional_hit']:.4f}")
        add_result("XGBoost NLP-only", s, "reg", m)

    # Muat TS-only (05) + Gabungan (04) untuk tabel 4-arah
    for path, label in [("05_ts_only_results.json", "XGBoost TS-only"),
                        ("04_gabungan_results.json", "XGBoost Gabungan (TS+NLP)")]:
        fp = RESULTS_DIR / path
        if fp.exists():
            g = json.loads(fp.read_text(encoding="utf-8"))
            for s in ["val", "test"]:
                if "classification" in g and s in g["classification"]:
                    c = {k: v for k, v in g["classification"][s].items()
                         if k not in ("classification_report", "confusion_matrix")}
                    add_result(label, s, "clf", c)
                if "regression" in g and s in g["regression"]:
                    add_result(label, s, "reg", g["regression"][s])
            print(f"[nlp_only] {label} dimuat dari {fp.name}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(all_results)
    prefix = args.output_prefix
    res.round(6).to_csv(RESULTS_DIR / f"{prefix}_comparison.csv", index=False)

    out = {
        "model": "XGBoost NLP-only (tanpa kurs)",
        "features": {"lexicon": FEAT_LEXICON, "tfidf_svd": len(tfidf_cols)},
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

    cm = np.array(clf_metrics["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticks(range(3))
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("XGBoost NLP-only: Confusion Matrix (Test)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[nlp_only] Tersimpan: {prefix}_comparison.csv, {prefix}_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
