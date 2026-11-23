"""Early fusion multimodal (butir 2a.i): concat fitur TS + teks → satu XGBoost.

Rancangan fusion tingkat fitur. Semua modalitas digabung menjadi satu vektor
sebelum masuk ke satu model tabular (XGBoost), lalu dilatih bersama.

Parametrik pada *modalitas teks* agar satu skrip melayani seluruh matriks ablasi:
  --text-group basic      -> FEAT_LEXICON (11) + tfidf_svd_* (50)   [Model B]
  --text-group embedding  -> kolom emb_* dari --embedding-csv        [Model C]
  --text-group both       -> lexicon + tfidf_svd + emb_*             [Model C+]
  --text-group none       -> tanpa teks (ekuivalen TS-only, uji sanity)

Dengan begitu Model B dan Model C dilatih oleh kode dan hiperparameter yang
sama persis; satu-satunya yang berubah adalah himpunan fitur teks — sehingga
perbandingan ablasi benar-benar apple-to-apple.

Target ganda identik pipeline tim: `target` 3-class + `kurs_besok` (via
`delta_besok`). Evaluasi via src/models/metrics.py. Naive same-rows via
src/models/naive_baseline.py.

CLI:
  python -m src.models.fusion_early                         # Model B (basic)
  python -m src.models.fusion_early --text-group embedding \
      --embedding-csv data/feature_extracted/10_finbert_daily_features.csv \
      --output-prefix 10_fusion_early_finbert                 # Model C
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

# Blok fitur per modalitas (sama persis dengan 04/05/06 agar ablasi murni).
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
    return sorted([c for c in df.columns if c.startswith("tfidf_svd_")],
                  key=lambda v: int(v.split("_")[-1]))


def _embedding_cols(df: pd.DataFrame) -> list[str]:
    """Kolom embedding yang di-merge dari --embedding-csv (prefix emb_)."""
    return sorted([c for c in df.columns if c.startswith("emb_")],
                  key=lambda v: int(v.split("_")[-1]) if v.split("_")[-1].isdigit() else 0)


def load_dataset(embedding_csv: str | None) -> pd.DataFrame:
    """Muat model_dataset.csv, opsional merge embedding harian pada `date`.

    File embedding diharapkan punya kolom `date` + kolom fitur. Kolom fitur
    (selain date/news_date) di-rename ke prefix `emb_1..emb_N` agar seragam
    lintas sumber embedding (word2vec/fasttext/glove/finbert/...).
    """
    df = pd.read_csv(FEATURE_DIR / "model_dataset.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    if embedding_csv:
        emb = pd.read_csv(embedding_csv)
        if "date" not in emb.columns:
            raise ValueError(f"{embedding_csv} wajib punya kolom `date`.")
        emb["date"] = pd.to_datetime(emb["date"], errors="coerce")
        feat_cols = [c for c in emb.columns if c not in ("date", "news_date", "split")]
        emb = emb[["date"] + feat_cols].rename(
            columns={c: f"emb_{i + 1}" for i, c in enumerate(feat_cols)}
        )
        before = len(df)
        df = df.merge(emb, on="date", how="left")
        n_missing = int(df[[c for c in df.columns if c.startswith("emb_")]]
                        .iloc[:, 0].isna().sum()) if feat_cols else 0
        print(f"[fusion_early] merge embedding: {len(feat_cols)} dim, "
              f"{before} baris, {n_missing} hari tanpa embedding (diisi 0).")
        emb_now = [c for c in df.columns if c.startswith("emb_")]
        df[emb_now] = df[emb_now].fillna(0.0)
    return df


def build_feature_list(df: pd.DataFrame, text_group: str) -> list[str]:
    text = {
        "none": [],
        "basic": FEAT_LEXICON + _tfidf_cols(df),
        "embedding": _embedding_cols(df),
        "both": FEAT_LEXICON + _tfidf_cols(df) + _embedding_cols(df),
    }[text_group]
    if text_group in ("embedding", "both") and not _embedding_cols(df):
        raise ValueError(
            "text-group butuh kolom emb_* tetapi --embedding-csv tidak diberikan "
            "atau kosong. Sertakan file embedding harian."
        )
    return FEAT_TS + text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Early fusion TS+teks (satu XGBoost).")
    p.add_argument("--text-group", default="basic",
                   choices=["none", "basic", "embedding", "both"],
                   help="Modalitas teks yang digabung (default: basic = Model B).")
    p.add_argument("--embedding-csv", default=None,
                   help="CSV embedding harian (kolom date + fitur) untuk Model C.")
    p.add_argument("--output-prefix", default="10_fusion_early",
                   help="Prefix file output di results/ (default: 10_fusion_early).")
    args = p.parse_args(argv)

    df = load_dataset(args.embedding_csv)

    if "target" not in df.columns:
        print("PERINGATAN: kolom `target` dihitung ulang dari kurs_besok.")
        pct_besok = (df["kurs_besok"] / df["kurs_jisdor"] - 1) * 100
        df["target"] = pct_besok.apply(lambda v: label_direction(v, THRESHOLD_PCT))
    df["delta_besok"] = df["kurs_besok"] - df["kurs_jisdor"]

    feats = build_feature_list(df, args.text_group)
    missing = [c for c in feats + ["kurs_jisdor", "kurs_besok", "target", "split"]
               if c not in df.columns]
    if missing:
        print(f"[fusion_early] ERROR: kolom hilang: {missing}", file=sys.stderr)
        return 1
    print(f"[fusion_early] text-group={args.text_group} | {len(feats)} fitur "
          f"(TS={len(FEAT_TS)}, teks={len(feats) - len(FEAT_TS)})")

    bounds = df.groupby("split")["date"].agg(["min", "max", "size"])
    print(bounds.to_string())
    assert bounds.loc["train", "max"] < bounds.loc["val", "min"] < bounds.loc["test", "min"], \
        "Split tidak kronologis"

    splits = {s: df[df["split"] == s].sort_values("date").copy()
              for s in ["train", "val", "test"]}
    tr = splits["train"]
    X = {s: d[feats] for s, d in splits.items()}
    y_clf = {s: d["target"].map(LABEL2ID).astype(int) for s, d in splits.items()}
    y_reg = {s: d["kurs_besok"].astype(float) for s, d in splits.items()}
    y_delta = {s: d["delta_besok"].astype(float) for s, d in splits.items()}
    kurs_today = {s: d["kurs_jisdor"].astype(float).to_numpy() for s, d in splits.items()}

    all_results: list[dict] = []

    def add_result(model, split, task, metrics):
        all_results.append({"model": model, "split": split, "task": task, **{
            k: v for k, v in metrics.items()
            if k not in ("classification_report", "confusion_matrix")}})

    # 1) Naive same-rows (acuan di setiap tabel)
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

    # 2) Early fusion — klasifikasi
    w_train = compute_sample_weight(class_weight="balanced", y=y_clf["train"])
    clf = make_xgb_classifier()
    clf.fit(X["train"], y_clf["train"], sample_weight=w_train,
            eval_set=[(X["val"], y_clf["val"])], verbose=False)
    print("Best iteration (clf):", clf.best_iteration)
    clf_metrics = {}
    pred_clf = {}
    for s in ["val", "test"]:
        pred = pd.Series(clf.predict(X[s])).map(ID2LABEL).to_numpy()
        pred_clf[s] = pred
        m = evaluate_classifier(splits[s]["target"].to_numpy(), pred, include_kappa=True)
        clf_metrics[s] = m
        add_result(f"Early Fusion ({args.text_group})", s, "clf", m)
        print(f"[early clf {s}] acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"kappa={m.get('kappa')}")

    # 3) Early fusion — regresi via delta
    reg = make_xgb_regressor()
    reg.fit(X["train"], y_delta["train"],
            eval_set=[(X["val"], y_delta["val"])], verbose=False)
    print("Best iteration (reg):", reg.best_iteration)
    reg_metrics = {}
    pred_reg = {}
    for s in ["val", "test"]:
        pred_reg[s] = kurs_today[s] + reg.predict(X[s])
        m = evaluate_regression(y_reg[s], pred_reg[s], rmse_naive=rmse_naive[s],
                                include_mape=True)
        m["directional_hit"] = directional_hit_from_regression(
            kurs_today[s], y_reg[s], pred_reg[s])
        reg_metrics[s] = m
        add_result(f"Early Fusion ({args.text_group})", s, "reg", m)
        print(f"[early reg {s}] RMSE={m['rmse']:.2f} skill={m['skill_vs_naive']:+.4f} "
              f"hit={m['directional_hit']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    pd.DataFrame(all_results).round(6).to_csv(
        RESULTS_DIR / f"{prefix}_comparison.csv", index=False)

    # Prediksi per-tanggal (val+test) untuk analisis error & korelasi
    pred_frames = pd.concat([splits["val"], splits["test"]])
    save_predictions(
        RESULTS_DIR / f"{prefix}_predictions.csv",
        model=f"Early Fusion ({args.text_group})",
        date=pred_frames["date"],
        split=pred_frames["split"],
        y_true_clf=np.concatenate([splits["val"]["target"].to_numpy(),
                                   splits["test"]["target"].to_numpy()]),
        y_pred_clf=np.concatenate([pred_clf["val"], pred_clf["test"]]),
        kurs_today=np.concatenate([kurs_today["val"], kurs_today["test"]]),
        y_true_reg=np.concatenate([y_reg["val"].to_numpy(), y_reg["test"].to_numpy()]),
        y_pred_reg=np.concatenate([pred_reg["val"], pred_reg["test"]]),
    )

    out = {
        "model": f"Early Fusion ({args.text_group})",
        "fusion": "early (concat fitur -> satu XGBoost)",
        "features": {"ts": FEAT_TS, "text_group": args.text_group,
                     "n_features": len(feats),
                     "embedding_csv": args.embedding_csv},
        "best_iteration": {"clf": int(clf.best_iteration), "reg": int(reg.best_iteration)},
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
    ax.set_title(f"Early Fusion ({args.text_group}): Confusion Matrix (Test)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"[fusion_early] Tersimpan: {prefix}_comparison.csv, {prefix}_results.json, "
          f"{prefix}_predictions.csv, {prefix}_confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
