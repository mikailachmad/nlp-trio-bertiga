"""Late fusion multimodal (butir 2a.ii): sub-model unimodal terpisah lalu gabung.

Berbeda dari early fusion (concat fitur), di sini tiap modalitas dilatih oleh
model terpisah, lalu *output*-nya yang digabung:

  Klasifikasi : P_gabung = w * P_ts + (1-w) * P_teks   (rata-rata probabilitas)
  Regresi     : d_gabung = w * d_ts + (1-w) * d_teks   (rata-rata selisih kurs)

Bobot ``w`` dipilih pada split VAL (grid 0..1) — klasifikasi memaksimalkan
Macro-F1, regresi meminimalkan RMSE — lalu dikunci dan dievaluasi sekali pada
TEST (kronologis ketat, Test tidak dipakai memilih w).

Modalitas teks parametrik, sama seperti fusion_early:
  --text-group basic      -> lexicon + tfidf_svd   [Model B late]
  --text-group embedding  -> emb_* (--embedding-csv) [Model C late]
  --text-group both       -> lexicon + tfidf_svd + emb_*

Sub-model TS, sub-model teks, dan hasil gabungan ketiganya dicatat agar
kontribusi tiap modalitas terlihat. Evaluasi via src/models/metrics.py.

CLI:
  python -m src.models.fusion_late --output-prefix 11_fusion_late_basic
  python -m src.models.fusion_late --text-group embedding \
      --embedding-csv data/feature_extracted/10_finbert_daily_features.csv \
      --output-prefix 11_fusion_late_finbert
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
from sklearn.metrics import f1_score
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
from src.models.fusion_early import (  # noqa: E402
    FEAT_TS,
    LABEL2ID,
    ID2LABEL,
    build_feature_list,
    load_dataset,
    make_xgb_classifier,
    make_xgb_regressor,
)

RESULTS_DIR = _REPO_ROOT / "results"


def _text_only_features(df: pd.DataFrame, text_group: str) -> list[str]:
    """Fitur teks saja (buang blok TS dari daftar gabungan)."""
    return [c for c in build_feature_list(df, text_group) if c not in FEAT_TS]


def _pick_weight_clf(p_ts, p_txt, y_true_val) -> tuple[float, float]:
    """Pilih w yang memaksimalkan Macro-F1 pada val (grid 0..1 step 0.05)."""
    best_w, best_f1 = 1.0, -1.0
    for w in np.linspace(0.0, 1.0, 21):
        p = w * p_ts + (1 - w) * p_txt
        pred = pd.Series(p.argmax(axis=1)).map(ID2LABEL).to_numpy()
        f1 = f1_score(y_true_val, pred, average="macro", labels=LABEL_ORDER)
        if f1 > best_f1:
            best_f1, best_w = f1, float(w)
    return best_w, round(best_f1, 4)


def _pick_weight_reg(d_ts, d_txt, y_true_val, kurs_val) -> tuple[float, float]:
    """Pilih w yang meminimalkan RMSE pada val (grid 0..1 step 0.05)."""
    best_w, best_rmse = 1.0, np.inf
    for w in np.linspace(0.0, 1.0, 21):
        pred = kurs_val + (w * d_ts + (1 - w) * d_txt)
        rmse = float(np.sqrt(np.mean((y_true_val - pred) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, float(w)
    return best_w, round(best_rmse, 4)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Late fusion TS+teks (gabung output).")
    p.add_argument("--text-group", default="basic",
                   choices=["basic", "embedding", "both"],
                   help="Modalitas teks sub-model (default: basic).")
    p.add_argument("--embedding-csv", default=None,
                   help="CSV embedding harian (kolom date + fitur) untuk Model C.")
    p.add_argument("--output-prefix", default="11_fusion_late")
    args = p.parse_args(argv)

    df = load_dataset(args.embedding_csv)
    if "target" not in df.columns:
        pct_besok = (df["kurs_besok"] / df["kurs_jisdor"] - 1) * 100
        df["target"] = pct_besok.apply(lambda v: label_direction(v, THRESHOLD_PCT))
    df["delta_besok"] = df["kurs_besok"] - df["kurs_jisdor"]

    feat_text = _text_only_features(df, args.text_group)
    missing = [c for c in FEAT_TS + feat_text +
               ["kurs_jisdor", "kurs_besok", "target", "split"] if c not in df.columns]
    if missing:
        print(f"[fusion_late] ERROR: kolom hilang: {missing}", file=sys.stderr)
        return 1
    print(f"[fusion_late] text-group={args.text_group} | "
          f"TS={len(FEAT_TS)} fitur, teks={len(feat_text)} fitur")

    bounds = df.groupby("split")["date"].agg(["min", "max", "size"])
    assert bounds.loc["train", "max"] < bounds.loc["val", "min"] < bounds.loc["test", "min"], \
        "Split tidak kronologis"

    splits = {s: df[df["split"] == s].sort_values("date").copy()
              for s in ["train", "val", "test"]}
    tr = splits["train"]
    Xts = {s: d[FEAT_TS] for s, d in splits.items()}
    Xtx = {s: d[feat_text] for s, d in splits.items()}
    y_clf = {s: d["target"].map(LABEL2ID).astype(int) for s, d in splits.items()}
    y_reg = {s: d["kurs_besok"].astype(float).to_numpy() for s, d in splits.items()}
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
        add_result("Naive Persistence of Direction", s, "clf",
                   evaluate_classifier(d["target"].to_numpy(),
                                       predict_persistence(d["target"]), include_kappa=True))
        add_result("Naive Majority-Class", s, "clf",
                   evaluate_classifier(d["target"].to_numpy(),
                                       predict_majority(tr["target"], len(d)), include_kappa=True))
        r_r = evaluate_regression(y_reg[s], kurs_today[s], include_mape=True)
        rmse_naive[s] = r_r["rmse"]
        r_r["skill_vs_naive"] = 0.0
        add_result("Naive Persistence (Random Walk)", s, "reg", r_r)

    # ---- Sub-model KLASIFIKASI (TS & teks terpisah) -> probabilitas ----
    w_train = compute_sample_weight(class_weight="balanced", y=y_clf["train"])
    clf_ts = make_xgb_classifier()
    clf_ts.fit(Xts["train"], y_clf["train"], sample_weight=w_train,
               eval_set=[(Xts["val"], y_clf["val"])], verbose=False)
    clf_tx = make_xgb_classifier()
    clf_tx.fit(Xtx["train"], y_clf["train"], sample_weight=w_train,
               eval_set=[(Xtx["val"], y_clf["val"])], verbose=False)
    proba_ts = {s: clf_ts.predict_proba(Xts[s]) for s in ["val", "test"]}
    proba_tx = {s: clf_tx.predict_proba(Xtx[s]) for s in ["val", "test"]}

    # Sub-model individual (diagnostik kontribusi modalitas)
    for name, proba in (("Late sub-TS", proba_ts), ("Late sub-teks", proba_tx)):
        for s in ["val", "test"]:
            pred = pd.Series(proba[s].argmax(axis=1)).map(ID2LABEL).to_numpy()
            add_result(name, s, "clf",
                       evaluate_classifier(splits[s]["target"].to_numpy(), pred, include_kappa=True))

    w_clf, val_f1 = _pick_weight_clf(proba_ts["val"], proba_tx["val"],
                                     splits["val"]["target"].to_numpy())
    print(f"[late clf] w*(TS)={w_clf:.2f} -> val macroF1={val_f1:.4f}")
    clf_metrics = {}
    pred_clf = {}
    for s in ["val", "test"]:
        p = w_clf * proba_ts[s] + (1 - w_clf) * proba_tx[s]
        pred = pd.Series(p.argmax(axis=1)).map(ID2LABEL).to_numpy()
        pred_clf[s] = pred
        m = evaluate_classifier(splits[s]["target"].to_numpy(), pred, include_kappa=True)
        clf_metrics[s] = m
        add_result(f"Late Fusion ({args.text_group})", s, "clf", m)
        print(f"[late clf {s}] acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"kappa={m.get('kappa')}")

    # ---- Sub-model REGRESI (TS & teks terpisah) -> delta ----
    reg_ts = make_xgb_regressor()
    reg_ts.fit(Xts["train"], y_delta["train"],
               eval_set=[(Xts["val"], y_delta["val"])], verbose=False)
    reg_tx = make_xgb_regressor()
    reg_tx.fit(Xtx["train"], y_delta["train"],
               eval_set=[(Xtx["val"], y_delta["val"])], verbose=False)
    delta_ts = {s: reg_ts.predict(Xts[s]) for s in ["val", "test"]}
    delta_tx = {s: reg_tx.predict(Xtx[s]) for s in ["val", "test"]}

    for name, delta in (("Late sub-TS", delta_ts), ("Late sub-teks", delta_tx)):
        for s in ["val", "test"]:
            pred = kurs_today[s] + delta[s]
            m = evaluate_regression(y_reg[s], pred, rmse_naive=rmse_naive[s], include_mape=True)
            m["directional_hit"] = directional_hit_from_regression(kurs_today[s], y_reg[s], pred)
            add_result(name, s, "reg", m)

    w_reg, val_rmse = _pick_weight_reg(delta_ts["val"], delta_tx["val"],
                                       y_reg["val"], kurs_today["val"])
    print(f"[late reg] w*(TS)={w_reg:.2f} -> val RMSE={val_rmse:.2f}")
    reg_metrics = {}
    pred_reg = {}
    for s in ["val", "test"]:
        pred = kurs_today[s] + (w_reg * delta_ts[s] + (1 - w_reg) * delta_tx[s])
        pred_reg[s] = pred
        m = evaluate_regression(y_reg[s], pred, rmse_naive=rmse_naive[s], include_mape=True)
        m["directional_hit"] = directional_hit_from_regression(kurs_today[s], y_reg[s], pred)
        reg_metrics[s] = m
        add_result(f"Late Fusion ({args.text_group})", s, "reg", m)
        print(f"[late reg {s}] RMSE={m['rmse']:.2f} skill={m['skill_vs_naive']:+.4f} "
              f"hit={m['directional_hit']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    pd.DataFrame(all_results).round(6).to_csv(
        RESULTS_DIR / f"{prefix}_comparison.csv", index=False)

    # Prediksi per-tanggal (val+test) untuk analisis error & korelasi
    pred_frames = pd.concat([splits["val"], splits["test"]])
    save_predictions(
        RESULTS_DIR / f"{prefix}_predictions.csv",
        model=f"Late Fusion ({args.text_group})",
        date=pred_frames["date"],
        split=pred_frames["split"],
        y_true_clf=np.concatenate([splits["val"]["target"].to_numpy(),
                                   splits["test"]["target"].to_numpy()]),
        y_pred_clf=np.concatenate([pred_clf["val"], pred_clf["test"]]),
        kurs_today=np.concatenate([kurs_today["val"], kurs_today["test"]]),
        y_true_reg=np.concatenate([y_reg["val"], y_reg["test"]]),
        y_pred_reg=np.concatenate([pred_reg["val"], pred_reg["test"]]),
    )

    out = {
        "model": f"Late Fusion ({args.text_group})",
        "fusion": "late (sub-model unimodal -> gabung probabilitas/delta)",
        "weights": {"clf_w_ts": w_clf, "reg_w_ts": w_reg},
        "features": {"ts": FEAT_TS, "text_group": args.text_group,
                     "n_text_features": len(feat_text),
                     "embedding_csv": args.embedding_csv},
        "classification": clf_metrics,
        "regression": reg_metrics,
        "rmse_naive_same_rows": rmse_naive,
    }
    with open(RESULTS_DIR / f"{prefix}_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    cm = np.array(clf_metrics["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticks(range(3)); ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title(f"Late Fusion ({args.text_group}): Confusion Matrix (Test)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"[fusion_late] Tersimpan: {prefix}_comparison.csv, {prefix}_results.json, "
          f"{prefix}_confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
