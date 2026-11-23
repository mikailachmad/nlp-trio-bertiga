"""Analisis error & korelasi lintas model (butir 3c.iii).

Membandingkan prediksi per-tanggal dua model pada split yang sama untuk
menjawab: *pada tanggal mana fitur NLP membantu menghindari sinyal palsu
pasar, dan pada tanggal mana teks justru menambah noise?*

Input: dua file ``*_predictions.csv`` hasil save_predictions() —
  - baseline : biasanya Model A (XGBoost TS-only), tanpa NLP.
  - candidate: model ber-NLP (early/late fusion basic, atau Model C nanti).

Keduanya dipadankan pada kolom `date` (split yang dipilih). Analisis:

1. Klasifikasi — kontingensi arah per tanggal:
     NLP MENOLONG : baseline salah, candidate benar.
     NLP MERUSAK  : baseline benar, candidate salah (teks menambah noise).
     + uji McNemar (apakah selisih menolong/merusak signifikan).
2. Regresi — directional hit & |error| per tanggal; korelasi antar-error
   dua model (seberapa independen error-nya), dan selisih |error|.
3. Korelasi sinyal NLP vs arah aktual: fitur leksikon harian (VADER / LM)
   dari model_dataset.csv dikorelasikan (point-biserial/Pearson) dengan
   arah aktual (naik/turun) pada tanggal yang sama.

Output (results/): tabel tanggal menolong & merusak, ringkasan JSON, dan
tiga plot (timeline error, distribusi |error|, diagram kontingensi).

CLI:
  python -m src.models.error_correlation \
      --baseline results/05_ts_only_predictions.csv \
      --candidate results/10_fusion_early_basic_predictions.csv \
      --output-prefix 12_errcorr_early_basic
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

FEATURE_DIR = _REPO_ROOT / "data" / "feature_extracted"
RESULTS_DIR = _REPO_ROOT / "results"

# Kolom sinyal NLP (leksikon harian) untuk korelasi vs arah aktual.
NLP_SIGNAL_COLS = [
    "vader_title_mean", "vader_body_mean", "vader_neg_share",
    "lm_negative_mean", "lm_positive_mean", "lm_net_mean",
]


def _mcnemar(help_n: int, hurt_n: int) -> dict:
    """Uji McNemar (binomial eksak) pada sel discordant menolong vs merusak.

    H0: P(menolong) = P(merusak). Hanya pakai pasangan discordant (b, c).
    Return p-value dua-sisi (binomial, p=0.5) + arah efek.
    """
    from math import comb

    n = help_n + hurt_n
    if n == 0:
        return {"p_value": 1.0, "n_discordant": 0, "note": "tak ada pasangan discordant"}
    k = min(help_n, hurt_n)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2 * tail)
    return {"p_value": round(float(p), 4), "n_discordant": int(n),
            "favored": "candidate (NLP menolong)" if help_n > hurt_n
            else ("baseline (NLP merusak)" if hurt_n > help_n else "seimbang")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Analisis error & korelasi lintas model.")
    p.add_argument("--baseline", default="results/05_ts_only_predictions.csv",
                   help="CSV prediksi model baseline (default: Model A TS-only).")
    p.add_argument("--candidate", default="results/10_fusion_early_basic_predictions.csv",
                   help="CSV prediksi model ber-NLP yang dibandingkan.")
    p.add_argument("--split", default="test", choices=["test", "val", "all"],
                   help="Split yang dianalisis (default: test).")
    p.add_argument("--output-prefix", default="12_errcorr",
                   help="Prefix output di results/.")
    args = p.parse_args(argv)

    base = pd.read_csv(_REPO_ROOT / args.baseline)
    cand = pd.read_csv(_REPO_ROOT / args.candidate)
    if args.split != "all":
        base = base[base["split"] == args.split]
        cand = cand[cand["split"] == args.split]
    base_name = base["model"].iloc[0] if len(base) else "baseline"
    cand_name = cand["model"].iloc[0] if len(cand) else "candidate"

    key = ["date"]
    m = base.merge(cand, on=key, suffixes=("_base", "_cand"))
    if m.empty:
        print("[errcorr] ERROR: tidak ada tanggal beririsan antar dua file.",
              file=sys.stderr)
        return 1
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date").reset_index(drop=True)
    print(f"[errcorr] {base_name}  vs  {cand_name}  | split={args.split} | "
          f"{len(m)} tanggal beririsan")

    # ---- 1) Klasifikasi: tanggal NLP menolong vs merusak ----
    helped = m[(m["correct_clf_base"] == 0) & (m["correct_clf_cand"] == 1)]
    hurt = m[(m["correct_clf_base"] == 1) & (m["correct_clf_cand"] == 0)]
    both_ok = int(((m["correct_clf_base"] == 1) & (m["correct_clf_cand"] == 1)).sum())
    both_no = int(((m["correct_clf_base"] == 0) & (m["correct_clf_cand"] == 0)).sum())
    mcnemar = _mcnemar(len(helped), len(hurt))
    print(f"[errcorr-clf] NLP menolong={len(helped)}  merusak={len(hurt)}  "
          f"sama-benar={both_ok}  sama-salah={both_no}  "
          f"McNemar p={mcnemar['p_value']} ({mcnemar['favored']})")

    cols_out = ["date", "y_true_clf_base", "y_pred_clf_base", "y_pred_clf_cand",
                "err_reg_base", "err_reg_cand"]
    helped[cols_out].to_csv(RESULTS_DIR / f"{args.output_prefix}_dates_nlp_helped.csv",
                            index=False)
    hurt[cols_out].to_csv(RESULTS_DIR / f"{args.output_prefix}_dates_nlp_hurt.csv",
                          index=False)

    # ---- 2) Regresi: korelasi error & directional hit ----
    abs_base = m["err_reg_base"].abs()
    abs_cand = m["err_reg_cand"].abs()
    err_corr = float(np.corrcoef(m["err_reg_base"], m["err_reg_cand"])[0, 1])
    reg = {
        "mae_base": round(float(abs_base.mean()), 4),
        "mae_cand": round(float(abs_cand.mean()), 4),
        "dir_hit_base": round(float(m["dir_hit_base"].mean()), 4),
        "dir_hit_cand": round(float(m["dir_hit_cand"].mean()), 4),
        "error_correlation": round(err_corr, 4),
        "dates_cand_closer": int((abs_cand < abs_base).sum()),
        "dates_base_closer": int((abs_base < abs_cand).sum()),
    }
    print(f"[errcorr-reg] MAE base={reg['mae_base']} cand={reg['mae_cand']}  "
          f"dirhit base={reg['dir_hit_base']} cand={reg['dir_hit_cand']}  "
          f"corr(err)={reg['error_correlation']}")

    # ---- 3) Korelasi sinyal NLP vs arah aktual ----
    nlp_corr: dict[str, float] = {}
    ds = pd.read_csv(FEATURE_DIR / "model_dataset.csv")
    ds["date"] = pd.to_datetime(ds["date"], errors="coerce")
    avail = [c for c in NLP_SIGNAL_COLS if c in ds.columns]
    if avail:
        j = m[["date", "dir_true_base"]].merge(
            ds[["date"] + avail], on="date", how="left")
        for c in avail:
            x = j[c].to_numpy(dtype=float)
            y = j["dir_true_base"].to_numpy(dtype=float)  # arah aktual (+1/-1/0)
            ok = ~np.isnan(x)
            if ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                nlp_corr[c] = round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 4)
        print(f"[errcorr-nlp] korelasi sinyal NLP vs arah aktual: {nlp_corr}")

    # ---- Ringkasan JSON ----
    summary = {
        "baseline": base_name,
        "candidate": cand_name,
        "split": args.split,
        "n_dates": int(len(m)),
        "classification": {
            "nlp_helped": int(len(helped)),
            "nlp_hurt": int(len(hurt)),
            "both_correct": both_ok,
            "both_wrong": both_no,
            "net_help": int(len(helped) - len(hurt)),
            "mcnemar": mcnemar,
        },
        "regression": reg,
        "nlp_signal_vs_direction_corr": nlp_corr,
    }
    with open(RESULTS_DIR / f"{args.output_prefix}_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- Plot 1: timeline |error| dua model, tandai help/hurt ----
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(m["date"], abs_base, label=f"|error| {base_name}", lw=1.2, alpha=0.8)
    ax.plot(m["date"], abs_cand, label=f"|error| {cand_name}", lw=1.2, alpha=0.8)
    ax.scatter(helped["date"], [0] * len(helped), marker="^", color="green",
               label="NLP menolong (clf)", zorder=5)
    ax.scatter(hurt["date"], [0] * len(hurt), marker="v", color="red",
               label="NLP merusak (clf)", zorder=5)
    ax.set_title(f"Error timeline & kontribusi NLP ({args.split})")
    ax.set_ylabel("|error| kurs (rupiah)")
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{args.output_prefix}_error_timeline.png", dpi=150)
    plt.close()

    # ---- Plot 2: distribusi |error| ----
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bins = np.linspace(0, float(max(abs_base.max(), abs_cand.max())), 25)
    ax.hist(abs_base, bins=bins, alpha=0.6, label=base_name)
    ax.hist(abs_cand, bins=bins, alpha=0.6, label=cand_name)
    ax.set_xlabel("|error| kurs (rupiah)")
    ax.set_ylabel("Jumlah hari")
    ax.set_title(f"Distribusi error ({args.split})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{args.output_prefix}_error_distribution.png", dpi=150)
    plt.close()

    # ---- Plot 3: kontingensi klasifikasi ----
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    cats = ["NLP\nmenolong", "NLP\nmerusak", "sama\nbenar", "sama\nsalah"]
    vals = [len(helped), len(hurt), both_ok, both_no]
    colors = ["green", "red", "steelblue", "grey"]
    ax.bar(cats, vals, color=colors, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("Jumlah hari")
    ax.set_title(f"Kontribusi NLP per tanggal (clf, {args.split})\n"
                 f"McNemar p={mcnemar['p_value']}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{args.output_prefix}_contingency.png", dpi=150)
    plt.close()

    print(f"[errcorr] Tersimpan prefix '{args.output_prefix}': "
          "dates_nlp_helped.csv, dates_nlp_hurt.csv, summary.json, "
          "error_timeline.png, error_distribution.png, contingency.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
