"""Integrasi hasil (Tugas 3): matriks ablasi Tugas 2 vs Tugas 3 dalam satu tabel.

Mengumpulkan metrik test dari seluruh file ``results/*_results.json`` menjadi satu
tabel perbandingan apple-to-apple (macro-F1 arah + RMSE regresi + skill vs naive),
diurutkan dari generasi Tugas 2 (baseline) ke Tugas 3 (lanjutan). Membuktikan
secara terukur apakah representasi transformer / feature-engineering meningkatkan
akurasi arah atau menurunkan error - sesuai butir "Matriks Ablasi & Diskusi".

Output:
  results/ablation_matrix_tugas2_vs_tugas3.csv
  results/ablation_matrix_tugas2_vs_tugas3.png

CLI:
  python -m src.models.build_ablation_matrix
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = _REPO_ROOT / "results"

# Baris matriks: (label, generasi, sumber). Sumber bisa file JSON standalone
# atau (file ablasi, nama varian di dalamnya).
ROWS = [
    ("Naive Persistence of Direction", "Baseline", ("05_ts_only_results.json", "__naive_clf__")),
    ("Model A - TS-only (4 fitur)", "Tugas 2", "05_ts_only_results.json"),
    ("Model B - TS + NLP dasar", "Tugas 2", "06_nlp_only_results.json"),
    ("Gabungan TS+NLP (early, awal)", "Tugas 2", "04_gabungan_results.json"),
    ("Model A+ - TS diperkaya + tuned", "Tugas 3", "05b_ts_tuned_results.json"),
    ("Model C - TS + Transformer", "Tugas 3", ("04b_ablasi_results.json", "Model C: TS + Transformer")),
    ("Model C+ - TS + NLP + Transformer", "Tugas 3", ("04b_ablasi_results.json", "Model C+: TS + NLP Dasar + Transformer")),
    ("Fusion Early (TS+NLP)", "Tugas 3", "10_fusion_early_basic_results.json"),
    ("Fusion Late (TS+NLP)", "Tugas 3", "11_fusion_late_basic_results.json"),
]


def _load(name: str) -> dict:
    p = RESULTS_DIR / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _metrics_from(source, split="test") -> dict:
    """Ambil metrik klasifikasi+regresi test dari sumber (file atau varian ablasi)."""
    if isinstance(source, tuple):
        fname, variant = source
        d = _load(fname)
        if variant == "__naive_clf__":
            # Naive persistence: dari comparison.csv agar seragam
            comp = RESULTS_DIR / "05_ts_only_comparison.csv"
            if comp.exists():
                c = pd.read_csv(comp)
                row = c[(c["model"] == "Naive Persistence of Direction") &
                        (c["split"] == split) & (c["task"] == "clf")]
                rw = c[(c["model"] == "Naive Persistence (Random Walk)") &
                       (c["split"] == split) & (c["task"] == "reg")]
                return {
                    "macro_f1": float(row["macro_f1"].iloc[0]) if len(row) else np.nan,
                    "rmse": float(rw["rmse"].iloc[0]) if len(rw) else np.nan,
                    "skill_vs_naive": 0.0,
                    "directional_hit": float(rw["directional_hit"].iloc[0]) if len(rw) and "directional_hit" in rw else np.nan,
                }
            return {}
        d = d.get("variants", {}).get(variant, {})
    else:
        d = _load(source)
    clf = d.get("classification", {}).get(split, {})
    reg = d.get("regression", {}).get(split, {})
    return {
        "macro_f1": clf.get("macro_f1"),
        "rmse": reg.get("rmse"),
        "skill_vs_naive": reg.get("skill_vs_naive"),
        "directional_hit": reg.get("directional_hit"),
    }


def main(argv=None) -> int:
    rows = []
    for label, gen, source in ROWS:
        m = _metrics_from(source)
        rows.append({"model": label, "generasi": gen, **m})
    df = pd.DataFrame(rows)

    # Acuan naive untuk penanda "mengalahkan baseline"
    naive_f1 = df.loc[df["model"].str.startswith("Naive"), "macro_f1"].iloc[0]
    df["kalahkan_naive_clf"] = df["macro_f1"] > naive_f1
    df["kalahkan_naive_reg"] = df["skill_vs_naive"].fillna(-1) > 0

    out_csv = RESULTS_DIR / "ablation_matrix_tugas2_vs_tugas3.csv"
    df.round(4).to_csv(out_csv, index=False)

    # Tabel teks ke stdout
    print("\n=== MATRIKS ABLASI: Tugas 2 vs Tugas 3 (split TEST) ===")
    show = df.copy()
    show["macro_f1"] = show["macro_f1"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    show["rmse"] = show["rmse"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    show["skill_vs_naive"] = show["skill_vs_naive"].map(lambda v: f"{v:+.4f}" if pd.notna(v) else "-")
    show["directional_hit"] = show["directional_hit"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    print(show[["generasi", "model", "macro_f1", "rmse", "skill_vs_naive", "directional_hit"]].to_string(index=False))

    # Visual: bar macro-F1 + garis naive
    plot = df[~df["model"].str.startswith("Naive")].copy()
    colors = {"Tugas 2": "#94a3b8", "Tugas 3": "#2563eb"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    bar_colors = [colors.get(g, "#cbd5e1") for g in plot["generasi"]]
    ax[0].barh(plot["model"], plot["macro_f1"], color=bar_colors)
    ax[0].axvline(naive_f1, color="red", ls="--", lw=1.2, label=f"Naive persistence ({naive_f1:.3f})")
    ax[0].set_title("Macro-F1 arah (Test) - makin kanan makin baik")
    ax[0].legend(); ax[0].invert_yaxis()
    ax[1].barh(plot["model"], plot["skill_vs_naive"].fillna(0), color=bar_colors)
    ax[1].axvline(0, color="red", ls="--", lw=1.2, label="Random walk (skill=0)")
    ax[1].set_title("Skill regresi vs naive (Test) - >0 mengalahkan random walk")
    ax[1].legend(); ax[1].invert_yaxis()
    plt.tight_layout()
    out_png = RESULTS_DIR / "ablation_matrix_tugas2_vs_tugas3.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"\n[matrix] Tersimpan: {out_csv.name}, {out_png.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
