"""Run the complete NLP + JISDOR modeling pipeline in order.

Pipeline:
    cleaning -> temporal alignment -> chronological split -> feature extraction -> modeling

Run from anywhere inside/outside the repository:
    python run_pipeline.py

The script executes the existing Jupyter notebooks in-memory, so the notebooks
are not overwritten with execution outputs. All artifacts produced by those
notebooks are written to the repository's data/ and results/ directories as
usual.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


PIPELINE = [
    ("Cleaning & preprocessing", Path("src/cleaning/Cleaning_article.ipynb")),
    ("Temporal alignment JISDOR", Path("src/01_align_merge.ipynb")),
    ("Chronological split + EDA", Path("src/02_split_data.ipynb")),
    ("Feature extraction", Path("src/03_feature_extraction.ipynb")),
    ("Modeling", Path("src/04_modeling.ipynb")),
]

# Files that must exist before the pipeline can start.
REQUIRED_INPUTS = [
    Path("data/raw/news/cnbc_articles_final.csv"),
    Path("data/raw/Informasi Kurs Jisdor.xlsx"),
    Path("data/raw/Loughran-McDonald_MasterDictionary_1993-2025.csv"),
]

# Generated files/directories only. Raw source data are never deleted.
GENERATED_PATHS = [
    Path("data/processed/cnbc_articles_clean.csv"),
    Path("data/processed/articles_aligned.csv"),
    Path("data/processed/daily_kurs.csv"),
    Path("data/processed/jisdor_target.csv"),
    Path("data/split/articles_split.csv"),
    Path("data/split/daily_kurs_split.csv"),
    Path("data/split/split_info.json"),
    Path("data/feature_extracted/features_lexicon.csv"),
    Path("data/feature_extracted/tfidf_X.npz"),
    Path("data/feature_extracted/tfidf_index.csv"),
    Path("data/feature_extracted/tfidf_vocab.csv"),
    Path("data/feature_extracted/model_dataset.csv"),
    Path("results/04_gabungan_comparison.csv"),
    Path("results/04_gabungan_results.json"),
    Path("results/04_gabungan_confusion_matrix.png"),
    Path("results/04_gabungan_predicted_vs_actual.png"),
    Path("results/04_gabungan_feature_importance.png"),
    Path("results/04_model_comparison.csv"),
    Path("results/04_all_results.json"),
    Path("results/04_model_comparison_chart.png"),
    Path("results/04_predicted_vs_actual.png"),
    Path("results/04_feature_importance.png"),
    Path("results/04_confusion_matrices.png"),
]


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by looking for data/ and src/."""
    start = (start or Path(__file__)).resolve()
    start_dir = start if start.is_dir() else start.parent
    for directory in (start_dir, *start_dir.parents):
        if (directory / "data").is_dir() and (directory / "src").is_dir():
            return directory
    raise FileNotFoundError(
        "Tidak dapat menemukan root repository. Pastikan script berada di "
        "dalam repository nlp-trio-bertiga."
    )


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_inputs(repo_root: Path) -> None:
    """Verify that raw inputs and all pipeline notebooks exist."""
    print_header("CHECK INPUT")

    missing = [repo_root / p for p in REQUIRED_INPUTS if not (repo_root / p).exists()]
    missing_notebooks = [
        repo_root / notebook_path
        for _, notebook_path in PIPELINE
        if not (repo_root / notebook_path).exists()
    ]

    if missing or missing_notebooks:
        if missing:
            print("Input yang hilang:")
            for path in missing:
                print(f"  - {path}")
        if missing_notebooks:
            print("Notebook yang hilang:")
            for path in missing_notebooks:
                print(f"  - {path}")
        raise FileNotFoundError("Input/notebook pipeline belum lengkap.")

    # Useful sanity information for the new scrape.
    news_file = repo_root / "data/raw/news/cnbc_articles_final.csv"
    try:
        import pandas as pd

        news_df = pd.read_csv(news_file, usecols=["url"], nrows=None)
        print(f"Artikel raw terdeteksi: {len(news_df):,}")
    except Exception as exc:
        print(f"Peringatan: tidak dapat menghitung jumlah artikel raw: {exc}")

    print("Semua input utama dan notebook tersedia.")


def clean_generated_outputs(repo_root: Path) -> None:
    """Delete only known generated artifacts; never delete raw data or source code."""
    print_header("CLEAN DERIVED OUTPUTS")
    removed = 0
    for relative_path in GENERATED_PATHS:
        path = repo_root / relative_path
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"Dihapus: {relative_path}")
            removed += 1
    print(f"Total artefak lama dihapus: {removed}")


def execute_notebook(repo_root: Path, label: str, notebook_path: Path, timeout: int) -> float:
    """Execute one notebook with repo root as cwd; do not overwrite the notebook file."""
    full_path = repo_root / notebook_path
    print_header(f"RUN: {label}\n{notebook_path}")

    with full_path.open("r", encoding="utf-8") as fh:
        notebook = nbformat.read(fh, as_version=4)

    # Use the current Python environment/kernel.
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(repo_root)}},
    )

    started = time.perf_counter()
    try:
        client.execute(cwd=str(repo_root))
    except CellExecutionError as exc:
        elapsed = time.perf_counter() - started
        print(f"\nGAGAL pada notebook: {notebook_path} ({elapsed:.1f} detik)")
        print(str(exc))
        raise

    elapsed = time.perf_counter() - started
    print(f"Selesai: {notebook_path} ({elapsed:.1f} detik)")
    return elapsed


def check_expected_outputs(repo_root: Path) -> None:
    """Check that the key artifact from each stage exists."""
    print_header("CHECK PIPELINE OUTPUTS")

    expected = [
        Path("data/processed/cnbc_articles_clean.csv"),
        Path("data/processed/articles_aligned.csv"),
        Path("data/processed/daily_kurs.csv"),
        Path("data/split/articles_split.csv"),
        Path("data/split/daily_kurs_split.csv"),
        Path("data/feature_extracted/features_lexicon.csv"),
        Path("data/feature_extracted/model_dataset.csv"),
    ]

    missing = []
    for relative_path in expected:
        path = repo_root / relative_path
        if path.exists():
            print(f"OK   {relative_path}")
        else:
            print(f"MISS {relative_path}")
            missing.append(relative_path)

    # Check for modeling output (supports both new 04_gabungan_comparison.csv and legacy name)
    has_model_res = False
    for res_name in ["results/04_gabungan_comparison.csv", "results/04_model_comparison.csv"]:
        if (repo_root / res_name).exists():
            print(f"OK   {res_name}")
            has_model_res = True
            break
    if not has_model_res:
        print("MISS results/04_gabungan_comparison.csv")
        missing.append(Path("results/04_gabungan_comparison.csv"))

    if missing:
        raise FileNotFoundError(
            "Pipeline selesai tetapi ada output penting yang tidak terbentuk: "
            + ", ".join(map(str, missing))
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Jalankan pipeline NLP + JISDOR secara berurutan."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Path root repository. Default: otomatis dari lokasi script.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout maksimum per notebook dalam detik (default: 3600).",
    )
    parser.add_argument(
        "--clean-outputs",
        action="store_true",
        help="Hapus artefak hasil pipeline lama sebelum menjalankan ulang.",
    )
    args = parser.parse_args()

    repo_root = (args.repo.resolve() if args.repo else find_repo_root())
    os.chdir(repo_root)

    print_header("NLP-TRIO-BERTIGA: FULL PIPELINE")
    print(f"Repository : {repo_root}")
    print(f"Python     : {sys.executable}")
    print(f"Timeout    : {args.timeout}s/notebook")
    print("Urutan     : cleaning -> align -> split -> feature extraction -> modeling")

    check_inputs(repo_root)

    if args.clean_outputs:
        clean_generated_outputs(repo_root)

    timings: list[tuple[str, float]] = []
    pipeline_started = time.perf_counter()

    try:
        for label, notebook_path in PIPELINE:
            elapsed = execute_notebook(repo_root, label, notebook_path, args.timeout)
            timings.append((label, elapsed))
    except Exception:
        print_header("PIPELINE BERHENTI")
        print("Pipeline dihentikan karena salah satu tahap gagal.")
        print("Output dari tahap sebelumnya tetap dipertahankan untuk debugging.")
        return 1

    check_expected_outputs(repo_root)

    total = time.perf_counter() - pipeline_started
    print_header("PIPELINE SELESAI")
    for label, elapsed in timings:
        print(f"{label:30s}: {elapsed:8.1f} detik")
    print(f"{'Total':30s}: {total:8.1f} detik")
    print("\nArtefak utama:")
    print("  data/processed/cnbc_articles_clean.csv")
    print("  data/processed/articles_aligned.csv")
    print("  data/split/articles_split.csv")
    print("  data/split/daily_kurs_split.csv")
    print("  data/feature_extracted/model_dataset.csv")
    print("  results/04_gabungan_comparison.csv (dan 04_model_comparison.csv)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
