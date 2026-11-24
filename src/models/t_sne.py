"""t-SNE visualizations for daily LDA topic distributions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_DIR = _REPO_ROOT / "data" / "feature_extracted"
DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"
DEFAULT_DAILY_FEATURES = DEFAULT_FEATURE_DIR / "10_lda_topics_daily_features.csv"
DEFAULT_MODEL_DATASET = DEFAULT_FEATURE_DIR / "model_dataset.csv"
DEFAULT_TOPIC_TERMS = DEFAULT_RESULTS_DIR / "10_lda_topics_topic_terms.csv"


def save_lda_topic_tsne(
    daily_topic_features: pd.DataFrame,
    model_data: pd.DataFrame,
    topic_columns: list[str],
    topic_terms: pd.DataFrame,
    output_dir: str | Path,
    output_prefix: str = "10_lda_topics",
) -> dict:
    """Project training-day LDA topic probabilities to 2D and save plots/data.

    Only days with at least one article are included. Validation and test rows
    are deliberately excluded from fitting and plotting the projection.
    """
    required_model_columns = {"date", "split", "target", "ret_std5"}
    missing_model_columns = required_model_columns.difference(model_data.columns)
    if missing_model_columns:
        raise ValueError(
            "Model dataset is missing t-SNE context columns: "
            f"{sorted(missing_model_columns)}"
        )
    missing_topic_columns = set(topic_columns).difference(daily_topic_features.columns)
    if missing_topic_columns:
        raise ValueError(
            f"Daily topic features are missing columns: {sorted(missing_topic_columns)}"
        )
    if len(topic_columns) < 2:
        raise ValueError("t-SNE requires at least two LDA topic features.")
    missing_term_columns = {"topic_id", "top_terms"}.difference(topic_terms.columns)
    if missing_term_columns:
        raise ValueError(
            f"Topic terms are missing columns: {sorted(missing_term_columns)}"
        )

    train_context = model_data.loc[
        model_data["split"].eq("train"),
        ["date", "split", "target", "ret_std5"],
    ]
    context_rows = len(train_context)
    duplicate_context = train_context.duplicated(["date", "split"], keep=False)
    if duplicate_context.any():
        duplicate_groups = train_context.loc[duplicate_context].groupby(
            ["date", "split"], dropna=False
        )
        conflicting_context = duplicate_groups[["target", "ret_std5"]].nunique(
            dropna=False
        ).gt(1).any(axis=1)
        if conflicting_context.any():
            raise ValueError(
                "Repeated training dates have conflicting target/volatility values."
            )
        train_context = train_context.drop_duplicates(["date", "split"])
    train_topics = daily_topic_features.loc[
        daily_topic_features["split"].eq("train"),
        ["date", "split", *topic_columns],
    ]
    train = train_context.merge(
        train_topics,
        on=["date", "split"],
        how="inner",
        validate="one_to_one",
    )
    unmatched_context_rows = len(train_context) - len(train)
    train = train.sort_values("date").reset_index(drop=True)

    topic_values = train[topic_columns].to_numpy(dtype=float)
    volatility = pd.to_numeric(train["ret_std5"], errors="coerce").to_numpy()
    if not np.isfinite(topic_values).all() or not np.isfinite(volatility).all():
        raise ValueError("Training topic probabilities and ret_std5 must be finite.")

    has_articles = topic_values.sum(axis=1) > 0
    train = train.loc[has_articles].reset_index(drop=True)
    topic_values = topic_values[has_articles]
    if len(train) < 3:
        raise ValueError("At least three training days with news are required for t-SNE.")

    perplexity = min(30.0, float(len(train) - 1))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1_000,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(topic_values)

    topic_ids = np.argmax(topic_values, axis=1) + 1
    projected = train[["date", "split", "target", "ret_std5"]].copy()
    projected["dominant_topic"] = topic_ids
    projected["tsne_1"] = coordinates[:, 0]
    projected["tsne_2"] = coordinates[:, 1]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = output_dir / f"{output_prefix}_tsne_coordinates.csv"
    topic_plot_path = output_dir / f"{output_prefix}_tsne_topics.png"
    volatility_plot_path = output_dir / f"{output_prefix}_tsne_volatility.png"
    projected.to_csv(coordinates_path, index=False)

    terms_by_topic = {
        int(row.topic_id): str(row.top_terms)
        for row in topic_terms.itertuples(index=False)
    }
    color_map = plt.get_cmap("tab20", len(topic_columns))
    fig, ax = plt.subplots(figsize=(10, 7))
    for topic_id in sorted(np.unique(topic_ids)):
        mask = topic_ids == topic_id
        terms = terms_by_topic.get(int(topic_id), "")
        short_terms = ", ".join(terms.split(" | ")[:3])
        label = f"Topik {topic_id}"
        if short_terms:
            label += f": {short_terms}"
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=color_map(int(topic_id) - 1),
            label=label,
            s=24,
            alpha=0.7,
            edgecolors="none",
        )
    ax.set_title("t-SNE Distribusi Topik LDA Harian (Train)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Topik dominan", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(topic_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    volatility = train["ret_std5"].to_numpy(dtype=float)
    q25, q75 = np.quantile(volatility, [0.25, 0.75])
    high_volatility = volatility >= q75
    low_volatility = (volatility <= q25) & ~high_volatility
    medium_volatility = ~(high_volatility | low_volatility)
    groups = (
        (high_volatility, "#e74c3c", f"Tinggi (Q4, ≥{q75:.4f})"),
        (low_volatility, "#2ecc71", f"Rendah (Q1, ≤{q25:.4f})"),
        (medium_volatility, "#bdc3c7", "Menengah"),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for mask, color, label in groups:
        if mask.any():
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                color=color,
                label=label,
                s=24,
                alpha=0.6,
                edgecolors="none",
            )
    ax.set_title("t-SNE Topik LDA: Hari Volatil vs Tenang (Train)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Volatilitas")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(volatility_plot_path, dpi=150)
    plt.close(fig)

    return {
        "coordinates": str(coordinates_path),
        "topic_plot": str(topic_plot_path),
        "volatility_plot": str(volatility_plot_path),
        "training_days_with_news": int(len(train)),
        "excluded_no_news_days": int((~has_articles).sum()),
        "excluded_missing_topic_days": int(unmatched_context_rows),
        "collapsed_duplicate_context_rows": int(context_rows - len(train_context)),
        "perplexity": perplexity,
        "random_state": 42,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate t-SNE plots from saved daily LDA topic features."
    )
    parser.add_argument("--daily-features", type=Path, default=DEFAULT_DAILY_FEATURES)
    parser.add_argument("--model-dataset", type=Path, default=DEFAULT_MODEL_DATASET)
    parser.add_argument("--topic-terms", type=Path, default=DEFAULT_TOPIC_TERMS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-prefix", default="10_lda_topics")
    args = parser.parse_args(argv)

    input_paths = {
        "daily topic features": args.daily_features,
        "model dataset": args.model_dataset,
        "topic terms": args.topic_terms,
    }
    missing_paths = [f"{label}: {path}" for label, path in input_paths.items()
                     if not path.is_file()]
    if missing_paths:
        print(
            "[t_sne] ERROR: input file tidak ditemukan:\n  "
            + "\n  ".join(missing_paths),
            file=sys.stderr,
        )
        return 1

    daily_topics = pd.read_csv(args.daily_features)
    model_data = pd.read_csv(args.model_dataset)
    topic_terms = pd.read_csv(args.topic_terms)
    for frame in (daily_topics, model_data):
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    topic_columns = sorted(
        (
            column for column in daily_topics.columns
            if column.startswith("topic_")
            and column.rsplit("_", 1)[-1].isdigit()
        ),
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )
    if len(topic_columns) < 2:
        print(
            "[t_sne] ERROR: ditemukan kurang dari dua kolom topic_* pada "
            f"{args.daily_features}.",
            file=sys.stderr,
        )
        return 1

    try:
        artifacts = save_lda_topic_tsne(
            daily_topic_features=daily_topics,
            model_data=model_data,
            topic_columns=topic_columns,
            topic_terms=topic_terms,
            output_dir=args.results_dir,
            output_prefix=args.output_prefix,
        )
    except ValueError as exc:
        print(f"[t_sne] ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"[t_sne] Proyeksi train: {artifacts['training_days_with_news']} "
        f"hari dengan berita; {artifacts['excluded_no_news_days']} hari tanpa "
        f"berita dan {artifacts['excluded_missing_topic_days']} hari tanpa fitur "
        f"LDA dikecualikan; {artifacts['collapsed_duplicate_context_rows']} "
        f"baris konteks duplikat digabung; perplexity={artifacts['perplexity']}."
    )
    print(
        "[t_sne] Tersimpan:\n"
        f"  topik: {artifacts['topic_plot']}\n"
        f"  volatilitas: {artifacts['volatility_plot']}\n"
        f"  koordinat: {artifacts['coordinates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())