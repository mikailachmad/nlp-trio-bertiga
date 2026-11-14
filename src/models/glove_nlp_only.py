"""Pretrained GloVe NLP-only experiment for next-day JISDOR prediction.

Word vectors are read from the Stanford GloVe 6B 100d text file. On first run,
the official archive is downloaded and only vectors used by the aligned corpus
are cached locally.

Run from the repository root:
    python -m src.models.glove_nlp_only
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.jisdor.target_formulation import THRESHOLD_PCT  # noqa: E402
from src.models.metrics import (  # noqa: E402
    LABEL_ORDER,
    directional_hit_from_regression,
    evaluate_classifier,
    evaluate_regression,
)
from src.models.naive_baseline import predict_majority, predict_persistence  # noqa: E402
from src.models.word2vec_nlp_only import (  # noqa: E402
    DEFAULT_ARTICLE_SPLITS,
    DEFAULT_ARTICLES,
    DEFAULT_FEATURE_DIR,
    DEFAULT_MODEL_DATASET,
    DEFAULT_RESULTS_DIR,
    ID2LABEL,
    LABEL2ID,
    LEXICON_FEATURES,
    SPLITS,
    _add_result,
    _validate_dataset,
    load_article_text,
    make_classifier,
    make_regressor,
    tokenize,
)

DEFAULT_GLOVE_CACHE = (
    _REPO_ROOT / "data" / "features" / "artifacts" / "glove_6b_100d.npz"
)
DEFAULT_ARCHIVE = (
    _REPO_ROOT / "data" / "features" / "pretrained" / "glove.6B.zip"
)
GLOVE_ARCHIVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_MEMBER = "glove.6B.100d.txt"
GLOVE_DIMENSION = 100
GLOVE_SOURCE_NAME = "Stanford GloVe 6B 100d"


class GloveVectors:
    """A filtered, cached subset of pretrained GloVe vectors."""

    def __init__(self, vectors: dict[str, np.ndarray], source: str) -> None:
        self.vectors = vectors
        self.source = source

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens = np.asarray(list(self.vectors), dtype=str)
        vectors = (
            np.stack(list(self.vectors.values())).astype(np.float32)
            if self.vectors
            else np.empty((0, GLOVE_DIMENSION), dtype=np.float32)
        )
        np.savez_compressed(
            path,
            tokens=tokens,
            vectors=vectors,
            metadata=np.array(
                json.dumps(
                    {
                        "source": self.source,
                        "dimension": GLOVE_DIMENSION,
                        "format": "GloVe text vectors",
                    }
                )
            ),
        )

    @classmethod
    def load(cls, path: Path) -> GloveVectors:
        with np.load(path, allow_pickle=False) as artifact:
            tokens = artifact["tokens"].tolist()
            vectors = artifact["vectors"]
            metadata = json.loads(artifact["metadata"].item())
        if vectors.ndim != 2 or vectors.shape != (len(tokens), GLOVE_DIMENSION):
            raise ValueError(f"Invalid cached GloVe matrix in {path}.")
        return cls(
            {token: vectors[index] for index, token in enumerate(tokens)},
            source=metadata["source"],
        )


def _read_glove_stream(
    stream: TextIO, needed_tokens: set[str]
) -> dict[str, np.ndarray]:
    """Read only requested words from a whitespace-delimited GloVe text stream."""
    vectors: dict[str, np.ndarray] = {}
    for line_number, line in enumerate(stream, start=1):
        fields = line.rstrip().split()
        if not fields:
            continue
        if line_number == 1 and len(fields) == 2:
            try:
                _, declared_dimension = map(int, fields)
            except ValueError:
                pass
            else:
                if declared_dimension != GLOVE_DIMENSION:
                    raise ValueError(
                        f"GloVe file declares {declared_dimension} dimensions; "
                        f"expected {GLOVE_DIMENSION}."
                    )
                continue
        word = fields[0]
        if word not in needed_tokens:
            continue
        if len(fields) != GLOVE_DIMENSION + 1:
            raise ValueError(
                f"Malformed GloVe vector for {word!r} at line {line_number}."
            )
        try:
            vector = np.asarray(fields[1:], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric GloVe vector for {word!r} at line {line_number}."
            ) from exc
        if not np.isfinite(vector).all():
            raise ValueError(
                f"GloVe vector for {word!r} contains non-finite values."
            )
        vectors[word] = vector
        if len(vectors) == len(needed_tokens):
            break
    return vectors


def _open_glove_source(path: Path) -> tuple[TextIO, object | None]:
    """Open either a GloVe text file or the official Stanford zip archive."""
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        try:
            member = archive.open(GLOVE_MEMBER)
        except (KeyError, OSError):
            archive.close()
            raise
        return io.TextIOWrapper(member, encoding="utf-8"), archive
    return path.open("r", encoding="utf-8"), None


def _download_archive(path: Path) -> None:
    """Download Stanford's pretrained archive to a local, ignored cache path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        GLOVE_ARCHIVE_URL,
        headers={"User-Agent": "nlp-trio-bertiga/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0"))
            received = 0
            with partial_path.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    received += len(chunk)
                    if total:
                        print(
                            f"\r[ GloVe download ] {received / total:.1%}",
                            end="",
                            flush=True,
                        )
        if total and received != total:
            raise OSError(
                f"Incomplete GloVe download: received {received} of {total} bytes."
            )
        partial_path.replace(path)
        print(f"\nGloVe archive cached at {path}")
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def _cached_vectors(path: Path) -> GloveVectors | None:
    if not path.exists():
        return None
    return GloveVectors.load(path)


def load_pretrained_vectors(
    needed_tokens: set[str],
    cache_path: Path,
    local_glove_path: Path | None,
    archive_path: Path,
    download: bool = True,
) -> tuple[GloveVectors, str]:
    """Load cached GloVe, or extract required words from a local/downloaded source."""
    cached = _cached_vectors(cache_path)
    desired_source = str(local_glove_path) if local_glove_path else GLOVE_SOURCE_NAME
    if cached is not None and cached.source == desired_source and needed_tokens.issubset(cached.vectors):
        print(f"Using cached GloVe vectors: {cache_path}")
        return cached, str(cache_path)

    if cached is not None and cached.source != desired_source:
        cached = None
    missing = needed_tokens.difference(cached.vectors if cached else ())
    source_path = local_glove_path
    if source_path is None:
        if not archive_path.exists():
            if not download:
                raise FileNotFoundError(
                    "No cached GloVe vectors or archive were found. Provide "
                    "--glove-path, allow download, or run without --download."
                )
            print(
                "Downloading Stanford GloVe 6B archive (approximately 822 MB); "
                "it will be kept in the local cache."
            )
            _download_archive(archive_path)
        source_path = archive_path
    if not source_path.is_file():
        raise FileNotFoundError(f"GloVe source file does not exist: {source_path}")

    print(
        f"Reading GloVe vectors for {len(missing):,} corpus tokens from "
        f"{source_path}."
    )
    stream, archive = _open_glove_source(source_path)
    try:
        extracted = _read_glove_stream(stream, missing)
    finally:
        stream.close()
        if archive is not None:
            archive.close()

    merged = dict(cached.vectors) if cached else {}
    merged.update(extracted)
    if not merged:
        raise ValueError(
            "No corpus tokens matched the GloVe source. Check that the selected "
            "file contains 100-dimensional English GloVe vectors."
        )
    vectors = GloveVectors(merged, source=desired_source)
    vectors.save(cache_path)
    print(
        f"Cached {len(extracted):,} new GloVe vectors "
        f"({len(merged):,} total) at {cache_path}."
    )
    return vectors, str(source_path)


def collect_corpus_tokens(articles: pd.DataFrame) -> set[str]:
    """Collect words that need pretrained vectors without fitting on their labels."""
    title_col, body_col = next(
        (
            pair
            for pair in (
                ("title_bow", "body_bow"),
                ("title_clean", "body_clean"),
            )
            if set(pair).issubset(articles.columns)
        ),
        (None, None),
    )
    if title_col is None or body_col is None:
        raise ValueError("Aligned articles need title/body text columns.")
    tokens: set[str] = set()
    for title, body in zip(articles[title_col], articles[body_col]):
        if isinstance(title, str):
            tokens.update(tokenize(title))
        if isinstance(body, str):
            tokens.update(tokenize(body))
    if not tokens:
        raise ValueError("No tokens found in aligned article text.")
    return tokens


def build_daily_embeddings(
    articles: pd.DataFrame,
    model_dataset: pd.DataFrame,
    vectors: GloveVectors,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Mean-pool pretrained in-vocabulary token vectors per trading date."""
    title_col, body_col = next(
        (
            pair
            for pair in (
                ("title_bow", "body_bow"),
                ("title_clean", "body_clean"),
            )
            if set(pair).issubset(articles.columns)
        ),
        (None, None),
    )
    if title_col is None or body_col is None:
        raise ValueError("Aligned articles need title/body text columns.")

    dimensions = GLOVE_DIMENSION
    daily = np.zeros((len(model_dataset), dimensions), dtype=np.float64)
    vector_counts = np.zeros(len(model_dataset), dtype=np.int64)
    token_counts = np.zeros(len(model_dataset), dtype=np.int64)
    date_to_row = {
        date: index for index, date in enumerate(model_dataset["date"].tolist())
    }

    for row in articles.itertuples(index=False):
        row_index = date_to_row.get(row.trading_date)
        if row_index is None:
            continue
        article_tokens: list[str] = []
        for value in (getattr(row, title_col), getattr(row, body_col)):
            article_tokens.extend(tokenize(value))
        token_counts[row_index] += len(article_tokens)
        for token in article_tokens:
            vector = vectors.vectors.get(token)
            if vector is not None:
                daily[row_index] += vector
                vector_counts[row_index] += 1

    nonzero_counts = vector_counts > 0
    daily[nonzero_counts] /= vector_counts[nonzero_counts, None]
    feature_columns = [
        f"glove_{index + 1:03d}" for index in range(dimensions)
    ]
    features = pd.DataFrame(daily.astype(np.float32), columns=feature_columns)
    features.insert(0, "split", model_dataset["split"].to_numpy())
    features.insert(0, "date", model_dataset["date"].to_numpy())

    coverage: dict[str, dict[str, float | int]] = {}
    for split in SPLITS:
        mask = model_dataset["split"].to_numpy() == split
        total = int(token_counts[mask].sum())
        known = int(vector_counts[mask].sum())
        coverage[split] = {
            "tokens_total": total,
            "tokens_in_vocabulary": known,
            "token_coverage": round(known / total, 6) if total else 0.0,
            "days_without_known_tokens": int((vector_counts[mask] == 0).sum()),
        }
    return features, coverage


def _plot_confusion_matrix(
    matrix: Iterable[Iterable[int]], output_path: Path
) -> None:
    cm = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(cm, cmap="Blues")
    axis.set_xticks(range(len(LABEL_ORDER)), labels=LABEL_ORDER)
    axis.set_yticks(range(len(LABEL_ORDER)), labels=LABEL_ORDER)
    axis.set_xlabel("Prediksi")
    axis.set_ylabel("Aktual")
    axis.set_title("XGBoost GloVe NLP-only: Confusion Matrix (Test)")
    for row in range(len(LABEL_ORDER)):
        for column in range(len(LABEL_ORDER)):
            axis.text(
                column,
                row,
                cm[row, column],
                ha="center",
                va="center",
                color="white" if cm[row, column] > cm.max() / 2 else "black",
            )
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluasi XGBoost NLP-only dengan pretrained GloVe."
    )
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--article-splits", type=Path, default=DEFAULT_ARTICLE_SPLITS)
    parser.add_argument("--model-dataset", type=Path, default=DEFAULT_MODEL_DATASET)
    parser.add_argument("--glove-path", type=Path)
    parser.add_argument("--glove-cache", type=Path, default=DEFAULT_GLOVE_CACHE)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output-prefix", default="09_glove")
    args = parser.parse_args(argv)

    data = _validate_dataset(pd.read_csv(args.model_dataset))
    articles = load_article_text(args.articles, args.article_splits, data)
    needed_tokens = collect_corpus_tokens(articles)
    vectors, vector_source = load_pretrained_vectors(
        needed_tokens=needed_tokens,
        cache_path=args.glove_cache,
        local_glove_path=args.glove_path,
        archive_path=args.archive_path,
        download=not args.no_download,
    )
    embeddings, coverage = build_daily_embeddings(articles, data, vectors)

    feature_path = (
        DEFAULT_FEATURE_DIR / f"{args.output_prefix}_daily_features.csv"
    )
    DEFAULT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(feature_path, index=False)
    embedding_columns = [
        column for column in embeddings.columns if column.startswith("glove_")
    ]
    feature_columns = LEXICON_FEATURES + embedding_columns
    data = data.merge(
        embeddings,
        on=["date", "split"],
        how="left",
        validate="one_to_one",
    )
    splits = {
        split: data[data["split"] == split].sort_values("date").copy()
        for split in SPLITS
    }
    X = {split: splits[split][feature_columns] for split in SPLITS}
    train = splits["train"]
    y_class = {
        split: splits[split]["target"].map(LABEL2ID).astype(int)
        for split in SPLITS
    }
    y_rate = {
        split: splits[split]["kurs_besok"].astype(float)
        for split in SPLITS
    }
    y_delta = {
        split: (
            splits[split]["kurs_besok"] - splits[split]["kurs_jisdor"]
        ).astype(float)
        for split in SPLITS
    }
    rate_today = {
        split: splits[split]["kurs_jisdor"].astype(float).to_numpy()
        for split in SPLITS
    }

    results: list[dict] = []
    naive_rmse: dict[str, float] = {}
    for split in ("val", "test"):
        sample = splits[split]
        persistence = evaluate_classifier(
            sample["target"].to_numpy(),
            predict_persistence(sample["target"]),
            include_kappa=True,
        )
        majority = evaluate_classifier(
            sample["target"].to_numpy(),
            predict_majority(train["target"], len(sample)),
            include_kappa=True,
        )
        _add_result(results, "Naive Persistence of Direction", split, "clf", persistence)
        _add_result(results, "Naive Majority-Class", split, "clf", majority)
        naive_regression = evaluate_regression(
            y_rate[split], rate_today[split], include_mape=True
        )
        naive_rmse[split] = naive_regression["rmse"]
        naive_regression["skill_vs_naive"] = 0.0
        _add_result(
            results, "Naive Persistence (Random Walk)", split, "reg", naive_regression
        )

    classifier = make_classifier()
    class_weights = compute_sample_weight(class_weight="balanced", y=y_class["train"])
    classifier.fit(
        X["train"],
        y_class["train"],
        sample_weight=class_weights,
        eval_set=[(X["val"], y_class["val"])],
        verbose=False,
    )
    predicted_class = {
        split: pd.Series(classifier.predict(X[split]))
        .map(ID2LABEL)
        .to_numpy()
        for split in ("val", "test")
    }
    classification_metrics = {}
    for split in ("val", "test"):
        metrics = evaluate_classifier(
            splits[split]["target"].to_numpy(),
            predicted_class[split],
            include_kappa=True,
        )
        classification_metrics[split] = metrics
        _add_result(results, "XGBoost GloVe NLP-only", split, "clf", metrics)
        print(
            f"[GloVe clf {split}] accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    regressor = make_regressor()
    regressor.fit(
        X["train"],
        y_delta["train"],
        eval_set=[(X["val"], y_delta["val"])],
        verbose=False,
    )
    predicted_rate = {
        split: rate_today[split] + regressor.predict(X[split])
        for split in ("val", "test")
    }
    regression_metrics = {}
    for split in ("val", "test"):
        metrics = evaluate_regression(
            y_rate[split],
            predicted_rate[split],
            rmse_naive=naive_rmse[split],
            include_mape=True,
        )
        metrics["directional_hit"] = directional_hit_from_regression(
            rate_today[split], y_rate[split], predicted_rate[split]
        )
        regression_metrics[split] = metrics
        _add_result(results, "XGBoost GloVe NLP-only", split, "reg", metrics)
        print(
            f"[GloVe reg {split}] rmse={metrics['rmse']:.2f} "
            f"skill={metrics['skill_vs_naive']:+.4f}"
        )

    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = DEFAULT_RESULTS_DIR / f"{args.output_prefix}_comparison.csv"
    pd.DataFrame(results).round(6).to_csv(comparison_path, index=False)
    model_path = (
        str(args.glove_cache.relative_to(_REPO_ROOT))
        if args.glove_cache.is_relative_to(_REPO_ROOT)
        else str(args.glove_cache)
    )
    output = {
        "model": "XGBoost GloVe NLP-only (tanpa fitur time-series)",
        "embedding": {
            "algorithm": GLOVE_SOURCE_NAME,
            "vector_size": GLOVE_DIMENSION,
            "training": "pretrained, tidak di-fit ulang pada split proyek",
            "aggregation": "mean pooling per trading_date",
            "oov": "diabaikan; hari tanpa token dikenal memakai vektor nol",
            "vocabulary_size": len(vectors.vectors),
            "coverage": coverage,
        },
        "features": {
            "lexicon": LEXICON_FEATURES,
            "glove_dimensions": len(embedding_columns),
        },
        "target": {
            "classification": (
                f"target (NAIK/STABIL/TURUN, threshold {THRESHOLD_PCT:.2f}%)"
            ),
            "regression": "kurs_besok (dilatih pada selisih, dikonversi ke level)",
        },
        "best_iteration": {
            "clf": int(classifier.best_iteration),
            "reg": int(regressor.best_iteration),
        },
        "classification": classification_metrics,
        "regression": regression_metrics,
        "rmse_naive_same_rows": naive_rmse,
        "artifacts": {
            "embedding_cache": model_path,
            "vector_source": vector_source,
            "daily_features": str(feature_path.relative_to(_REPO_ROOT)),
            "comparison": str(comparison_path.relative_to(_REPO_ROOT)),
        },
    }
    results_path = DEFAULT_RESULTS_DIR / f"{args.output_prefix}_results.json"
    with results_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=False, indent=2)
    _plot_confusion_matrix(
        classification_metrics["test"]["confusion_matrix"],
        DEFAULT_RESULTS_DIR / f"{args.output_prefix}_confusion_matrix.png",
    )
    print(f"GloVe vectors cached: {args.glove_cache}")
    print(f"Results saved: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
