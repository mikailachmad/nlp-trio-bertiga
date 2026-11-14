"""FastText NLP-only experiment for next-day JISDOR prediction.

This implements FastText-style Skip-gram with Negative Sampling using NumPy and
SciPy, learning character n-gram vectors from train-split articles only.

Run from the repository root:
    python -m src.models.fasttext_nlp_only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
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

DEFAULT_MODEL_PATH = (
    _REPO_ROOT / "data" / "features" / "artifacts" / "fasttext_sgns.npz"
)
NEGATIVE_SAMPLES = 5
TRAIN_BATCH_SIZE = 4096


def _fnv1a_hash(text: str, bucket_count: int) -> int:
    """Return a stable FastText-style hash without Python's randomized hash()."""
    value = 2166136261
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value % bucket_count


def character_ngram_buckets(
    token: str,
    bucket_count: int,
    min_n: int,
    max_n: int,
) -> np.ndarray:
    """Hash boundary-marked character n-grams for known and unseen words."""
    marked = f"<{token}>"
    buckets = {
        _fnv1a_hash(marked[start : start + size], bucket_count)
        for size in range(min_n, max_n + 1)
        for start in range(max(0, len(marked) - size + 1))
    }
    if not buckets:
        buckets.add(_fnv1a_hash(marked, bucket_count))
    return np.fromiter(sorted(buckets), dtype=np.int32)


@dataclass
class FastTextModel:
    token_to_id: dict[str, int]
    subword_matrix: sparse.csr_matrix
    subword_vectors: np.ndarray
    counts: np.ndarray
    bucket_count: int
    min_n: int
    max_n: int
    epochs: int
    window: int
    min_count: int
    negative_samples: int
    vector_cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def vector_for(self, token: str) -> np.ndarray:
        """Return a word vector, including a subword-only vector for OOV words."""
        cached = self.vector_cache.get(token)
        if cached is not None:
            return cached
        token_id = self.token_to_id.get(token)
        if token_id is not None:
            row = self.subword_matrix.getrow(token_id)
        else:
            buckets = character_ngram_buckets(
                token, self.bucket_count, self.min_n, self.max_n
            )
            row = sparse.csr_matrix(
                (
                    np.full(buckets.size, 1.0 / buckets.size, dtype=np.float32),
                    buckets,
                    np.array([0, buckets.size], dtype=np.int32),
                ),
                shape=(1, self.bucket_count),
            )
        vector = np.asarray(row @ self.subword_vectors).reshape(-1)
        self.vector_cache[token] = vector
        return vector

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens = np.asarray(
            [
                token
                for token, _ in sorted(
                    self.token_to_id.items(), key=lambda item: item[1]
                )
            ],
            dtype=str,
        )
        metadata = {
            "algorithm": "FastText Skip-gram with Negative Sampling (SGNS)",
            "epochs": self.epochs,
            "window": self.window,
            "min_count": self.min_count,
            "negative_samples": self.negative_samples,
            "vector_size": int(self.subword_vectors.shape[1]),
            "bucket_count": self.bucket_count,
            "character_ngrams": [self.min_n, self.max_n],
        }
        np.savez_compressed(
            path,
            tokens=tokens,
            subword_vectors=self.subword_vectors,
            counts=self.counts,
            metadata=np.array(json.dumps(metadata)),
        )

    @classmethod
    def load(cls, path: Path) -> FastTextModel:
        with np.load(path, allow_pickle=False) as artifact:
            tokens = artifact["tokens"].tolist()
            counts = artifact["counts"]
            vectors = artifact["subword_vectors"]
            metadata = json.loads(artifact["metadata"].item())
        token_to_id = {token: index for index, token in enumerate(tokens)}
        bucket_count = int(metadata["bucket_count"])
        min_n, max_n = metadata["character_ngrams"]
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for token_id, token in enumerate(tokens):
            buckets = character_ngram_buckets(
                token, bucket_count, int(min_n), int(max_n)
            )
            rows.extend([token_id] * buckets.size)
            columns.extend(buckets.tolist())
            values.extend([1.0 / buckets.size] * buckets.size)
        matrix = sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(len(tokens), bucket_count),
            dtype=np.float32,
        )
        return cls(
            token_to_id=token_to_id,
            subword_matrix=matrix,
            subword_vectors=vectors,
            counts=counts,
            bucket_count=bucket_count,
            min_n=int(min_n),
            max_n=int(max_n),
            epochs=int(metadata["epochs"]),
            window=int(metadata["window"]),
            min_count=int(metadata["min_count"]),
            negative_samples=int(metadata["negative_samples"]),
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -15.0, 15.0)))


def _training_pairs(
    sentences: list[list[str]],
    token_to_id: dict[str, int],
    counts: np.ndarray,
    window: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total_tokens = counts.sum()
    keep_probability = np.minimum(
        1.0,
        (np.sqrt(counts / (1e-4 * total_tokens)) + 1)
        * (1e-4 * total_tokens / counts),
    )
    pair_centers: list[np.ndarray] = []
    pair_contexts: list[np.ndarray] = []
    for sentence in sentences:
        ids = np.fromiter(
            (token_to_id[token] for token in sentence if token in token_to_id),
            dtype=np.int32,
        )
        if ids.size < 2:
            continue
        ids = ids[rng.random(ids.size) < keep_probability[ids]]
        if ids.size < 2:
            continue
        for offset in range(1, min(window, ids.size - 1) + 1):
            left = ids[:-offset]
            right = ids[offset:]
            pair_centers.extend((left, right))
            pair_contexts.extend((right, left))
    if not pair_centers:
        raise ValueError(
            "No FastText Skip-gram pairs remain after filtering/subsampling."
        )
    return np.concatenate(pair_centers), np.concatenate(pair_contexts)


def train_fasttext(
    sentences: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    bucket_count: int,
    min_n: int = 3,
    max_n: int = 6,
    seed: int = 42,
) -> FastTextModel:
    """Fit FastText-style subword Skip-gram vectors using training sentences."""
    if min_n <= 0 or max_n < min_n:
        raise ValueError("Character n-gram sizes must satisfy 0 < min_n <= max_n.")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")

    token_counts_map = Counter(token for sentence in sentences for token in sentence)
    vocabulary = {
        token: count
        for token, count in token_counts_map.items()
        if count >= min_count
    }
    if len(vocabulary) < 2:
        raise ValueError(
            "FastText needs at least two training tokens after applying min_count."
        )
    ordered_tokens = sorted(vocabulary, key=lambda token: (-vocabulary[token], token))
    token_to_id = {token: index for index, token in enumerate(ordered_tokens)}
    counts = np.asarray(
        [vocabulary[token] for token in ordered_tokens], dtype=np.float64
    )

    subword_rows: list[int] = []
    subword_columns: list[int] = []
    subword_values: list[float] = []
    for token_id, token in enumerate(ordered_tokens):
        buckets = character_ngram_buckets(token, bucket_count, min_n, max_n)
        weight = 1.0 / buckets.size
        subword_rows.extend([token_id] * buckets.size)
        subword_columns.extend(buckets.tolist())
        subword_values.extend([weight] * buckets.size)
    subword_matrix = sparse.csr_matrix(
        (subword_values, (subword_rows, subword_columns)),
        shape=(len(ordered_tokens), bucket_count),
        dtype=np.float32,
    )
    row_sums = np.asarray(subword_matrix.sum(axis=1)).reshape(-1)
    subword_matrix = sparse.diags(1.0 / row_sums) @ subword_matrix
    subword_matrix = subword_matrix.tocsr().astype(np.float32)

    rng = np.random.default_rng(seed)
    centers, contexts = _training_pairs(
        sentences, token_to_id, counts, window, rng
    )
    subword_vectors = rng.uniform(
        -0.5 / vector_size,
        0.5 / vector_size,
        size=(bucket_count, vector_size),
    ).astype(np.float32)
    output_vectors = np.zeros((len(ordered_tokens), vector_size), dtype=np.float32)
    negative_distribution = counts**0.75
    negative_distribution /= negative_distribution.sum()

    pair_count = centers.size
    for epoch in range(epochs):
        order = rng.permutation(pair_count)
        for start in range(0, pair_count, TRAIN_BATCH_SIZE):
            batch_indices = order[start : start + TRAIN_BATCH_SIZE]
            batch_centers = centers[batch_indices]
            batch_contexts = contexts[batch_indices]
            negatives = rng.choice(
                len(ordered_tokens),
                size=(batch_indices.size, NEGATIVE_SAMPLES),
                p=negative_distribution,
            )

            batch_subwords = subword_matrix[batch_centers]
            center_vectors = np.asarray(batch_subwords @ subword_vectors)
            positive_vectors = output_vectors[batch_contexts].copy()
            negative_vectors = output_vectors[negatives].copy()
            positive_gradient = _sigmoid(
                np.einsum("ij,ij->i", center_vectors, positive_vectors)
            ) - 1
            negative_gradient = _sigmoid(
                np.einsum("ij,ikj->ik", center_vectors, negative_vectors)
            )
            center_gradient = (
                positive_gradient[:, None] * positive_vectors
                + np.einsum("ik,ikj->ij", negative_gradient, negative_vectors)
            )
            positive_vector_gradient = positive_gradient[:, None] * center_vectors
            negative_vector_gradient = (
                negative_gradient[:, :, None] * center_vectors[:, None, :]
            )

            progress = (epoch * pair_count + start) / (epochs * pair_count)
            learning_rate = max(0.0001, 0.025 * (1 - progress))
            coo = batch_subwords.tocoo()
            subword_gradient = (
                center_gradient[coo.row] * coo.data[:, None]
            )
            np.add.at(
                subword_vectors,
                coo.col,
                -learning_rate * subword_gradient,
            )
            np.add.at(
                output_vectors,
                batch_contexts,
                -learning_rate * positive_vector_gradient,
            )
            np.add.at(
                output_vectors,
                negatives.ravel(),
                -learning_rate
                * negative_vector_gradient.reshape(-1, vector_size),
            )
        print(
            f"[FastText] epoch {epoch + 1}/{epochs}; "
            f"training pairs={pair_count:,}"
        )

    return FastTextModel(
        token_to_id=token_to_id,
        subword_matrix=subword_matrix,
        subword_vectors=subword_vectors,
        counts=counts.astype(np.int64),
        bucket_count=bucket_count,
        min_n=min_n,
        max_n=max_n,
        epochs=epochs,
        window=window,
        min_count=min_count,
        negative_samples=NEGATIVE_SAMPLES,
    )


def build_daily_embeddings(
    articles: pd.DataFrame,
    model_dataset: pd.DataFrame,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    bucket_count: int,
    min_n: int,
    max_n: int,
) -> tuple[FastTextModel, pd.DataFrame]:
    text_column_pairs = (
        ("title_bow", "body_bow"),
        ("title_clean", "body_clean"),
    )
    title_col, body_col = next(
        (pair for pair in text_column_pairs if set(pair).issubset(articles.columns)),
        (None, None),
    )
    if title_col is None or body_col is None:
        raise ValueError("Aligned articles need title/body text columns.")
    articles = articles.copy()
    articles["tokens"] = [
        tokenize(
            " ".join(
                value for value in (title, body) if isinstance(value, str)
            )
        )
        for title, body in zip(articles[title_col], articles[body_col])
    ]
    training_sentences = articles.loc[
        articles["split"] == "train", "tokens"
    ].tolist()
    training_sentences = [tokens for tokens in training_sentences if tokens]
    if not training_sentences:
        raise ValueError("No non-empty training articles are available for FastText.")

    model = train_fasttext(
        sentences=training_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        bucket_count=bucket_count,
        min_n=min_n,
        max_n=max_n,
    )
    feature_columns = [f"fasttext_{index + 1:03d}" for index in range(vector_size)]
    daily = np.zeros((len(model_dataset), vector_size), dtype=np.float32)
    counts = np.zeros(len(model_dataset), dtype=np.int64)
    date_to_row = {
        date: index for index, date in enumerate(model_dataset["date"].tolist())
    }
    for row in articles.itertuples(index=False):
        row_index = date_to_row.get(row.trading_date)
        if row_index is None:
            continue
        token_vectors = [model.vector_for(token) for token in row.tokens]
        if token_vectors:
            daily[row_index] += np.sum(token_vectors, axis=0)
            counts[row_index] += len(token_vectors)
    nonzero_counts = counts > 0
    daily[nonzero_counts] /= counts[nonzero_counts, None]
    features = pd.DataFrame(daily, columns=feature_columns)
    features.insert(0, "split", model_dataset["split"].to_numpy())
    features.insert(0, "date", model_dataset["date"].to_numpy())
    return model, features


def _plot_confusion_matrix(matrix: Sequence[Sequence[int]], output_path: Path) -> None:
    cm = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(cm, cmap="Blues")
    axis.set_xticks(range(len(LABEL_ORDER)), labels=LABEL_ORDER)
    axis.set_yticks(range(len(LABEL_ORDER)), labels=LABEL_ORDER)
    axis.set_xlabel("Prediksi")
    axis.set_ylabel("Aktual")
    axis.set_title("XGBoost FastText NLP-only: Confusion Matrix (Test)")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Latih dan evaluasi XGBoost NLP-only dengan FastText."
    )
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--article-splits", type=Path, default=DEFAULT_ARTICLE_SPLITS)
    parser.add_argument("--model-dataset", type=Path, default=DEFAULT_MODEL_DATASET)
    parser.add_argument("--output-prefix", default="08_fasttext")
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bucket-count", type=int, default=50_000)
    parser.add_argument("--min-ngram", type=int, default=3)
    parser.add_argument("--max-ngram", type=int, default=6)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)
    if min(
        args.vector_size,
        args.window,
        args.min_count,
        args.epochs,
        args.bucket_count,
        args.min_ngram,
    ) <= 0:
        parser.error("Embedding dimensions, counts, epochs and n-gram sizes must be positive.")
    if args.max_ngram < args.min_ngram:
        parser.error("--max-ngram must be greater than or equal to --min-ngram.")

    data = _validate_dataset(pd.read_csv(args.model_dataset))
    articles = load_article_text(args.articles, args.article_splits, data)
    model, embeddings = build_daily_embeddings(
        articles,
        data,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        bucket_count=args.bucket_count,
        min_n=args.min_ngram,
        max_n=args.max_ngram,
    )
    model.save(args.model_path)
    DEFAULT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = (
        DEFAULT_FEATURE_DIR / f"{args.output_prefix}_daily_features.csv"
    )
    embeddings.to_csv(feature_path, index=False)

    embedding_columns = [
        column for column in embeddings.columns if column.startswith("fasttext_")
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
        _add_result(results, "XGBoost FastText NLP-only", split, "clf", metrics)
        print(
            f"[FastText clf {split}] accuracy={metrics['accuracy']:.4f} "
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
        _add_result(results, "XGBoost FastText NLP-only", split, "reg", metrics)
        print(
            f"[FastText reg {split}] rmse={metrics['rmse']:.2f} "
            f"skill={metrics['skill_vs_naive']:+.4f}"
        )

    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = DEFAULT_RESULTS_DIR / f"{args.output_prefix}_comparison.csv"
    pd.DataFrame(results).round(6).to_csv(comparison_path, index=False)
    model_path = (
        str(args.model_path.relative_to(_REPO_ROOT))
        if args.model_path.is_relative_to(_REPO_ROOT)
        else str(args.model_path)
    )
    output = {
        "model": "XGBoost FastText NLP-only (tanpa fitur time-series)",
        "embedding": {
            "algorithm": "FastText Skip-gram with Negative Sampling (SGNS)",
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "bucket_count": args.bucket_count,
            "character_ngrams": [args.min_ngram, args.max_ngram],
            "training_articles": int((articles["split"] == "train").sum()),
            "vocabulary_size": len(model.token_to_id),
            "aggregation": "mean pooling per trading_date",
            "oov": "character n-gram vectors",
        },
        "features": {
            "lexicon": LEXICON_FEATURES,
            "fasttext_dimensions": len(embedding_columns),
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
            "embedding_model": model_path,
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
    print(f"FastText model saved: {args.model_path}")
    print(f"Results saved: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
