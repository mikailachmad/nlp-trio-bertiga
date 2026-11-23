"""Train 1,000-dimensional Word2Vec, FastText, and GloVe word embeddings.

Word2Vec and FastText are trained with Gensim. Gensim does not implement a
GloVe trainer, so the GloVe model is trained with the Mittens implementation
of the original GloVe objective.

Run from the repository root:
    python -m src.models.static_embeddings.gensim_1000d
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import FastText, Word2Vec
from mittens import GloVe

from src.models.static_embeddings import (
    DEFAULT_ARTICLE_SPLITS,
    DEFAULT_ARTICLES,
    _REPO_ROOT,
    tokenize,
)

DEFAULT_ARTIFACT_DIR = (
    _REPO_ROOT / "data" / "features" / "artifacts" / "static_embeddings_1000d"
)
DEFAULT_FEATURE_DIR = (
    _REPO_ROOT / "data" / "feature_extracted" / "static_embeddings_1000d"
)
SPLITS = ("train", "val", "test")
VECTOR_SIZE = 1000
SEED = 42


def load_clean_split_articles(
    articles_path: Path, article_splits_path: Path
) -> pd.DataFrame:
    """Load articles with valid date and split labels, rejecting split leakage."""
    articles = pd.read_csv(
        articles_path,
        usecols=["url", "trading_date", "title_bow", "body_bow"],
    )
    split_map = pd.read_csv(article_splits_path, usecols=["url", "split"])
    valid_split = split_map["split"].isin(SPLITS)
    invalid_split_rows = int((~valid_split).sum())
    split_map = split_map.loc[valid_split].copy()

    conflicting_urls = (
        split_map.groupby("url", dropna=False)["split"].nunique().gt(1)
    )
    if conflicting_urls.any():
        examples = conflicting_urls[conflicting_urls].index[:5].tolist()
        raise ValueError(
            "Article split file assigns multiple splits to the same URL; "
            f"examples: {examples}"
        )
    split_map = split_map.drop_duplicates(subset="url")
    if split_map["url"].isna().any():
        raise ValueError("Article split file has missing URLs in valid split rows.")

    articles["trading_date"] = pd.to_datetime(
        articles["trading_date"], errors="coerce"
    ).dt.normalize()
    invalid_date_rows = int(articles["trading_date"].isna().sum())
    articles = articles.loc[articles["trading_date"].notna()].copy()
    articles = articles.merge(
        split_map,
        on="url",
        how="inner",
        validate="many_to_one",
    )
    if articles.empty:
        raise ValueError("No articles with valid dates and split labels were found.")
    if articles[["title_bow", "body_bow"]].isna().any().any():
        raise ValueError("Aligned article text is missing for valid split records.")

    split_counts = articles.groupby("split").size().to_dict()
    missing_splits = set(SPLITS).difference(split_counts)
    if missing_splits:
        raise ValueError(f"No valid articles found for splits: {sorted(missing_splits)}")
    date_split_counts = articles.groupby("trading_date")["split"].nunique()
    if date_split_counts.gt(1).any():
        dates = date_split_counts[date_split_counts.gt(1)].index[:5].tolist()
        raise ValueError(
            "One trading date maps to multiple data splits; "
            f"examples: {dates}"
        )

    if invalid_split_rows or invalid_date_rows:
        print(
            "WARNING: excluded malformed source rows: "
            f"{invalid_split_rows:,} rows with invalid split labels and "
            f"{invalid_date_rows:,} articles with invalid dates."
        )
    print(
        "Using valid split articles: "
        + ", ".join(f"{split}={split_counts[split]:,}" for split in SPLITS)
    )
    return articles.sort_values(["trading_date", "url"]).reset_index(drop=True)


def _article_sentences(articles: pd.DataFrame) -> list[list[str]]:
    return [
        tokenize(
            " ".join(
                text
                for text in (row.title_bow, row.body_bow)
                if isinstance(text, str)
            )
        )
        for row in articles.itertuples(index=False)
    ]


def train_gensim_models(
    sentences: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    workers: int,
    fasttext_bucket: int,
) -> tuple[Word2Vec, FastText]:
    """Train SGNS Word2Vec and FastText models with Gensim."""
    training_sentences = [sentence for sentence in sentences if sentence]
    if not training_sentences:
        raise ValueError("No non-empty training sentences are available.")

    word2vec = Word2Vec(
        sentences=training_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        negative=5,
        seed=SEED,
        epochs=epochs,
    )
    fasttext = FastText(
        sentences=training_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        negative=5,
        seed=SEED,
        epochs=epochs,
        min_n=3,
        max_n=6,
        bucket=fasttext_bucket,
    )
    if word2vec.wv.vector_size != vector_size:
        raise RuntimeError("Gensim Word2Vec returned an unexpected vector size.")
    if fasttext.wv.vector_size != vector_size:
        raise RuntimeError("Gensim FastText returned an unexpected vector size.")
    return word2vec, fasttext


def _build_glove_cooccurrence(
    sentences: list[list[str]],
    min_count: int,
    window: int,
    max_vocab: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build a dense, symmetric distance-weighted co-occurrence matrix."""
    dictionary = Dictionary(sentences)
    dictionary.filter_extremes(no_below=min_count, no_above=1.0, keep_n=max_vocab)
    tokens = [dictionary[index] for index in range(len(dictionary))]
    if len(tokens) < 2:
        raise ValueError("GloVe requires at least two tokens in its vocabulary.")

    cooccurrence = np.zeros((len(tokens), len(tokens)), dtype=np.float32)
    for sentence in sentences:
        ids = np.fromiter(
            (
                dictionary.token2id[token]
                for token in sentence
                if token in dictionary.token2id
            ),
            dtype=np.int32,
        )
        for distance in range(1, min(window, max(0, ids.size - 1)) + 1):
            left = ids[:-distance]
            right = ids[distance:]
            np.add.at(cooccurrence, (left, right), 1.0 / distance)
            np.add.at(cooccurrence, (right, left), 1.0 / distance)

    counts = np.asarray(
        [dictionary.cfs[dictionary.token2id[token]] for token in tokens],
        dtype=np.int64,
    )
    return tokens, counts, cooccurrence


def train_glove(
    sentences: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    max_vocab: int,
    epochs: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Train a 1,000d GloVe model from train-split co-occurrence counts."""
    tokens, counts, cooccurrence = _build_glove_cooccurrence(
        sentences,
        min_count=min_count,
        window=window,
        max_vocab=max_vocab,
    )
    np.random.seed(SEED)
    random.seed(SEED)
    model = GloVe(
        n=vector_size,
        max_iter=epochs,
        learning_rate=0.05,
        display_progress=10,
    )
    vectors = np.asarray(model.fit(cooccurrence), dtype=np.float32)
    if vectors.shape != (len(tokens), vector_size):
        raise RuntimeError(
            f"GloVe returned shape {vectors.shape}; "
            f"expected {(len(tokens), vector_size)}."
        )
    return tokens, counts, vectors


def _save_vectors(
    path: Path,
    tokens: list[str],
    vectors: np.ndarray,
    counts: np.ndarray,
    metadata: dict,
) -> None:
    if vectors.shape != (len(tokens), VECTOR_SIZE):
        raise ValueError(
            f"Expected a {VECTOR_SIZE}-dimensional vector matrix; got {vectors.shape}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tokens=np.asarray(tokens, dtype=str),
        vectors=vectors.astype(np.float32, copy=False),
        counts=counts.astype(np.int64, copy=False),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def _save_daily_features(
    articles: pd.DataFrame,
    vectors_by_model: dict[str, tuple[set[str], object]],
    feature_dir: Path,
) -> dict[str, dict[str, float | int]]:
    """Mean-pool token vectors by date and save a separate 1,000d CSV per model."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    coverage: dict[str, dict[str, float | int]] = {}
    article_tokens = _article_sentences(articles)

    for model_name, (vocabulary, vector_lookup) in vectors_by_model.items():
        daily_sums: dict[pd.Timestamp, np.ndarray] = {}
        daily_counts: Counter[pd.Timestamp] = Counter()
        split_by_date: dict[pd.Timestamp, str] = {}
        token_total = 0
        token_known = 0
        token_vectors_used = 0

        for row, tokens in zip(articles.itertuples(index=False), article_tokens):
            date = row.trading_date
            split_by_date[date] = row.split
            token_total += len(tokens)
            if model_name == "fasttext":
                known_tokens = tokens
                token_known += sum(token in vocabulary for token in tokens)
            else:
                known_tokens = [token for token in tokens if token in vocabulary]
                token_known += len(known_tokens)
            token_vectors_used += len(known_tokens)
            if not known_tokens:
                continue
            vectors = np.asarray(
                [vector_lookup(token) for token in known_tokens], dtype=np.float32
            )
            daily_sums.setdefault(date, np.zeros(VECTOR_SIZE, dtype=np.float64))
            daily_sums[date] += vectors.sum(axis=0, dtype=np.float64)
            daily_counts[date] += len(known_tokens)

        dates = sorted(split_by_date)
        rows = np.zeros((len(dates), VECTOR_SIZE), dtype=np.float32)
        for index, date in enumerate(dates):
            if daily_counts[date]:
                rows[index] = daily_sums[date] / daily_counts[date]
        feature_data = pd.DataFrame(
            rows,
            columns=[f"{model_name}_{index + 1:04d}" for index in range(VECTOR_SIZE)],
        )
        feature_data.insert(0, "split", [split_by_date[date] for date in dates])
        feature_data.insert(0, "date", dates)
        feature_path = feature_dir / f"{model_name}_daily_1000d.csv"
        feature_data.to_csv(feature_path, index=False)
        coverage[model_name] = {
            "tokens_total": token_total,
            "tokens_in_vocabulary": token_known,
            "tokens_with_vectors": token_vectors_used,
            "token_coverage": round(token_vectors_used / token_total, 6)
            if token_total
            else 0.0,
            "days": len(dates),
            "days_without_known_tokens": sum(
                daily_counts[date] == 0 for date in dates
            ),
            "daily_features": str(feature_path),
        }
    return coverage


def train_and_save(
    articles_path: Path = DEFAULT_ARTICLES,
    article_splits_path: Path = DEFAULT_ARTICLE_SPLITS,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    vector_size: int = VECTOR_SIZE,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 5,
    workers: int = 1,
    fasttext_bucket: int = 50_000,
    glove_max_vocab: int = 5_000,
    glove_epochs: int = 30,
) -> dict:
    if vector_size != VECTOR_SIZE:
        raise ValueError(f"This run is configured to produce {VECTOR_SIZE}d vectors.")
    if min(window, min_count, epochs, workers, fasttext_bucket, glove_max_vocab, glove_epochs) <= 0:
        raise ValueError("Training parameters must be positive.")

    articles = load_clean_split_articles(articles_path, article_splits_path)
    articles["tokens"] = _article_sentences(articles)
    train_articles = articles.loc[articles["split"] == "train"]
    training_sentences = [
        sentence for sentence in train_articles["tokens"].tolist() if sentence
    ]
    if not training_sentences:
        raise ValueError("No non-empty train articles are available.")
    print(
        f"Training on {len(training_sentences):,} train articles "
        f"and {sum(map(len, training_sentences)):,} tokens."
    )

    word2vec, fasttext = train_gensim_models(
        training_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=workers,
        fasttext_bucket=fasttext_bucket,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    word2vec_path = artifact_dir / "word2vec_gensim_1000d.model"
    word2vec.save(str(word2vec_path))
    _save_vectors(
        artifact_dir / "word2vec_gensim_1000d.npz",
        word2vec.wv.index_to_key,
        word2vec.wv.vectors,
        np.asarray(
            [word2vec.wv.get_vecattr(token, "count") for token in word2vec.wv.index_to_key]
        ),
        {
            "algorithm": "Gensim Word2Vec Skip-gram with Negative Sampling",
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "epochs": epochs,
            "seed": SEED,
        },
    )

    fasttext_path = artifact_dir / "fasttext_gensim_1000d.model"
    fasttext.save(str(fasttext_path))
    _save_vectors(
        artifact_dir / "fasttext_gensim_1000d.npz",
        fasttext.wv.index_to_key,
        fasttext.wv.vectors,
        np.asarray(
            [fasttext.wv.get_vecattr(token, "count") for token in fasttext.wv.index_to_key]
        ),
        {
            "algorithm": "Gensim FastText Skip-gram with Negative Sampling",
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "epochs": epochs,
            "bucket": fasttext_bucket,
            "character_ngrams": [3, 6],
            "seed": SEED,
        },
    )

    glove_tokens, glove_counts, glove_vectors = train_glove(
        training_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        max_vocab=glove_max_vocab,
        epochs=glove_epochs,
    )
    _save_vectors(
        artifact_dir / "glove_mittens_1000d.npz",
        glove_tokens,
        glove_vectors,
        glove_counts,
        {
            "algorithm": "GloVe (Mittens, original weighted least-squares objective)",
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "max_vocab": glove_max_vocab,
            "epochs": glove_epochs,
            "seed": SEED,
        },
    )

    glove_lookup = {token: glove_vectors[index] for index, token in enumerate(glove_tokens)}
    vectors_by_model = {
        "word2vec": (
            set(word2vec.wv.key_to_index),
            lambda token: word2vec.wv[token],
        ),
        "fasttext": (
            set(fasttext.wv.key_to_index),
            lambda token: fasttext.wv[token],
        ),
        "glove": (set(glove_lookup), lambda token: glove_lookup[token]),
    }
    coverage = _save_daily_features(articles, vectors_by_model, feature_dir)
    summary = {
        "vector_size": vector_size,
        "seed": SEED,
        "training": {
            "train_articles": len(training_sentences),
            "train_tokens": sum(map(len, training_sentences)),
            "split_articles": articles["split"].value_counts().to_dict(),
            "window": window,
            "min_count": min_count,
            "gensim_epochs": epochs,
            "gensim_workers": workers,
            "fasttext_bucket": fasttext_bucket,
            "glove_vocabulary_limit": glove_max_vocab,
            "glove_epochs": glove_epochs,
        },
        "models": {
            "word2vec": {
                "vocabulary_size": len(word2vec.wv),
                "model": str(word2vec_path),
                "vectors": str(artifact_dir / "word2vec_gensim_1000d.npz"),
                **coverage["word2vec"],
            },
            "fasttext": {
                "vocabulary_size": len(fasttext.wv),
                "model": str(fasttext_path),
                "vectors": str(artifact_dir / "fasttext_gensim_1000d.npz"),
                **coverage["fasttext"],
            },
            "glove": {
                "vocabulary_size": len(glove_tokens),
                "vectors": str(artifact_dir / "glove_mittens_1000d.npz"),
                **coverage["glove"],
            },
        },
    }
    summary_path = artifact_dir / "static_embeddings_1000d_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved 1,000d embedding artifacts and summary to {artifact_dir}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train 1,000-dimensional Word2Vec, FastText, and GloVe embeddings."
    )
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--article-splits", type=Path, default=DEFAULT_ARTICLE_SPLITS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fasttext-bucket", type=int, default=50_000)
    parser.add_argument("--glove-max-vocab", type=int, default=5_000)
    parser.add_argument("--glove-epochs", type=int, default=30)
    args = parser.parse_args(argv)
    train_and_save(
        articles_path=args.articles,
        article_splits_path=args.article_splits,
        artifact_dir=args.artifact_dir,
        feature_dir=args.feature_dir,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
        fasttext_bucket=args.fasttext_bucket,
        glove_max_vocab=args.glove_max_vocab,
        glove_epochs=args.glove_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
