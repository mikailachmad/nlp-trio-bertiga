"""Reusable static embeddings and experiment entry points.

The embedding transformers expose a common ``fit``/``transform`` interface.
Use :func:`make_embedding` to select an embedding by name, or
:func:`run_embedding_experiment` to execute its complete NLP-only experiment.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin

from src.models.metrics import LABEL_ORDER

_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ARTICLES = _REPO_ROOT / "data" / "processed" / "articles_aligned.csv"
DEFAULT_ARTICLE_SPLITS = _REPO_ROOT / "data" / "split" / "articles_split.csv"
DEFAULT_MODEL_DATASET = _REPO_ROOT / "data" / "feature_extracted" / "model_dataset.csv"
DEFAULT_FEATURE_DIR = _REPO_ROOT / "data" / "feature_extracted"
DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"
DEFAULT_MODEL_PATH = (
    _REPO_ROOT / "data" / "features" / "artifacts" / "word2vec_sgns.npz"
)

LEXICON_FEATURES = [
    "vader_title_mean",
    "vader_body_mean",
    "vader_body_min",
    "vader_neg_share",
    "lm_negative_mean",
    "lm_positive_mean",
    "lm_uncertainty_mean",
    "lm_litigious_mean",
    "lm_constraining_mean",
    "lm_net_mean",
    "no_news",
]
LABEL2ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE)
SPLITS = ("train", "val", "test")
NEGATIVE_SAMPLES = 5
TRAIN_BATCH_SIZE = 4096

_EMBEDDING_EXPERIMENTS = {
    "word2vec": "src.models.static_embeddings.word2vec_nlp_only",
    "fasttext": "src.models.static_embeddings.fasttext_nlp_only",
    "glove": "src.models.static_embeddings.glove_nlp_only",
}


def tokenize(text: object) -> list[str]:
    """Tokenize text with the shared embedding tokenizer."""
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def _documents(documents: Iterable[str | Sequence[str]] | str | Sequence[str]):
    if isinstance(documents, str):
        return [documents]
    items = list(documents)
    if items and all(isinstance(item, str) for item in items):
        return items
    return items


def _tokens(document: str | Sequence[str]) -> list[str]:
    if isinstance(document, str):
        return tokenize(document)
    return [token.lower() for token in document if isinstance(token, str)]


def _vector_matrix(vectors: list[np.ndarray], dimension: int) -> np.ndarray:
    if not vectors:
        return np.empty((0, dimension), dtype=np.float32)
    return np.stack(vectors).astype(np.float32, copy=False)


def make_classifier() -> xgb.XGBClassifier:
    """Build the shared XGBoost classifier used by embedding experiments."""
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABEL_ORDER),
        eval_metric="mlogloss",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )


def make_regressor() -> xgb.XGBRegressor:
    """Build the shared XGBoost regressor used by embedding experiments."""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )


def load_article_text(
    articles_path: Path,
    article_splits_path: Path,
    model_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the pipeline's article split labels to aligned article text."""
    articles = pd.read_csv(articles_path)
    split_map = pd.read_csv(article_splits_path, usecols=["url", "split"])
    if "url" not in articles.columns:
        raise ValueError(f"{articles_path} must contain a `url` column.")
    if split_map["url"].duplicated().any():
        raise ValueError(f"{article_splits_path} contains duplicate article URLs.")
    if split_map["split"].isna().any() or not split_map["split"].isin(SPLITS).all():
        raise ValueError(
            f"{article_splits_path} split values must be train, val, or test."
        )

    text_columns = set(articles.columns)
    if not (
        {"title_bow", "body_bow"}.issubset(text_columns)
        or {"title_clean", "body_clean"}.issubset(text_columns)
    ):
        raise ValueError(
            f"{articles_path} must contain title/body bow or clean text columns."
        )
    if "trading_date" not in articles.columns:
        raise ValueError(f"{articles_path} must contain a `trading_date` column.")

    articles = articles.merge(
        split_map, on="url", how="inner", validate="many_to_one"
    )
    articles["trading_date"] = pd.to_datetime(
        articles["trading_date"], errors="coerce"
    ).dt.normalize()
    if articles["trading_date"].isna().any():
        raise ValueError(
            f"{articles_path} has split articles with invalid trading dates."
        )

    dates_by_split = model_dataset[["date", "split"]].drop_duplicates()
    articles = articles.merge(
        dates_by_split.rename(
            columns={"date": "trading_date", "split": "dataset_split"}
        ),
        on="trading_date",
        how="inner",
        validate="many_to_one",
    )
    if (articles["split"] != articles["dataset_split"]).any():
        raise ValueError("Article split labels do not match model_dataset.csv.")
    return articles


def _validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "split",
        "target",
        "kurs_jisdor",
        "kurs_besok",
        *LEXICON_FEATURES,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"model_dataset.csv is missing required columns: {missing}")
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if data["date"].isna().any():
        raise ValueError("model_dataset.csv contains invalid dates.")
    if data["date"].duplicated().any():
        raise ValueError("model_dataset.csv must contain one row per date.")
    if not data["split"].isin(SPLITS).all():
        raise ValueError("model_dataset.csv split values must be train, val, or test.")
    if data["target"].isna().any() or not data["target"].isin(LABEL_ORDER).all():
        raise ValueError(f"target values must be in {LABEL_ORDER}.")
    numeric_columns = required - {"date", "split", "target"}
    if data[list(numeric_columns)].isna().any().any():
        raise ValueError("Model dataset has missing numeric targets/features.")

    data = data.sort_values("date").reset_index(drop=True)
    ranges = data.groupby("split")["date"].agg(["min", "max"])
    if not set(SPLITS).issubset(ranges.index):
        raise ValueError("model_dataset.csv must contain train, val, and test rows.")
    if not (
        ranges.loc["train", "max"] < ranges.loc["val", "min"]
        and ranges.loc["val", "max"] < ranges.loc["test", "min"]
    ):
        raise ValueError("Train, validation, and test must be chronological.")
    return data


def _add_result(
    results: list[dict], model_name: str, split: str, task: str, metrics: dict
) -> None:
    results.append(
        {
            "model": model_name,
            "split": split,
            "task": task,
            **{
                key: value
                for key, value in metrics.items()
                if key not in {"classification_report", "confusion_matrix"}
            },
        }
    )


class Word2VecEmbedding(BaseEstimator, TransformerMixin):
    """Trainable Word2Vec SGNS transformer for integration into NLP pipelines."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 5,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.model_ = None

    def fit(self, X, y=None):
        from .word2vec_nlp_only import train_word2vec

        self.model_ = train_word2vec(
            [_tokens(document) for document in _documents(X)],
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.seed,
        )
        return self

    def transform(self, X) -> np.ndarray:
        model = self._require_fitted()
        vectors = []
        for document in _documents(X):
            token_vectors = [
                model.vectors[model.token_to_id[token]]
                for token in _tokens(document)
                if token in model.token_to_id
            ]
            vectors.append(
                np.mean(token_vectors, axis=0, dtype=np.float32)
                if token_vectors
                else np.zeros(self.vector_size, dtype=np.float32)
            )
        return _vector_matrix(vectors, self.vector_size)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, path: str | Path) -> None:
        self._require_fitted().save(Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Word2VecEmbedding:
        from .word2vec_nlp_only import Word2VecModel

        with np.load(path, allow_pickle=False) as artifact:
            tokens = artifact["tokens"].tolist()
            vectors = artifact["vectors"]
            counts = artifact["counts"]
            metadata = json.loads(artifact["metadata"].item())
        instance = cls(
            vector_size=int(metadata["vector_size"]),
            window=int(metadata["window"]),
            min_count=int(metadata["min_count"]),
            epochs=int(metadata["epochs"]),
        )
        instance.model_ = Word2VecModel(
            token_to_id={token: index for index, token in enumerate(tokens)},
            vectors=vectors,
            counts=counts,
            epochs=instance.epochs,
            window=instance.window,
            min_count=instance.min_count,
            negative_samples=int(metadata["negative_samples"]),
        )
        return instance

    @property
    def vocabulary_size(self) -> int:
        return len(self._require_fitted().token_to_id)

    def _require_fitted(self):
        if self.model_ is None:
            raise RuntimeError("Word2VecEmbedding must be fitted before use.")
        return self.model_


class FastTextEmbedding(BaseEstimator, TransformerMixin):
    """Trainable FastText-style SGNS transformer with character subwords."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 5,
        bucket_count: int = 50_000,
        min_n: int = 3,
        max_n: int = 6,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.bucket_count = bucket_count
        self.min_n = min_n
        self.max_n = max_n
        self.seed = seed
        self.model_ = None

    def fit(self, X, y=None):
        from .fasttext_nlp_only import train_fasttext

        self.model_ = train_fasttext(
            [_tokens(document) for document in _documents(X)],
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            bucket_count=self.bucket_count,
            min_n=self.min_n,
            max_n=self.max_n,
            seed=self.seed,
        )
        return self

    def transform(self, X) -> np.ndarray:
        model = self._require_fitted()
        vectors = []
        for document in _documents(X):
            tokens = _tokens(document)
            vectors.append(
                np.mean(
                    [model.vector_for(token) for token in tokens],
                    axis=0,
                    dtype=np.float32,
                )
                if tokens
                else np.zeros(self.vector_size, dtype=np.float32)
            )
        return _vector_matrix(vectors, self.vector_size)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, path: str | Path) -> None:
        self._require_fitted().save(Path(path))

    @classmethod
    def load(cls, path: str | Path) -> FastTextEmbedding:
        from .fasttext_nlp_only import FastTextModel

        model = FastTextModel.load(Path(path))
        instance = cls(
            vector_size=model.subword_vectors.shape[1],
            window=model.window,
            min_count=model.min_count,
            epochs=model.epochs,
            bucket_count=model.bucket_count,
            min_n=model.min_n,
            max_n=model.max_n,
        )
        instance.model_ = model
        return instance

    @property
    def vocabulary_size(self) -> int:
        return len(self._require_fitted().token_to_id)

    def _require_fitted(self):
        if self.model_ is None:
            raise RuntimeError("FastTextEmbedding must be fitted before use.")
        return self.model_


class GloVeEmbedding(BaseEstimator, TransformerMixin):
    """Pretrained GloVe 100d transformer with local vector caching."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        glove_path: str | Path | None = None,
        archive_path: str | Path | None = None,
        download: bool = True,
    ) -> None:
        self.cache_path = cache_path
        self.glove_path = glove_path
        self.archive_path = archive_path
        self.download = download
        self.vectors_ = None
        self.source_ = None
        self.missing_words_ = set()

    def fit(self, X, y=None):
        from .glove_nlp_only import (
            DEFAULT_ARCHIVE,
            DEFAULT_GLOVE_CACHE,
            load_pretrained_vectors,
        )

        documents = [_tokens(document) for document in _documents(X)]
        needed = {token for document in documents for token in document}
        if not needed:
            raise ValueError("GloVeEmbedding requires at least one token.")
        cache_path = Path(self.cache_path or DEFAULT_GLOVE_CACHE)
        archive_path = Path(self.archive_path or DEFAULT_ARCHIVE)
        local_path = Path(self.glove_path) if self.glove_path is not None else None
        self.vectors_, self.source_ = load_pretrained_vectors(
            needed_tokens=needed,
            cache_path=cache_path,
            local_glove_path=local_path,
            archive_path=archive_path,
            download=self.download,
        )
        self.missing_words_ = needed.difference(self.vectors_.vectors)
        return self

    def transform(self, X) -> np.ndarray:
        vectors = self._require_fitted()
        dimensions = 100
        rows = []
        for document in _documents(X):
            known = [
                vectors.vectors[token]
                for token in _tokens(document)
                if token in vectors.vectors
            ]
            rows.append(
                np.mean(known, axis=0, dtype=np.float32)
                if known
                else np.zeros(dimensions, dtype=np.float32)
            )
        return _vector_matrix(rows, dimensions)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @classmethod
    def load(cls, path: str | Path) -> GloVeEmbedding:
        from .glove_nlp_only import GloveVectors

        instance = cls(cache_path=path, download=False)
        instance.vectors_ = GloveVectors.load(Path(path))
        instance.source_ = instance.vectors_.source
        return instance

    @property
    def vocabulary_size(self) -> int:
        return len(self._require_fitted().vectors)

    def _require_fitted(self):
        if self.vectors_ is None:
            raise RuntimeError("GloVeEmbedding must be fitted or loaded before use.")
        return self.vectors_


def make_embedding(name: str, **kwargs):
    """Create one embedding transformer by canonical method name."""
    normalized_name = name.strip().lower()
    embedding_types = {
        "word2vec": Word2VecEmbedding,
        "fasttext": FastTextEmbedding,
        "glove": GloVeEmbedding,
    }
    try:
        embedding_type = embedding_types[normalized_name]
    except KeyError as exc:
        supported = ", ".join(sorted(embedding_types))
        raise ValueError(
            f"Unknown static embedding {name!r}; choose from: {supported}."
        ) from exc
    return embedding_type(**kwargs)


def run_embedding_experiment(name: str, argv: Sequence[str] | None = None) -> int:
    """Run the selected end-to-end NLP-only experiment from the main module."""
    normalized_name = name.strip().lower()
    try:
        module_name = _EMBEDDING_EXPERIMENTS[normalized_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_EMBEDDING_EXPERIMENTS))
        raise ValueError(
            f"Unknown static embedding experiment {name!r}; choose from: {supported}."
        ) from exc
    module = importlib.import_module(module_name)
    return module.main(list(argv) if argv is not None else None)


__all__ = [
    "Word2VecEmbedding",
    "FastTextEmbedding",
    "GloVeEmbedding",
    "make_embedding",
    "run_embedding_experiment",
    "DEFAULT_ARTICLES",
    "DEFAULT_ARTICLE_SPLITS",
    "DEFAULT_MODEL_DATASET",
    "DEFAULT_FEATURE_DIR",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_MODEL_PATH",
    "LEXICON_FEATURES",
    "LABEL2ID",
    "ID2LABEL",
    "LABEL_ORDER",
    "SPLITS",
    "TOKEN_RE",
    "NEGATIVE_SAMPLES",
    "TRAIN_BATCH_SIZE",
    "tokenize",
    "make_classifier",
    "make_regressor",
    "load_article_text",
    "_validate_dataset",
    "_add_result",
    "_REPO_ROOT",
]
