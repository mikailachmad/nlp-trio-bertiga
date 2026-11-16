"""Reusable LDA topic modeling utilities for geopolitical news articles."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class LDATopicModel:
    """Train-once LDA topic model with document-topic probability inference."""

    def __init__(
        self,
        n_topics: int = 10,
        max_features: int = 5_000,
        min_df: int = 3,
        max_df: float = 0.85,
        max_iter: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_topics = n_topics
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.max_iter = max_iter
        self.random_state = random_state
        self.vectorizer_: CountVectorizer | None = None
        self.model_: LatentDirichletAllocation | None = None

    def fit(self, documents: Iterable[str], y=None) -> LDATopicModel:
        """Fit vocabulary and LDA on the supplied training documents only."""
        documents = _normalize_documents(documents)
        if not documents or not any(document.strip() for document in documents):
            raise ValueError("LDA requires at least one non-empty training document.")
        if self.n_topics < 2:
            raise ValueError("n_topics must be at least 2.")
        if self.max_features < 1 or self.min_df < 1 or self.max_iter < 1:
            raise ValueError("max_features, min_df, and max_iter must be positive.")
        if not 0 < self.max_df <= 1:
            raise ValueError("max_df must be in the interval (0, 1].")

        vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r"(?u)\b\w\w+\b",
            lowercase=True,
        )
        document_term_matrix = vectorizer.fit_transform(documents)
        if document_term_matrix.shape[1] < self.n_topics:
            raise ValueError(
                f"Only {document_term_matrix.shape[1]} terms remain after "
                f"vectorization; cannot fit {self.n_topics} topics. Reduce "
                "min_df/n_topics or increase max_features."
            )

        model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method="batch",
            random_state=self.random_state,
            evaluate_every=-1,
        )
        model.fit(document_term_matrix)
        self.vectorizer_ = vectorizer
        self.model_ = model
        return self

    def transform(self, documents: Iterable[str]) -> np.ndarray:
        """Return one topic-probability row per document; empty/OOV docs are zero."""
        vectorizer, model = self._require_fitted()
        documents = _normalize_documents(documents)
        document_term_matrix = vectorizer.transform(documents)
        topic_probabilities = np.zeros(
            (len(documents), self.n_topics), dtype=np.float32
        )
        nonempty = np.asarray(document_term_matrix.getnnz(axis=1)).ravel() > 0
        if nonempty.any():
            topic_probabilities[nonempty] = model.transform(
                document_term_matrix[nonempty]
            ).astype(np.float32, copy=False)
        return topic_probabilities

    def fit_transform(self, documents: Iterable[str], y=None) -> np.ndarray:
        documents = _normalize_documents(documents)
        return self.fit(documents, y).transform(documents)

    def top_terms(self, n_terms: int = 15) -> list[list[str]]:
        """Return the highest-weight terms for each fitted topic."""
        vectorizer, model = self._require_fitted()
        if n_terms < 1:
            raise ValueError("n_terms must be positive.")
        terms = vectorizer.get_feature_names_out()
        return [
            terms[topic.argsort()[-n_terms:][::-1]].tolist()
            for topic in model.components_
        ]

    def save(self, path: str | Path) -> None:
        vectorizer, model = self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "n_topics": self.n_topics,
                "max_features": self.max_features,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                "vectorizer": vectorizer,
                "model": model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> LDATopicModel:
        artifact = joblib.load(Path(path))
        instance = cls(
            n_topics=artifact["n_topics"],
            max_features=artifact["max_features"],
            min_df=artifact["min_df"],
            max_df=artifact["max_df"],
            max_iter=artifact["max_iter"],
            random_state=artifact["random_state"],
        )
        instance.vectorizer_ = artifact["vectorizer"]
        instance.model_ = artifact["model"]
        return instance

    @property
    def vocabulary_size(self) -> int:
        vectorizer, _ = self._require_fitted()
        return len(vectorizer.vocabulary_)

    def _require_fitted(
        self,
    ) -> tuple[CountVectorizer, LatentDirichletAllocation]:
        if self.vectorizer_ is None or self.model_ is None:
            raise RuntimeError("LDATopicModel must be fitted or loaded before use.")
        return self.vectorizer_, self.model_


def select_topic_count(
    documents: Sequence[str],
    candidates: Sequence[int] = (8, 10, 12, 15),
    *,
    max_features: int = 5_000,
    min_df: int = 3,
    max_df: float = 0.85,
    max_iter: int = 20,
    random_state: int = 42,
    top_n_terms: int = 10,
) -> pd.DataFrame:
    """Rank candidate topic counts with UMass coherence on training documents."""
    documents = _normalize_documents(documents)
    if not documents or not any(document.strip() for document in documents):
        raise ValueError("Topic-count selection requires non-empty training documents.")
    candidate_counts = sorted(set(candidates))
    if not candidate_counts or candidate_counts[0] < 2:
        raise ValueError("Topic-count candidates must contain integers >= 2.")
    if top_n_terms < 2:
        raise ValueError("top_n_terms must be at least 2.")

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )
    document_term_matrix = vectorizer.fit_transform(documents)
    if document_term_matrix.shape[1] < candidate_counts[-1]:
        raise ValueError(
            f"Only {document_term_matrix.shape[1]} terms remain after "
            f"vectorization, fewer than the largest K={candidate_counts[-1]}."
        )

    binary_matrix = document_term_matrix.copy()
    binary_matrix.data = np.ones_like(binary_matrix.data)
    document_frequencies = np.asarray(binary_matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    records: list[dict[str, float | int]] = []
    for n_topics in candidate_counts:
        model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method="batch",
            random_state=random_state,
            evaluate_every=-1,
        )
        model.fit(document_term_matrix)
        top_indices = np.argsort(model.components_, axis=1)[:, -top_n_terms:][:, ::-1]
        topic_scores = []
        top_words = []
        for indices in top_indices:
            topic_words = [str(terms[index]) for index in indices]
            top_words.append(" | ".join(topic_words))
            pair_scores = []
            for later_rank in range(1, len(indices)):
                later_index = int(indices[later_rank])
                later_doc_frequency = document_frequencies[later_index]
                for earlier_rank in range(later_rank):
                    earlier_index = int(indices[earlier_rank])
                    cooccurrence = float(
                        binary_matrix[:, earlier_index]
                        .multiply(binary_matrix[:, later_index])
                        .sum()
                    )
                    pair_scores.append(
                        np.log(
                            (cooccurrence + 1.0)
                            / max(float(later_doc_frequency), 1.0)
                        )
                    )
            topic_scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)
        records.append(
            {
                "n_topics": n_topics,
                "umass_coherence": float(np.mean(topic_scores)),
                "top_terms": top_words,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["umass_coherence", "n_topics"], ascending=[False, True]
    ).reset_index(drop=True)


def _normalize_documents(documents: Iterable[str]) -> list[str]:
    if isinstance(documents, str):
        return [documents]
    normalized = []
    for document in documents:
        if document is None:
            normalized.append("")
        elif isinstance(document, str):
            normalized.append(document)
        else:
            raise TypeError(
                "LDA documents must be strings or None; "
                f"received {type(document).__name__}."
            )
    return normalized
