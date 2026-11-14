"""Word2Vec NLP-only experiment for next-day JISDOR prediction.

Word2Vec is trained on training-split articles only. Article vectors are mean
pooled by trading date and combined with the existing daily lexicon features.

Run from the repository root:
    python -m src.models.static_embeddings.word2vec_nlp_only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from src.jisdor.target_formulation import THRESHOLD_PCT  # noqa: E402
from src.models.metrics import (  # noqa: E402
    LABEL_ORDER,
    directional_hit_from_regression,
    evaluate_classifier,
    evaluate_regression,
)
from src.models.naive_baseline import predict_majority, predict_persistence  # noqa: E402
from src.models.static_embeddings import (  # noqa: E402
    _REPO_ROOT,
    DEFAULT_ARTICLE_SPLITS,
    DEFAULT_ARTICLES,
    DEFAULT_FEATURE_DIR,
    DEFAULT_MODEL_DATASET,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_DIR,
    ID2LABEL,
    LABEL2ID,
    TOKEN_RE,
    LEXICON_FEATURES,
    SPLITS,
    TRAIN_BATCH_SIZE,
    NEGATIVE_SAMPLES,
)


def tokenize(text: object) -> list[str]:
    """Return lowercase word tokens from a preprocessed article field."""
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


@dataclass
class Word2VecModel:
    """Skip-gram with negative-sampling input vectors and token vocabulary."""

    token_to_id: dict[str, int]
    vectors: np.ndarray
    counts: np.ndarray
    epochs: int
    window: int
    min_count: int
    negative_samples: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "algorithm": "Skip-gram with Negative Sampling (SGNS)",
            "epochs": self.epochs,
            "window": self.window,
            "min_count": self.min_count,
            "negative_samples": self.negative_samples,
            "vector_size": int(self.vectors.shape[1]),
        }
        tokens = np.asarray(
            [
                token
                for token, _ in sorted(
                    self.token_to_id.items(), key=lambda item: item[1]
                )
            ],
            dtype=str,
        )
        np.savez_compressed(
            path,
            tokens=tokens,
            vectors=self.vectors,
            counts=self.counts,
            metadata=np.array(json.dumps(metadata)),
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -15.0, 15.0)))


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    seed: int = 42,
) -> Word2VecModel:
    """Train deterministic Skip-gram with Negative Sampling on train sentences."""
    counts = Counter(token for sentence in sentences for token in sentence)
    vocabulary = {
        token: count for token, count in counts.items() if count >= min_count
    }
    if len(vocabulary) < 2:
        raise ValueError(
            "Word2Vec needs at least two training tokens after applying min_count."
        )

    ordered_tokens = sorted(vocabulary, key=lambda token: (-vocabulary[token], token))
    token_to_id = {token: index for index, token in enumerate(ordered_tokens)}
    token_counts = np.asarray(
        [vocabulary[token] for token in ordered_tokens], dtype=np.float64
    )
    total_tokens = token_counts.sum()
    keep_probability = np.minimum(
        1.0,
        (np.sqrt(token_counts / (1e-4 * total_tokens)) + 1)
        * (1e-4 * total_tokens / token_counts),
    )

    rng = np.random.default_rng(seed)
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
            "No Skip-gram training pairs remain after filtering/subsampling."
        )
    centers = np.concatenate(pair_centers)
    contexts = np.concatenate(pair_contexts)
    del pair_centers, pair_contexts

    vocabulary_size = len(ordered_tokens)
    input_vectors = rng.uniform(
        -0.5 / vector_size,
        0.5 / vector_size,
        size=(vocabulary_size, vector_size),
    ).astype(np.float32)
    output_vectors = np.zeros_like(input_vectors)
    negative_distribution = token_counts**0.75
    negative_distribution /= negative_distribution.sum()

    pair_count = centers.size
    for epoch in range(epochs):
        order = rng.permutation(pair_count)
        for start in range(0, pair_count, TRAIN_BATCH_SIZE):
            batch_indices = order[start : start + TRAIN_BATCH_SIZE]
            batch_centers = centers[batch_indices]
            batch_contexts = contexts[batch_indices]
            batch_size = batch_indices.size
            negatives = rng.choice(
                vocabulary_size,
                size=(batch_size, NEGATIVE_SAMPLES),
                p=negative_distribution,
            )

            center_vectors = input_vectors[batch_centers].copy()
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
                + np.einsum(
                    "ik,ikj->ij", negative_gradient, negative_vectors
                )
            )
            positive_vector_gradient = positive_gradient[:, None] * center_vectors
            negative_vector_gradient = (
                negative_gradient[:, :, None] * center_vectors[:, None, :]
            )

            progress = (epoch * pair_count + start) / (epochs * pair_count)
            learning_rate = max(0.0001, 0.025 * (1 - progress))
            np.add.at(
                input_vectors,
                batch_centers,
                -learning_rate * center_gradient,
            )
            np.add.at(
                output_vectors,
                batch_contexts,
                -learning_rate * positive_vector_gradient,
            )
            np.add.at(
                output_vectors,
                negatives.ravel(),
                -learning_rate * negative_vector_gradient.reshape(
                    -1, vector_size
                ),
            )
        print(
            f"[Word2Vec] epoch {epoch + 1}/{epochs}; "
            f"training pairs={pair_count:,}"
        )

    return Word2VecModel(
        token_to_id=token_to_id,
        vectors=input_vectors,
        counts=token_counts.astype(np.int64),
        epochs=epochs,
        window=window,
        min_count=min_count,
        negative_samples=NEGATIVE_SAMPLES,
    )


def make_classifier() -> xgb.XGBClassifier:
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
    required_text = {"url"}
    if not required_text.issubset(articles.columns):
        raise ValueError(f"{articles_path} must contain a `url` column.")
    if split_map["url"].duplicated().any():
        raise ValueError(f"{article_splits_path} contains duplicate article URLs.")
    if split_map["split"].isna().any() or not split_map["split"].isin(SPLITS).all():
        raise ValueError(
            f"{article_splits_path} split values must be train, val, or test."
        )

    text_fields = [
        column
        for column in ("title_bow", "body_bow", "title_clean", "body_clean")
        if column in articles.columns
    ]
    if not any(column in text_fields for column in ("title_bow", "title_clean")):
        raise ValueError(f"{articles_path} must contain title_bow or title_clean.")
    if not any(column in text_fields for column in ("body_bow", "body_clean")):
        raise ValueError(f"{articles_path} must contain body_bow or body_clean.")
    if "trading_date" not in articles.columns:
        raise ValueError(f"{articles_path} must contain a `trading_date` column.")

    articles = articles.merge(
        split_map,
        on="url",
        how="inner",
        validate="many_to_one",
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
        dates_by_split.rename(columns={"date": "trading_date", "split": "dataset_split"}),
        on="trading_date",
        how="inner",
        validate="many_to_one",
    )
    mismatch = articles["split"] != articles["dataset_split"]
    if mismatch.any():
        raise ValueError("Article split labels do not match model_dataset.csv.")
    return articles


def build_daily_embeddings(
    articles: pd.DataFrame,
    model_dataset: pd.DataFrame,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
) -> tuple[Word2VecModel, pd.DataFrame]:
    """Train on train articles and mean-pool article tokens by trading date."""
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
        raise ValueError("No non-empty training articles are available for Word2Vec.")

    model = train_word2vec(
        sentences=training_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
    )

    feature_columns = [f"word2vec_{index + 1:03d}" for index in range(vector_size)]
    daily = np.zeros((len(model_dataset), vector_size), dtype=np.float32)
    counts = np.zeros(len(model_dataset), dtype=np.int64)
    date_to_row = {
        date: index for index, date in enumerate(model_dataset["date"].tolist())
    }

    for row in articles.itertuples(index=False):
        row_index = date_to_row.get(row.trading_date)
        if row_index is None:
            continue
        token_vectors = [
            model.vectors[model.token_to_id[token]]
            for token in row.tokens
            if token in model.token_to_id
        ]
        if token_vectors:
            daily[row_index] += np.sum(token_vectors, axis=0)
            counts[row_index] += len(token_vectors)

    nonzero_counts = counts > 0
    daily[nonzero_counts] /= counts[nonzero_counts, None]
    features = pd.DataFrame(daily, columns=feature_columns)
    features.insert(0, "split", model_dataset["split"].to_numpy())
    features.insert(0, "date", model_dataset["date"].to_numpy())
    return model, features


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
    if data[list(required - {"date", "split", "target"})].isna().any().any():
        raise ValueError("Model dataset has missing numeric targets/features.")

    data = data.sort_values("date").reset_index(drop=True)
    split_ranges = data.groupby("split")["date"].agg(["min", "max"])
    if not set(SPLITS).issubset(split_ranges.index):
        raise ValueError("model_dataset.csv must contain train, val, and test rows.")
    if not (
        split_ranges.loc["train", "max"] < split_ranges.loc["val", "min"]
        and split_ranges.loc["val", "max"] < split_ranges.loc["test", "min"]
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


def _plot_confusion_matrix(matrix: Sequence[Sequence[int]], output_path: Path) -> None:
    cm = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(cm, cmap="Blues")
    axis.set_xticks(range(len(LABEL_ORDER)), labels=LABEL_ORDER)
    axis.set_yticks(range(len(LABEL_ORDER)), labels=LABEL_ORDER)
    axis.set_xlabel("Prediksi")
    axis.set_ylabel("Aktual")
    axis.set_title("XGBoost Word2Vec NLP-only: Confusion Matrix (Test)")
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
        description="Latih dan evaluasi XGBoost NLP-only dengan Word2Vec."
    )
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--article-splits", type=Path, default=DEFAULT_ARTICLE_SPLITS)
    parser.add_argument("--model-dataset", type=Path, default=DEFAULT_MODEL_DATASET)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-prefix", default="07_word2vec")
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)
    if min(args.vector_size, args.window, args.min_count, args.epochs) <= 0:
        parser.error("--vector-size, --window, --min-count, and --epochs must be positive.")

    data = _validate_dataset(pd.read_csv(args.model_dataset))
    articles = load_article_text(args.articles, args.article_splits, data)
    model, embeddings = build_daily_embeddings(
        articles,
        data,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(
        args.feature_dir / f"{args.output_prefix}_daily_features.csv", index=False
    )

    embedding_columns = [
        column for column in embeddings.columns if column.startswith("word2vec_")
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
        split: (splits[split]["kurs_besok"] - splits[split]["kurs_jisdor"]).astype(float)
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
        _add_result(results, "XGBoost Word2Vec NLP-only", split, "clf", metrics)
        print(
            f"[Word2Vec clf {split}] accuracy={metrics['accuracy']:.4f} "
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
        _add_result(results, "XGBoost Word2Vec NLP-only", split, "reg", metrics)
        print(
            f"[Word2Vec reg {split}] rmse={metrics['rmse']:.2f} "
            f"skill={metrics['skill_vs_naive']:+.4f}"
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.results_dir / f"{args.output_prefix}_comparison.csv"
    results_path = args.results_dir / f"{args.output_prefix}_results.json"
    confusion_path = args.results_dir / f"{args.output_prefix}_confusion_matrix.png"
    pd.DataFrame(results).round(6).to_csv(comparison_path, index=False)
    output = {
        "model": "XGBoost Word2Vec NLP-only (tanpa fitur time-series)",
        "embedding": {
            "algorithm": "Word2Vec Skip-gram with Negative Sampling (SGNS)",
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "training_articles": int((articles["split"] == "train").sum()),
            "vocabulary_size": len(model.token_to_id),
            "aggregation": "mean pooling per trading_date",
        },
        "features": {"lexicon": LEXICON_FEATURES, "word2vec_dimensions": len(embedding_columns)},
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
            "embedding_model": str(args.model_path.relative_to(_REPO_ROOT))
            if args.model_path.is_relative_to(_REPO_ROOT)
            else str(args.model_path),
            "daily_features": (
                str(args.feature_dir / f"{args.output_prefix}_daily_features.csv")
            ),
            "comparison": str(comparison_path),
            "results": str(results_path),
            "confusion_matrix": str(confusion_path),
        },
    }
    with results_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=False, indent=2)
    _plot_confusion_matrix(
        classification_metrics["test"]["confusion_matrix"],
        confusion_path,
    )
    print(
        f"Word2Vec model saved: {args.model_path}\n"
        f"Results saved under {args.results_dir / args.output_prefix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
