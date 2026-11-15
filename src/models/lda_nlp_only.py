"""LDA geopolitical topic features and incremental USD/IDR evaluation.

Run from the repository root:
    python -m src.models.lda_nlp_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from src.jisdor.target_formulation import THRESHOLD_PCT
from src.models.metrics import (
    LABEL_ORDER,
    directional_hit_from_regression,
    evaluate_classifier,
    evaluate_regression,
)
from src.models.naive_baseline import predict_majority, predict_persistence
from src.models.static_embeddings import (
    DEFAULT_ARTICLE_SPLITS,
    DEFAULT_ARTICLES,
    DEFAULT_FEATURE_DIR,
    DEFAULT_MODEL_DATASET,
    DEFAULT_RESULTS_DIR,
    ID2LABEL,
    LABEL2ID,
    LEXICON_FEATURES,
    SPLITS,
    _REPO_ROOT,
    load_article_text,
    make_classifier,
    make_regressor,
    _validate_dataset,
)
from src.models.topic_modeling import LDATopicModel, select_topic_count


DEFAULT_MODEL_PATH = (
    _REPO_ROOT / "data" / "features" / "artifacts" / "lda_geopolitical_topics.joblib"
)


def build_article_text(articles: pd.DataFrame) -> pd.DataFrame:
    """Build a consistent title+body text field and retain article metadata."""
    text_column_pairs = (
        ("title_bow", "body_bow"),
        ("title_clean", "body_clean"),
    )
    title_col, body_col = next(
        (pair for pair in text_column_pairs if set(pair).issubset(articles.columns)),
        (None, None),
    )
    if title_col is None:
        raise ValueError("Aligned articles need title/body BoW or clean text columns.")
    required = {"url", "trading_date", "split"}
    missing = required.difference(articles.columns)
    if missing:
        raise ValueError(f"Aligned articles are missing columns: {sorted(missing)}")

    frame = articles[["url", "trading_date", "split", title_col, body_col]].copy()
    frame["title"] = frame[title_col].fillna("").astype(str)
    frame["document"] = (
        frame["title"]
        + " "
        + frame[body_col].fillna("").astype(str)
    ).str.strip()
    return frame[["url", "trading_date", "split", "title", "document"]]


def _topic_count_candidates(value: str) -> list[int]:
    try:
        candidates = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Topic count must be comma-separated integers, e.g. 8,10,12,15."
        ) from exc
    if not candidates or candidates[0] < 2:
        raise argparse.ArgumentTypeError("Each topic count must be at least 2.")
    return candidates


def _plain_records(frame: pd.DataFrame) -> list[dict]:
    """Convert selection output into JSON-safe scalar/list values."""
    records = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                key: value if not isinstance(value, np.generic) else value.item()
                for key, value in row.items()
            }
        )
    return records


def _fit_and_score_models(
    model_data: pd.DataFrame,
    topic_features: pd.DataFrame,
    topic_columns: list[str],
) -> tuple[list[dict], dict, dict]:
    svd_columns = sorted(
        [column for column in model_data.columns if column.startswith("tfidf_svd_")],
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    baseline_features = LEXICON_FEATURES + svd_columns
    required = baseline_features + [
        "date",
        "split",
        "target",
        "kurs_jisdor",
        "kurs_besok",
    ]
    missing = sorted(set(required).difference(model_data.columns))
    if missing:
        raise ValueError(f"Model dataset is missing required columns: {missing}")
    if not svd_columns:
        raise ValueError("Model dataset contains no tfidf_svd_* baseline features.")

    data = model_data.merge(
        topic_features,
        on=["date", "split"],
        how="left",
        validate="one_to_one",
    )
    if data[topic_columns].isna().any().any():
        raise ValueError("Topic features are missing after date/split alignment.")
    data = data.sort_values("date").reset_index(drop=True)
    splits = {
        split: data[data["split"] == split].sort_values("date").copy()
        for split in SPLITS
    }
    X = {
        "Baseline NLP": {
            split: splits[split][baseline_features] for split in SPLITS
        },
        "Baseline NLP + LDA topics": {
            split: splits[split][baseline_features + topic_columns]
            for split in SPLITS
        },
    }
    y_class = {
        split: splits[split]["target"].map(LABEL2ID).astype(int)
        for split in SPLITS
    }
    y_next = {
        split: splits[split]["kurs_besok"].astype(float).to_numpy()
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
    classification_results: dict[str, dict] = {}
    regression_results: dict[str, dict] = {}
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
            predict_majority(splits["train"]["target"], len(sample)),
            include_kappa=True,
        )
        for model_name, metrics in (
            ("Naive Persistence of Direction", persistence),
            ("Naive Majority-Class", majority),
        ):
            results.append(
                {
                    "model": model_name,
                    "split": split,
                    "task": "clf",
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"classification_report", "confusion_matrix"}
                    },
                }
            )
        naive_regression = evaluate_regression(
            y_next[split], rate_today[split], include_mape=True
        )
        naive_rmse[split] = naive_regression["rmse"]
        naive_regression["skill_vs_naive"] = 0.0
        results.append(
            {
                "model": "Naive Persistence (Random Walk)",
                "split": split,
                "task": "reg",
                **naive_regression,
            }
        )

    for model_name, feature_sets in X.items():
        classifier = make_classifier()
        weights = compute_sample_weight(
            class_weight="balanced", y=y_class["train"]
        )
        classifier.fit(
            feature_sets["train"],
            y_class["train"],
            sample_weight=weights,
            eval_set=[(feature_sets["val"], y_class["val"])],
            verbose=False,
        )
        classification_results[model_name] = {}
        for split in ("val", "test"):
            prediction = (
                pd.Series(classifier.predict(feature_sets[split]))
                .map(ID2LABEL)
                .to_numpy()
            )
            metrics = evaluate_classifier(
                splits[split]["target"].to_numpy(),
                prediction,
                include_kappa=True,
            )
            classification_results[model_name][split] = metrics
            results.append(
                {
                    "model": model_name,
                    "split": split,
                    "task": "clf",
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"classification_report", "confusion_matrix"}
                    },
                }
            )
            print(
                f"[{model_name} clf {split}] "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"accuracy={metrics['accuracy']:.4f}"
            )

        regressor = make_regressor()
        regressor.fit(
            feature_sets["train"],
            y_delta["train"],
            eval_set=[(feature_sets["val"], y_delta["val"])],
            verbose=False,
        )
        regression_results[model_name] = {}
        for split in ("val", "test"):
            prediction = rate_today[split] + regressor.predict(feature_sets[split])
            metrics = evaluate_regression(
                y_next[split],
                prediction,
                rmse_naive=naive_rmse[split],
                include_mape=True,
            )
            metrics["directional_hit"] = directional_hit_from_regression(
                rate_today[split], y_next[split], prediction
            )
            regression_results[model_name][split] = metrics
            results.append(
                {
                    "model": model_name,
                    "split": split,
                    "task": "reg",
                    **metrics,
                }
            )
            print(
                f"[{model_name} reg {split}] rmse={metrics['rmse']:.2f} "
                f"skill={metrics['skill_vs_naive']:+.4f}"
            )

    nested_results = {
        "classification": classification_results,
        "regression": regression_results,
        "rmse_naive_same_rows": naive_rmse,
    }
    return results, nested_results, {"baseline_features": baseline_features}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ekstraksi topik LDA dan evaluasi fitur topik NLP-only."
    )
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--article-splits", type=Path, default=DEFAULT_ARTICLE_SPLITS)
    parser.add_argument("--model-dataset", type=Path, default=DEFAULT_MODEL_DATASET)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-prefix", default="10_lda_topics")
    parser.add_argument("--topic-counts", type=_topic_count_candidates, default=[8, 10, 12, 15])
    parser.add_argument("--n-topics", type=int)
    parser.add_argument(
        "--select-topics-only",
        action="store_true",
        help="Evaluate candidate K using train-only coherence, then exit before val/test.",
    )
    parser.add_argument("--max-features", type=int, default=5_000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=0.85)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--top-terms", type=int, default=15)
    args = parser.parse_args(argv)
    if args.n_topics is not None and args.n_topics < 2:
        parser.error("--n-topics must be at least 2.")
    if args.max_features < 1 or args.min_df < 1 or args.max_iter < 1:
        parser.error("--max-features, --min-df, and --max-iter must be positive.")
    if not 0 < args.max_df <= 1:
        parser.error("--max-df must be in the interval (0, 1].")
    if args.top_terms < 2:
        parser.error("--top-terms must be at least 2.")

    model_data = _validate_dataset(pd.read_csv(args.model_dataset))
    articles = load_article_text(args.articles, args.article_splits, model_data)
    article_frame = build_article_text(articles)
    training_documents = article_frame.loc[
        article_frame["split"] == "train", "document"
    ].tolist()
    candidates = [args.n_topics] if args.n_topics is not None else args.topic_counts
    selection = select_topic_count(
        training_documents,
        candidates,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        max_iter=args.max_iter,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.results_dir / f"{args.output_prefix}_topic_counts.csv"
    if args.select_topics_only or args.n_topics is None or not candidate_path.exists():
        selection.to_csv(candidate_path, index=False)
    candidate_report = pd.read_csv(candidate_path)
    if args.select_topics_only:
        print(
            "Train-only topic-count selection saved. Inspect candidate top terms "
            "and choose K before running validation/test evaluation."
        )
        print(selection[["n_topics", "umass_coherence"]].to_string(index=False))
        return 0

    selected_topics = (
        args.n_topics
        if args.n_topics is not None
        else int(selection.iloc[0]["n_topics"])
    )
    print(
        f"Using K={selected_topics}; candidate coherence is computed from "
        "train articles only."
    )

    model = LDATopicModel(
        n_topics=selected_topics,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        max_iter=args.max_iter,
    ).fit(training_documents)
    documents = article_frame["document"].tolist()
    topic_probabilities = model.transform(documents)
    topic_columns = [
        f"topic_{topic_id:03d}" for topic_id in range(1, selected_topics + 1)
    ]
    topic_frame = pd.DataFrame(topic_probabilities, columns=topic_columns)
    max_probability = topic_probabilities.max(axis=1)
    topic_frame.insert(
        0,
        "dominant_topic",
        np.where(max_probability > 0, topic_probabilities.argmax(axis=1) + 1, -1),
    )
    topic_frame.insert(1, "topic_confidence", max_probability)
    article_topics = pd.concat(
        [article_frame.reset_index(drop=True), topic_frame], axis=1
    )
    daily_article_topics = article_topics.loc[
        article_topics[topic_columns].sum(axis=1) > 0
    ]
    daily_topics = (
        daily_article_topics.groupby("trading_date")[topic_columns]
        .mean()
        .reindex(model_data["date"], fill_value=0.0)
        .reset_index(drop=True)
    )
    daily_topics.insert(0, "split", model_data["split"].to_numpy())
    daily_topics.insert(0, "date", model_data["date"].to_numpy())
    daily_topics[topic_columns] = daily_topics[topic_columns].astype(np.float32)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    article_path = args.feature_dir / f"{args.output_prefix}_article_topics.csv"
    daily_path = args.feature_dir / f"{args.output_prefix}_daily_features.csv"
    terms_path = args.results_dir / f"{args.output_prefix}_topic_terms.csv"
    comparison_path = args.results_dir / f"{args.output_prefix}_comparison.csv"
    results_path = args.results_dir / f"{args.output_prefix}_results.json"

    article_topics.to_csv(article_path, index=False)
    daily_topics.to_csv(daily_path, index=False)
    terms = pd.DataFrame(
        {
            "topic_id": range(1, selected_topics + 1),
            "top_terms": [" | ".join(topic) for topic in model.top_terms(args.top_terms)],
        }
    )
    terms.to_csv(terms_path, index=False)

    results, score_details, model_details = _fit_and_score_models(
        model_data,
        daily_topics[["date", "split", *topic_columns]],
        topic_columns,
    )
    pd.DataFrame(results).round(6).to_csv(comparison_path, index=False)
    coverage = {
        split: int(
            (article_topics["split"].eq(split)
             & article_topics[topic_columns].sum(axis=1).gt(0)).sum()
        )
        for split in SPLITS
    }
    output = {
        "model": "LDA geopolitical topics, NLP-only incremental evaluation",
        "topic_model": {
            "algorithm": "sklearn LatentDirichletAllocation",
            "n_topics": selected_topics,
            "selection": (
                "highest train-only UMass coherence among configured candidates"
                if args.n_topics is None
                else "selected manually by --n-topics; coherence measured on train only"
            ),
            "vocabulary_size": model.vocabulary_size,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "max_iter": args.max_iter,
            "random_state": 42,
            "daily_aggregation": "mean article topic probabilities; equal article weight",
            "articles_with_topic_distribution": coverage,
            "no_news_days": {
                split: int(
                    (
                        model_data["split"].eq(split)
                        & ~model_data["date"].isin(
                            daily_article_topics["trading_date"].unique()
                        )
                    ).sum()
                )
                for split in SPLITS
            },
            "topic_terms": terms.to_dict(orient="records"),
            "candidate_selection": _plain_records(candidate_report),
        },
        "features": {
            "baseline": model_details["baseline_features"],
            "lda_topic_dimensions": topic_columns,
            "comparison": "baseline NLP versus baseline NLP + LDA topic distributions",
        },
        "target": {
            "classification": (
                f"target (NAIK/STABIL/TURUN, threshold {THRESHOLD_PCT:.2f}%)"
            ),
            "regression": "kurs_besok trained on delta, converted to rate level",
        },
        **score_details,
        "artifacts": {
            "lda_model": str(args.model_path),
            "article_topics": str(article_path),
            "daily_topic_features": str(daily_path),
            "topic_count_selection": str(candidate_path),
            "topic_terms": str(terms_path),
            "comparison": str(comparison_path),
        },
    }
    results_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "Saved LDA artifacts:\n"
        f"  model: {args.model_path}\n"
        f"  topics: {terms_path}\n"
        f"  comparison: {comparison_path}\n"
        f"  results: {results_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
