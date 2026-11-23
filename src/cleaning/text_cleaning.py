"""Text cleaning untuk artikel berita.

Cleaning mempertahankan kapitalisasi dan tanda baca pada ``title_clean`` dan
``body_clean`` agar teks tetap dapat digunakan untuk analisis sentimen. Versi
bag-of-words disediakan terpisah untuk TF-IDF dan Loughran-McDonald.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


MIN_PARAGRAPH_WORDS = 4
NEGATIONS = {"no", "not", "nor", "never", "none", "cannot", "without"}
STOPWORDS = (set(ENGLISH_STOP_WORDS) - NEGATIONS) - {"us"}

QUOTE_MAP = str.maketrans({
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": " - ",
})

INVISIBLE_RE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]"
)
URL_RE = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
DATELINE_RE = re.compile(r"^[A-Z][A-Za-z.,' ]{2,40}\s+\((?:Reuters|AP)\)\s*-\s*")
ACRONYM_RE = re.compile(r"\b(?:[A-Z]\.){2,}")
TITLE_SUFFIX_RE = re.compile(r"\s+[-|]\s+CNBC\s*$", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z]{2,}")

BOILERPLATE_PATTERNS = (
    r"^(watch|read more|read|related|listen|see also|more from|sign up|subscribe|click here|follow)\b\s*[:\-]",
    r"\bcnbc pro\b",
    r"\bsubscribe to\b",
    r"\bsign up for\b",
    r"\bnewsletter\b",
    r"contributed to this (report|story|article)",
    r"^(disclaimer|copyright)\b",
    r"all rights reserved",
    r"^this (story|article) is (still )?developing",
    r"^(get|receive) the latest",
)
BOILERPLATE_RE = tuple(re.compile(pattern, re.IGNORECASE) for pattern in BOILERPLATE_PATTERNS)
BOILERPLATE_MAX_WORDS = 40

__all__ = [
    "clean_title",
    "clean_text",
    "to_bow_text",
    "clean_structured_record",
    "batch_clean_html",
    "clean_dataframe",
]


def _as_text(value: Any) -> str:
    """Convert nullable values to text without turning nulls into the word 'nan'."""
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return str(value)


def _base_normalize(text: Any) -> str:
    value = html.unescape(_as_text(text))
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(QUOTE_MAP)
    value = INVISIBLE_RE.sub("", value)
    return ACRONYM_RE.sub(lambda match: match.group().replace(".", ""), value)


def clean_title(title: Any) -> str:
    """Normalize a title and remove the trailing CNBC suffix."""
    value = _base_normalize(title)
    value = TITLE_SUFFIX_RE.sub("", value)
    return SPACE_RE.sub(" ", value).strip()


def clean_text(text: Any) -> str:
    """Clean article text while retaining useful casing and punctuation.

    URL/email removal, dateline cleanup, short/boilerplate paragraph removal,
    and stable paragraph deduplication follow ``Cleaning_article.ipynb``.
    """
    value = _base_normalize(text)
    value = URL_RE.sub(" ", value)
    value = EMAIL_RE.sub(" ", value)

    paragraphs = []
    for paragraph in value.split("\n"):
        paragraph = SPACE_RE.sub(" ", paragraph).strip()
        if not paragraph:
            continue
        if not paragraphs:
            paragraph = DATELINE_RE.sub("", paragraph).strip()
        word_count = len(paragraph.split())
        if word_count < MIN_PARAGRAPH_WORDS:
            continue
        if word_count < BOILERPLATE_MAX_WORDS and any(
            pattern.search(paragraph) for pattern in BOILERPLATE_RE
        ):
            continue
        paragraphs.append(paragraph)

    return "\n".join(dict.fromkeys(paragraphs))


def to_bow_text(text: Any) -> str:
    """Create the notebook's lowercase, stopword-filtered BoW text."""
    value = _as_text(text).lower()
    value = value.replace("can't", "cannot").replace("won't", "will not")
    value = re.sub(r"n't\b", " not", value)
    value = re.sub(r"'s\b", "", value)
    value = value.replace("'", "")
    tokens = WORD_RE.findall(value)
    return " ".join(token for token in tokens if token not in STOPWORDS)


def _iter_jsonld_articles(soup: BeautifulSoup) -> Iterable[Mapping[str, Any]]:
    def visit(value: Any) -> Iterable[Mapping[str, Any]]:
        if isinstance(value, list):
            for item in value:
                yield from visit(item)
        elif isinstance(value, dict):
            yield value
            yield from visit(value.get("@graph", []))

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        source = script.string or script.get_text()
        if not source:
            continue
        try:
            data = json.loads(source)
        except (json.JSONDecodeError, TypeError):
            continue
        yield from visit(data)


def _extract_metadata(soup: BeautifulSoup) -> dict[str, Any]:
    title = author = published_at = None

    for item in _iter_jsonld_articles(soup):
        item_type = item.get("@type", [])
        item_types = item_type if isinstance(item_type, list) else [item_type]
        if not any(
            isinstance(value, str) and value.lower() in {"article", "newsarticle"}
            for value in item_types
        ) and not item.get("headline"):
            continue

        title = title or item.get("headline")
        published_at = published_at or item.get("datePublished")
        author_data = item.get("author")
        if not author and isinstance(author_data, dict):
            author = author_data.get("name")
        elif not author and isinstance(author_data, list):
            names = [
                value.get("name")
                for value in author_data
                if isinstance(value, dict) and value.get("name")
            ]
            author = ", ".join(names) if names else None

    if not title:
        title = (
            soup.select_one('meta[property="og:title"]') or soup.find("h1") or soup.find("title")
        )
        if title:
            title = title.get("content") if title.name == "meta" else title.get_text(" ", strip=True)

    if not author:
        author_tag = soup.select_one(".Author-authorName, [rel='author']")
        author = author_tag.get_text(" ", strip=True) if author_tag else None

    if not published_at:
        time_tag = soup.find("time", attrs={"datetime": True})
        published_at = time_tag.get("datetime") if time_tag else None

    return {"title": title, "author": author, "published_at": published_at}


def _extract_html_body(soup: BeautifulSoup) -> str:
    jsonld_body = next(
        (
            item["articleBody"]
            for item in _iter_jsonld_articles(soup)
            if isinstance(item.get("articleBody"), str) and item["articleBody"].strip()
        ),
        "",
    )
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    content = soup.select_one(
        ".ArticleBody-articleBody, .article-content, [class*='ArticleBody'], article"
    ) or soup
    paragraphs = content.find_all("p")
    text = "\n".join(
        paragraph.get_text(" ", strip=True)
        for paragraph in paragraphs
        if paragraph.get_text(strip=True)
    )
    if not text:
        text = content.get_text("\n", strip=True)

    return text or jsonld_body


def clean_structured_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a cleaned copy of a structured article record."""
    if not isinstance(record, Mapping):
        raise TypeError("Record harus berupa mapping/dict.")

    result = dict(record)
    title = _as_text(result.get("title")).strip()
    if not title:
        title = _as_text(result.get("discovery_title")).strip()
    body = _as_text(result.get("body"))
    result["title_clean"] = clean_title(title)
    result["body_clean"] = clean_text(body)
    result["title"] = result["title_clean"]
    result["body"] = result["body_clean"]
    result["title_bow"] = to_bow_text(result["title_clean"])
    result["body_bow"] = to_bow_text(result["body_clean"])
    return result


def batch_clean_html(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Clean a batch of HTML or structured article records."""
    if isinstance(items, (str, bytes, Mapping)):
        raise TypeError("Input harus berupa iterable record, bukan teks atau record tunggal.")

    cleaned = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("Setiap item harus berupa mapping/dict.")

        result = dict(item)
        source_html = result.get("html")
        if isinstance(source_html, str) and source_html.strip():
            soup = BeautifulSoup(source_html, "html.parser")
            metadata = _extract_metadata(soup)
            for key, value in metadata.items():
                if not _as_text(result.get(key)).strip() and value:
                    result[key] = value
            if not result.get("title") and result.get("discovery_title"):
                result["title"] = result["discovery_title"]
            result["body"] = _extract_html_body(soup)
        result.pop("html", None)
        cleaned.append(clean_structured_record(result))
    return cleaned


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Add the notebook's cleaned and BoW text columns to a DataFrame copy."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame.")
    if "body" not in data.columns:
        raise ValueError("Kolom 'body' wajib tersedia.")

    result = data.copy()
    titles = (
        result["title"].combine_first(result["discovery_title"])
        if "title" in result.columns and "discovery_title" in result.columns
        else result["title"]
        if "title" in result.columns
        else result["discovery_title"]
        if "discovery_title" in result.columns
        else pd.Series("", index=result.index, dtype="object")
    )
    result["title_clean"] = titles.map(clean_title)
    result["body_clean"] = result["body"].map(clean_text)
    result["title_bow"] = result["title_clean"].map(to_bow_text)
    result["body_bow"] = result["body_clean"].map(to_bow_text)
    return result
