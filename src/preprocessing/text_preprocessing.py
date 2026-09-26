"""Text preprocessing untuk berita terpilih (hasil cleaning + filtering).

Alur Preprocessing:
    Buat minimal dua versi teks:
        - text_clean_light
        - text_clean_classical
        
    A. Light preprocessing
        Digunakan untuk transformer, sentiment model, NER, dan embedding:

        - Unicode NFC
        - Menghapus HTML
        - Menghapus karakter kontrol
        - Menormalkan spasi
        - Mengganti URL dan email dengan placeholder
        - Mempertahankan kapitalisasi, tanda baca, negasi, nama, angka, dan urutan kata

    B. Classical preprocessing
        Digunakan untuk TF-IDF, LDA, atau bag-of-words:

        - Light preprocessing
        - Lowercase atau casefold
        - Tokenisasi
        - Lematisasi
        - Optional stop-word removal
        - Filter token yang terlalu pendek
        - Bentuk unigram dan bigram

    Jangan hapus kata negasi. Hindari stemming sebagai pilihan utama.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    _PANDAS_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    _NLTK_AVAILABLE = True
except ImportError:
    nltk = None  # type: ignore
    stopwords = None  # type: ignore
    wordnet = None  # type: ignore
    WordNetLemmatizer = None  # type: ignore
    _NLTK_AVAILABLE = False


__all__ = [
    "split_paragraphs",
    "jaccard_similarity",
    "deduplicate_paragraphs",
    "mask_special_entities",
    "case_fold",
    "extract_preserved_entities",
    "preprocess_light",
    "light_preprocess",
    "lemmatize_tokens",
    "build_ngrams",
    "remove_stopwords_preserving_negations",
    "preprocess_classical",
    "classical_preprocess",
    "preprocess_text",
    "preprocess_article",
    "preprocess_documents",
    "preprocess_dataframe",
    "NEGATION_WORDS",
]


# ---------------------------------------------------------------------------
# Konstanta & Pola Regex
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
URL_RE = re.compile(
    r"https?://\S+|www\.\S+|(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|gov|edu|io|id|co)\b\S*",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
WORD_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z]+)?\b")

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# Daftar kata negasi (English & Indonesian) yang TIDAK BOLEH dihapus
NEGATION_WORDS: Set[str] = {
    # English negations & contractions
    "no", "not", "nor", "neither", "never", "none", "nobody", "nowhere", "nothing",
    "hardly", "scarcely", "barely", "without", "cannot",
    "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "wont", "won't", "wouldnt", "wouldn't",
    "cant", "can't", "couldnt", "couldn't",
    "shouldnt", "shouldn't",
    "isnt", "isn't", "arent", "aren't",
    "wasnt", "wasn't", "werent", "weren't",
    "havent", "haven't", "hasnt", "hasn't", "hadnt", "hadn't",
    # Indonesian negations
    "tidak", "bukan", "jangan", "belum", "tak", "tiada", "tanpa",
}

# Fallback stopword set standar jika nltk tidak terpasang
FALLBACK_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

_LEMMATIZER_INSTANCE: Optional[Any] = None
_STOPWORDS_CACHE: Optional[Set[str]] = None


def _init_nltk_resources() -> None:
    """Inisialisasi lazy loader untuk NLTK Lemmatizer dan Stopwords."""
    global _LEMMATIZER_INSTANCE, _STOPWORDS_CACHE
    if not _NLTK_AVAILABLE:
        return

    if _LEMMATIZER_INSTANCE is None:
        try:
            _LEMMATIZER_INSTANCE = WordNetLemmatizer()
            # Test wordnet load
            _LEMMATIZER_INSTANCE.lemmatize("wars")
        except Exception:
            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                _LEMMATIZER_INSTANCE = WordNetLemmatizer()
            except Exception:
                _LEMMATIZER_INSTANCE = None

    if _STOPWORDS_CACHE is None:
        try:
            sw = set(stopwords.words("english"))
        except Exception:
            try:
                nltk.download("stopwords", quiet=True)
                sw = set(stopwords.words("english"))
            except Exception:
                sw = set(FALLBACK_STOPWORDS)
        # Pastikan kata negasi TIDAK ada di stopwords yang akan dihapus
        _STOPWORDS_CACHE = sw - NEGATION_WORDS


# ---------------------------------------------------------------------------
# Helper Deduplikasi Paragraf
# ---------------------------------------------------------------------------

def split_paragraphs(text: str) -> List[str]:
    """Pecah teks menjadi blok paragraf berdasarkan newline."""
    if not text:
        return []
    parts = re.split(r"\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _shingles(tokens: List[str], n: int = 3) -> set:
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(a: str, b: str) -> float:
    """Kemiripan Jaccard 3-shingle antara dua paragraf (case-insensitive)."""
    ta = re.findall(r"[a-z0-9]+", (a or "").lower())
    tb = re.findall(r"[a-z0-9]+", (b or "").lower())
    sa, sb = _shingles(ta), _shingles(tb)
    if not sa or not sb:
        return 1.0 if (a or "").strip() == (b or "").strip() else 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate_paragraphs(
    paragraphs: List[str], jaccard_threshold: float = 0.8
) -> Tuple[List[str], int, int]:
    """Hapus duplikasi persis (exact) dan near-duplikasi (Jaccard >= threshold).

    Returns:
        (unik, n_exact_dihapus, n_near_dihapus)
    """
    unique: List[str] = []
    seen_exact: Set[str] = set()
    n_exact = 0
    n_near = 0

    for para in paragraphs:
        key = re.sub(r"\s+", " ", (para or "").strip().lower())
        if not key:
            continue
        if key in seen_exact:
            n_exact += 1
            continue
        is_near = any(
            jaccard_similarity(para, kept) >= jaccard_threshold for kept in unique
        )
        if is_near:
            n_near += 1
            continue
        seen_exact.add(key)
        unique.append(para.strip())

    return unique, n_exact, n_near


# ---------------------------------------------------------------------------
# Helper Masking, Case Fold, dan Retensi
# ---------------------------------------------------------------------------

def mask_special_entities(
    text: str, url_placeholder: str = "URL", email_placeholder: str = "EMAIL"
) -> str:
    """Ganti URL dan email dengan placeholder khusus.
    
    Email diganti lebih dulu agar komponen domain email tidak terpotong URL regex.
    """
    if not text:
        return ""
    masked = EMAIL_RE.sub(email_placeholder, text)
    masked = URL_RE.sub(url_placeholder, masked)
    return masked


def case_fold(text: str) -> str:
    """Penyeragaman seluruh huruf ke huruf kecil (lowercase / case folding)."""
    return (text or "").lower()


def extract_preserved_entities(text: str) -> Dict[str, List[str]]:
    """Ekstrak nama berkapital dan angka yang dipertahankan di teks."""
    source = text or ""
    capitals = sorted(set(_ENTITY_RE.findall(source)))
    numbers = sorted(set(_NUMBER_RE.findall(source)))
    return {"capitalized_candidates": capitals, "numbers": numbers}


# ---------------------------------------------------------------------------
# A. Light Preprocessing
# ---------------------------------------------------------------------------

def preprocess_light(
    text: str,
    deduplicate_paras: bool = True,
    jaccard_threshold: float = 0.8,
    url_placeholder: str = "URL",
    email_placeholder: str = "EMAIL",
    preserve_newlines: bool = False,
) -> str:
    """A. Light Preprocessing untuk transformer, sentiment model, NER, dan embedding.

    Langkah-langkah:
      1. Unicode NFC normalization.
      2. Menghapus tag HTML dan unescape entitas HTML.
      3. Menghapus karakter kontrol (ASCII / Unicode non-printable).
      4. Mengganti URL dan email dengan placeholder (default: 'URL' dan 'EMAIL').
      5. Deduplikasi paragraf (exact + near duplication).
      6. Menormalkan spasi.
      7. Mempertahankan kapitalisasi, tanda baca, negasi, nama, angka, dan urutan kata.
    """
    if not text:
        return ""

    # 1. Unicode NFC
    cleaned = unicodedata.normalize("NFC", str(text))

    # 2. Menghapus HTML
    cleaned = html.unescape(cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)

    # 3. Menghapus karakter kontrol (tetap izinkan newline dan tab sementara)
    cleaned = CONTROL_CHAR_RE.sub(" ", cleaned)

    # 4. Mengganti URL dan email dengan placeholder
    cleaned = mask_special_entities(
        cleaned,
        url_placeholder=url_placeholder,
        email_placeholder=email_placeholder,
    )

    # 5. Deduplikasi paragraf jika teks terdiri dari beberapa baris/paragraf
    if deduplicate_paras:
        paras = split_paragraphs(cleaned)
        if len(paras) > 1:
            unique_paras, _, _ = deduplicate_paragraphs(paras, jaccard_threshold)
            separator = "\n\n" if preserve_newlines else " "
            cleaned = separator.join(unique_paras)

    # 6. Menormalkan spasi
    # Mengubah non-breaking spaces (\u00a0), zero-width spaces (\u200b), dan spasi berulang
    cleaned = cleaned.replace("\u00a0", " ").replace("\u200b", "")
    if preserve_newlines:
        cleaned = re.sub(r"[^\S\r\n]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    else:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


# Alias untuk kemudahan penggunaan
light_preprocess = preprocess_light


# ---------------------------------------------------------------------------
# B. Classical Preprocessing
# ---------------------------------------------------------------------------

def _get_wordnet_pos(treebank_tag: str) -> str:
    """Petakan POS tag Treebank ke format WordNet (NOUN, VERB, ADJ, ADV)."""
    if not _NLTK_AVAILABLE or wordnet is None:
        return "n"
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lematisasi daftar token kata dengan NLTK WordNet Lemmatizer.
    
    Menggunakan POS tagging untuk hasil lematisasi optimal (menghindari stemming).
    Jika POS tagging / WordNet tidak tersedia, fallback ke aturan lematisasi dasar.
    """
    if not tokens:
        return []

    _init_nltk_resources()
    lemmatizer = _LEMMATIZER_INSTANCE

    if lemmatizer is None:
        # Fallback lematisasi sederhana jika NLTK tidak tersedia
        res = []
        for t in tokens:
            if t.endswith("ies") and len(t) > 4:
                res.append(t[:-3] + "y")
            elif t.endswith("es") and len(t) > 3 and not t.endswith(("ss", "us", "is")):
                res.append(t[:-2])
            elif t.endswith("s") and len(t) > 2 and not t.endswith(("ss", "us", "is")):
                res.append(t[:-1])
            else:
                res.append(t)
        return res

    try:
        # Jalankan POS tagger untuk lematisasi akurat berdasarkan kelas kata
        pos_tags = nltk.pos_tag(tokens)
        lemmatized: List[str] = []
        for word, tag in pos_tags:
            wn_pos = _get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos=wn_pos)
            lemmatized.append(lemma)
        return lemmatized
    except Exception:
        # Fallback jika pos_tag bermasalah: lematisasi verb lalu noun
        return [
            lemmatizer.lemmatize(lemmatizer.lemmatize(t, pos="v"), pos="n")
            for t in tokens
        ]


def remove_stopwords_preserving_negations(tokens: List[str]) -> List[str]:
    """Hapus stopwords umum dengan mempertahankan kata-kata negasi."""
    _init_nltk_resources()
    sw = _STOPWORDS_CACHE if _STOPWORDS_CACHE is not None else (FALLBACK_STOPWORDS - NEGATION_WORDS)
    return [tok for tok in tokens if tok not in sw]


def build_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Bentuk n-gram (default: bigram) dari urutan token kata."""
    if len(tokens) < n:
        return []
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def preprocess_classical(
    text: str,
    remove_stopwords: bool = True,
    min_token_len: int = 3,
    include_bigrams: bool = True,
    lemmatize: bool = True,
    deduplicate_paras: bool = True,
    jaccard_threshold: float = 0.8,
) -> Dict[str, Any]:
    """B. Classical Preprocessing untuk TF-IDF, LDA, atau Bag-of-Words.

    Langkah-langkah:
      1. Light preprocessing terlebih dahulu.
      2. Lowercase / casefold.
      3. Tokenisasi kata (menghapus tanda baca, menjaga kata & angka).
      4. Lematisasi (WordNet POS-aware, hindari stemming).
      5. Optional stop-word removal (kata negasi DIJAMIN TIDAK DIHAPUS).
      6. Filter token yang terlalu pendek (panjang < min_token_len, kecuali kata negasi).
      7. Pembentukan unigram dan bigram.

    Returns:
        Dict dengan kunci:
          - 'text_clean_classical': string representasi fitur kata untuk BoW/TF-IDF
          - 'unigrams': list token unigram bersih
          - 'bigrams': list token bigram
          - 'tokens': gabungan unigram dan bigram (atau unigram saja jika include_bigrams=False)
    """
    # 1. Light preprocessing
    light_text = preprocess_light(
        text,
        deduplicate_paras=deduplicate_paras,
        jaccard_threshold=jaccard_threshold,
    )

    if not light_text:
        return {
            "text_clean_classical": "",
            "unigrams": [],
            "bigrams": [],
            "tokens": [],
        }

    # 2. Lowercase / casefold
    lowered = case_fold(light_text)

    # 3. Tokenisasi
    raw_tokens = WORD_TOKEN_RE.findall(lowered)

    # Bersihkan possessive suffix seperti "china's" -> "china"
    tokens: List[str] = [
        tok[:-2] if tok.endswith("'s") and len(tok) > 2 else tok
        for tok in raw_tokens
    ]

    # 4. Lematisasi
    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    # 5. Stop-word removal (mempertahankan kata negasi)
    if remove_stopwords:
        tokens = remove_stopwords_preserving_negations(tokens)

    # 6. Filter token yang terlalu pendek (kata negasi seperti 'no' tetap dipertahankan)
    filtered_tokens: List[str] = [
        tok for tok in tokens
        if len(tok) >= min_token_len or tok in NEGATION_WORDS
    ]

    # 7. Bentuk unigram dan bigram
    unigrams = list(filtered_tokens)
    bigrams = build_ngrams(filtered_tokens, n=2) if include_bigrams else []

    combined_tokens = unigrams + bigrams if include_bigrams else unigrams
    text_clean_classical = " ".join(combined_tokens)

    return {
        "text_clean_classical": text_clean_classical,
        "unigrams": unigrams,
        "bigrams": bigrams,
        "tokens": combined_tokens,
    }


# Alias untuk kemudahan penggunaan
classical_preprocess = preprocess_classical


# ---------------------------------------------------------------------------
# Pipeline Lengkap Preprocessing Teks & Artikel
# ---------------------------------------------------------------------------

def preprocess_text(
    text: str,
    jaccard_threshold: float = 0.8,
    remove_stopwords: bool = True,
    min_token_len: int = 3,
    include_bigrams: bool = True,
    lemmatize: bool = True,
) -> Dict[str, Any]:
    """Pipeline preprocessing lengkap untuk satu teks berita.

    Menghasilkan dua versi teks:
      - text_clean_light (untuk transformer, NER, sentiment analysis)
      - text_clean_classical (untuk TF-IDF, BoW, LDA)
    """
    raw_text = str(text or "")
    light_text = preprocess_light(
        raw_text,
        deduplicate_paras=True,
        jaccard_threshold=jaccard_threshold,
    )
    classical_res = preprocess_classical(
        light_text,
        remove_stopwords=remove_stopwords,
        min_token_len=min_token_len,
        include_bigrams=include_bigrams,
        lemmatize=lemmatize,
        deduplicate_paras=False,  # sudah dideduplikasi di tahap light
    )
    preserved = extract_preserved_entities(light_text)

    return {
        "text_clean_light": light_text,
        "text_clean_classical": classical_res["text_clean_classical"],
        "tokens_classical": classical_res["tokens"],
        "unigrams": classical_res["unigrams"],
        "bigrams": classical_res["bigrams"],
        "text_final": light_text,  # alias kompatibilitas pipeline
        "preserved_entities": preserved,
    }


TEXT_FIELDS = ("title", "heading", "body")


def preprocess_article(
    article: Union[Dict[str, Any], str],
    jaccard_threshold: float = 0.8,
    remove_stopwords: bool = True,
    min_token_len: int = 3,
    include_bigrams: bool = True,
    lemmatize: bool = True,
) -> Dict[str, Any]:
    """Preprocess satu artikel (berupa dict atau string).
    
    Menambahkan field 'text_clean_light', 'text_clean_classical', 'tokens_classical',
    'unigrams', 'bigrams', dan 'text_final'.
    """
    # Jika input langsung berupa string teks tunggal
    if isinstance(article, str):
        res = preprocess_text(
            article,
            jaccard_threshold=jaccard_threshold,
            remove_stopwords=remove_stopwords,
            min_token_len=min_token_len,
            include_bigrams=include_bigrams,
            lemmatize=lemmatize,
        )
        return {
            "text_original": article,
            **res,
        }

    # Jika input berupa dictionary / structured record
    out = dict(article)
    candidate_parts: List[str] = []

    # Cek field standar: body, title, heading
    for field in TEXT_FIELDS:
        val = article.get(field)
        if isinstance(val, str) and val.strip():
            candidate_parts.append(val.strip())

    # Fallback jika tidak ada field TEXT_FIELDS, cari teks pada kunci umum lain
    if not candidate_parts:
        for fallback_key in ("text", "content", "discovery_title"):
            val = article.get(fallback_key)
            if isinstance(val, str) and val.strip():
                candidate_parts.append(val.strip())

    combined_raw = "\n\n".join(candidate_parts)
    res = preprocess_text(
        combined_raw,
        jaccard_threshold=jaccard_threshold,
        remove_stopwords=remove_stopwords,
        min_token_len=min_token_len,
        include_bigrams=include_bigrams,
        lemmatize=lemmatize,
    )

    out["text_clean_light"] = res["text_clean_light"]
    out["text_clean_classical"] = res["text_clean_classical"]
    out["tokens_classical"] = res["tokens_classical"]
    out["unigrams"] = res["unigrams"]
    out["bigrams"] = res["bigrams"]
    out["text_final"] = res["text_final"]
    out["preserved_entities"] = res["preserved_entities"]

    return out


def preprocess_dataframe(
    df: Any,
    text_col: str = "body",
    jaccard_threshold: float = 0.8,
    remove_stopwords: bool = True,
    min_token_len: int = 3,
    include_bigrams: bool = True,
    lemmatize: bool = True,
) -> Any:
    """Preprocess pandas DataFrame, menambahkan kolom hasil preprocessing.

    Kolom baru:
      - text_clean_light
      - text_clean_classical
      - tokens_classical
      - unigrams
      - bigrams
      - text_final
    """
    if not _PANDAS_AVAILABLE or not isinstance(df, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame.")

    df_out = df.copy()
    series = df_out[text_col].fillna("").astype(str) if text_col in df_out.columns else pd.Series([""] * len(df_out))

    processed_records = [
        preprocess_text(
            t,
            jaccard_threshold=jaccard_threshold,
            remove_stopwords=remove_stopwords,
            min_token_len=min_token_len,
            include_bigrams=include_bigrams,
            lemmatize=lemmatize,
        )
        for t in series
    ]

    df_out["text_clean_light"] = [r["text_clean_light"] for r in processed_records]
    df_out["text_clean_classical"] = [r["text_clean_classical"] for r in processed_records]
    df_out["tokens_classical"] = [r["tokens_classical"] for r in processed_records]
    df_out["unigrams"] = [r["unigrams"] for r in processed_records]
    df_out["bigrams"] = [r["bigrams"] for r in processed_records]
    df_out["text_final"] = [r["text_final"] for r in processed_records]

    return df_out


def preprocess_documents(
    docs: Union[Iterable[Any], Any],
    jaccard_threshold: float = 0.8,
    remove_stopwords: bool = True,
    min_token_len: int = 3,
    include_bigrams: bool = True,
    lemmatize: bool = True,
) -> Any:
    """Preprocess kumpulan dokumen dalam berbagai format.

    Format yang didukung:
      - pandas.DataFrame: mengembalikan DataFrame dengan kolom-kolom baru
      - pandas.Series: mengembalikan list of dicts hasil pemrosesan
      - List[str]: kumpulan string teks -> mengembalikan list of dicts
      - List[Dict[str, Any]]: kumpulan artikel dict -> mengembalikan list of dicts
      - str atau dict tunggal: mengembalikan satu dict
    """
    # 1. Kasus pandas DataFrame
    if _PANDAS_AVAILABLE and isinstance(docs, pd.DataFrame):
        # Tentukan kolom teks utama (body atau text)
        col = "body" if "body" in docs.columns else ("text" if "text" in docs.columns else docs.columns[0])
        return preprocess_dataframe(
            docs,
            text_col=col,
            jaccard_threshold=jaccard_threshold,
            remove_stopwords=remove_stopwords,
            min_token_len=min_token_len,
            include_bigrams=include_bigrams,
            lemmatize=lemmatize,
        )

    # 2. Kasus item tunggal (str atau dict)
    if isinstance(docs, str):
        return preprocess_article(
            docs,
            jaccard_threshold=jaccard_threshold,
            remove_stopwords=remove_stopwords,
            min_token_len=min_token_len,
            include_bigrams=include_bigrams,
            lemmatize=lemmatize,
        )

    if isinstance(docs, dict):
        return preprocess_article(
            docs,
            jaccard_threshold=jaccard_threshold,
            remove_stopwords=remove_stopwords,
            min_token_len=min_token_len,
            include_bigrams=include_bigrams,
            lemmatize=lemmatize,
        )

    # 3. Kasus pandas Series
    if _PANDAS_AVAILABLE and isinstance(docs, pd.Series):
        docs_list = docs.fillna("").astype(str).tolist()
        return [
            preprocess_article(
                d,
                jaccard_threshold=jaccard_threshold,
                remove_stopwords=remove_stopwords,
                min_token_len=min_token_len,
                include_bigrams=include_bigrams,
                lemmatize=lemmatize,
            )
            for d in docs_list
        ]

    # 4. Kasus List atau Iterable (List[str] atau List[Dict])
    if isinstance(docs, Iterable):
        return [
            preprocess_article(
                d,
                jaccard_threshold=jaccard_threshold,
                remove_stopwords=remove_stopwords,
                min_token_len=min_token_len,
                include_bigrams=include_bigrams,
                lemmatize=lemmatize,
            )
            for d in docs
        ]

    raise TypeError(f"Tipe input {type(docs)} tidak didukung untuk preprocess_documents.")
