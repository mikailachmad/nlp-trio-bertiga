"""
Text filtering untuk seleksi relevansi berita terhadap topik geopolitik.

Alur Data Filtering:
    1. Filter fitur wajib yang kosong (author, title, published_date, body, discovery_title, url)
    2. Filter relevansi geopolitik:
        a. Recall tinggi menggunakan kata kunci
            Gunakan kelompok kosakata, bukan hanya kata "geopolitics".
            Gabungkan judul dan isi, tetapi beri tanda agar keduanya tetap dapat dibedakan:
            
        b. Klasifikasi relevansi
            0 = tidak relevan
            1 = relevan geopolitik, dampak ekonomi rendah
            2 = relevan geopolitik dan berpotensi memengaruhi pasar

            Lakukan anotasi manual acak dan terstratifikasi berdasarkan tahun serta sumber. Setelah itu, latih classifier seperti:

            - Logistic Regression + TF-IDF
            - Linear SVM + TF-IDF
            - Transformer multilingual
            - Zero-shot classification sebagai alat bantu, bukan sebagai ground truth

            TF-IDF dengan model linear adalah baseline yang baik karena relatif cepat dan fitur pentingnya dapat diperiksa. Pipeline scikit-learn juga membantu menggabungkan transformasi dan model dalam alur yang dapat divalidasi secara konsisten.
"""

import re

import pandas as pd


REQUIRED_FIELDS = (
    "author",
    "title",
    "published_at",
    "body",
    "discovery_title",
    "url",
)


geopolitical_keywords = {
    "conflict": [
        "war", "conflict", "military", "missile",
        "airstrike", "invasion", "ceasefire",
        "troops", "defense"
    ],
    "sanction": [
        "sanction", "embargo", "asset freeze",
        "export control", "trade restriction"
    ],
    "diplomacy": [
        "diplomatic", "summit", "bilateral",
        "treaty", "negotiation", "foreign minister"
    ],
    "trade": [
        "tariff", "trade war", "trade dispute",
        "export ban", "import restriction"
    ],
    "political_risk": [
        "election", "coup", "protest",
        "political crisis", "government collapse"
    ]
}


def _build_keyword_pattern():
    terms = sorted(
        {term for group in geopolitical_keywords.values() for term in group},
        key=len,
        reverse=True,
    )
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(term) for term in terms) + r")(?!\w)",
        re.IGNORECASE,
    )


_KEYWORD_PATTERN = _build_keyword_pattern()


def filter_required_fields(data):
    
    """
    Return a copy containing only rows with all required values populated.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    valid = data.loc[:, REQUIRED_FIELDS].notna().all(axis=1)
    
    for field in REQUIRED_FIELDS:
        valid &= data[field].astype("string").str.strip().ne("")
        
    return data.loc[valid].copy()


def find_geopolitical_keywords(title, body, discovery_title):
    
    """
    Return distinct keyword matches, retaining title/body provenance.
    """
    matches = {"title": [], "body": [], "discovery_title": []}
    
    for field, text in (("title", title), ("body", body), ("discovery_title", discovery_title)):
        if isinstance(text, str):
            matches[field] = list(
                dict.fromkeys(
                    match.group(0).lower()
                    for match in _KEYWORD_PATTERN.finditer(text)
                )
            )
            
    return matches


def filter_geopolitical_relevance(data):
    
    """
    Keep rows whose title or body contains a geopolitical keyword.
    """
    missing = [field for field in ("title", "body", "discovery_title") if field not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    result = data.copy()
    result["geopolitical_matches"] = [
        find_geopolitical_keywords(title, body, discovery_title)
        for title, body, discovery_title in zip(result["title"], result["body"], result["discovery_title"])
    ]
    
    relevant = result["geopolitical_matches"].map(
        lambda matches: bool(matches["title"] or matches["body"] or matches["discovery_title"])
    )
    
    return result.loc[relevant].copy()


def filter_news(data):
    """
    Apply required-field validation followed by keyword relevance filtering.
    """
    
    return filter_geopolitical_relevance(filter_required_fields(data))