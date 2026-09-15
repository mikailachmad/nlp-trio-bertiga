"""
Text filtering untuk seleksi relevansi berita terhadap topik geopolitik & nilai tukar.

Alur (sesuai README / repo workflow — bagian "Data Filtering"):
  1. Penetapan keyword per topik (geopolitik, nilai tukar rupiah).
  2. Perhitungan probabilitas keterkaitan tiap dokumen terhadap kosa kata
     keyword masing-masing topik menggunakan TF-IDF.
  3. Pemetaan skor probabilitas ke ruang 2D (geopolitik × nilai tukar).
  4. Pengelompokan (clustering) dokumen ke 4 kategori via K-Means:
       Kategori 1: geopolitik tinggi & nilai tukar tinggi   → dipilih
       Kategori 2: geopolitik tinggi & nilai tukar rendah
       Kategori 3: geopolitik rendah & nilai tukar tinggi
       Kategori 4: geopolitik rendah & nilai tukar rendah
  5. Pemilihan dokumen di kluster geopolitik-tinggi & nilai-tukar-tinggi.

Dependensi: hanya stdlib + numpy (sudah ada di requirements.txt).
"""


from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "KEYWORDS_GEOPOLITIK",
    "KEYWORDS_KURS",
    "tokenize",
    "compute_tf",
    "compute_idf",
    "compute_tfidf_scores",
    "topic_probability",
    "kmeans_cluster",
    "assign_categories",
    "filter_documents",
]

# ---------------------------------------------------------------------------
# 1. Penetapan keyword berdasarkan topik pembahasan
# ---------------------------------------------------------------------------

KEYWORDS_GEOPOLITIK: List[str] = [
    "konflik", "perang", "ancaman", "perdamaian", "diplomasi",
    "geopolitik", "global", "dunia", "internasional",
    "amerika", "united states", "us", "as",
    "china", "rusia", "ukraina",
    "timur tengah", "iran", "palestina", "gaza", "rafah", "israel",
    "persaingan", "klaim", "presiden", "perdana menteri",
    "ekspor", "impor", "perdagangan", "pemerintah",
]

KEYWORDS_KURS: List[str] = [
    "rupiah", "dolar as", "nilai tukar", "kurs",
    "ekonomi", "perdagangan", "keuangan", "peningkatan",
    "inflasi", "mata uang", "global",
    "penguatan", "menguat", "pelemahan", "melemah",
    "suku bunga", "pasar",
    "ekspor", "impor", "devisa", "valuta", "bank", "dana",
]


# ---------------------------------------------------------------------------
# 2. TF-IDF scoring terhadap keyword masing-masing topik
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\s+[a-z0-9]+)?", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """
    Tokenisasi sederhana: lowercase, hapus tanda baca, hasilkan unigram + bigram.
    """


    text_lower = (text or "").lower()
    words = re.findall(r"[a-z0-9]+", text_lower)

    tokens = list(words)

    # Tambahkan bigram agar keyword multi-kata seperti
    # "timur tengah", "dolar as", "nilai tukar" bisa terdeteksi.
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i + 1]}")

    return tokens


def _doc_text(doc: Dict[str, Any]) -> str:
    """
    Gabungkan field teks dari dokumen terstruktur (output cleaning).
    """


    parts: List[str] = []

    for field in ("title", "heading", "body"):
        val = doc.get(field, "") or ""
        if isinstance(val, str) and val.strip():
            parts.append(val)

    return " ".join(parts)


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Term Frequency: frekuensi relatif tiap token dalam dokumen.
    """


    if not tokens:
        return {}

    counts = Counter(tokens)
    total = len(tokens)

    return {term: count / total for term, count in counts.items()}


def compute_idf(corpus_tokens: List[List[str]], vocabulary: List[str]) -> Dict[str, float]:
    """
    Inverse Document Frequency (smooth) untuk subset vocabulary.

    IDF(t) = log((N + 1) / (df(t) + 1)) + 1   (sklearn-style smooth IDF)
    """


    n = len(corpus_tokens)
    idf: Dict[str, float] = {}

    for term in vocabulary:
        df = sum(1 for doc_tokens in corpus_tokens if term in set(doc_tokens))
        idf[term] = math.log((n + 1) / (df + 1)) + 1

    return idf


def compute_tfidf_scores(
    docs: List[Dict[str, Any]],
    keywords: List[str],
) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Hitung skor TF-IDF tiap dokumen terhadap satu set keyword.

    Returns:
        (scores, details)
        - scores: list float skor agregat per dokumen
        - details: list dict {keyword: tfidf_value} per dokumen
    """


    # Tokenisasi seluruh corpus
    corpus_tokens: List[List[str]] = [tokenize(_doc_text(doc)) for doc in docs]

    # IDF hanya dihitung untuk keyword yang relevan
    idf = compute_idf(corpus_tokens, keywords)

    scores: List[float] = []
    details: List[Dict[str, float]] = []

    for doc_tokens in corpus_tokens:
        tf = compute_tf(doc_tokens)
        doc_detail: Dict[str, float] = {}
        doc_score = 0.0

        for kw in keywords:
            tfidf_val = tf.get(kw, 0.0) * idf.get(kw, 0.0)
            doc_detail[kw] = round(tfidf_val, 6)
            doc_score += tfidf_val

        scores.append(round(doc_score, 6))
        details.append(doc_detail)

    return scores, details


# ---------------------------------------------------------------------------
# 3. Pemetaan skor probabilitas ke ruang 2D
# ---------------------------------------------------------------------------

def topic_probability(
    docs: List[Dict[str, Any]],
) -> Tuple[List[float], List[float]]:
    """
    Hitung probabilitas keterkaitan tiap dokumen pada kedua topik.

    Probabilitas = skor TF-IDF dinormalisasi ke [0, 1] menggunakan
    min-max scaling per topik agar bisa dipetakan ke ruang 2D.

    Returns:
        (p_geopolitik, p_kurs) — masing-masing list float panjang N.
    """


    scores_geo, _ = compute_tfidf_scores(docs, KEYWORDS_GEOPOLITIK)
    scores_kurs, _ = compute_tfidf_scores(docs, KEYWORDS_KURS)

    def _minmax(values: List[float]) -> List[float]:
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            # Semua dokumen bernilai sama → beri 0.5 agar netral.
            return [0.5] * len(values)
        return [round((v - lo) / (hi - lo), 6) for v in values]

    return _minmax(scores_geo), _minmax(scores_kurs)


# ---------------------------------------------------------------------------
# 4. K-Means clustering
# ---------------------------------------------------------------------------

def kmeans_cluster(
    points: np.ndarray,
    k: int = 4,
    seed: int = 0,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means sederhana (numpy saja, tanpa sklearn).

    Args:
        points: array (N, 2) — titik 2D (p_geopolitik, p_kurs).
        k: jumlah kluster.
        seed: seed random agar deterministik.
        max_iter: iterasi maksimum.

    Returns:
        (labels, centroids) — label kluster tiap titik, posisi centroid.
    """


    rng = np.random.RandomState(seed)
    n = points.shape[0]

    if n == 0:
        return np.array([], dtype=int), np.empty((0, 2))

    if n <= k:
        # Dokumen lebih sedikit dari k: tiap dokumen jadi klusternya sendiri.
        labels = np.arange(n, dtype=int)
        centroids = points.copy()
        return labels, centroids

    # Inisialisasi centroid: K-Means++ sederhana.
    indices: List[int] = [rng.randint(n)]

    for _ in range(1, k):
        dists = np.min(
            [np.sum((points - points[idx]) ** 2, axis=1) for idx in indices],
            axis=0,
        )
        probs = dists / dists.sum()
        indices.append(rng.choice(n, p=probs))

    centroids = points[indices].astype(float)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assignment
        dists = np.array(
            [np.sum((points - c) ** 2, axis=1) for c in centroids]
        )  # (k, N)
        new_labels = np.argmin(dists, axis=0)

        # Update centroids
        new_centroids = np.empty_like(centroids)

        for j in range(k):
            members = points[new_labels == j]
            if len(members) == 0:
                new_centroids[j] = centroids[j]
            else:
                new_centroids[j] = members.mean(axis=0)

        if np.array_equal(new_labels, labels):
            labels = new_labels
            centroids = new_centroids
            break

        labels = new_labels
        centroids = new_centroids

    return labels, centroids


# ---------------------------------------------------------------------------
# 5. Penentuan kategori & seleksi dokumen
# ---------------------------------------------------------------------------

def assign_categories(
    p_geo: List[float],
    p_kurs: List[float],
    labels: np.ndarray,
    centroids: np.ndarray,
    thresh_geo: float = 0.15,
    thresh_kurs: float = 0.15,
) -> List[int]:
    """
    Petakan label kluster ke kategori semantik 1-4 berdasarkan centroid.

    Logika:
      - Tiap centroid dievaluasi: apakah rata-rata geopolitik & kurs-nya
        tergolong *tinggi* (>= threshold) atau *rendah*.
      - Kategori diurutkan:
          1 = geo tinggi + kurs tinggi
          2 = geo tinggi + kurs rendah
          3 = geo rendah + kurs tinggi
          4 = geo rendah + kurs rendah
    """


    n_clusters = centroids.shape[0]

    # Hitung rata-rata p_geo dan p_kurs tiap kluster
    p_geo_arr = np.array(p_geo)
    p_kurs_arr = np.array(p_kurs)

    cluster_means: List[Tuple[float, float]] = []

    for j in range(n_clusters):
        mask = labels == j

        if mask.sum() == 0:
            cluster_means.append((centroids[j, 0], centroids[j, 1]))
        else:
            cluster_means.append((
                float(p_geo_arr[mask].mean()),
                float(p_kurs_arr[mask].mean()),
            ))

    # Map kluster → kategori
    cluster_to_cat: Dict[int, int] = {}

    for j, (mean_geo, mean_kurs) in enumerate(cluster_means):
        geo_high = mean_geo >= thresh_geo
        kurs_high = mean_kurs >= thresh_kurs

        if geo_high and kurs_high:
            cluster_to_cat[j] = 1
        elif geo_high and not kurs_high:
            cluster_to_cat[j] = 2
        elif not geo_high and kurs_high:
            cluster_to_cat[j] = 3
        else:
            cluster_to_cat[j] = 4

    # Jika dua kluster mendapat kategori yang sama (bisa terjadi ketika
    # distribusi miring), bedakan berdasarkan magnitude skor gabungan.
    cat_clusters: Dict[int, List[int]] = {}

    for j, cat in cluster_to_cat.items():
        cat_clusters.setdefault(cat, []).append(j)

    # Assign ke per-dokumen
    categories: List[int] = [cluster_to_cat.get(int(lbl), 4) for lbl in labels]

    return categories


def filter_documents(
    docs: List[Dict[str, Any]],
    thresh_geo: Optional[float] = None,
    thresh_kurs: Optional[float] = None,
    keep_only_category_1: bool = True,
    k: int = 4,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pipeline filtering lengkap sesuai README.

    Args:
        docs: list dokumen terstruktur (output cleaning).
        thresh_geo: ambang batas "tinggi" geopolitik (default 0.15).
        thresh_kurs: ambang batas "tinggi" nilai tukar (default 0.15).
        keep_only_category_1: jika True, hanya kembalikan kategori 1.
        k: jumlah kluster K-Means.
        seed: seed agar deterministik.

    Returns:
        (selected, scored_all)
        - selected: dokumen terpilih (kategori 1).
        - scored_all: semua dokumen + skor/kategori (untuk inspeksi).
    """


    if thresh_geo is None:
        thresh_geo = 0.15
    if thresh_kurs is None:
        thresh_kurs = 0.15

    if not docs:
        return [], []

    # Langkah 2-3: hitung probabilitas TF-IDF 2D
    p_geo, p_kurs = topic_probability(docs)

    # Langkah 4: K-Means clustering
    points = np.column_stack([p_geo, p_kurs])
    labels, centroids = kmeans_cluster(points, k=k, seed=seed)

    # Langkah 5: assign kategori semantik & seleksi
    categories = assign_categories(
        p_geo, p_kurs, labels, centroids,
        thresh_geo=thresh_geo, thresh_kurs=thresh_kurs,
    )

    # Perkaya dokumen dengan metadata scoring
    scored_all: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs):
        enriched = dict(doc)
        enriched["p_geopolitik"] = round(p_geo[i], 4)
        enriched["p_kurs"] = round(p_kurs[i], 4)
        enriched["cluster"] = int(labels[i])
        enriched["kategori"] = categories[i]
        scored_all.append(enriched)

    # Pilih kategori 1 (geopolitik tinggi + kurs tinggi)
    if keep_only_category_1:
        selected = [d for d in scored_all if d["kategori"] == 1]
    else:
        selected = list(scored_all)

    return selected, scored_all
