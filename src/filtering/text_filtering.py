"""Text filtering: pilih berita relevan geopolitik + nilai tukar.

Alur (sesuai README / repo workflow):
  1. Keyword per topik (geopolitik & nilai tukar rupiah).
  2. Hitung probabilitas keterkaitan tiap dokumen terhadap kosakata keyword
     masing-masing topik:  P = (#keyword unik yang muncul) / (#kosakata topik).
  3. Petakan skor ke ruang 2D (p_geopolitik, p_kurs).
  4. Cluster titik 2D dengan K-Means (k=4) -> petakan tiap cluster ke 1 dari
     4 kategori (tinggi/rendah x tinggi/rendah) berdasar posisi centroid.
  5. Pilih dokumen kategori 1 (geopolitik tinggi + kurs tinggi).

Hanya memakai stdlib agar ringan (tanpa sklearn).
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Tuple

__all__ = [
    "KEYWORDS_GEOPOLITIK",
    "KEYWORDS_KURS",
    "N_CLUSTERS",
    "DEFAULT_THRESH_GEO",
    "DEFAULT_THRESH_KURS",
    "tokenize",
    "keyword_hits",
    "relevance_score",
    "score_document",
    "score_documents",
    "kmeans_2d",
    "cluster_documents",
    "filter_high_high",
    "filter_documents",
    "CATEGORY_LABELS",
]


# ---------------------------------------------------------------------------
# 1. Keyword per topik (verbatim dari README workflow)
# ---------------------------------------------------------------------------

KEYWORDS_GEOPOLITIK: List[str] = [
    "konflik", "perang", "ancaman", "perdamaian", "diplomasi", "geopolitik",
    "global", "dunia", "internasional", "amerika", "united states", "us",
    "as", "china", "rusia", "ukraina", "timur tengah", "iran", "palestina",
    "gaza", "rafah", "israel", "persaingan", "klaim", "presiden",
    "perdana menteri", "ekspor", "impor", "perdagangan", "pemerintah",
]

KEYWORDS_KURS: List[str] = [
    "rupiah", "dolar as", "nilai tukar", "kurs", "ekonomi", "perdagangan",
    "keuangan", "peningkatan", "inflasi", "mata uang", "global", "penguatan",
    "menguat", "pelemahan", "melemah", "suku bunga", "pasar", "ekspor",
    "impor", "devisa", "valuta", "bank", "dana",
]

CATEGORY_LABELS = {
    1: "1_geopolitik_tinggi_kurs_tinggi",
    2: "2_geopolitik_tinggi_kurs_rendah",
    3: "3_geopolitik_rendah_kurs_tinggi",
    4: "4_geopolitik_rendah_kurs_rendah",
}

#: Jumlah cluster K-Means (langkah 4 workflow: 4 kategori).
N_CLUSTERS = 4

#: Ambang absolut default untuk "tinggi": dokumen cocok >= 15% kosakata topik
#: (~4-5 keyword). Absolut (bukan rata-rata korpus) agar definisi "tinggi"
#: stabil dan kategori 1 tidak kosong pada korpus kecil/miring.
DEFAULT_THRESH_GEO = 0.15
DEFAULT_THRESH_KURS = 0.15


# ---------------------------------------------------------------------------
# 2. Probabilitas keterkaitan
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenisasi sederhana: lowercase + ambil run alfanumerik.
    """

    return _TOKEN_RE.findall((text or "").lower())


def _phrase_pattern(phrase: str) -> re.Pattern:
    """
    Regex pencocokan frasa utuh (word-boundary) case-insensitive.
    """

    norm = re.escape(_normalize(phrase))
    norm = norm.replace(r"\ ", r"\s+")

    return re.compile(rf"(?<![a-z0-9]){norm}(?![a-z0-9])")


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    """
    Kembalikan keyword unik yang muncul di teks (pencocokan frasa utuh).

    Frasa multi-kata seperti "timur tengah" / "suku bunga" dicocokkan sebagai
    satu kesatuan; keyword satu kata butuh word-boundary agar "as" tidak match
    di dalam "pasar".
    """

    lowered = _normalize(text)
    hits: List[str] = []

    for kw in keywords:
        if _phrase_pattern(kw).search(lowered):
            hits.append(_normalize(kw))

    return hits


def relevance_score(text: str, keywords: List[str]) -> float:
    """
    P(topik | dokumen) ~= (#keyword unik muncul) / (#kosakata topik).
    """

    if not keywords:
        return 0.0

    return len(keyword_hits(text, keywords)) / len(keywords)


def document_text(doc: Dict[str, Any]) -> str:
    """
    Gabungkan title + heading + body untuk penakaran relevansi.
    """

    return " ".join(
        str(doc.get(k, "") or "") for k in ("title", "heading", "body")
    )


def score_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tambahkan p_geopolitik, p_kurs, hits, dan koordinat 2D ke dokumen.
    """

    text = document_text(doc)
    hits_geo = keyword_hits(text, KEYWORDS_GEOPOLITIK)
    hits_kurs = keyword_hits(text, KEYWORDS_KURS)
    p_geo = len(hits_geo) / len(KEYWORDS_GEOPOLITIK)
    p_kurs = len(hits_kurs) / len(KEYWORDS_KURS)
    scored = dict(doc)
    scored.update(
        {
            "p_geopolitik": round(p_geo, 4),
            "p_kurs": round(p_kurs, 4),
            "hits_geopolitik": hits_geo,
            "hits_kurs": hits_kurs,
            "score_2d": [round(p_geo, 4), round(p_kurs, 4)],
        }
    )

    return scored


def score_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Langkah 2-3: skor semua dokumen + koordinat 2D.
    """

    return [score_document(d) for d in docs]


# ---------------------------------------------------------------------------
# 4-5. K-Means clustering (k=4) + seleksi kategori 1
# ---------------------------------------------------------------------------

def _sq_dist(a: Tuple[float, float], b: List[float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def kmeans_2d(
    points: List[Tuple[float, float]],
    k: int = N_CLUSTERS,
    max_iter: int = 100,
    seed: int = 0,
) -> Tuple[List[int], List[List[float]]]:
    """K-Means pada titik 2D (implementasi stdlib, inisialisasi k-means++).

    Args:
        points: koordinat (p_geopolitik, p_kurs) tiap dokumen.
        k: jumlah cluster (default :data:`N_CLUSTERS` = 4).
        max_iter: batas iterasi Lloyd.
        seed: seed RNG agar hasil deterministik.

    Returns:
        (labels, centroids): label cluster 0..k-1 per titik dan koordinat
        centroid tiap cluster.
    """
    n = len(points)
    if n == 0:
        return [], []
    k = max(1, min(int(k), n))
    rng = random.Random(seed)

    # Inisialisasi k-means++: centroid pertama acak, sisanya proporsional
    # terhadap kuadrat jarak ke centroid terdekat.
    centroids: List[List[float]] = [list(points[rng.randrange(n)])]
    while len(centroids) < k:
        d2 = [min(_sq_dist(p, c) for c in centroids) for p in points]
        total = sum(d2)
        if total <= 0.0:  # semua titik identik -> duplikasi titik acak
            centroids.append(list(points[rng.randrange(n)]))
            continue
        r = rng.random() * total
        cum = 0.0
        for i, v in enumerate(d2):
            cum += v
            if cum >= r:
                centroids.append(list(points[i]))
                break

    labels = [0] * n
    for _ in range(max_iter):
        new_labels = []
        for p in points:
            best, best_d = 0, _sq_dist(p, centroids[0])
            for ci in range(1, k):
                d = _sq_dist(p, centroids[ci])
                if d < best_d:
                    best, best_d = ci, d
            new_labels.append(best)

        sums = [[0.0, 0.0] for _ in range(k)]
        counts = [0] * k
        for p, lab in zip(points, new_labels):
            sums[lab][0] += p[0]
            sums[lab][1] += p[1]
            counts[lab] += 1

        new_centroids: List[List[float]] = []
        for ci in range(k):
            if counts[ci] > 0:
                new_centroids.append(
                    [sums[ci][0] / counts[ci], sums[ci][1] / counts[ci]]
                )
            else:
                # Cluster kosong -> muat ulang di titik terjauh dari centroidnya.
                far = max(
                    range(n),
                    key=lambda i: min(_sq_dist(points[i], c) for c in centroids),
                )
                new_centroids.append(list(points[far]))

        shift = sum(
            (new_centroids[i][d] - centroids[i][d]) ** 2
            for i in range(k)
            for d in (0, 1)
        )
        centroids, labels = new_centroids, new_labels
        if shift < 1e-12:
            break

    return labels, centroids


def _quadrant_category(cx: float, cy: float, ref_g: float, ref_k: float) -> int:
    if cx >= ref_g and cy >= ref_k:
        return 1
    if cx >= ref_g and cy < ref_k:
        return 2
    if cx < ref_g and cy >= ref_k:
        return 3
    return 4


def cluster_documents(
    scored_docs: List[Dict[str, Any]],
    thresh_geo: float | None = None,
    thresh_kurs: float | None = None,
    *,
    k: int = N_CLUSTERS,
    seed: int = 0,
    max_iter: int = 100,
) -> List[Dict[str, Any]]:
    """Langkah 4: cluster skor 2D dengan K-Means scikit-learn (k=4).

    Tiap cluster K-Means dipetakan ke satu kategori 1-4 berdasar posisi
    centroid-nya relatif terhadap ambang absolut (default 0.15 per sumbu,
    bisa dioverride via ``thresh_geo``/``thresh_kurs``). Ambang absolut
    dipakai agar "tinggi" punya definisi operasional yang stabil — acuan
    rata-rata korpus terbukti mengosongkan kategori 1 pada korpus kecil
    atau miring.

    Catatan: dua cluster bisa terpetakan ke kategori yang sama bila kedua
    centroid jatuh di kuadran yang sama — hal yang wajar pada K-Means dan
    menandakan kategori tersebut terbelah dua secara alami.
    """
    if not scored_docs:
        return []

    ref_g = DEFAULT_THRESH_GEO if thresh_geo is None else float(thresh_geo)
    ref_k = DEFAULT_THRESH_KURS if thresh_kurs is None else float(thresh_kurs)

    geos = [float(d.get("p_geopolitik", 0.0)) for d in scored_docs]
    kurs = [float(d.get("p_kurs", 0.0)) for d in scored_docs]
    points = list(zip(geos, kurs))
    labels, centroids = kmeans_2d(points, k=k, max_iter=max_iter, seed=seed)

    cluster_to_cat = {
        ci: _quadrant_category(cx, cy, ref_g, ref_k)
        for ci, (cx, cy) in enumerate(centroids)
    }

    clustered: List[Dict[str, Any]] = []
    for d, lab in zip(scored_docs, labels):
        cat = cluster_to_cat[lab]
        out = dict(d)
        out["cluster"] = lab
        out["centroid"] = [round(centroids[lab][0], 4), round(centroids[lab][1], 4)]
        out["kategori"] = cat
        out["kategori_label"] = CATEGORY_LABELS[cat]
        clustered.append(out)
    return clustered


def filter_high_high(clustered_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Langkah 5: hanya dokumen kategori 1 (geopolitik & kurs tinggi).
    """

    return [d for d in clustered_docs if d.get("kategori") == 1]


def filter_documents(
    docs: List[Dict[str, Any]],
    thresh_geo: float | None = None,
    thresh_kurs: float | None = None,
    keep_only_category_1: bool = True,
    *,
    k: int = N_CLUSTERS,
    seed: int = 0,
    max_iter: int = 100,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pipeline filtering penuh (skor 2D -> K-Means k=4 -> seleksi kategori 1).

    Returns:
        (selected, scored_all): selected = kategori 1 (atau semua berlabel
        jika keep_only_category_1=False), scored_all = semua dokumen berskor.
    """

    scored = cluster_documents(
        score_documents(docs),
        thresh_geo=thresh_geo,
        thresh_kurs=thresh_kurs,
        k=k,
        seed=seed,
        max_iter=max_iter,
    )

    if keep_only_category_1:
        return filter_high_high(scored), scored

    return scored, scored
