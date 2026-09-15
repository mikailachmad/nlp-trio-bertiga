"""Text preprocessing untuk berita terpilih (hasil cleaning + filtering).

Alur (sesuai README / repo workflow):
  1. Deduplikasi paragraf: exact + near-duplication tingkat blok (Jaccard).
  2. Penandaan entitas khusus: URL -> "URL", email -> "EMAIL".
  3. Case folding: lowercase seluruh teks.
  4. Retensi entitas & stopwords: nama, angka, stopword DIPERTAHANKAN
     (tidak ada penghapusan); entitas diekstrak ke field terpisah sebagai bukti.

Hanya memakai stdlib.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

__all__ = [
    "split_paragraphs",
    "jaccard_similarity",
    "deduplicate_paragraphs",
    "mask_special_entities",
    "case_fold",
    "extract_preserved_entities",
    "preprocess_text",
    "preprocess_article",
    "preprocess_documents",
]


# ---------------------------------------------------------------------------
# 1. Deduplikasi paragraf (exact + near)
# ---------------------------------------------------------------------------

def split_paragraphs(text: str) -> List[str]:
    """
    Pecah teks menjadi blok paragraf (pemisah newline / newline ganda).
    """
    
    if not text:
        return []
    
    # Normalisasi pemisah dulu agar satu newline pun jadi batas blok.
    parts = re.split(r"\n+", text)
    
    return [p.strip() for p in parts if p.strip()]


def _shingles(tokens: List[str], n: int = 3) -> set:
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(a: str, b: str) -> float:
    """
    Kemiripan Jaccard 3-shingle antara dua paragraf (case-insensitive).
    """
    
    ta = re.findall(r"[a-z0-9]+", (a or "").lower())
    tb = re.findall(r"[a-z0-9]+", (b or "").lower())
    sa, sb = _shingles(ta), _shingles(tb)
    
    if not sa or not sb:
        return 1.0 if (a or "").strip() == (b or "").strip() else 0.0
 
    return len(sa & sb) / len(sa | sb)


def deduplicate_paragraphs(
    paragraphs: List[str], jaccard_threshold: float = 0.8
) -> Tuple[List[str], int, int]:
    """
    Hapus duplikat persis + near-duplikat (Jaccard >= threshold).

    Returns:
        (unik, n_exact_dihapus, n_near_dihapus)
    """
    
    unique: List[str] = []
    seen_exact: set = set()
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
# 2. Penandaan entitas khusus
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+|(?:[a-z0-9-]+\.)+(?:com|id|net|org|co|go|io)\S*", re.IGNORECASE)


def mask_special_entities(text: str) -> str:
    """
    Ubah email -> EMAIL dan URL -> URL (email dulu agar tidak tertelan URL).
    """
    
    if not text:
        return ""
    
    masked = EMAIL_RE.sub("EMAIL", text)
    masked = URL_RE.sub("URL", masked)
    
    return masked


# ---------------------------------------------------------------------------
# 3-4. Case folding + retensi
# ---------------------------------------------------------------------------

def case_fold(text: str) -> str:
    """Penyeragaman huruf kecil (langkah 3)."""
    
    return (text or "").lower()


_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def extract_preserved_entities(text: str) -> Dict[str, List[str]]:
    """
    Ekstrak nama/angka yang DIPERTAHANKAN di teks (langkah 4: retensi).

    Workflow menetapkan stopword, nama, dan angka tidak dihapus, jadi fungsi
    ini hanya mencatat bukti retensi tanpa memodifikasi teks.
    """
    
    source = text or ""
    capitals = sorted(set(_ENTITY_RE.findall(source)))
    numbers = sorted(set(_NUMBER_RE.findall(source)))
    
    return {"capitalized_candidates": capitals, "numbers": numbers}


def preprocess_text(
    text: str, jaccard_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Pipeline preprocessing satu string: dedup -> mask -> fold + bukti retensi.
    """
    
    paragraphs = split_paragraphs(text or "")
    unique, n_exact, n_near = deduplicate_paragraphs(paragraphs, jaccard_threshold)
    deduped = "\n\n".join(unique)
    masked = mask_special_entities(deduped)
    preserved = extract_preserved_entities(masked)
    final = case_fold(masked)
    
    return {
        "n_paragraf_awal": len(paragraphs),
        "n_paragraf_akhir": len(unique),
        "n_exact_dihapus": n_exact,
        "n_near_dihapus": n_near,
        "preserved_entities": preserved,
        "final": final,
    }


TEXT_FIELDS = ("title", "heading", "body")


def preprocess_article(
    article: Dict[str, Any], jaccard_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Preprocess satu artikel terstruktur; tambah field *_clean + text_final.
    """
    
    out = dict(article)
    combined_paras: List[str] = []
    
    for field in TEXT_FIELDS:
        res = preprocess_text(
            str(article.get(field, "") or ""), jaccard_threshold=jaccard_threshold
        )
        out[f"{field}_clean"] = res["final"]
        out[f"{field}_preprocess_info"] = {
            k: v for k, v in res.items() if k != "final"
        }
        if res["final"]:
            combined_paras.append(res["final"])
    
    # Body final = gabungan blok unik agar deduplikasi lintas field ikut tertangani.
    merged, n_exact, n_near = deduplicate_paragraphs(
        split_paragraphs("\n\n".join(combined_paras)), jaccard_threshold
    )
    
    out["text_final"] = "\n\n".join(merged)
    out["preprocess_summary"] = {
        "n_exact_dihapus": n_exact,
        "n_near_dihapus": n_near,
        "retensi": "stopword, nama entitas, dan angka dipertahankan",
    }
    
    return out


def preprocess_documents(
    docs: List[Dict[str, Any]], jaccard_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Preprocess batch dokumen.
    """
    
    return [preprocess_article(d, jaccard_threshold) for d in docs]
