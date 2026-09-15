"""
Text cleaning untuk berita geopolitik.

Alur (sesuai README / repo workflow):
  1. Ekstraksi komponen utama: headline dari ``<script type="application/ld+json">``,
     konten dari elemen ``<p>`` dan ``<h2>``.
  2. Pembersihan tag HTML (isolasi paragraf murni).
  3. Perbaikan karakter korup / tidak valid.
  4. Normalisasi teks & spasi (Unicode NFKC + collapse whitespace).
  5. Strukturisasi ke dict: title, heading, body, author, publish_date.

Catatan README: jika dataset awal sudah berupa paragraf mentah yang bersih,
tahap ini boleh dilewati (lihat :func:`is_already_clean_record`).

Hanya memakai stdlib (re, json, html, unicodedata, html.parser) agar bisa
jalan tanpa instalasi tambahan.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "extract_headline",
    "extract_author_and_date",
    "extract_body",
    "strip_html_tags",
    "fix_characters",
    "normalize_text",
    "structure_article",
    "clean_html",
    "clean_structured_record",
    "is_already_clean_record",
    "batch_clean_html",
]


# ---------------------------------------------------------------------------
# 1. Ekstraksi komponen utama
# ---------------------------------------------------------------------------

LD_JSON_SCRIPT_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)

TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
H1_TAG_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
META_RE_TMPL = (
    r'<meta[^>]*?(?:name|property)=["\']{key}["\'][^>]*?content=["\'](.*?)["\']'
    r"|<meta[^>]*?content=[\"'](.*?)[\"'][^>]*?(?:name|property)=[\"']{key}[\"']"
)


def _iter_ld_json_objects(html_text: str) -> List[Dict[str, Any]]:
    
    """
    Ambil semua objek JSON-LD dari tag script.
    """
    objects: List[Dict[str, Any]] = []
    
    for match in LD_JSON_SCRIPT_RE.finditer(html_text or ""):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            objects.extend([o for o in parsed if isinstance(o, dict)])
        elif isinstance(parsed, dict):
            # Tangani @graph ala schema.org
            if isinstance(parsed.get("@graph"), list):
                objects.extend([o for o in parsed["@graph"] if isinstance(o, dict)])
            else:
                objects.append(parsed)
                
    return objects


def _pick_ld_value(obj: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        if obj.get(key) not in (None, ""):
            return obj[key]
        
    return None


def extract_headline(html_text: str) -> str:
    
    """
    Ambil headline: prioritas JSON-LD (headline/title/name), fallback <title>/<h1>.
    """
    for obj in _iter_ld_json_objects(html_text or ""):
        # NewsArticle sering nested: {"@type": "NewsArticle", "headline": ...}
        candidates: List[Dict[str, Any]] = [obj]
        if isinstance(obj.get("mainEntity"), dict):
            candidates.append(obj["mainEntity"])
        for cand in candidates:
            val = _pick_ld_value(cand, ("headline", "title", "name"))
            if isinstance(val, str) and val.strip():
                return strip_html_tags(val).strip()
            
    for pattern in (H1_TAG_RE, TITLE_TAG_RE):
        m = pattern.search(html_text or "")
        if m and m.group(1).strip():
            return strip_html_tags(m.group(1)).strip()
        
    return ""


def _meta_content(html_text: str, key: str) -> str:
    pattern = META_RE_TMPL.format(key=re.escape(key))
    
    m = re.search(pattern, html_text or "", re.IGNORECASE | re.DOTALL)
    
    if not m:
        return ""
    
    return (m.group(1) or m.group(2) or "").strip()


def extract_author_and_date(html_text: str) -> Tuple[str, str]:
    
    """
    Ambil (author, publish_date) dari JSON-LD, fallback ke meta tag.
    """
    author, publish_date = "", ""
    
    for obj in _iter_ld_json_objects(html_text or ""):
        raw_author = _pick_ld_value(obj, ("author",))
        if raw_author and not author:
            
            if isinstance(raw_author, dict):
                author = str(raw_author.get("name", "") or "")
            elif isinstance(raw_author, list):
                names = [
                    a.get("name", "") if isinstance(a, dict) else str(a)
                    for a in raw_author
                ]
                author = ", ".join(n for n in names if n)
            else:
                author = str(raw_author)
        
        raw_date = _pick_ld_value(obj, ("datePublished", "dateCreated", "uploadDate"))
        
        if raw_date and not publish_date:
            publish_date = str(raw_date)
        
        if author and publish_date:
            break
        
    if not author:
        author = (
            _meta_content(html_text or "", "author")
            or _meta_content(html_text or "", "article:author")
        )
    if not publish_date:
        publish_date = (
            _meta_content(html_text or "", "article:published_time")
            or _meta_content(html_text or "", "publish_date")
            or _meta_content(html_text or "", "date")
        )
        
    return strip_html_tags(author).strip(), strip_html_tags(publish_date).strip()


class _ParagraphExtractor(HTMLParser):
    """
    Kumpulkan teks dari <p> dan <h2> saja (langkah 1 workflow).
    """

    def __init__(self) -> None:
        super().__init__()
        self._current: Optional[str] = None
        self._buf: List[str] = []
        self.paragraphs: List[str] = []
        self.headings: List[str] = []
        self._skip = False  # abaikan isi <script> / <style>

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in ("script", "style"):
            self._skip = True
            return
        if tag in ("p", "h2"):
            self._current = tag
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("script", "style"):
            self._skip = False
            return
        if tag == self._current:
            text = "".join(self._buf).strip()
            if text:
                if tag == "h2":
                    self.headings.append(text)
                else:
                    self.paragraphs.append(text)
            self._current = None
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._skip or self._current is None:
            return
        self._buf.append(data)


def extract_body(html_text: str) -> Tuple[List[str], str]:
    """
    Ekstrak (headings, body) dari <h2> dan <p>. 
    Body = paragraf digabung newline.
    """
    
    parser = _ParagraphExtractor()
    try:
        parser.feed(html_text or "")
    except Exception:
        pass
    
    body = "\n\n".join(p for p in parser.paragraphs if p.strip())
    
    return parser.headings, body


# ---------------------------------------------------------------------------
# 2-4. Pembersihan tag, perbaikan karakter, normalisasi
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<[^>]+>")

# Pemetaan mojibake / tanda kutip pintar yang umum muncul di hasil scraping.
_MOJIBAKE_MAP = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â ": " ",
    "Â": "",
    "\u00a0": " ",  # nbsp
    "\u200b": "",  # zero-width space
    "\ufeff": "",  # BOM
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
}

WHITESPACE_RE = re.compile(r"\s+")


def strip_html_tags(text: str) -> str:
    
    """
    Hilangkan sisa tag HTML (langkah 2) + unescape entitas HTML.
    """
    if not text:
        return ""
    no_tags = TAG_RE.sub("", text)
    
    return html.unescape(no_tags)


def fix_characters(text: str) -> str:
    
    """
    Perbaiki karakter korup/tidak valid (langkah 3).
    """
    if not text:
        return ""
    
    fixed = html.unescape(text)
    
    for bad, good in _MOJIBAKE_MAP.items():
        if bad in fixed:
            fixed = fixed.replace(bad, good)
            
    # Buang karakter kontrol tak tercetak kecuali newline/tab.
    fixed = "".join(
        ch for ch in fixed if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
    )
    
    return fixed


def normalize_text(text: str) -> str:
    
    """
    Normalisasi Unicode NFKC + ubah newline/tab/indentasi jadi spasi tunggal.

    Untuk struktur multi-paragraf gunakan ``preserve_paragraphs=True`` versi
    :func:`normalize_paragraphs`.
    """
    
    if not text:
        return ""
    
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    normalized = WHITESPACE_RE.sub(" ", normalized)
    
    return normalized.strip()


def normalize_paragraphs(text: str) -> str:
    """
    Varian normalisasi yang mempertahankan batas paragraf (newline ganda).
    """
    
    if not text:
        return ""
    
    text = unicodedata.normalize("NFKC", fix_characters(strip_html_tags(text)))
    paras = [WHITESPACE_RE.sub(" ", p).strip() for p in re.split(r"\n\s*\n", text)]
    
    return "\n\n".join(p for p in paras if p)


# ---------------------------------------------------------------------------
# 5. Strukturisasi
# ---------------------------------------------------------------------------

ARTICLE_FIELDS = ("title", "heading", "body", "author", "publish_date")


def structure_article(
    title: str = "",
    heading: str | List[str] = "",
    body: str = "",
    author: str = "",
    publish_date: str = "",
    url: str = "",
    source: str = "",
) -> Dict[str, str]:
    
    """
    Susun hasil cleaning ke format structured (langkah 5).
    """
    
    if isinstance(heading, list):
        heading = "\n".join(h for h in heading if h.strip())
        
    return {
        "title": normalize_text(fix_characters(strip_html_tags(title or ""))),
        "heading": normalize_text(fix_characters(strip_html_tags(heading or ""))),
        "body": normalize_paragraphs(body or ""),
        "author": normalize_text(fix_characters(strip_html_tags(author or ""))),
        "publish_date": normalize_text(publish_date or ""),
        "url": (url or "").strip(),
        "source": (source or "").strip(),
    }


def clean_html(
    html_text: str, url: str = "", source: str = ""
) -> Dict[str, str]:
    
    """
    Pipeline penuh cleaning untuk satu dokumen HTML mentah (langkah 1-5).
    """
    
    title_raw = extract_headline(html_text or "")
    headings, body_raw = extract_body(html_text or "")
    author_raw, date_raw = extract_author_and_date(html_text or "")
    
    return structure_article(
        title=title_raw,
        heading=headings,
        body=body_raw,
        author=author_raw,
        publish_date=date_raw,
        url=url,
        source=source,
    )


def is_already_clean_record(record: Dict[str, Any]) -> bool:
    
    """
    True jika record sudah berupa paragraf mentah bersih (tanpa tag HTML).

    Sesuai catatan README: tahap cleaning boleh dilewati untuk data seperti ini.
    """
    
    if not isinstance(record, dict):
        return False
    
    has_body = bool(str(record.get("body", "") or "").strip())
    combined = " ".join(str(record.get(k, "") or "") for k in ("title", "body"))
    
    return has_body and "<" not in combined and ">" not in combined


def clean_structured_record(record: Dict[str, Any]) -> Dict[str, str]:
    
    """
    Bersihkan record yang sudah terstruktur (tanpa HTML): fix + normalisasi saja.
    """
    
    return structure_article(
        title=str(record.get("title", "") or ""),
        heading=str(record.get("heading", "") or ""),
        body=str(record.get("body", "") or ""),
        author=str(record.get("author", "") or ""),
        publish_date=str(record.get("publish_date", "") or ""),
        url=str(record.get("url", "") or ""),
        source=str(record.get("source", "") or ""),
    )


def batch_clean_html(
    items: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    
    """Bersihkan banyak dokumen. Tiap item: {"html": ..., "url": ..., "source": ...}.

    Jika item sudah berupa record bersih (ada "body" tanpa HTML), lewati
    ekstraksi dan hanya normalisasi (sesuai catatan README).
    """
    
    cleaned: List[Dict[str, str]] = []
    
    for item in items:
        if is_already_clean_record(item):
            cleaned.append(clean_structured_record(item))
        else:
            cleaned.append(
                clean_html(
                    item.get("html", ""),
                    url=item.get("url", ""),
                    source=item.get("source", ""),
                )
            )
            
    return cleaned