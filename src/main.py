"""Main program: integrasi cleaning -> filtering -> preprocessing.

Contoh:
python src/main.py --demo
python src/main.py --stage all --input data/raw/news --output data/processed/news_final.json
python src/main.py --stage cleaning --input data/raw/news/artikel.html --output data/processed/news_clean.json

Format input yang didukung:
  - file .html tunggal (artikel mentah hasil scraping)
  - direktori berisi file .html
  - file .json berisi list record, masing-masing {"html": ...} atau
    record terstruktur {"title": ..., "body": ...} (paragraf mentah bersih)

Output: JSON list artikel final (default seluruh tahap; --stage memilih tahap).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Agar bisa dijalankan sebagai `python src/main.py` maupun `python -m src.main`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from cleaning.text_cleaning import batch_clean_html
    from filtering.text_filtering import filter_documents
    from preprocessing.text_preprocessing import preprocess_documents
except ImportError:  # fallback saat dijalankan sebagai `python -m src.main`
    from src.cleaning.text_cleaning import batch_clean_html  # type: ignore
    from src.filtering.text_filtering import filter_documents  # type: ignore
    from src.preprocessing.text_preprocessing import preprocess_documents  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "news_final.json"


# ---------------------------------------------------------------------------
# Demo data (3 artikel: relevan ganda, ekonomi saja, tidak relevan)
# ---------------------------------------------------------------------------

def demo_items() -> List[Dict[str, str]]:
    html_relevan = """
    <html><head><title>Fallback Title</title>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "NewsArticle",
     "headline": "Konflik Timur Tengah Picu Pelemahan Rupiah",
     "author": {"@type": "Person", "name": "Aji Pratama"},
     "datePublished": "2024-05-01"}
    </script></head><body>
    <h2>Dolar AS Menguat, Pasar Keuangan Bergejolak</h2>
    <p>Konflik perang di Gaza dan Rafah meningkatkan ancaman global.</p>
    <p>Presiden Amerika dan pemerintah China membahas diplomasi perdamaian.</p>
    <p>Nilai tukar rupiah melemah, kurs dolar AS naik, inflasi dan suku bunga melonjak.</p>
    <p>Info lebih lanjut di https://example.com/berita/1 dan redaksi@example.com.</p>
    <p>Nilai tukar rupiah melemah, kurs dolar AS naik, inflasi dan suku bunga melonjak.</p>
    </body></html>
    """
    html_ekonomi = """
    <html><body>
    <h2>Bank Sentral Jaga Stabilitas Moneter</h2>
    <p>Bank melaporkan peningkatan devisa dan dana valuta.</p>
    <p>Pasar keuangan domestik stabil, ekonomi tumbuh, perdagangan impor ekspor naik.</p>
    </body></html>
    """
    html_sport = """
    <html><body>
    <h2>Hasil Pertandingan Sepak Bola Tadi Malam</h2>
    <p>Skor akhir pertandingan sepak bola tadi malam sangat seru dan menghibur penonton stadion.</p>
    </body></html>
    """
    return [
        {"html": html_relevan, "source": "demo"},
        {"html": html_ekonomi, "source": "demo"},
        {"html": html_sport, "source": "demo"},
    ]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_items(input_path: Path) -> List[Dict[str, str]]:
    if input_path.is_file() and input_path.suffix.lower() == ".html":
        return [{"html": input_path.read_text(encoding="utf-8"), "source": input_path.name}]
    
    if input_path.is_dir():
        items = []
        
        for f in sorted(input_path.glob("*.html")):
            items.append({"html": f.read_text(encoding="utf-8"), "source": f.name})
        
        if not items:
            raise FileNotFoundError(f"Tidak ada file .html di {input_path}")
        
        return items
    
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        
        if isinstance(data, dict):
            data = [data]
        
        if not isinstance(data, list):
            raise ValueError("JSON input harus list of records.")
        
        return data
    
    raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")


def save_json(docs: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv_preview(docs: List[Dict[str, Any]], path: Path) -> None:
    """
    CSV ringkas: title, kategori, skor, text_final (dipotong 500 char).
    """
    
    if not docs:
        return
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["title", "kategori", "p_geopolitik", "p_kurs", "text_final"]
        )
        w.writeheader()
        
        for d in docs:
            w.writerow(
                {
                    "title": d.get("title", ""),
                    "kategori": d.get("kategori", ""),
                    "p_geopolitik": d.get("p_geopolitik", ""),
                    "p_kurs": d.get("p_kurs", ""),
                    "text_final": str(d.get("text_final", ""))[:500],
                }
            )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    items: List[Dict[str, str]],
    stage: str = "all",
    thresh_geo: float | None = None,
    thresh_kurs: float | None = None,
    jaccard_threshold: float = 0.8,
    k: int = 4,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Jalankan tahap sesuai pilihan; kembalikan dict hasil tiap tahap.
    """
    
    result: Dict[str, Any] = {"cleaned": [], "scored_all": [], "selected": [], "final": []}

    cleaned = batch_clean_html(items)
    result["cleaned"] = cleaned
    print(f"[cleaning] {len(cleaned)} dokumen -> title/body terstruktur.")

    if stage == "cleaning":
        result["final"] = cleaned
        return result

    selected, scored_all = filter_documents(
        cleaned, thresh_geo=thresh_geo, thresh_kurs=thresh_kurs,
        keep_only_category_1=True, k=k, seed=seed,
    )
    
    result["scored_all"] = scored_all
    result["selected"] = selected
    cats = [d.get("kategori") for d in scored_all]
    
    print(f"[filtering] {len(scored_all)} diskoring, {len(selected)} kategori-1 "
          f"(distribusi kategori: {{1:{cats.count(1)}, 2:{cats.count(2)}, "
          f"3:{cats.count(3)}, 4:{cats.count(4)}}}).")
    
    for d in scored_all:
        print(f"  - kat={d.get('kategori')} cluster={d.get('cluster')} "
              f"p_geo={d.get('p_geopolitik')} "
              f"p_kurs={d.get('p_kurs')} | {(d.get('title') or '')[:60]}")

    if stage == "filtering":
        result["final"] = selected
        return result

    final = preprocess_documents(selected, jaccard_threshold=jaccard_threshold)
    result["final"] = final
    
    print(f"[preprocessing] {len(final)} dokumen final (dedup+mask+fold).")
    
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline berita: cleaning -> filtering -> preprocessing")
    parser.add_argument("--stage", choices=["cleaning", "filtering", "preprocessing", "all"],
                        default="all", help="Tahap yang dijalankan (default: all).")
    parser.add_argument("--input", type=str, default="",
                        help="File .html / .json atau direktori .html.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Path output JSON.")
    parser.add_argument("--csv", type=str, default="",
                        help="Path output CSV ringkas (opsional).")
    parser.add_argument("--save-stages", action="store_true",
                        help="Simpan juga artefak per tahap di sebelah output: "
                             "<nama>_clean.json, <nama>_scored.json, <nama>_selected.json.")
    parser.add_argument("--demo", action="store_true", help="Jalankan dengan 3 artikel contoh.")
    parser.add_argument("--thresh-geo", type=float, default=None,
                        help="Ambang 'tinggi' geopolitik (default: 0.15).")
    parser.add_argument("--thresh-kurs", type=float, default=None,
                        help="Ambang 'tinggi' kurs (default: 0.15).")
    parser.add_argument("--clusters", type=int, default=4,
                        help="Jumlah cluster K-Means (default: 4).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed K-Means agar deterministik (default: 0).")
    parser.add_argument("--jaccard", type=float, default=0.8)
    args = parser.parse_args(argv)

    if args.demo or not args.input:
        items: List[Dict[str, str]] = demo_items()
        if not args.demo:
            print("Tidak ada --input: memakai data demo bawaan (setara --demo).")
    else:
        items = load_items(Path(args.input))

    # Stage preprocessing saja = input diasumsikan sudah terstruktur bersih.
    if args.stage == "preprocessing":
        from cleaning.text_cleaning import clean_structured_record  # noqa: E402

        cleaned_like = [
            clean_structured_record(it) if "body" in it else it for it in items
        ]
        final = preprocess_documents(cleaned_like, args.jaccard)
        result = {"final": final}
        print(f"[preprocessing] {len(final)} dokumen final.")
    
    else:
        result = run_pipeline(
            items, stage=args.stage, thresh_geo=args.thresh_geo,
            thresh_kurs=args.thresh_kurs, jaccard_threshold=args.jaccard,
            k=args.clusters, seed=args.seed,
        )

    out_path = Path(args.output)
    save_json(result["final"], out_path)
    print(f"Tersimpan: {out_path} ({len(result['final'])} dokumen).")

    if args.save_stages:
        stem = out_path.with_suffix("")
        for key, suffix in (("cleaned", "_clean"), ("scored_all", "_scored"),
                            ("selected", "_selected")):
            docs = result.get(key, [])
            if docs:
                stage_path = Path(str(stem) + suffix + ".json")
                save_json(docs, stage_path)
                print(f"Tersimpan: {stage_path} ({len(docs)} dokumen).")
    
    if args.csv:
        save_csv_preview(result["final"], Path(args.csv))
        print(f"CSV ringkas: {args.csv}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
