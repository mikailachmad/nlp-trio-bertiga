"""
Visualisasi tren kurs USD/IDR (BI JISDOR).

Modul ini menyediakan tiga fungsi utama yang bersifat modular dan independen:

  1. ``plot_trend_harian(df)``
     Line chart resolusi harian seluruh rentang data.

  2. ``plot_trend_tahunan(df)``
     Bar chart rata-rata kurs per tahun, memudahkan melihat tren tahunan.

  3. ``plot_with_events(df, events)``
     Overlay event/berita geopolitik di atas chart harian.
     Scaffold ini sudah siap dipakai begitu dataset berita tersedia;
     input ``events`` berupa list dict dengan kunci ``date`` dan ``label``.

Semua fungsi menghasilkan dua format output:
  - PNG (via matplotlib) untuk keperluan laporan statis.
  - HTML interaktif (via plotly) untuk eksplorasi data (zoom, hover).

Cara pakai:
  python -m src.jisdor.visualize_jisdor
  python -m src.jisdor.visualize_jisdor --output-dir outputs/

Dependensi:
  pip install pandas matplotlib plotly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError: 
    _PLOTLY_AVAILABLE = False

__all__ = [
    "load_jisdor_for_plot",
    "plot_trend_harian",
    "plot_trend_tahunan",
    "plot_with_events",
]


# Konstanta path default
_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV  = _REPO_ROOT / "data" / "processed" / "jisdor_clean.csv"
DEFAULT_OUTDIR = _REPO_ROOT / "data" / "processed"

COL_DATE = "tanggal"
COL_RATE = "kurs_jisdor"

# Palet warna konsisten di seluruh chart
_COLOR_MAIN   = "#1f4e79"   # biru tua — garis utama kurs
_COLOR_ACCENT = "#e74c3c"   # merah — marker event berita
_COLOR_BAR    = "#2980b9"   # biru medium — bar tahunan
_COLOR_GRID   = "#e0e0e0"   # abu-abu — gridline


# Loader
def load_jisdor_for_plot(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """
    Baca CSV JISDOR dan pastikan kolom sudah dalam tipe yang benar untuk plot.

    Parameters
    ----------
    path : str | Path
        Path ke file CSV (kolom: tanggal, kurs_jisdor).

    Returns
    -------
    pd.DataFrame
        DataFrame dengan ``tanggal`` (datetime64) dan ``kurs_jisdor`` (int/float).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df[COL_RATE] = pd.to_numeric(df[COL_RATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_RATE]).sort_values(COL_DATE).reset_index(drop=True)
    return df


# 1. plot_trend_harian — line chart resolusi harian
def plot_trend_harian(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTDIR,
    show: bool = False,
) -> Dict[str, Path]:
    """
    Buat line chart kurs harian USD/IDR (JISDOR) seluruh rentang data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom ``tanggal`` dan ``kurs_jisdor``.
    output_dir : str | Path
        Direktori penyimpanan output.
    show : bool
        Jika True, tampilkan chart secara interaktif (untuk Jupyter/IDE).

    Returns
    -------
    dict
        ``{"png": Path, "html": Path}`` — path file yang tersimpan.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tgl_awal = df[COL_DATE].min().strftime("%B %Y")
    tgl_akhir = df[COL_DATE].max().strftime("%B %Y")
    judul     = f"Kurs USD/IDR (JISDOR) — {tgl_awal} s.d. {tgl_akhir}"

    #  Matplotlib (PNG) 
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df[COL_DATE], df[COL_RATE], linewidth=1.2, color=_COLOR_MAIN)
    ax.fill_between(df[COL_DATE], df[COL_RATE], alpha=0.08, color=_COLOR_MAIN)

    ax.set_title(judul, fontsize=13, pad=14)
    ax.set_xlabel("Tanggal", labelpad=8)
    ax.set_ylabel("Kurs (IDR per USD)", labelpad=8)
    ax.grid(True, color=_COLOR_GRID, linewidth=0.7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    png_path = output_dir / "jisdor_trend_harian.png"
    fig.savefig(png_path, dpi=150)
    print(f"[visualize] PNG tersimpan: {png_path}")
    if show:
        plt.show()
    plt.close(fig)

    # Plotly (HTML interaktif) 
    html_path: Optional[Path] = None
    if _PLOTLY_AVAILABLE:
        fig_px = px.line(
            df,
            x=COL_DATE,
            y=COL_RATE,
            title=judul,
            labels={COL_DATE: "Tanggal", COL_RATE: "Kurs (IDR/USD)"},
            template="plotly_white",
        )
        fig_px.update_traces(line=dict(color=_COLOR_MAIN, width=1.5))
        fig_px.update_layout(
            hovermode="x unified",
            yaxis_tickformat=",",
            font=dict(family="Arial, sans-serif", size=12),
        )
        html_path = output_dir / "jisdor_trend_harian.html"
        fig_px.write_html(str(html_path))
        print(f"[visualize] HTML interaktif tersimpan: {html_path}")
    else:
        print("[visualize] Plotly tidak tersedia; HTML tidak dibuat (pip install plotly).")

    return {"png": png_path, "html": html_path}

# 2. plot_trend_tahunan — bar chart rata-rata per tahun
def plot_trend_tahunan(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTDIR,
    show: bool = False,
) -> Dict[str, Path]:
    """
    Buat bar chart rata-rata kurs tahunan untuk menampilkan tren jangka panjang.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom ``tanggal`` dan ``kurs_jisdor``.
    output_dir : str | Path
        Direktori penyimpanan output.
    show : bool
        Jika True, tampilkan chart secara interaktif.

    Returns
    -------
    dict
        ``{"png": Path, "html": Path}`` — path file yang tersimpan.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Agregasi rata-rata per tahun
    df_year = (
        df.copy()
        .assign(tahun=df[COL_DATE].dt.year)
        .groupby("tahun")[COL_RATE]
        .mean()
        .round(0)
        .astype(int)
        .reset_index()
    )

    judul = "Rata-rata Kurs USD/IDR (JISDOR) per Tahun"

    # Matplotlib (PNG) 
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        df_year["tahun"].astype(str),
        df_year[COL_RATE],
        color=_COLOR_BAR,
        edgecolor="white",
        width=0.6,
    )

    # Label nilai di atas batang
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(judul, fontsize=13, pad=14)
    ax.set_xlabel("Tahun", labelpad=8)
    ax.set_ylabel("Rata-rata Kurs (IDR per USD)", labelpad=8)
    ax.grid(axis="y", color=_COLOR_GRID, linewidth=0.7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_axisbelow(True)
    plt.tight_layout()

    png_path = output_dir / "jisdor_trend_tahunan.png"
    fig.savefig(png_path, dpi=150)
    print(f"[visualize] PNG tersimpan: {png_path}")
    if show:
        plt.show()
    plt.close(fig)

    #  Plotly (HTML interaktif)
    html_path: Optional[Path] = None
    if _PLOTLY_AVAILABLE:
        fig_px = px.bar(
            df_year,
            x="tahun",
            y=COL_RATE,
            title=judul,
            labels={"tahun": "Tahun", COL_RATE: "Rata-rata Kurs (IDR/USD)"},
            template="plotly_white",
            text=COL_RATE,
        )
        fig_px.update_traces(marker_color=_COLOR_BAR, textposition="outside")
        fig_px.update_layout(
            yaxis_tickformat=",",
            font=dict(family="Arial, sans-serif", size=12),
            xaxis=dict(type="category"),
        )
        html_path = output_dir / "jisdor_trend_tahunan.html"
        fig_px.write_html(str(html_path))
        print(f"[visualize] HTML interaktif tersimpan: {html_path}")
    else:
        print("[visualize] Plotly tidak tersedia; HTML tidak dibuat (pip install plotly).")

    return {"png": png_path, "html": html_path}

# 3. plot_with_events — overlay berita geopolitik di atas chart harian
def plot_with_events(
    df: pd.DataFrame,
    events: List[Dict[str, Any]],
    output_dir: str | Path = DEFAULT_OUTDIR,
    show: bool = False,
) -> Dict[str, Path]:
    """
    Buat chart harian dengan overlay marker event/berita geopolitik.

    Scaffold ini siap dipakai begitu dataset berita tersedia. Saat ini
    menerima parameter ``events`` sebagai list dict sehingga integrasi
    dengan output ``align_news_to_kurs`` cukup satu baris konversi.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame kurs harian (kolom: tanggal, kurs_jisdor).
    events : list[dict]
        Daftar event berita. Setiap dict memiliki:
        - ``date`` (str | datetime | date): tanggal efektif trading.
        - ``label`` (str): label singkat event (misal judul berita, maks 40 karakter).
        Contoh:
            [
                {"date": "2022-02-24", "label": "Rusia invasi Ukraina"},
                {"date": "2023-10-07", "label": "Konflik Hamas-Israel"},
            ]
    output_dir : str | Path
        Direktori penyimpanan output.
    show : bool
        Jika True, tampilkan chart secara interaktif.

    Returns
    -------
    dict
        ``{"png": Path, "html": Path}`` — path file yang tersimpan.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tgl_awal  = df[COL_DATE].min().strftime("%B %Y")
    tgl_akhir = df[COL_DATE].max().strftime("%B %Y")
    judul     = f"Kurs JISDOR & Event Geopolitik — {tgl_awal} s.d. {tgl_akhir}"

    # Konversi events ke DataFrame agar mudah join
    if events:
        df_ev = pd.DataFrame(events)
        df_ev["date"] = pd.to_datetime(df_ev["date"], errors="coerce")
        # Ambil kurs pada tanggal event (jika ada)
        df_ev = df_ev.merge(
            df.rename(columns={COL_DATE: "date", COL_RATE: "_rate"}),
            on="date",
            how="left",
        )
    else:
        df_ev = pd.DataFrame(columns=["date", "label", "_rate"])

    # Matplotlib (PNG)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df[COL_DATE], df[COL_RATE], linewidth=1.2, color=_COLOR_MAIN, zorder=2)
    ax.fill_between(df[COL_DATE], df[COL_RATE], alpha=0.07, color=_COLOR_MAIN)

    # Marker event
    if not df_ev.empty and "_rate" in df_ev.columns:
        ev_valid = df_ev.dropna(subset=["_rate"])
        ax.scatter(
            ev_valid["date"],
            ev_valid["_rate"],
            color=_COLOR_ACCENT,
            s=60,
            zorder=5,
            label="Event Geopolitik",
        )
        for _, row in ev_valid.iterrows():
            ax.annotate(
                str(row["label"])[:40],
                xy=(row["date"], row["_rate"]),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=7,
                color=_COLOR_ACCENT,
                rotation=30,
            )

    ax.set_title(judul, fontsize=13, pad=14)
    ax.set_xlabel("Tanggal", labelpad=8)
    ax.set_ylabel("Kurs (IDR per USD)", labelpad=8)
    ax.grid(True, color=_COLOR_GRID, linewidth=0.7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")
    if not df_ev.empty:
        ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    png_path = output_dir / "jisdor_with_events.png"
    fig.savefig(png_path, dpi=150)
    print(f"[visualize] PNG tersimpan: {png_path}")
    if show:
        plt.show()
    plt.close(fig)

    # Plotly (HTML interaktif)
    html_path: Optional[Path] = None
    if _PLOTLY_AVAILABLE:
        fig_go = go.Figure()

        # Garis kurs
        fig_go.add_trace(
            go.Scatter(
                x=df[COL_DATE],
                y=df[COL_RATE],
                mode="lines",
                name="Kurs JISDOR",
                line=dict(color=_COLOR_MAIN, width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>Kurs: %{y:,}<extra></extra>",
            )
        )

        # Scatter event
        if not df_ev.empty and "_rate" in df_ev.columns:
            ev_valid = df_ev.dropna(subset=["_rate"])
            fig_go.add_trace(
                go.Scatter(
                    x=ev_valid["date"],
                    y=ev_valid["_rate"],
                    mode="markers+text",
                    name="Event Geopolitik",
                    marker=dict(color=_COLOR_ACCENT, size=10, symbol="circle"),
                    text=ev_valid["label"].str[:30],
                    textposition="top center",
                    textfont=dict(size=9),
                    hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>Kurs: %{y:,}<extra></extra>",
                )
            )

        fig_go.update_layout(
            title=judul,
            xaxis_title="Tanggal",
            yaxis_title="Kurs (IDR/USD)",
            yaxis_tickformat=",",
            template="plotly_white",
            hovermode="x unified",
            font=dict(family="Arial, sans-serif", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        html_path = output_dir / "jisdor_with_events.html"
        fig_go.write_html(str(html_path))
        print(f"[visualize] HTML interaktif tersimpan: {html_path}")
    else:
        print("[visualize] Plotly tidak tersedia; HTML tidak dibuat (pip install plotly).")

    return {"png": png_path, "html": html_path}


# CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualisasi tren kurs JISDOR (PNG + HTML interaktif)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_CSV),
        help=f"Path CSV kurs (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help=f"Direktori output (default: {DEFAULT_OUTDIR}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Tampilkan chart secara interaktif (matplotlib).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        df = load_jisdor_for_plot(args.input)
    except FileNotFoundError as exc:
        print(f"[visualize] ERROR: {exc}", file=sys.stderr)
        return 1

    out = Path(args.output_dir)

    print("[visualize] Membuat chart harian...")
    plot_trend_harian(df, output_dir=out, show=args.show)

    print("[visualize] Membuat chart tahunan...")
    plot_trend_tahunan(df, output_dir=out, show=args.show)

    print("[visualize] Membuat chart harian + overlay event (tanpa events — scaffold)...")
    plot_with_events(df, events=[], output_dir=out, show=args.show)

    print("[visualize] Selesai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
