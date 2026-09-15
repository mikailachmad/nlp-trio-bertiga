"""
Fungsi penyelarasan temporal: berita geopolitik <-> trading day JISDOR.

Strategi penyelarasan (sesuai docs/temporal_alignment_strategy.md):

  +--------------------------------------------------+------------------------------------------+
  | Kondisi Berita Terbit                            | Aturan                                   |
  +--------------------------------------------------+------------------------------------------+
  | Hari kerja, sebelum jam tutup pasar (< 16:00 WIB)| Digabung ke kurs hari itu juga           |
  | Hari kerja, >= 16:00 WIB                         | Digabung ke kurs hari kerja berikutnya   |
  | Sabtu / Minggu                                   | Digabung ke Senin (atau hari kerja berikutnya) |
  | Hari libur nasional                              | Digabung ke hari kerja pertama setelah libur |
  | Libur panjang                                    | Semua berita diagregat ke hari kerja pertama setelah libur berakhir |
  +--------------------------------------------------+------------------------------------------+

Dukungan timezone:
  - WIB  (UTC+7)  -- zona barat, meliputi Jawa, Sumatra, Kalimantan barat/tengah
  - WITA (UTC+8)  -- zona tengah, meliputi Bali, NTB, NTT, Kalimantan timur/selatan
  - WIT  (UTC+9)  -- zona timur, meliputi Maluku, Papua

  Semua timestamp berita dikonversi ke WIB sebelum menerapkan aturan batas
  jam 16:00, karena pasar IDR (BI / JISDOR) beroperasi di zona WIB.

Dependensi tambahan:
  pip install holidays pandas

Cara pakai:
  from src.jisdor.temporal_alignment import get_effective_trading_date, align_news_to_kurs
"""

from __future__ import annotations

import warnings as _warnings
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Set

import pandas as pd

try:
    import holidays as _holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:  
    _HOLIDAYS_AVAILABLE = False

__all__ = [
    "TIMEZONE_WIB",
    "TIMEZONE_WITA",
    "TIMEZONE_WIT",
    "get_indonesia_holidays",
    "is_trading_day",
    "next_trading_day",
    "get_effective_trading_date",
    "align_news_to_kurs",
]

# Konstanta timezone Indonesia

TIMEZONE_WIB  = timezone(timedelta(hours=7), name="WIB")   # UTC+7
TIMEZONE_WITA = timezone(timedelta(hours=8), name="WITA")  # UTC+8
TIMEZONE_WIT  = timezone(timedelta(hours=9), name="WIT")   # UTC+9

# Peta nama string ke objek timezone
_TZ_MAP = {
    "WIB":  TIMEZONE_WIB,
    "WITA": TIMEZONE_WITA,
    "WIT":  TIMEZONE_WIT,
}

# Jam tutup pasar dalam WIB
_MARKET_CLOSE_HOUR_WIB = 16


# 1. Daftar hari libur nasional Indonesia
def get_indonesia_holidays(years: List[int]) -> Set[date]:
    """
    Kembalikan set tanggal hari libur nasional Indonesia untuk tahun-tahun
    yang diminta.

    Menggunakan library ``holidays`` (pip install holidays) jika tersedia.
    Jika tidak terinstal, kembalikan set kosong dan tampilkan peringatan.

    Parameters
    ----------
    years : list[int]
        Daftar tahun yang ingin dimuat hari liburnya.

    Returns
    -------
    set[date]
        Kumpulan tanggal libur nasional Indonesia.
    """
    if not _HOLIDAYS_AVAILABLE:
        _warnings.warn(
            "Library 'holidays' tidak terinstal. Hari libur nasional tidak "
            "diperhitungkan. Jalankan: pip install holidays",
            stacklevel=2,
        )
        return set()

    result: Set[date] = set()
    for year in years:
        result.update(_holidays_lib.Indonesia(years=year).keys())
    return result


# 2. Cek apakah suatu tanggal adalah trading day
def is_trading_day(d: date, holiday_set: Set[date]) -> bool:
    """
    Tentukan apakah tanggal ``d`` adalah trading day JISDOR.

    Trading day = hari kerja (Senin–Jumat) yang BUKAN hari libur nasional Indonesia.

    Parameters
    ----------
    d : date
        Tanggal yang diperiksa.
    holiday_set : set[date]
        Kumpulan tanggal hari libur nasional (dari :func:`get_indonesia_holidays`).

    Returns
    -------
    bool
    """
    # weekday(): 0=Senin, ..., 4=Jumat, 5=Sabtu, 6=Minggu
    return d.weekday() < 5 and d not in holiday_set

# 3. Cari hari trading berikutnya
def next_trading_day(d: date, holiday_set: Set[date]) -> date:
    """
    Kembalikan hari trading *terdekat setelah* tanggal ``d`` (tidak termasuk d).

    Parameters
    ----------
    d : date
        Tanggal referensi (eksklusif).
    holiday_set : set[date]
        Set hari libur.

    Returns
    -------
    date
    """
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate, holiday_set):
        candidate += timedelta(days=1)
    return candidate

# 4. Fungsi utama: get_effective_trading_date
def get_effective_trading_date(
    news_timestamp: datetime,
    holiday_set: Set[date],
    source_tz: str = "WIB",
) -> date:
    """
    Tentukan trading day efektif untuk satu timestamp berita.

    Logika penyelarasan (sesuai strategi di docs/):

    1. Konversi timestamp berita ke WIB (pasar IDR beroperasi di WIB).
    2. Ambil tanggal dan jam dalam WIB.
    3. Jika tanggal tersebut adalah trading day:
       - Jam < 16:00 WIB  -> pakai tanggal hari itu.
       - Jam >= 16:00 WIB -> pakai hari trading berikutnya.
    4. Jika tanggal bukan trading day (akhir pekan / libur):
       -> pakai hari trading berikutnya.

    Parameters
    ----------
    news_timestamp : datetime
        Waktu penerbitan berita. Boleh tz-aware maupun tz-naive.
        Jika tz-naive, diasumsikan sudah dalam timezone ``source_tz``.
    holiday_set : set[date]
        Set hari libur nasional (dari :func:`get_indonesia_holidays`).
    source_tz : str
        Timezone asal berita: ``"WIB"``, ``"WITA"``, atau ``"WIT"``.
        Digunakan saat ``news_timestamp`` bersifat tz-naive.

    Returns
    -------
    date
        Tanggal trading day efektif yang harus dipasangkan dengan kurs JISDOR.

    Raises
    ------
    ValueError
        Jika ``source_tz`` bukan salah satu dari ``"WIB"``, ``"WITA"``, ``"WIT"``.
    """
    if source_tz not in _TZ_MAP:
        raise ValueError(
            f"source_tz tidak valid: '{source_tz}'. "
            f"Pilihan yang tersedia: {list(_TZ_MAP.keys())}"
        )

    tz_src = _TZ_MAP[source_tz]

    # Lampirkan timezone jika tz-naive
    if news_timestamp.tzinfo is None:
        ts_aware = news_timestamp.replace(tzinfo=tz_src)
    else:
        ts_aware = news_timestamp

    # Konversi ke WIB untuk menerapkan aturan jam 16:00
    ts_wib = ts_aware.astimezone(TIMEZONE_WIB)
    news_date: date = ts_wib.date()
    news_hour: int  = ts_wib.hour

    if is_trading_day(news_date, holiday_set):
        if news_hour < _MARKET_CLOSE_HOUR_WIB:
            # Pasar masih buka -> digabung ke kurs hari ini
            return news_date
        else:
            # Pasar sudah tutup -> efek baru terasa esok hari
            return next_trading_day(news_date, holiday_set)
    else:
        # Akhir pekan atau hari libur -> efek tertunda ke hari trading berikutnya
        return next_trading_day(news_date, holiday_set)

# 5. Fungsi join: align_news_to_kurs
def align_news_to_kurs(
    df_news: pd.DataFrame,
    df_kurs: pd.DataFrame,
    news_timestamp_col: str = "publish_date",
    kurs_date_col: str = "tanggal",
    source_tz: str = "WIB",
    holiday_set: Optional[Set[date]] = None,
) -> pd.DataFrame:
    """
    Gabungkan dataset berita dengan dataset kurs JISDOR berdasarkan
    trading day efektif.

    Setiap baris berita mendapatkan kolom ``effective_trading_date`` dan
    kemudian di-join ke kurs JISDOR pada tanggal tersebut menggunakan
    ``pd.merge`` (exact match pada tanggal).

    Parameters
    ----------
    df_news : pd.DataFrame
        Dataset berita yang sudah di-cleaning dan filtering.
        Harus memiliki kolom ``publish_date`` (atau nama lain via
        ``news_timestamp_col``) berisi timestamp penerbitan.
    df_kurs : pd.DataFrame
        Dataset kurs JISDOR hasil :func:`preprocess_jisdor`.
        Harus memiliki kolom ``tanggal`` (datetime64) dan ``kurs_jisdor``.
    news_timestamp_col : str
        Nama kolom timestamp di ``df_news`` (default: ``"publish_date"``).
    kurs_date_col : str
        Nama kolom tanggal di ``df_kurs`` (default: ``"tanggal"``).
    source_tz : str
        Timezone asal berita: ``"WIB"``, ``"WITA"``, atau ``"WIT"``.
    holiday_set : set[date] | None
        Set hari libur. Jika None, akan dibuat otomatis dari rentang tahun
        yang ada di ``df_kurs``.

    Returns
    -------
    pd.DataFrame
        DataFrame berita yang diperkaya dengan kolom:
        - ``effective_trading_date``: tanggal trading yang dipasangkan.
        - ``kurs_jisdor``: nilai kurs pada tanggal tersebut (NaN jika tidak ada).

    Notes
    -----
    Berita yang effective_trading_date-nya melewati rentang data kurs (misal
    berita terbaru setelah data kurs berakhir) akan memiliki ``kurs_jisdor`` NaN.
    """
    df_news  = df_news.copy()
    df_kurs  = df_kurs.copy()

    # Pastikan kolom tanggal kurs bertipe datetime
    if not pd.api.types.is_datetime64_any_dtype(df_kurs[kurs_date_col]):
        df_kurs[kurs_date_col] = pd.to_datetime(df_kurs[kurs_date_col])

    # Buat holiday_set otomatis jika tidak disuplai
    if holiday_set is None:
        years = df_kurs[kurs_date_col].dt.year.unique().tolist()
        # Sertakan tahun dari df_news juga
        if news_timestamp_col in df_news.columns:
            ts_series = pd.to_datetime(df_news[news_timestamp_col], errors="coerce")
            years += ts_series.dt.year.dropna().astype(int).unique().tolist()
        holiday_set = get_indonesia_holidays(sorted(set(years)))

    # Parse timestamp berita
    df_news[news_timestamp_col] = pd.to_datetime(
        df_news[news_timestamp_col], errors="coerce"
    )

    # Hitung effective_trading_date per baris berita
    def _apply_effective(ts) -> Optional[date]:
        if pd.isna(ts):
            return None
        # Jika timestamp sudah timezone-aware, abaikan source_tz
        if ts.tzinfo is not None:
            return get_effective_trading_date(ts, holiday_set, source_tz=source_tz)
        return get_effective_trading_date(ts, holiday_set, source_tz=source_tz)

    df_news["effective_trading_date"] = df_news[news_timestamp_col].apply(_apply_effective)

    # Buat kolom join di kurs: date (bukan datetime) agar bisa match
    df_kurs["_kurs_date_key"] = df_kurs[kurs_date_col].dt.date
    df_news["_news_date_key"] = df_news["effective_trading_date"]

    # Join exact berdasarkan tanggal
    df_merged = df_news.merge(
        df_kurs[["_kurs_date_key", "kurs_jisdor"]],
        left_on="_news_date_key",
        right_on="_kurs_date_key",
        how="left",
    )

    # Bersihkan kolom sementara
    df_merged.drop(columns=["_news_date_key", "_kurs_date_key"], inplace=True)

    jumlah_nan = df_merged["kurs_jisdor"].isna().sum()
    if jumlah_nan > 0:
        print(
            f"[temporal_alignment] Peringatan: {jumlah_nan} berita tidak memiliki "
            f"data kurs (tanggal di luar rentang JISDOR atau belum trading)."
        )

    print(
        f"[temporal_alignment] {len(df_news)} berita -> "
        f"{df_merged['kurs_jisdor'].notna().sum()} berhasil dipasangkan dengan kurs."
    )

    return df_merged
