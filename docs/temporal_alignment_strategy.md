## Aturan/Strategi Penyelarasan untuk Menangani Pergeseran Zona Waktu dan Non-Trading Days

Trading day adalah hari kerja (yang meliputi Senin-Jumat). Yang bukan termasuk trading day adalah hari Sabtu, Minggu, serta tanggal dalam daftar hari libur nasional Indonesia. Strategi penyelarasan data temporal untuk menangani beberapa kondisi pergeseran waktu dan hari non-trading yang kami usulkan adalah sebagai berikut:

| Kondisi Berita Terbit                             | Aturan                                                                                    | Alasan                                                           |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Hari kerja, sebelum jam tutup pasar (< 16:00 WIB) | Digabung ke kurs hari itu juga                                                            | Pasar masih bisa bereaksi di hari yang sama                      |
| Hari kerja, setelah jam tutup pasar (≥ 16:00 WIB) | Digabung ke kurs hari kerja berikutnya                                                    | Reaksi pasar baru mungkin terjadi keesokan hari                  |
| Sabtu/Minggu                                      | Digabung ke kurs hari Senin (atau hari kerja berikutnya jika Senin libur)                 | Tidak ada sesi trading di akhir pekan                            |
| Hari libur nasional                               | Digabung ke kurs hari kerja pertama setelah libur                                         | Sama seperti akhir pekan, market tutup                           |
| Libur panjang                                     | Semua berita di periode libur diagregat ke kurs hari kerja pertama setelah libur berakhir | Efek berita menumpuk terserap sekaligus saat market buka kembali |

Implementasi yang kami lakukan adalah dengan membuat fungsi `get_effective_trading_date()` di `src/temporal_alignment.py`, yang menerima timestamp berita dan mengembalikan tanggal trading day yang sesuai. Fungsi ini lalu dipakai sebagai key join saat `merge_asof` menggabungkan dataset berita dengan dataset kurs JISDOR.
