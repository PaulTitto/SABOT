from datetime import datetime
from typing import Tuple

BULAN_INDONESIA = {
    1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
    5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
    9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
}

def date_format(date: str) -> Tuple[str, str]:
    try:
        dt = datetime.strptime(date.strip(), '%d/%m/%Y')  # Changed to slash
        indo_date = f"{dt.day} {BULAN_INDONESIA[dt.month]} {dt.year}"
        date_iso = dt.strftime('%Y-%m-%d')
        return indo_date, date_iso
    except:
        return "Data is unidentified", ""