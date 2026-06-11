import re
from typing import Optional
from datetime import datetime, timedelta


QUARTER_START = "28/03/2026"


def get_lesson_id_by_date(date_iso: str) -> str | None:
    try:
        start = datetime.strptime(
            QUARTER_START,
            "%d/%m/%Y"
        )

        target = datetime.strptime(
            date_iso,
            "%Y-%m-%d"
        )

        delta_days = (
            target - start
        ).days

        if delta_days < 0:
            return None

        week = (delta_days // 7) + 1
        day = (delta_days % 7) + 1

        return f"2026-q2-w{week:02d}-d{day}"

    except Exception:
        return None
def get_today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def extract_date_from_question(question: str) -> Optional[str]:
    q = question.lower().strip()
    today = datetime.now()

    relative_map = {
        # Hari ini
        r'\bhari ini\b': 0,
        r'\bsekarang\b': 0,
        r'\bsaat ini\b': 0,

        # Kemarin
        r'\bkemarin\b': -1,
        r'\bsehari sebelumnya\b': -1,
        r'\bhari sebelumnya\b': -1,

        # Besok
        r'\bbesok\b': 1,
        r'\bhari berikutnya\b': 1,
        r'\bkeesokan harinya\b': 1,
        r'\bbesok harinya\b': 1,

        # Lusa
        r'\blusa\b': 2,
        r'\bdua hari lagi\b': 2,

        # Kemarin lusa
        r'\bkemarin lusa\b': -2,
        r'\bdua hari lalu\b': -2,
        r'\bdua hari yang lalu\b': -2,
    }

    for pattern, delta in relative_map.items():
        if re.search(pattern, q):
            target = today + timedelta(days=delta)
            return target.strftime("%Y-%m-%d")

    if re.search(r'\bminggu ini\b', q):
        days_since_saturday = (today.weekday() + 2) % 7
        saturday = today - timedelta(days=days_since_saturday)
        return saturday.strftime("%Y-%m-%d")

    if re.search(r'\bminggu lalu\b|\bminggu kemarin\b|\bminggu sebelumnya\b', q):
        days_since_saturday = (today.weekday() + 2) % 7
        saturday = today - timedelta(days=days_since_saturday + 7)
        return saturday.strftime("%Y-%m-%d")

    if re.search(r'\bminggu depan\b|\bminggu berikutnya\b|\bminggu selanjutnya\b', q):
        days_since_saturday = (today.weekday() + 2) % 7
        saturday = today - timedelta(days=days_since_saturday - 7)
        return saturday.strftime("%Y-%m-%d")
    iso_match = re.search(r'\d{4}-\d{2}-\d{2}', question)
    if iso_match:
        return iso_match.group()

    bulan_map = {
        'januari': 1, 'februari': 2, 'maret': 3, 'april': 4,
        'mei': 5, 'juni': 6, 'juli': 7, 'agustus': 8,
        'september': 9, 'oktober': 10, 'november': 11, 'desember': 12
    }
    pattern = r'(\d{1,2})\s+(' + '|'.join(bulan_map.keys()) + r')\s+(\d{4})'
    match = re.search(pattern, q)
    if match:
        day, month_name, year = match.groups()
        month = bulan_map[month_name]
        return f"{year}-{month:02d}-{int(day):02d}"

    hari_map = {
        'sabtu':    5,
        'minggu':   6,
        'senin':    0,
        'selasa':   1,
        'rabu':     2,
        'kamis':    3,
        'jumat':    4,
        "jum'at":   4,
    }

    for nama_hari, target_weekday in hari_map.items():
        if re.search(rf'\b{nama_hari}\b', q):
            current_weekday = today.weekday()
            delta = (target_weekday - current_weekday) % 7
            if delta == 0 and re.search(r'\blalu\b|\bkemarin\b|\bsebelumnya\b', q):
                delta = -7
            target = today + timedelta(days=delta)
            return target.strftime("%Y-%m-%d")

    return None