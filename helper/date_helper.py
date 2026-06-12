import re
from datetime import datetime, timedelta
from typing import Optional

# ==========================================
# QUARTER CONFIG
# ==========================================

QUARTER_CODE = "2026-q2"

# Siklus Sabat Triwulan ini dimulai pada hari Sabtu, 28 Maret 2026
QUARTER_START = datetime.strptime("28/03/2026", "%d/%m/%Y")
QUARTER_END = datetime.strptime("26/06/2026", "%d/%m/%Y")


def get_today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ==========================================
# BIAS CLEANER & CONVERTER HELPERS
# ==========================================

def clean_typo_keywords(text: str) -> str:
    """Membersihkan typo massal yang sering diketik user di HP"""
    t = text.lower()
    t = t.replace("sahat", "sabat").replace("sabbat", "sabat").replace("sabath", "sabat")
    t = t.replace("ringkasn", "ringkasan").replace("ingkasan", "ringkasan")
    return t


def konversi_nama_hari_ke_sabat_day(question: str) -> Optional[int]:
    """
    Mengonversi nama hari (Senin-Minggu) ke nomor hari Sekolah Sabat.
    Siklus Sekolah Sabat: Sabtu adalah Hari 1, Jumat adalah Hari 7.
    """
    q = question.lower()
    hari_map = {
        "sabtu": 1,
        "minggu": 2,
        "senin": 3,
        "selasa": 4,
        "rabu": 5,
        "kamis": 6,
        "jumat": 7
    }
    for nama_hari, angka_day in hari_map.items():
        if re.search(rf"\b{nama_hari}\b", q):
            return angka_day
    return None


def extract_relative_week_delta(question: str) -> int:
    """Mendeteksi pergeseran pekan (Past, Now, Future)"""
    q = question.lower()

    future_patterns = [r"minggu\s+depan", r"minggu\s+besok", r"next\s+week", r"pelajaran\s+berikutnya"]
    if any(re.search(p, q) for p in future_patterns):
        return 1

    past_patterns = [r"minggu\s+lalu", r"minggu\s+kemarin", r"last\s+week", r"minggu\s+sebelumnya"]
    if any(re.search(p, q) for p in past_patterns):
        return -1

    return 0


def extract_date_from_question(question: str) -> Optional[str]:
    q = question.lower().strip()
    today = datetime.now()

    relative_map = {
        r"\bhari ini\b": 0,
        r"\bsekarang\b": 0,
        r"\bsaat ini\b": 0,
        r"\bkemarin\b": -1,
        r"\bsehari sebelumnya\b": -1,
        r"\bbesok\b": 1,
        r"\bkeesokan harinya\b": 1,
        r"\blusa\b": 2,
        r"\bdua hari lalu\b": -2,
        r"\bdua hari yang lalu\b": -2,
    }

    for pattern, delta in relative_map.items():
        if re.search(pattern, q):
            target = today + timedelta(days=delta)
            return target.strftime("%Y-%m-%d")

    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", q)
    if iso_match:
        return iso_match.group()

    bulan_map = {
        "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
        "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12
    }

    pattern = (
        r"(\d{1,2})\s+"
        r"(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)"
        r"\s+(\d{4})"
    )

    match = re.search(pattern, q)
    if match:
        day, month_name, year = match.groups()
        month = bulan_map[month_name]
        return f"{year}-{month:02d}-{int(day):02d}"

    return None


# ==========================================
# CORE ID GENERATORS
# ==========================================

def get_lesson_id_by_date(date_iso: str) -> Optional[str]:
    try:
        target = datetime.strptime(date_iso, "%Y-%m-%d")
        if target < QUARTER_START or target > QUARTER_END:
            return None

        delta_days = (target - QUARTER_START).days
        week = (delta_days // 7) + 1
        day = (delta_days % 7) + 1

        return f"{QUARTER_CODE}-w{week:02d}-d{day}"
    except:
        return None


def get_current_lesson_id_with_delta(week_delta: int = 0) -> Optional[str]:
    today = datetime.now() + timedelta(weeks=week_delta)
    if today < QUARTER_START or today > QUARTER_END:
        return None

    delta_days = (today - QUARTER_START).days
    week = (delta_days // 7) + 1
    day = (delta_days % 7) + 1

    return f"{QUARTER_CODE}-w{week:02d}-d{day}"


def extract_week_day(question: str) -> Optional[dict]:
    q = question.lower()
    patterns = [
        r"minggu\s*(\d+)\s*hari\s*(\d+)",
        r"week\s*(\d+)\s*day\s*(\d+)",
        r"w(\d+)[-\s]?d(\d+)",
        r"pelajaran\s*(\d+)[-\s](\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            week, day = match.groups()
            w_val = int(week)
            d_val = int(day)

            # PROTEKSI ANGKA KEMBAR: Jika user typo ngetik day > 7 (misal w11 d11)
            # Batasi paksa day maksimal ke hari ke-7
            if d_val > 7:
                d_val = 7

            return {"week": w_val, "day": d_val}

    return None


def is_memory_verse_question(question: str) -> bool:
    q = question.lower()
    keywords = ["ayat hafalan", "ayat emas", "memory verse", "ayat minggu ini", "ayat pelajaran"]
    return any(k in q for k in keywords)


def is_summary_question(question: str) -> bool:
    q = question.lower()
    keywords = ["ringkas", "ringkasan", "summary", "inti pelajaran", "kesimpulan", "garis besar"]
    return any(k in q for k in keywords)


def is_any_week_keyword(question: str) -> bool:
    q = question.lower()
    if "ayat" in q:
        return False
    return "minggu" in q or "week" in q


# ==========================================
# MAIN REWRITE FUNCTION
# ==========================================

def rewrite_question(question: str) -> str:
    # Bersihkan string awal dari whitespace dan typo pengetikan keyboard
    original = clean_typo_keywords(question.strip())

    # 1. Cek tanggal eksplisit atau kata penunjuk hari "hari ini", "besok", dll.
    date_iso = extract_date_from_question(original)
    if date_iso:
        lesson_id = get_lesson_id_by_date(date_iso)
        if lesson_id:
            if is_summary_question(original):
                return f"[LESSON_ID={lesson_id}] [DATE={date_iso}] [FOCUS=SUMMARY] {original}"
            return f"[LESSON_ID={lesson_id}] [DATE={date_iso}] {original}"

    # 2. Cek gabungan format nama hari teks dan angka minggu (Contoh: "Pelajaran Kamis minggu 11")
    nama_hari_day = konversi_nama_hari_ke_sabat_day(original)
    if nama_hari_day:
        # Cari angka minggunya dari string teks
        week_match = re.search(r"(?:minggu|week|w)\s*(\d+)", original)
        if week_match:
            week_val = int(week_match.group(1))
            lesson_id = f"{QUARTER_CODE}-w{week_val:02d}-d{nama_hari_day}"
            if is_summary_question(original):
                return f"[LESSON_ID={lesson_id}] [FOCUS=SUMMARY] {original}"
            return f"[LESSON_ID={lesson_id}] {original}"

    # 3. Cek format angka hardcode terstruktur (Contoh: w11-d5 atau minggu 11 hari 5)
    lesson = extract_week_day(original)
    if lesson:
        lesson_id = f"{QUARTER_CODE}-w{lesson['week']:02d}-d{lesson['day']}"
        if is_summary_question(original):
            return f"[LESSON_ID={lesson_id}] [FOCUS=SUMMARY] {original}"
        return f"[LESSON_ID={lesson_id}] {original}"

    # 4. Cek timeline relatif rombongan mingguan (Minggu ini, Minggu depan, Minggu lalu)
    if is_any_week_keyword(original):
        week_delta = extract_relative_week_delta(original)
        current_id = get_current_lesson_id_with_delta(week_delta)

        if current_id:
            # Kupas kode hari (-dX) agar cakupan RAG meluas ke dokumen satu minggu penuh
            week_id = re.sub(r'-d\d+', '', current_id)

            if is_summary_question(original):
                return f"[LESSON_ID={week_id}] [FOCUS=SUMMARY] {original}"
            return f"[LESSON_ID={week_id}] {original}"

    # 5. Fallback ke Filter Ayat Hafalan
    if is_memory_verse_question(original):
        return f"[FOCUS=MEMORY_VERSE] {original}"

    # 6. Fallback ke Filter Ringkasan Biasa
    if is_summary_question(original):
        return f"[FOCUS=SUMMARY] {original}"

    return original


def extract_lesson_id(question: str) -> str | None:
    m = re.search(r'\d{4}-q\d+-w\d+(-d\d+)?', question)
    return m.group(0) if m else None