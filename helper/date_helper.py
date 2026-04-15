import datetime

def extract_date_from_question(question: str) -> Optional[str]:
    iso_match = re.search(r'\d{4}-\d{2}-\d{2}', question)
    if iso_match:
        return iso_match.group()
    bulan_map = {
        'januari': 1, 'februari': 2, 'maret': 3, 'april': 4,
        'mei': 5, 'juni': 6, 'juli': 7, 'agustus': 8,
        'september': 9, 'oktober': 10, 'november': 11, 'desember': 12
    }
    pattern = r'(\d{1,2})\s+(' + '|'.join(bulan_map.keys()) + r')\s+(\d{4})'
    match = re.search(pattern, question.lower())
    if match:
        day, month_name, year = match.groups()
        month = bulan_map[month_name]
        return f"{year}-{month:02d}-{int(day):02d}"
    return None

def get_today_iso():
    return datetime.now().strftime("%Y-%m-%d")