import re
from typing import Tuple


def memorized_verses(text: str) -> Tuple[str, str]:
    pattern_verses = [
        r'Ayat Hafalan\s*[“"](.+?)[”"]\s*[\(（](.+?)[\)）]',
        r'Ayat Hafalan\s*(.+?)\s*[\(（](.+?)[\)）]'
    ]
    for pattern in pattern_verses:
        is_true = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if is_true:
            content= is_true.group(1).strip()
            reference = is_true.group(2).strip()
            return  content, reference
    return "",""