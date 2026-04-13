import re
def cleaner_text(text: str) -> str:
    """
    Clean the text by removing punctuation.
    """
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[*>`_-`"]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub('[ \t]+$', '', text, flags=re.MULTILINE)
    return text.strip()