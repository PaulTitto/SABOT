import re
from typing import Dict
import yaml

def take_metadata(content_md: str) -> Dict:
    metadata = {}
    yaml_match = re.match(r'^---\n(.*?)\n---\n', content_md, re.DOTALL)
    if yaml_match:
        yaml_str = yaml_match.group(1)
        try:
            data = yaml.safe_load(yaml_str)
            if data:
                if 'title' in data:  # Fixed typo
                    title = data['title']
                    if isinstance(title, dict):
                        metadata['judul'] = title.get('in', str(title))
                    else:
                        metadata['judul'] = str(title)
                if 'date' in data:
                    metadata['tanggal_mentah'] = str(data['date'])
                # Fixed condition
                if 'ayat_hafalan' in data:
                    mt = data['ayat_hafalan']
                    if isinstance(mt, dict):
                        metadata['ayat_hafalan'] = mt.get('in', '')
                    else:
                        metadata['ayat_hafalan'] = str(mt)
                elif 'memory_text' in data:
                    mt = data['memory_text']
                    if isinstance(mt, dict):
                        metadata['ayat_hafalan'] = mt.get('in', '')
                    else:
                        metadata['ayat_hafalan'] = str(mt)
        except Exception as e:
            print(f"Error parse YAML: {e}")
    return metadata