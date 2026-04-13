import os
import re
import time
import requests
from utils.cleaner_text import cleaner_text
from utils.date_format import date_format
from utils.extract_memorized_verses import memorized_verses
from utils.metadata_frontmatter import take_metadata

NAMA_HARI = {
    1: "Sabtu", 2: "Minggu", 3: "Senin", 4: "Selasa",
    5: "Rabu", 6: "Kamis", 7: "Jumat"
}


def download_and_save_data(y: int, q: int, w: int, tema_triwulan: str):
    tema_mingguan = ""  # Initialize
    HARI = range(1, 8)

    for d in HARI:
        url = f"https://raw.githubusercontent.com/Adventech/sabbath-school-lessons/refs/heads/stage/src/in/{y}-{q:02d}/{w:02d}/{d:02d}.md"

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"Gagal: {url} (HTTP {resp.status_code})")
                continue

            content_md = resp.text
            meta = take_metadata(content_md)
            content_without_yaml = re.sub(r'^---\n.*?\n---\n', '', content_md, flags=re.DOTALL)
            clean_text = cleaner_text(content_without_yaml)

            judul = meta.get("judul", "tidak ada judul")
            if d == 1:
                tema_mingguan = judul.lower()

            tanggal_mentah = meta.get("tanggal_mentah", "")
            indo_date, iso_date = date_format(tanggal_mentah) if tanggal_mentah else ("", "")
            hari_indo = NAMA_HARI[d]

            ayat_isi = meta.get('ayat_hafalan', '')
            ayat_ref = ""

            if ayat_isi:
                ref_match = re.search(r'[\(（]([^\)）]+)[\)）]', ayat_isi)
                if ref_match:
                    ayat_ref = ref_match.group(1)
                    ayat_isi = re.sub(r'\s*[\(（][^\)）]+[\)）]', '', ayat_isi).strip()
            else:
                ayat_isi, ayat_ref = memorized_verses(clean_text)

            # Moved outside the if block
            doc_id = f"{y}-q{q}-w{w:02d}-d{d}"
            dataset = f"""
[DOC]
id: {doc_id}
tanggal: {hari_indo}, {indo_date}
tanggal_iso: {iso_date}
minggu: {w:02d}
triwulan: {q}
judul: {judul}

[PELAJARAN]
judul_harian: {judul.lower()}
tema_mingguan: {tema_mingguan}
tema_triwulan: {tema_triwulan}

[AYAT]
hafalan: {ayat_ref}
isi_ayat: {ayat_isi}

[ISI]
{clean_text}
"""
            # Create directories and save
            output_dir = f"./{y}"
            folder_quartal = os.path.join(output_dir, f"{q:02d}")
            os.makedirs(folder_quartal, exist_ok=True)
            folder_mingguan = os.path.join(folder_quartal, f"{w:02d}")
            os.makedirs(folder_mingguan, exist_ok=True)
            file_path = os.path.join(folder_mingguan, f"{d:02d}.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(dataset)
            print(f"✅ {file_path}")
            time.sleep(0.2)

        except Exception as e:
            print(f"Error: {url} - {str(e)[:80]}")
            continue

