import requests
import yaml
import sys
from datetime import datetime
from utils.download_and_save_data import download_and_save_data

import os
import glob


def fetch_quarterly_theme(tahun: int, kuartal: int) -> str:
    """
    Fetch the quarterly theme from info.yml.

    Args:
        tahun (int): Year
        kuartal (int): Quarter (1-4)

    Returns:
        str: Quarterly theme or 'unknown' if fetch fails
    """
    info_url = f"https://raw.githubusercontent.com/Adventech/sabbath-school-lessons/refs/heads/stage/src/in/{tahun}-{kuartal:02d}/info.yml"
    tema_triwulan = "unknown"

    try:
        resp_info = requests.get(info_url, timeout=10)
        if resp_info.status_code == 200:
            info = yaml.safe_load(resp_info.text)
            tema = info.get('human_title', info.get('title', 'unknown'))

            if isinstance(tema, dict):
                tema_triwulan = tema.get('in', str(tema))
            else:
                tema_triwulan = str(tema)

            print(f"Tema triwulan: {tema_triwulan}")
        else:
            print(f"Gagal mengambil info.yml (HTTP {resp_info.status_code})")
    except requests.exceptions.Timeout:
        print(f"Timeout mengambil info.yml")
    except Exception as e:
        print(f"Error mengambil info.yml: {e}")

    return tema_triwulan


def main():
    """Main function to download and save lesson data."""
    TAHUN = 2026
    KUARTAL = 2
    MINGGU_MAX = 15

    print("=" * 60)
    print(f"Download Sabbath School Lessons")
    print(f"Tahun: {TAHUN}, Kuartal: {KUARTAL}")
    print(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    #  input
    if not (1 <= KUARTAL <= 4):
        print(f"Error: Kuartal harus 1-4, bukan {KUARTAL}")
        sys.exit(1)

    if MINGGU_MAX < 1:
        print(f"Error: MINGGU_MAX harus >= 1")
        sys.exit(1)

    tema_triwulan = fetch_quarterly_theme(TAHUN, KUARTAL)

    total_success = 0
    total_failed = 0

    try:
        for w in range(1, MINGGU_MAX + 1):
            print(f"\nMinggu {w:02d} / {MINGGU_MAX}")
            try:
                download_and_save_data(TAHUN, KUARTAL, w, tema_triwulan)
                total_success += 1
            except Exception as e:
                print(f"Error pada minggu {w}: {str(e)[:100]}")
                total_failed += 1
                continue

    except KeyboardInterrupt:
        print("\n⚠️  Download dihentikan oleh user")
        sys.exit(0)

    print("\n" + "=" * 60)
    print(f"Selesai!")
    print(f"Minggu berhasil: {total_success} / {MINGGU_MAX}")
    if total_failed > 0:
        print(f"Minggu gagal: {total_failed}")
    print(f"Data tersimpan di: ./{TAHUN}/{KUARTAL:02d}/")
    print("=" * 60)



def merge_weekly_txt_dataset():
    # Folder sesuai gambar kamu
    source_root = "./data"
    target_root = "./data-merge"

    if not os.path.exists(source_root):
        print(f"❌ Folder sumber '{source_root}' tidak ditemukan!")
        return

    print("=" * 60)
    print("Mulai Merging Dataset Txt...")
    print("=" * 60)

    pattern = os.path.join(source_root, "*", "*", "*")
    folder_minggu_list = sorted(glob.glob(pattern))

    for path_minggu in folder_minggu_list:
        if not os.path.isdir(path_minggu):
            continue

        parts = path_minggu.replace("\\", "/").split("/")

        nama_minggu = parts[-1]
        kuartal = parts[-2]
        tahun = parts[-3]

        file_hari = sorted(glob.glob(os.path.join(path_minggu, "*.txt")))

        if not file_hari:
            continue

        print(f"-> Memproses {tahun}/{kuartal}/{nama_minggu} ({len(file_hari)} hari)...")

        merged_text = []
        for file_path in file_hari:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    merged_text.append(content)
            except Exception as e:
                print(f"   ❌ Gagal membaca {os.path.basename(file_path)}: {e}")

        if merged_text:
            target_dir = os.path.join(target_root, tahun, kuartal)
            os.makedirs(target_dir, exist_ok=True)

            output_file = os.path.join(target_dir, f"week-{nama_minggu}.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n\n" + "=" * 40 + "\n\n".join(merged_text))

            print(f"   ✅ Berhasil merge -> {output_file}")

    print("=" * 60)
    print("Proses merge selesai!")
    print("=" * 60)


if __name__ == "__main__":
    # main()
    merge_weekly_txt_dataset()