import requests
import yaml
import sys
from datetime import datetime
from utils.download_and_save_data import download_and_save_data


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
    MINGGU_MAX = 1

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


if __name__ == "__main__":
    main()