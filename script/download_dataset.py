import requests
import yaml

from utils.download_and_save_data import download_and_save_data


def main():
    TAHUN = 2026
    KUARTAL = 1
    MINGGU_MAX = 1

    info_url = f"https://raw.githubusercontent.com/Adventech/sabbath-school-lessons/refs/heads/stage/src/in/{TAHUN}-{KUARTAL:02d}/info.yml"
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
    except Exception as e:
        print(f"Error mengambil info.yml: {e}")

    for w in range(1, MINGGU_MAX + 1):
        print(f"\nMinggu {w:02d} / {MINGGU_MAX}")
        download_and_save_data(TAHUN, KUARTAL, w, tema_triwulan)

    print("\n" + "=" * 60)
    print(f"Selesai! Data tersimpan di folder: ./{TAHUN}")


if __name__ == "__main__":
    main()