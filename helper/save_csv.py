import csv
import os


def save_to_csv(data, filename="research_performance_logs.csv"):
    file_exists = os.path.isfile(filename)

    fieldnames = [
        "timestamp",
        "model",  # Tambahkan ini
        "mode",
        "question",
        "answer",  # Tambahkan ini
        "llm_p_tokens",
        "llm_c_tokens",
        "embed_tokens",
        "cost_llm",
        "cost_embed",
        "total_cost",
        "latency",
        "call_count"  # Tambahkan ini
    ]

    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)