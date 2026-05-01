import asyncio
import os.path
import re
import csv
from datetime import datetime
from typing import Optional
from helper.calculate_cost import get_gemini_detailed_costs, get_deepseek_detailed_costs, get_openai_detailed_costs, \
    get_cost_by_model
from helper.save_csv import save_to_csv
import time

DATA_ROOT = "./data/"




def extract_doc_id(content: str) -> Optional[str]:
    match = re.search(r'^id:\s*(\S+)', content, re.MULTILINE)
    return match.group(1) if match else None

INGEST_LOG = "ingest_log.csv"

def extract_metadata(content: str):
    import re
    week = re.search(r'minggu:\s*(\d+)', content)
    date = re.search(r'tanggal_iso:\s*([\d-]+)', content)

    return {
        "week": week.group(1) if week else None,
        "date": date.group(1) if date else None,
    }

def _init_ingest_log():
    if not os.path.exists(INGEST_LOG):
        with open(INGEST_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "doc_id", "file_path", "status"])
            writer.writeheader()

def _log_ingest(doc_id: str, file_path: str, status: str):
    with open(INGEST_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "doc_id", "file_path", "status"])
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id or "unknown",
            "file_path": file_path,
            "status": status
        })

def get_ingested_doc_ids() -> set:
    if not os.path.exists(INGEST_LOG):
        return set()
    with open(INGEST_LOG, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["doc_id"] for row in reader if row["status"] == "SUCCESS"}


async def insert_documents_folder(MODEL, rag, root_dir: str, llm_tracker, embed_tracker):
    files = []
    for dir_path, _, file_names in os.walk(root_dir):
        for f in file_names:
            if f.endswith(".txt"):
                files.append(os.path.join(dir_path, f))

    print(f"Ditemukan {len(files)} file. Menyiapkan Batch Indexing...")

    all_contents = []
    all_ids = []
    all_file_paths = []
    metadata_results = []  # Tambahkan penampung metadata

    for path in files:
        with open(path, "r", encoding="utf-8") as fp:
            content = fp.read().strip()
        if not content: continue

        doc_id = extract_doc_id(content)
        # Ambil metadata (week dan date) menggunakan fungsi yang sudah kamu buat
        meta = extract_metadata(content)

        all_contents.append(content)
        all_ids.append(doc_id if doc_id else path)
        all_file_paths.append(path)
        metadata_results.append(meta)  # Masukkan metadata ke list

    llm_tracker.reset()
    embed_tracker.reset()
    start_time = time.time()

    try:
        print(f"Memulai indexing {len(all_contents)} dokumen secara bersamaan...")
        await rag.ainsert(input=all_contents, ids=all_ids, file_paths=all_file_paths)
        status = "SUCCESS"
    except Exception as e:
        status = f"ERROR: {e}"

    latency = time.time() - start_time
    metrics = get_cost_by_model(MODEL, llm_tracker.get_usage(), embed_tracker.get_usage())

    save_to_csv({
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "mode": "BATCH_INDEXING_WEEKLY",
        "question": f"Batch Indexing: {len(all_contents)} files",
        "answer": status,
        "latency": latency,
        "llm_p_tokens": metrics["llm_p"],
        "llm_c_tokens": metrics["llm_c"],
        "embed_tokens": metrics["emb_p"],
        "cost_llm": f"{metrics['c_llm']:.10f}",
        "cost_embed": f"{metrics['c_emb']:.10f}",
        "total_cost": f"{metrics['total']:.10f}",
        "call_count": llm_tracker.get_usage().get("call_count", 0)
    })

    print(f"[{status}] Batch Indexing Selesai | Total Cost: ${metrics['total']:.10f}")

    return metadata_results