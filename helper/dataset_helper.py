import asyncio
import os.path
import re
import csv
from datetime import datetime
from typing import Optional
from helper.calculate_cost import get_gemini_detailed_costs
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

    print(f"Ditemukan {len(files)} file")

    ingested_docs = []

    for path in files:
        with open(path, "r", encoding="utf-8") as fp:
            content = fp.read().strip()

        if not content:
            continue

        doc_id = extract_doc_id(content)
        meta = extract_metadata(content)

        llm_tracker.reset()
        embed_tracker.reset()

        start_time = time.time()

        try:
            await rag.ainsert(content)
            status = "SUCCESS"
        except Exception as e:
            if "already exists" in str(e).lower():
                status = "SKIPPED"
            else:
                status = f"ERROR: {e}"

        latency = time.time() - start_time

        metrics = get_gemini_detailed_costs(
            llm_tracker.get_usage(),
            embed_tracker.get_usage()
        )

        has_usage = metrics["total"] > 0

        if status == "SUCCESS" and has_usage:
            save_to_csv({
                "timestamp": datetime.now().isoformat(),
                "model": MODEL,
                "mode": "INDEXING_PER_FILE",
                "question": f"{doc_id} | week {meta['week']} | {meta['date']}",
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

            ingested_docs.append({
                "doc_id": doc_id,
                "week": meta["week"],
                "date": meta["date"]
            })

        print(f"[{status}] {doc_id} | cost: ${metrics['total']:.10f}")

        await asyncio.sleep(0.2)

    return ingested_docs