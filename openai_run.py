import asyncio
import os
import time
from datetime import datetime

from core.openai import init_openai_lightRAG, WORKING_DIR, llm_tracker
from core.embedding import embed_tracker
from docs.configure_logging import configure_logging
from helper.answer_question import answer_question
from helper.calculate_cost import get_openai_detailed_costs
from helper.dataset_helper import DATA_ROOT, insert_documents_folder
from helper.save_csv import save_to_csv

MODEL = "gpt-4.1-mini"


async def run_openai():
    configure_logging()
    print("Starting System Chat GPT...")

    rag = await init_openai_lightRAG()

    storage_check_path = os.path.join(WORKING_DIR, "kv_Store_full_text.json")

    if not os.path.exists(storage_check_path):
        print("--- DATABASE KOSONG: Memulai proses indexing... ---")
        llm_tracker.reset()
        embed_tracker.reset()

        start_time = time.time()
        docs = await insert_documents_folder(MODEL, rag, DATA_ROOT, llm_tracker, embed_tracker)
        weeks = sorted(set(d["week"] for d in docs if d["week"]))
        dates = sorted(d["date"] for d in docs if d["date"])
        week_range = f"{weeks[0]}-{weeks[-1]}" if weeks else "-"
        date_range = f"{dates[0]} to {dates[-1]}" if dates else "-"

        end_time = time.time()
        latency = end_time - start_time

        metrics = get_openai_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())

        if metrics["total"] >= 0:
            save_to_csv({
                "timestamp": datetime.now().isoformat(),
                "model": MODEL,
                "mode": "INDEXING",
                "question": f"Initial Ingest {DATA_ROOT}-{week_range}-{date_range} ",
                "answer": "SUCCESS",
                "latency": latency,
                "llm_p_tokens": metrics["llm_p"],
                "llm_c_tokens": metrics["llm_c"],
                "embed_tokens": metrics["emb_p"],
                "cost_llm": metrics["c_llm"],
                "cost_embed": metrics["c_emb"],
                "total_cost": metrics["total"],
                "call_count": llm_tracker.get_usage().get("call_count", 0)
            })
        print(f"Indexing selesai. Latency: {latency:.2f}s")
    else:
        print("--- DATABASE DITEMUKAN ---")

    question = "Ayat Judul pelajaran di Minggu ke 1"
    await answer_question(MODEL, rag, question, llm_tracker, embed_tracker)

    await rag.finalize_storages()
    print("\nProcess finished.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("API Key tidak ditemukan!")
    else:
        asyncio.run(run_openai())
