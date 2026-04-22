import asyncio
import os
import time

from core.gemini import WORKING_DIR
from docs.configure_logging import configure_logging
from helper.answer_question import answer_question
from helper.calculate_cost import get_gemini_detailed_costs
from helper.dataset_helper import DATA_ROOT, insert_documents_folder
from helper.save_csv import save_to_csv
from datetime import datetime
from core.gemini import llm_tracker
from core.embedding import embed_tracker

MODEL = "gemini-2.5-flash-lite"


async def run_gemini():
    configure_logging()
    print("Starting System Gemini...")

    rag = await init_gemini_lightRAG()
    storage_check_path = os.path.join(WORKING_DIR, "kv_store_full_text.json")

    if not os.path.exists(storage_check_path):
        print("--- DATABASE KOSONG: Memulai proses indexing dokumen... ---")

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

        metrics = get_gemini_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())

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

        print(f"Indexing selesai dalam {latency:.2f}s. Biaya dicatat ke CSV.")
    else:
        print("--- DATABASE DITEMUKAN: Menggunakan data lama (Hemat Biaya) ---")

    question = "Apa judul pelajaran minggu kesatu?"
    print(f"\nAsking: {question}")

    await answer_question(MODEL, rag, question, llm_tracker, embed_tracker)
    await rag.finalize_storages()
    print("\nAll processes finished. Check your CSV for research data.")


from core.gemini import init_gemini_lightRAG

async def export_data():
    rag = await init_gemini_lightRAG()
    await rag.aexport_data("graph_data_full.csv", file_format="csv", include_vector_data=True)

    print("Exported!")
    await rag.finalize_storages()


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("API Key Gemini tidak ditemukan! Setel di environment variable.")
    else:
        try:
            asyncio.run(run_gemini())
        except KeyboardInterrupt:
            print("\nProses dihentikan oleh pengguna.")
