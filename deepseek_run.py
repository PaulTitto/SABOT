import asyncio
import os
import time
from datetime import datetime


from core.deepseek import init_deepseek_lightRAG, WORKING_DIR, llm_tracker
from core.embedding import embed_tracker
from docs.configure_logging import configure_logging
from helper.answer_question import answer_question
from helper.calculate_cost import get_deepseek_detailed_costs
from helper.dataset_helper import DATA_ROOT, insert_documents_folder
from helper.save_csv import save_to_csv

MODEL = "deepseek-chat"

async def run_deepseek():
    configure_logging()
    print("Starting System Deepseek...")
    rag = await init_deepseek_lightRAG()
    storage_check_path = os.path.join(WORKING_DIR, "kv_Store_full_text.json")
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

        metrics = get_deepseek_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())

        has_usage = metrics["total"]>0
        if has_usage:
            save_to_csv({
                "timestamp": datetime.now().isoformat(),
                "model": MODEL,
                "mode": "INDEXING",
                "question": f"Initial Ingest {DATA_ROOT}-{week_range}-{date_range} ",
                "answer": "SUCCESS",
                "latency": latency,
                "llm_p_tokens": f"{metrics["llm_p"]:.10f}",
                "llm_c_tokens": f"{metrics["llm_c"]:.10f}",
                "embed_tokens": f"{metrics["emb_p"]:.10f}",
                "cost_llm": f"{metrics["c_llm"]:.10f}",
                "cost_embed": f"{metrics["c_emb"]:.10f}",
                "total_cost": f"{metrics["total"]:.10f}",
                "call_count": llm_tracker.get_usage().get("call_count", 0)
            })
        print(f"Indexing selesai dalam {latency:.2f}s. Biaya dicatat ke CSV.")
    else:
        print("--- DATABASE DITEMUKAN: Menggunakan data lama (Hemat Biaya) ---")

    # question = "Ayat hafalan minggu ini apa"
    question = "Ayat  Judul pelajaran di triwulan ini apa"
    await answer_question(MODEL, rag, question, llm_tracker, embed_tracker)
    await  rag.finalize_storages()
    print("\nAll processes finished. Check your CSV for research data.")



async def export_data():
    rag = await init_deepseek_lightRAG()
    await rag.aexport_data("graph_data_full.csv", file_format="csv", include_vector_data=True)

    print("Exported!")
    await rag.finalize_storages()



if __name__ == "__main__":
    if not os.getenv("DEEP_SEEK_API_KEY"):
        print("API Key Deep Seek tidak ditemukan! Setel di environment variable.")
    else:
        try:
            asyncio.run(run_deepseek())
        except KeyboardInterrupt:
            print("\nProses dihentikan oleh pengguna.")
