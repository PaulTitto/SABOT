# import os
# from datetime import time, datetime
#
# from core.gemini import llm_tracker, embed_tracker
# from helper.calculate_cost import get_gemini_detailed_costs
# from helper.save_csv import save_to_csv
#
#
# async def run_ingestion(rag, data_path):
#     storage_file = os.path.join(rag.working_dir, "kv_store_full_text.json")
#     if os.path.exists(storage_file):
#         print("--- Database ditemukan. Melewati proses indexing ---")
#         return
#
#     print(f"--- Memulai Indexing Dataset dari {data_path} ---")
#     llm_tracker.reset()
#     embed_tracker.reset()
#     start = time.time()
#
#     from helper.dataset_helper import insert_documents_folder
#     await insert_documents_folder(rag, data_path)
#
#     metrics = get_gemini_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())
#     save_to_csv({
#         "timestamp": datetime.now().isoformat(),
#         "mode": "INDEXING",
#         "question": f"Dataset: {data_path}",
#         "llm_p_tokens": metrics["llm_p"],
#         "llm_c_tokens": metrics["llm_c"],
#         "embed_tokens": metrics["emb_p"],
#         "cost_llm": metrics["c_llm"],
#         "cost_embed": metrics["c_emb"],
#         "total_cost": metrics["total"],
#         "latency": time.time() - start,
#         "call_count": llm_tracker.get_usage().get("call_count", 0)
#     })