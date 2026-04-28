"""
experiment_indexing_comparison.py (v3.0 - Fixed)
Eksperimen perbandingan strategi indexing untuk skripsi SABOT:
- Strategi A: SEPARATE  — insert harian satu per satu (sequential)
- Strategi B: MERGE     — gabung semua konten jadi satu dokumen, insert sekali
- Strategi C: BATCH     — insert list dokumen sekaligus via ainsert(list) (LightRAG native batch)

Fix:
- Token tracker menggunakan closure pattern (sesuai dokumentasi resmi LightRAG dev branch)
- Batch menggunakan ainsert(list) bukan asyncio.gather (ini batch LightRAG yang sebenarnya)
- Cost hanya dari LLM tokens, embedding dipisah dan tidak dicampur
"""

import asyncio
import os
import shutil
import time
import re
from datetime import datetime

from core.embedding import gemini_embedding_func
from lightrag.utils import TokenTracker
from lightrag.llm.gemini import gemini_complete_if_cache
from helper.calculate_cost import get_gemini_detailed_costs

FILE_D1 = "../data/2026/02/01/01.txt"
FILE_D2 = "../data/2026/02/01/02.txt"

WORKING_DIR_SEPARATE = "../exp_separate_storage"
WORKING_DIR_MERGE    = "../exp_merge_storage"
WORKING_DIR_BATCH    = "../exp_batch_storage"

MODEL = "gemini-2.5-flash-lite"
EXPERIMENT_CSV = "experiment_indexing_comparison.csv"


# --- Closure Pattern (Dokumentasi Resmi LightRAG dev branch) ---
def make_llm_func(tracker: TokenTracker):
    """
    Buat LLM function dengan tracker di-capture via closure.
    Tracker selalu terhubung ke setiap internal worker call,
    tidak bergantung pada llm_model_kwargs propagation.
    """
    async def _llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        **kwargs,
    ):
        kwargs.pop("token_tracker", None)  # hindari conflict jika LightRAG inject kwargs
        return await gemini_complete_if_cache(
            MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            token_tracker=tracker,
            api_key=os.getenv("GEMINI_API_KEY"),
            **kwargs,
        )
    return _llm_model_func


# --- Helper Functions ---
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_doc_id(content: str) -> str:
    match = re.search(r'^id:\s*(\S+)', content, re.MULTILINE)
    return match.group(1) if match else "unknown"

def cleanup_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_experiment(row: dict):
    import csv
    fieldnames = ["timestamp", "strategy", "file", "llm_p_tokens", "llm_c_tokens",
                  "llm_cost", "latency", "call_count"]
    file_exists = os.path.exists(EXPERIMENT_CSV)
    with open(EXPERIMENT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# --- Strategi A: SEPARATE ---
async def experiment_separate():
    print("\n" + "="*55)
    print("STRATEGI A: SEPARATE (Penyisipan Harian Sequential)")
    print("="*55)

    cleanup_dir(WORKING_DIR_SEPARATE)

    # Buat tracker baru dan llm_func dengan closure
    llm_tracker = TokenTracker()
    llm_func = make_llm_func(llm_tracker)

    from lightrag import LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR_SEPARATE,
        llm_model_func=llm_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=False,
        # Tidak perlu llm_model_kwargs — tracker via closure
    )
    await rag.initialize_storages()

    total_metrics = {"llm_p": 0, "llm_c": 0, "cost": 0.0, "latency": 0.0, "call_count": 0}

    for i, filepath in enumerate([FILE_D1, FILE_D2], start=1):
        content = read_file(filepath)
        doc_id = extract_doc_id(content)

        llm_tracker.reset()
        start = time.time()

        await rag.ainsert(content)

        latency = time.time() - start
        usage = llm_tracker.get_usage()

        # Hitung cost dari LLM tokens saja
        metrics = get_gemini_detailed_costs(usage, {})

        print(f"  [File {i}] ID: {doc_id} | LLM Cost: ${metrics['c_llm']:.6f} | "
              f"Tokens: {usage['prompt_tokens']:,} | Calls: {usage['call_count']} | {latency:.2f}s")

        row = {
            "timestamp": datetime.now().isoformat(),
            "strategy": "SEPARATE",
            "file": f"d{i}",
            "llm_p_tokens": usage["prompt_tokens"],
            "llm_c_tokens": usage["completion_tokens"],
            "llm_cost": metrics['c_llm'],
            "latency": round(latency, 3),
            "call_count": usage["call_count"],
        }
        save_experiment(row)

        total_metrics["llm_p"] += usage["prompt_tokens"]
        total_metrics["llm_c"] += usage["completion_tokens"]
        total_metrics["cost"] += metrics['c_llm']
        total_metrics["latency"] += latency
        total_metrics["call_count"] += usage["call_count"]

        await asyncio.sleep(0.5)

    await rag.finalize_storages()
    return total_metrics


# --- Strategi B: MERGE ---
async def experiment_merge():
    print("\n" + "="*55)
    print("STRATEGI B: MERGE (Gabung Semua File Jadi Satu Dokumen)")
    print("="*55)

    cleanup_dir(WORKING_DIR_MERGE)

    llm_tracker = TokenTracker()
    llm_func = make_llm_func(llm_tracker)

    from lightrag import LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR_MERGE,
        llm_model_func=llm_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=False,
    )
    await rag.initialize_storages()

    content1 = read_file(FILE_D1)
    content2 = read_file(FILE_D2)
    merged_content = f"{content1}\n\n---\n\n{content2}"
    merged_id = "merged_d1_d2"

    llm_tracker.reset()
    start = time.time()

    await rag.ainsert(merged_content, ids=merged_id)

    latency = time.time() - start
    usage = llm_tracker.get_usage()
    metrics = get_gemini_detailed_costs(usage, {})

    print(f"  [MERGE] ID: {merged_id} | LLM Cost: ${metrics['c_llm']:.6f} | "
          f"Tokens: {usage['prompt_tokens']:,} | Calls: {usage['call_count']} | {latency:.2f}s")

    row = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "MERGE",
        "file": "d1+d2_merged",
        "llm_p_tokens": usage["prompt_tokens"],
        "llm_c_tokens": usage["completion_tokens"],
        "llm_cost": metrics['c_llm'],
        "latency": round(latency, 3),
        "call_count": usage["call_count"],
    }
    save_experiment(row)

    await rag.finalize_storages()
    return {
        "llm_p": usage["prompt_tokens"],
        "llm_c": usage["completion_tokens"],
        "cost": metrics['c_llm'],
        "latency": latency,
        "call_count": usage["call_count"],
    }


# --- Strategi C: BATCH (LightRAG Native Batch via ainsert list) ---
async def experiment_batch():
    print("\n" + "="*55)
    print("STRATEGI C: BATCH (LightRAG Native ainsert dengan List)")
    print("="*55)

    cleanup_dir(WORKING_DIR_BATCH)

    llm_tracker = TokenTracker()
    llm_func = make_llm_func(llm_tracker)

    from lightrag import LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR_BATCH,
        llm_model_func=llm_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=False,
    )
    await rag.initialize_storages()

    all_contents = []
    all_ids = []
    all_paths = []
    for path in [FILE_D1, FILE_D2]:
        content = read_file(path)
        all_contents.append(content)
        all_ids.append(extract_doc_id(content))
        all_paths.append(path)

    llm_tracker.reset()
    start = time.time()

    # Ini batch LightRAG yang benar — bukan asyncio.gather
    await rag.ainsert(
        input=all_contents,
        ids=all_ids,
        file_paths=all_paths,
    )

    latency = time.time() - start
    usage = llm_tracker.get_usage()
    metrics = get_gemini_detailed_costs(usage, {})

    final_metrics = {
        "llm_p": usage["prompt_tokens"],
        "llm_c": usage["completion_tokens"],
        "cost": metrics['c_llm'],
        "latency": latency,
        "call_count": usage["call_count"],
    }

    print(f"  [BATCH] Indexed {len(all_contents)} files | LLM Cost: ${final_metrics['cost']:.6f} | "
          f"Tokens: {final_metrics['llm_p']:,} | Calls: {final_metrics['call_count']} | {latency:.2f}s")

    row = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "BATCH",
        "file": "d1+d2_batch",
        "llm_p_tokens": final_metrics["llm_p"],
        "llm_c_tokens": final_metrics["llm_c"],
        "llm_cost": final_metrics["cost"],
        "latency": round(latency, 3),
        "call_count": final_metrics["call_count"],
    }
    save_experiment(row)

    await rag.finalize_storages()
    return final_metrics


# --- Main ---
async def main():
    print("\n" + "🚀 MEMULAI EKSPERIMEN INDEXING SABOT".center(60))
    print(f"Model: {MODEL}")
    print("Strategi: SEPARATE | MERGE | BATCH (LightRAG native ainsert list)")
    print("Catatan: Cost = LLM tokens only, embedding tidak dihitung")

    # m_sep  = await experiment_separate()
    # m_merge = await experiment_merge()
    # m_bat  = \
    await experiment_batch()
    #
    # def pct(base, compare):
    #     if base == 0:
    #         return 0.0
    #     return ((base - compare) / base) * 100

    # print("\n" + "📊 HASIL PERBANDINGAN AKHIR".center(85))
    # print("-" * 85)
    # print(f"{'Metrik':<28} {'SEPARATE':>12} {'MERGE':>12} {'BATCH':>12}")
    # print("-" * 85)
    # print(f"{'LLM Prompt Tokens':<28} {m_sep['llm_p']:>12,} {m_merge['llm_p']:>12,} {m_bat['llm_p']:>12,}")
    # print(f"{'LLM Completion Tokens':<28} {m_sep['llm_c']:>12,} {m_merge['llm_c']:>12,} {m_bat['llm_c']:>12,}")
    # print(f"{'LLM Cost (USD)':<28} {m_sep['cost']:>12.6f} {m_merge['cost']:>12.6f} {m_bat['cost']:>12.6f}")
    # print(f"{'Latency (Detik)':<28} {m_sep['latency']:>12.2f} {m_merge['latency']:>12.2f} {m_bat['latency']:>12.2f}")
    # print(f"{'Call Count':<28} {m_sep['call_count']:>12} {m_merge['call_count']:>12} {m_bat['call_count']:>12}")
    # print("-" * 85)
    # print(f"{'Hemat vs SEPARATE (token)':<28} {'':>12} {m_sep['llm_p']-m_merge['llm_p']:>12,} {m_sep['llm_p']-m_bat['llm_p']:>12,}")
    # print(f"{'Hemat vs SEPARATE (cost)':<28} {'':>12} {m_sep['cost']-m_merge['cost']:>12.6f} {m_sep['cost']-m_bat['cost']:>12.6f}")
    # print(f"{'Hemat vs SEPARATE (%)':<28} {'':>12} {pct(m_sep['cost'], m_merge['cost']):>11.1f}% {pct(m_sep['cost'], m_bat['cost']):>11.1f}%")
    # print("=" * 85)
    #
    # best = min([("SEPARATE", m_sep), ("MERGE", m_merge), ("BATCH", m_bat)], key=lambda x: x[1]["cost"])
    # print(f"✅ KESIMPULAN: Strategi {best[0]} paling hemat biaya LLM.")
    print(f"✅ Data lengkap tersimpan di: {EXPERIMENT_CSV}")


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: API Key Gemini tidak ditemukan!")
    else:
        asyncio.run(main())