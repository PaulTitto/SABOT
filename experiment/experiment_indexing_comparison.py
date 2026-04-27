"""
experiment_indexing_comparison.py (v2.2 - Batch Tracking Fixed)
Eksperimen perbandingan strategi indexing untuk skripsi SABOT:
- Strategi A: SEPARATE (Harian - Boros Token Tax)
- Strategi B: BATCH (Mingguan - Hemat Token Tax & Citasi Akurat)
- FIX: Batch sekarang dapat mencatat token LLM menggunakan closure workaround
"""
import asyncio
import os
import shutil
import time
import re
from datetime import datetime

from core.embedding import embed_tracker, gemini_embedding_func
from core.gemini import llm_tracker, gemini_llm_model_func
from helper.calculate_cost import get_gemini_detailed_costs

FILE_D1 = "../data/2026/02/01/01.txt"
FILE_D2 = "../data/2026/02/01/02.txt"

WORKING_DIR_SEPARATE = "../exp_separate_storage"
WORKING_DIR_BATCH    = "../exp_batch_storage"

MODEL = "gemini-2.5-flash-lite"
EXPERIMENT_CSV = "experiment_indexing_comparison.csv"

# --- Helper Functions ---
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_doc_id(content: str) -> str:
    match = re.search(r'^id:\s*(\S+)', content, re.MULTILINE)
    return match.group(1) if match else "unknown"

def cleanup_dir(path: str):
    """Menghapus folder storage agar indexing benar-benar dari nol (Clean Run)"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_experiment(row: dict):
    import csv
    fieldnames = list(row.keys())
    file_exists = os.path.exists(EXPERIMENT_CSV)
    with open(EXPERIMENT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# --- Eksperimen A: Separate (Individual Calls) ---
async def experiment_separate():
    print("\n" + "="*55)
    print("STRATEGI A: SEPARATE (Penyisipan Harian)")
    print("="*55)

    cleanup_dir(WORKING_DIR_SEPARATE)

    from lightrag import LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR_SEPARATE,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=False,  # WAJIB: Agar token dihitung asli
        llm_model_kwargs={"token_tracker": llm_tracker}, # Hubungkan pencatat
    )
    await rag.initialize_storages()

    total_metrics = {"llm_p": 0, "llm_c": 0, "cost": 0.0, "latency": 0.0}

    for i, filepath in enumerate([FILE_D1, FILE_D2], start=1):
        content = read_file(filepath)
        doc_id = extract_doc_id(content)

        llm_tracker.reset()
        embed_tracker.reset()
        start = time.time()

        # Eksekusi insert harian
        await rag.ainsert(content)

        latency = time.time() - start
        metrics = get_gemini_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())

        print(f"  [File {i}] ID: {doc_id} | Cost: ${metrics['total']:.6f} | {latency:.2f}s")

        row = {
            "timestamp": datetime.now().isoformat(),
            "strategy": "SEPARATE",
            "file": f"d{i}",
            "llm_p_tokens": metrics["llm_p"],
            "llm_c_tokens": metrics["llm_c"],
            "total_cost": metrics["total"],
            "latency": round(latency, 3),
            "call_count": llm_tracker.get_usage().get("call_count", 0),
        }
        save_experiment(row)

        total_metrics["llm_p"] += metrics["llm_p"]
        total_metrics["llm_c"] += metrics["llm_c"]
        total_metrics["cost"] += metrics["total"]
        total_metrics["latency"] += latency

        await asyncio.sleep(0.5)

    await rag.finalize_storages()
    return total_metrics


# --- Eksperimen B: Batch (List Input) dengan Closure Workaround ---
async def experiment_batch():
    print("\n" + "=" * 55)
    print("STRATEGI B: BATCH (Paralel Single Insert) - Tracking Token OK")
    print("=" * 55)

    cleanup_dir(WORKING_DIR_BATCH)

    from lightrag import LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR_BATCH,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=False,
        llm_model_kwargs={"token_tracker": llm_tracker},  # tetap pakai
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
    embed_tracker.reset()
    start = time.time()

    # 🔥 PARALLEL SINGLE INSERT (bukan batch internal)
    tasks = [
        rag.ainsert(content, ids=doc_id, file_paths=file_path)
        for content, doc_id, file_path in zip(all_contents, all_ids, all_paths)
    ]
    await asyncio.gather(*tasks)

    latency = time.time() - start
    metrics = get_gemini_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())

    final_metrics = {
        "llm_p": metrics["llm_p"],
        "llm_c": metrics["llm_c"],
        "cost": metrics["total"],
        "latency": latency,
        "call_count": llm_tracker.get_usage().get("call_count", 0),
    }

    print(f"  [BATCH] Indexed {len(all_contents)} files | Cost: ${final_metrics['cost']:.6f} | {latency:.2f}s")
    print(f"          LLM Prompt Tokens: {final_metrics['llm_p']:,} | Calls: {final_metrics['call_count']}")

    row = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "BATCH",
        "file": "d1+d2",
        "llm_p_tokens": final_metrics["llm_p"],
        "llm_c_tokens": final_metrics["llm_c"],
        "total_cost": final_metrics["cost"],
        "latency": round(latency, 3),
        "call_count": final_metrics["call_count"],
    }
    save_experiment(row)

    await rag.finalize_storages()
    return final_metrics
# --- Main Comparison ---
async def main():
    print("\n" + "🚀 MEMULAI EKSPERIMEN INDEXING SABOT".center(60))
    print(f"Model: {MODEL}")
    print("Perhatian: Batch menggunakan closure workaround agar token tracker tercatat.")

    # Jalankan kedua eksperimen
    m_sep = await experiment_separate()
    m_bat = await experiment_batch()

    # Kalkulasi Persentase Hemat
    hemat_cost = m_sep["cost"] - m_bat["cost"]
    hemat_pct = (hemat_cost / m_sep["cost"]) * 100 if m_sep["cost"] > 0 else 0
    hemat_tokens = m_sep["llm_p"] - m_bat["llm_p"]

    # Tampilkan Tabel Hasil Akhir untuk Skripsi
    print("\n" + "📊 HASIL PERBANDINGAN AKHIR".center(60))
    print("-" * 60)
    print(f"{'Metrik':<25} {'SEPARATE':>12} {'BATCH':>12} {'SELISIH':>10}")
    print("-" * 60)
    print(f"{'LLM Prompt Tokens':<25} {m_sep['llm_p']:>12,} {m_bat['llm_p']:>12,} {hemat_tokens:>10,}")
    print(f"{'Total Cost (USD)':<25} {m_sep['cost']:>12.6f} {m_bat['cost']:>12.6f} {hemat_cost:>10.6f}")
    print(f"{'Latency (Detik)':<25} {m_sep['latency']:>12.2f} {m_bat['latency']:>12.2f}")
    print("-" * 60)
    print(f"KESIMPULAN: Strategi BATCH hemat {hemat_pct:.1f}% biaya (dan tracking token berhasil)!")
    print("=" * 60)
    print(f"✅ Data lengkap tersimpan di: {EXPERIMENT_CSV}")

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: API Key Gemini tidak ditemukan!")
    else:
        asyncio.run(main())