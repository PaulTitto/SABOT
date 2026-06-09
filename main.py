import os
import re
import csv
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from lightrag import LightRAG
from lightrag.utils import TokenTracker

from core.embedding import gemini_embedding_func, embed_tracker
from core.gemini import gemini_llm_model_func
from helper.calculate_cost import get_cost_by_model
from helper.chunk_graph_stats import count_graph_stats

BASE_DATA_DIR = "data-merge/2026/02"
WORKING_DIR = "../../final_boss_working_dir"
EXPERIMENT_CSV = "experiment_results_final_boss.csv"
model_key = "gemini"

CSV_FIELDNAMES = [
    "timestamp", "strategy", "week", "llm_model", "llm_developer",
    "files_count", "llm_p_tokens", "llm_c_tokens", "emb_p_tokens",
    "cost_llm", "cost_emb", "total_cost", "latency",
    "entity", "relation", "call_count",
]


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_doc_id(content: str) -> str:
    match = re.search(r'^id:\s*(\S+)', content, re.MULTILINE)
    if match:
        return match.group(1)
    first_line = content.splitlines()[0].strip() if content else ""
    return first_line[:20] or "UNKNOWN"


def save_experiment(row: dict):
    """Append one result row to the shared experiment CSV."""
    file_exists = os.path.exists(EXPERIMENT_CSV)
    with open(EXPERIMENT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_merged_weekly_files(base_dir: str) -> list[Path]:
    """Mengambil list file week-*.txt dari folder data-merge secara urut."""
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Folder merge tidak ditemukan: {base_dir}")

    files = sorted(
        [f for f in base.iterdir() if f.is_file() and f.name.startswith("week-") and f.suffix in (".txt", ".md")],
        key=lambda f: f.name
    )
    return files


def cleanup_dir(path: str):
    """Delete and recreate a directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


async def experiment_merge_batch() -> Dict:
    print(f"\n{'=' * 60}")
    print(f"  STRATEGI BATCH MERGE — Gemini")
    print(f"  Working dir : {WORKING_DIR}")
    print(f"{'=' * 60}")

    cleanup_dir(WORKING_DIR)

    llm_tracker = TokenTracker()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
        llm_model_max_async=2,
        embedding_func_max_async=4,
    )

    await rag.initialize_storages()

    all_contents = []
    all_ids = []
    all_file_path = []

    try:
        weekly_files = get_merged_weekly_files(BASE_DATA_DIR)
    except FileNotFoundError as e:
        print(f"Status ERROR: {e}")
        await rag.finalize_storages()
        return {}

    for filepath in weekly_files:
        content = read_file(str(filepath)).strip()
        if not content:
            continue

        doc_id = extract_doc_id(content)

        final_id = doc_id if doc_id and doc_id != "UNKNOWN" else filepath.stem

        all_contents.append(content)
        all_ids.append(final_id)
        all_file_path.append(str(filepath))

    print(f"  Memuat {len(all_contents)} file mingguan dari data-merge...")
    print(f"  Doc IDs: {', '.join(all_ids)}")

    llm_tracker.reset()
    embed_tracker.reset()
    start = time.time()

    try:
        print(f"\nMemulai indexing {len(all_contents)} dokumen minggu secara batch...")
        await rag.ainsert(
            input=all_contents,
            ids=all_ids,
            file_paths=all_file_path,
        )
        status = "SUCCESS"
    except Exception as e:
        status = f"ERROR: {e}"
        print(f"Status {status}")

    latency = time.time() - start
    metrics = get_cost_by_model(
        model_key,
        llm_tracker.get_usage(),
        embed_tracker.get_usage(),
    )

    graph = count_graph_stats(WORKING_DIR)
    print(
        f"  [{status}] {len(all_contents)} weeks | "
        f"Cost: ${metrics['total']:>10.6f} | "
        f"LLM: {metrics['llm_p']:>6,}p {metrics['llm_c']:>5,}c | "
        f"Emb: {metrics['emb_p']:>6,} | "
        f"Calls: {llm_tracker.get_usage().get('call_count', 0):>3} | "
        f"{latency:.2f}s | "
        f"E:{graph['entities']} R:{graph['relations']}"
    )

    save_experiment({
        "timestamp": datetime.now().isoformat(),
        "strategy": f"BATCH MERGE | {len(all_contents)} weeks | {status}",
        "week": f"All {len(all_contents)} Weeks",
        "llm_model": "gemini",
        "llm_developer": "Google",
        "files_count": len(all_contents),
        "llm_p_tokens": metrics["llm_p"],
        "llm_c_tokens": metrics["llm_c"],
        "emb_p_tokens": metrics["emb_p"],
        "cost_llm": metrics["c_llm"],
        "cost_emb": metrics["c_emb"],
        "total_cost": metrics["total"],
        "latency": round(latency, 3),
        "entity": graph["entities"],
        "relation": graph["relations"],
        "call_count": llm_tracker.get_usage().get("call_count", 0),
    })

    await rag.finalize_storages()

    print(f"\n  TOTAL BATCH MERGE (Gemini):")
    print(f"     Minggu diproses  : {len(all_contents)}")
    print(f"     Prompt Tokens    : {metrics['llm_p']:,}")
    print(f"     Completion Tokens: {metrics['llm_c']:,}")
    print(f"     Embed Tokens     : {metrics['emb_p']:,}")
    print(f"     Cost LLM         : ${metrics['c_llm']:.6f}")
    print(f"     Cost Embed       : ${metrics['c_emb']:.6f}")
    print(f"     Total Cost       : ${metrics['total']:.6f}")
    print(f"     Total Latency    : {latency:.2f}s")
    print(f"     Call Count       : {llm_tracker.get_usage().get('call_count', 0)}")
    print(f"     Status           : {status}")

    return metrics

async def experiment_separate_sequential() -> Dict:
    print(f"\n{'=' * 60}")
    print(f"  STRATEGI SEPARATE SEQUENTIAL — Gemini")
    print(f"  Working dir : {WORKING_DIR}")
    print(f"{'=' * 60}")

    # cleanup_dir(WORKING_DIR)

    llm_tracker = TokenTracker()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
        # llm_model_max_async=1,
        # embedding_func_max_async=2,
    )
    await rag.initialize_storages()

    try:
        weekly_files = get_merged_weekly_files(BASE_DATA_DIR)
    except FileNotFoundError as e:
        print(f"Status ERROR: {e}")
        await rag.finalize_storages()
        return {}

    totals = {
        "llm_p": 0, "llm_c": 0, "emb_p": 0,
        "c_llm": 0.0, "c_emb": 0.0, "total": 0.0,
        "latency": 0.0, "call_count": 0,
    }

    for i, filepath in enumerate(weekly_files, start=1):
        content = read_file(str(filepath)).strip()
        if not content:
            continue

        doc_id = extract_doc_id(content)
        final_id = doc_id if doc_id and doc_id != "UNKNOWN" else filepath.stem

        llm_tracker.reset()
        embed_tracker.reset()
        start = time.time()

        await rag.ainsert(content)

        latency = time.time() - start

        metrics = get_cost_by_model(
            model_key,
            llm_tracker.get_usage(),
            embed_tracker.get_usage(),
        )

        graph = count_graph_stats(WORKING_DIR)

        print(
            f"  [{i}/{len(weekly_files)}] ID: {final_id:20} | "
            f"Cost: ${metrics['total']:>10.6f} | "
            f"LLM: {metrics['llm_p']:>6,}p {metrics['llm_c']:>5,}c | "
            f"Emb: {metrics['emb_p']:>6,} | "
            f"Calls: {llm_tracker.get_usage().get('call_count', 0):>3} | "
            f"{latency:>6.2f}s | "
            f"E:{graph['entities']} R:{graph['relations']}"
        )

        save_experiment({
            "timestamp":     datetime.now().isoformat(),
            "strategy":      f"SEPARATE MANUAL LOOP | {final_id:20} [{i}/{len(weekly_files)}]",
            "week":          f"Week {final_id}",
            "llm_model":     "gemini",
            "llm_developer": "Google",
            "files_count":   1,
            "llm_p_tokens":  metrics["llm_p"],
            "llm_c_tokens":  metrics["llm_c"],
            "emb_p_tokens":  metrics["emb_p"],
            "cost_llm":      metrics["c_llm"],
            "cost_emb":      metrics["c_emb"],
            "total_cost":    metrics["total"],
            "latency":       round(latency, 3),
            "entity":        graph["entities"],
            "relation":      graph["relations"],
            "call_count":    llm_tracker.get_usage().get("call_count", 0),
        })

        for key in ("llm_p", "llm_c", "emb_p", "c_llm", "c_emb", "total"):
            totals[key] += metrics[key]
        totals["latency"]    += latency
        totals["call_count"] += llm_tracker.get_usage().get("call_count", 0)

        await asyncio.sleep(3.0)

    await rag.finalize_storages()

    print(f"\n TOTAL SEPARATE SEQUENTIAL (Gemini):")
    print(f"     Prompt Tokens    : {totals['llm_p']:,}")
    print(f"     Completion Tokens: {totals['llm_c']:,}")
    print(f"     Embed Tokens     : {totals['emb_p']:,}")
    print(f"     Cost LLM         : ${totals['c_llm']:.6f}")
    print(f"     Cost Embed       : ${totals['c_emb']:.6f}")
    print(f"     Total Cost       : ${totals['total']:.6f}")
    print(f"     Total Latency    : {totals['latency']:.2f}s")
    print(f"     Call Count       : {totals['call_count']}")

    return totals
if __name__ == "__main__":
    import asyncio

    asyncio.run(experiment_separate_sequential())