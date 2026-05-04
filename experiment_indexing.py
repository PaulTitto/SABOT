import asyncio
import csv
import os
import re
import shutil
import time
from datetime import datetime
from typing import Dict

from lightrag import LightRAG
from lightrag.llm.gemini import gemini_complete_if_cache
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import TokenTracker

from core.deepseek import deepseek_llm
from core.embedding import gemini_embedding_func, embed_tracker
from core.gemini import gemini_llm_model_func
from core.openai import openai_llm
from docs.configure_logging import configure_logging
from helper.calculate_cost import (
    get_cost_by_model,
    get_deepseek_detailed_costs,
    get_gemini_detailed_costs,
    get_openai_detailed_costs,
)
from helper.chunk_graph_stats import count_graph_stats
from helper.dataset_helper import extract_metadata
from server import rag

# ── Constants ─────────────────────────────────────────────────────────────────

SCHOOL_FILES = [
    "./data/2026/02/01/01.txt",
    "./data/2026/02/01/02.txt",
]
LLM_CONFIGS = {
    # "gemini-2.5-flash-lite": {
    #     "name": "Gemini 2.5 Flash-Lite",
    #     "developer": "Google",
    #     "context_window": 1048576,
    #     "input_cost_per_million": 0.10,
    #     "output_cost_per_million": 0.40,
    #     "cost_func": get_gemini_detailed_costs,
    # },
    "deepseek-v4-flash": {
        "name": "DeepSeek V4 Flash",
        "developer": "DeepSeek AI",
        "context_window": 1000000,
        "input_cost_per_million": 0.14,
        "output_cost_per_million": 0.28,
        "cost_func": get_deepseek_detailed_costs,
    },
    # "gpt-4.1-mini": {
    #     "name": "GPT-4.1 Mini",
    #     "developer": "OpenAI",
    #     "context_window": 1047576,
    #     "input_cost_per_million": 0.40,
    #     "output_cost_per_million": 1.60,
    #     "cost_func": get_openai_detailed_costs,
    # },
}


WORKING_DIRS = {
    # ("separate", "gemini-2.5-flash-lite"): "../exp_separate_gemini",
    ("separate", "deepseek-v4-flash"):      "../exp_separate_deepseek",
    ("separate", "gpt-4.1-mini"):           "../exp_separate_gpt4",
    ("merge",    "gemini-2.5-flash-lite"):  "../exp_merge_gemini",
    ("merge",    "deepseek-v4-flash"):      "../exp_merge_deepseek",
    ("merge",    "gpt-4.1-mini"):           "../exp_merge_gpt4",
    ("batch",    "gemini-2.5-flash-lite"):  "../exp_batch_gemini",
    ("batch",    "deepseek-v4-flash"):      "../exp_batch_deepseek",
    ("batch",    "gpt-4.1-mini"):           "../exp_batch_gpt4",
}

EXPERIMENT_CSV = "experiment_indexing_9combinations.csv"

CSV_FIELDNAMES = [
    "timestamp",
    "strategy",
    "llm_model",
    "llm_developer",
    "files_count",
    "llm_p_tokens",
    "llm_c_tokens",
    "emb_p_tokens",
    "cost_llm",
    "cost_emb",
    "total_cost",
    "latency",
    "entity",
    "relation",
    "call_count",
]



def gemini_llm_tracker_func(model: str, tracker: TokenTracker):
    async def _func(prompt, system_prompt=None, history_messages=None, **kwargs):
        kwargs.pop("token_tracker", None)
        return await gemini_complete_if_cache(
            model, prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            token_tracker=tracker,
            api_key=os.getenv("GEMINI_API_KEY"),
            **kwargs,
        )
    return _func


def openai_llm_tracker_func(model: str, tracker: TokenTracker):
    async def _func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
        if model == "gpt-4.1-mini":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
        if model == "deepseek-v4-flash":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL")

        kwargs.pop("token_tracker", None)

        return await openai_complete_if_cache(
            model, prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            token_tracker=tracker,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    return _func

def cleanup_dir(path: str):
    """Delete and recreate a directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_doc_id(content: str) -> str:
    """
    Mirrors dataset_helper.extract_doc_id():
    looks for 'id: <value>' in the document front-matter.
    Falls back to the first 20 chars of the first line.
    """
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


def _pick_llm_func(model_key: str):
    """Return the correct pre-built LLM function for a model key."""
    if model_key == "gpt-4.1-mini":
        return openai_llm
    elif model_key == "deepseek-v4-flash":
        return deepseek_llm
    elif model_key == "gemini-2.5-flash-lite":
        return gemini_llm_model_func
    else:
        raise ValueError(f"Unknown model_key: {model_key!r}")



async def experiment_separate(model_key: str) -> Dict:
    """
    Insert each file into LightRAG individually.
    Records per-file cost, tokens, latency, and graph stats after each insert.
    """
    llm_config = LLM_CONFIGS[model_key]
    working_dir = WORKING_DIRS[("separate", model_key)]

    print(f"\n{'='*60}")
    print(f"  STRATEGI SEPARATE — {llm_config['name']}")
    print(f"  Working dir : {working_dir}")
    print(f"{'='*60}")

    llm_func = _pick_llm_func(model_key)
    cleanup_dir(working_dir)

    llm_tracker = TokenTracker()

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
    )
    await rag.initialize_storages()

    totals = {
        "llm_p": 0, "llm_c": 0, "emb_p": 0,
        "c_llm": 0.0, "c_emb": 0.0, "total": 0.0,
        "latency": 0.0, "call_count": 0,
    }

    for i, filepath in enumerate(SCHOOL_FILES, start=1):
        content = read_file(filepath)
        doc_id  = extract_doc_id(content)

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

        graph = count_graph_stats(working_dir)

        print(
            f"  [{i}/{len(SCHOOL_FILES)}] ID: {doc_id:20} | "
            f"Cost: ${metrics['total']:>10.6f} | "
            f"LLM: {metrics['llm_p']:>6,}p {metrics['llm_c']:>5,}c | "
            f"Emb: {metrics['emb_p']:>6,} | "
            f"Calls: {llm_tracker.get_usage().get('call_count', 0):>3} | "
            f"{latency:>6.2f}s | "
            f"E:{graph['entities']} R:{graph['relations']}"
        )

        save_experiment({
            "timestamp":     datetime.now().isoformat(),
            "strategy":      f"SEPARATE | {doc_id:20} [{i}/{len(SCHOOL_FILES)}]",
            "llm_model":     llm_config["name"],
            "llm_developer": llm_config["developer"],
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

        await asyncio.sleep(0.3)

    await rag.finalize_storages()

    print(f"\n TOTAL SEPARATE ({llm_config['name']}):")
    print(f"     Prompt Tokens    : {totals['llm_p']:,}")
    print(f"     Completion Tokens: {totals['llm_c']:,}")
    print(f"     Embed Tokens     : {totals['emb_p']:,}")
    print(f"     Cost LLM         : ${totals['c_llm']:.6f}")
    print(f"     Cost Embed       : ${totals['c_emb']:.6f}")
    print(f"     Total Cost       : ${totals['total']:.6f}")
    print(f"     Total Latency    : {totals['latency']:.2f}s")
    print(f"     Call Count       : {totals['call_count']}")

    return totals



async def experiment_merge(model_key : str)-> Dict:
    llm_config = LLM_CONFIGS[model_key]
    working_dir = WORKING_DIRS[("merge", model_key)]

    print(f"\n{'=' * 60}")
    print(f"  STRATEGI MERGE — {llm_config['name']}")
    print(f"  Working dir : {working_dir}")
    print(f"{'=' * 60}")

    llm_func = _pick_llm_func(model_key)
    cleanup_dir(working_dir)

    llm_tracker = TokenTracker()

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
    )

    await rag.initialize_storages()

    merged_parts = []
    doc_ids = []
    for filepath in SCHOOL_FILES:
        content = read_file(filepath)
        doc_id = extract_doc_id(content)
        doc_ids.append(doc_id)
        merged_parts.append(content)

    merged_content = "\n\n".join(merged_parts)

    print(f"  Menggabungkan {len(SCHOOL_FILES)} file...")
    print(f"  Doc IDs: {', '.join(doc_ids)}")
    print(f"  Total chars: {len(merged_content):,}")

    llm_tracker.reset()
    embed_tracker.reset()
    start = time.time()

    await rag.ainsert(merged_content)

    latency = time.time() - start

    metrics = get_cost_by_model(
        model_key,
        llm_tracker.get_usage(),
        embed_tracker.get_usage(),
    )
    graph = count_graph_stats(working_dir)

    print(
        f"  [1/1] MERGED {len(SCHOOL_FILES)} files | "
        f"Cost: ${metrics['total']:>10.6f} | "
        f"LLM: {metrics['llm_p']:>6,}p {metrics['llm_c']:>5,}c | "
        f"Emb: {metrics['emb_p']:>6,} | "
        f"Calls: {llm_tracker.get_usage().get('call_count', 0):>3} | "
        f"{latency:>6.2f}s | "
        f"E:{graph['entities']} R:{graph['relations']}"
    )

    save_experiment({
        "timestamp": datetime.now().isoformat(),
        "strategy": f"MEREGE | {len(SCHOOL_FILES)} files",
        "llm_model": llm_config["name"],
        "llm_developer": llm_config["developer"],
        "files_count": len(SCHOOL_FILES),
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
    print(f"\n TOTAL MERGE ({llm_config['name']}):")
    print(f"     Prompt Tokens    : {metrics['llm_p']:,}")
    print(f"     Completion Tokens: {metrics['llm_c']:,}")
    print(f"     Embed Tokens     : {metrics['emb_p']:,}")
    print(f"     Cost LLM         : ${metrics['c_llm']:.6f}")
    print(f"     Cost Embed       : ${metrics['c_emb']:.6f}")
    print(f"     Total Cost       : ${metrics['total']:.6f}")
    print(f"     Total Latency    : {latency:.2f}s")
    print(f"     Call Count       : {llm_tracker.get_usage().get('call_count', 0)}")

    await rag.finalize_storages()

    return metrics


async def experiment_batch(model_key: str) -> Dict:
    llm_config = LLM_CONFIGS[model_key]
    working_dir = WORKING_DIRS[("batch", model_key)]

    print(f"\n{'=' * 60}")
    print(f"  STRATEGI BATCH — {llm_config['name']}")
    print(f"  Working dir : {working_dir}")
    print(f"{'=' * 60}")

    cleanup_dir(working_dir)

    llm_tracker = TokenTracker()

    if model_key == "gemini-2.5-flash-lite":
        llm_func = gemini_llm_tracker_func("gemini-2.5-flash-lite", llm_tracker)
    elif model_key == "deepseek-v4-flash":
        llm_func = openai_llm_tracker_func("deepseek-v4-flash", llm_tracker)
    elif model_key == "gpt-4.1-mini":
        llm_func = openai_llm_tracker_func("gpt-4.1-mini", llm_tracker)
    else:
        raise ValueError(f"Unknown model_key: {model_key!r}")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
    )

    await rag.initialize_storages()

    all_contents = []
    all_ids = []
    all_file_path = []
    metadata_results = []

    for filepath in SCHOOL_FILES:
        with open(filepath, "r", encoding="utf-8") as fp:
            content = fp.read().strip()
        if not content:
            continue

        doc_id = extract_doc_id(content)
        meta = extract_metadata(content)

        all_contents.append(content)
        all_ids.append(doc_id if doc_id else filepath)
        all_file_path.append(filepath)
        metadata_results.append(meta)

    print(f"  Memuat {len(all_contents)} dokumen...")
    print(f"  Doc IDs: {', '.join(all_ids)}")

    llm_tracker.reset()
    embed_tracker.reset()
    start = time.time()

    try:
        print(f"Memulai indexing {len(all_contents)} dokumen secara batch...")
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

    graph = count_graph_stats(working_dir)
    print(
        f"  [{status}] {len(all_contents)} files | "
        f"Cost: ${metrics['total']:>10.6f} | "
        f"LLM: {metrics['llm_p']:>6,}p {metrics['llm_c']:>5,}c | "
        f"Emb: {metrics['emb_p']:>6,} | "
        f"Calls: {llm_tracker.get_usage().get('call_count', 0):>3} | "
        f"{latency:>6.2f}s | "
        f"E:{graph['entities']} R:{graph['relations']}"
    )

    save_experiment({
        "timestamp": datetime.now().isoformat(),
        "strategy": f"BATCH | {len(all_contents)} files | {status}",
        "llm_model": llm_config["name"],
        "llm_developer": llm_config["developer"],
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

    print(f"\n  TOTAL BATCH ({llm_config['name']}):")
    print(f"     Dokumen diproses : {len(all_contents)}")
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


def validate_env():
    required_keys = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"]
    for key in required_keys:
        value = os.getenv(key, "").strip()
        if not value:
            raise ValueError(f"{key} is empty! Please set it in .env")
    print("✓ All API keys are configured")


if __name__ == "__main__":
    validate_env()
    configure_logging()
    for model_key in LLM_CONFIGS:
        # asyncio.run(experiment_separate(model_key))
        # asyncio.run(experiment_merge(model_key))
        asyncio.run(experiment_batch(model_key))