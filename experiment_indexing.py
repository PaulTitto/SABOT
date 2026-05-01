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
from helper.calculate_cost import (
    get_cost_by_model,
    get_deepseek_detailed_costs,
    get_gemini_detailed_costs,
    get_openai_detailed_costs,
)
from helper.chunk_graph_stats import count_graph_stats


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
    "deepseek-chat": {
        "name": "DeepSeek V4 Flash",
        "developer": "DeepSeek AI",
        "context_window": 1000000,
        "input_cost_per_million": 0.14,
        "output_cost_per_million": 0.28,
        "cost_func": get_deepseek_detailed_costs,
    },
    "gpt-4.1-mini": {
        "name": "GPT-4.1 Mini",
        "developer": "OpenAI",
        "context_window": 1047576,
        "input_cost_per_million": 0.40,
        "output_cost_per_million": 1.60,
        "cost_func": get_openai_detailed_costs,
    },
}

WORKING_DIRS = {
    ("separate", "gemini-2.5-flash-lite"): "../exp_separate_gemini",
    ("separate", "deepseek-chat"):      "../exp_separate_deepseek",
    ("separate", "gpt-4.1-mini"):           "../exp_separate_gpt4",
    ("merge",    "gemini-2.5-flash-lite"):  "../exp_merge_gemini",
    ("merge",    "deepseek-chat"):      "../exp_merge_deepseek",
    ("merge",    "gpt-4.1-mini"):           "../exp_merge_gpt4",
    ("batch",    "gemini-2.5-flash-lite"):  "../exp_batch_gemini",
    ("batch",    "deepseek-chat"):      "../exp_batch_deepseek",
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


# ── Tracker-aware LLM wrappers (for batch/ainsert strategies) ────────────────

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
    async def _func(prompt, system_prompt=None, history_messages=None, **kwargs):
        api_key = (
            os.getenv("OPENAI_API_KEY")
            if model == "gpt-4.1-mini"
            else os.getenv("DEEPSEEK_API_KEY")
        )
        kwargs.pop("token_tracker", None)
        return await openai_complete_if_cache(
            model, prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            token_tracker=tracker,
            api_key=api_key,
            **kwargs,
        )
    return _func


# ── Shared helpers ────────────────────────────────────────────────────────────

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
    elif model_key == "deepseek-chat":
        return deepseek_llm
    elif model_key == "gemini-2.5-flash-lite":
        return gemini_llm_model_func
    else:
        raise ValueError(f"Unknown model_key: {model_key!r}")


# ── Experiment: SEPARATE (one file at a time) ─────────────────────────────────

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
            "strategy":      "SEPARATE",
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

        # Accumulate totals
        for key in ("llm_p", "llm_c", "emb_p", "c_llm", "c_emb", "total"):
            totals[key] += metrics[key]
        totals["latency"]    += latency
        totals["call_count"] += llm_tracker.get_usage().get("call_count", 0)

        await asyncio.sleep(0.3)

    await rag.finalize_storages()

    print(f"\n  📊 TOTAL SEPARATE ({llm_config['name']}):")
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
    for model_key in LLM_CONFIGS:
        asyncio.run(experiment_separate(model_key))