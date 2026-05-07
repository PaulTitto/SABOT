import os
import json
import time
import asyncio

import numpy as np
import pandas as pd
from datetime import datetime
import google.genai as genai
from lightrag import QueryParam, LightRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

from core.embedding import gemini_embedding_func

MODEL_JUDGE = "gemini-2.5-flash"

def save_experiment(data):
    file_path = "evaluation_results.csv"
    df = pd.DataFrame([data])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


async def judge_metrics(client, question, context, response, reference):
    """
    LLM-as-a-Judge: Menilai output berdasarkan 3 metrik utama.
    """
    prompt = f"""
    Evaluasi jawaban Sistem RAG berdasarkan standar berikut:

    Context: {context}
    Question: {question}
    Response: {response}
    Reference: {reference}

    Berikan skor (0 atau 1) untuk:
    1. accuracy: Apakah informasi dalam jawaban benar secara faktual sesuai reference?
    2. relevance: Apakah jawaban menjawab inti pertanyaan?
    3. completeness: Apakah jawaban mencakup semua poin penting dari reference?

    Format JSON: {{"accuracy": 1, "relevance": 1, "completeness": 1, "reason": "..."}}
    """
    try:
        res = client.models.generate_content(
            model=MODEL_JUDGE,
            contents=prompt,
            config=genai.types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(res.text)
    except Exception as e:
        print(f"Error saat judging: {e}")
        return {"accuracy": 0, "relevance": 0, "completeness": 0}

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.5-flash-lite",
        **kwargs
    )

@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=2048,
    model_name="gemini-embedding-001",
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await gemini_embed.func(
        texts, api_key=os.getenv("GEMINI_API_KEY"), model="models/gemini-embedding-001"
    )
async def run_evaluation():
    working_dir = "../exp_merge_gemini"
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=gemini_embedding_func,
        llm_model_name="gemini-2.5-flash-lite"
    )
    await rag.initialize_storages()
    question_dataset = [
        {
            "id": "Q01",
            "question": "Bagaimana metafora 'getah' menjelaskan kerja Roh Kudus?",
            "reference": "Getah diibaratkan sebagai Roh Kudus yang mengalir dari akar menuju tunas untuk memicu pertumbuhan."
        }
    ]

    modes = ["hybrid", "local", "global", "naive","mix"]

    for item in question_dataset:
        print(f"\n--- Menguji Pertanyaan: {item['id']} ---")
        for mode in modes:
            start_time = time.time()
            try:
                response_text = await rag.aquery(item["question"], param=QueryParam(mode=mode))
                retrieved_contexts = "Nodes retrieved from LightRAG"
            except Exception as e:
                print(f"Error Query {mode}: {e}")
                response_text = "ERROR"
                retrieved_contexts = "N/A"

            judge = await judge_metrics(client, item["question"], retrieved_contexts, response_text, item["reference"])
            latency = time.time() - start_time
            save_experiment({
                "user_input": item["question"],
                "retrieved_contexts": retrieved_contexts,
                "response": response_text,
                "reference": item["reference"],
                "accuracy": judge["accuracy"],
                "relevance": judge["relevance"],
                "completeness": judge["completeness"],
                "strategy": mode,
                "latency": latency,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            print(
                f"Mode: {mode:7} | Acc: {judge['accuracy']} | Rel: {judge['relevance']} | Comp: {judge['completeness']}")

            await asyncio.sleep(1)


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("API Key belum diset di Environment!")
    else:
        asyncio.run(run_evaluation())