import os
import json
import time
import asyncio

import numpy as np
import pandas as pd
from datetime import datetime
from lightrag import QueryParam, LightRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs
from openai import OpenAI

from core.embedding import gemini_embedding_func


def save_experiment(data):
    file_path = "new_evaluation_results.csv"
    df = pd.DataFrame([data])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


async def judge_metrics(client, question, context, response, reference):
    """
    LLM-as-a-Judge menggunakan SumoPod: Menilai output berdasarkan standar evaluasi ketat.
    """
    prompt = f"""Evaluasi jawaban Sistem RAG berdasarkan standar kurikulum Sekolah Sabat:

Context: {context}
Question: {question}
Response: {response}
Reference: {reference}

Berikan skor (0 atau 1) untuk:
1. accuracy: Benar secara faktual sesuai Reference.
2. relevance: Menjawab inti pertanyaan tanpa bertele-tele.
3. completeness: Mencakup semua poin penting dari Reference.

HANYA JAWAB DENGAN JSON, TANPA PENJELASAN LAIN:
{{"accuracy": 1, "relevance": 1, "completeness": 1, "reason": "..."}}"""

    try:
        res = client.chat.completions.create(
            model="mimo-v2.5-pro",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3,
        )

        response_text = res.choices[0].message.content.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)

        return {
            "accuracy": result.get("accuracy", 0),
            "relevance": result.get("relevance", 0),
            "completeness": result.get("completeness", 0),
            "reason": result.get("reason", "No reason provided")
        }
    except json.JSONDecodeError as e:
        print(f"JSON decode error saat judging: {e}")
        return {
            "accuracy": 0,
            "relevance": 0,
            "completeness": 0,
            "reason": f"JSON decode error: {str(e)}"
        }
    except Exception as e:
        print(f"Error saat judging: {e}")
        return {
            "accuracy": 0,
            "relevance": 0,
            "completeness": 0,
            "reason": f"Error: {str(e)}"
        }


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
    working_dir = "../exp_merge_gemini_third"

    # Initialize SumoPod client untuk judging
    judge_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://ai.sumopod.com/v1"
    )

    # RAG menggunakan Gemini
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
            "question": "Apa judul pelajaran untuk hari Sabtu?",
            "reference": "Judul pelajaran untuk hari Sabtu (Hari ke-1) adalah 'Cek Realitas'."
        },
        {
            "id": "Q02",
            "question": "Mengapa 'Cek Realitas' (materi Sabtu) menjadi langkah awal yang penting sebelum kita membahas 'Kondisi Kita' (materi Minggu)?",
            "reference": "Karena kita tidak bisa bergerak ke tempat yang lebih baik sebelum melakukan kejujuran terhadap diri sendiri mengenai kondisi hubungan kita saat ini dan mendengarkan solusi yang Yesus tawarkan."
        },
    ]

    modes = ["hybrid", "local", "global", "naive", "mix"]

    for item in question_dataset:
        print(f"\n--- Menguji Pertanyaan: {item['id']} ---")
        for mode in modes:
            start_time = time.time()
            try:
                response_text = await rag.aquery(
                    item["question"],
                    param=QueryParam(
                        mode=mode,
                        user_prompt="Anda adalah asisten Sekolah Sabat. Jawab berdasarkan fakta database secara singkat."))
                retrieved_contexts = "Nodes retrieved from LightRAG"
            except Exception as e:
                print(f"Error Query {mode}: {e}")
                response_text = "ERROR"
                retrieved_contexts = "N/A"

            judge = await judge_metrics(judge_client, item["question"], retrieved_contexts, response_text,
                                        item["reference"])
            latency = time.time() - start_time

            save_experiment({
                "user_input": item["question"],
                "retrieved_contexts": retrieved_contexts,
                "response": response_text,
                "reference": item["reference"],
                "accuracy": judge["accuracy"],
                "relevance": judge["relevance"],
                "completeness": judge["completeness"],
                "reason": judge["reason"],
                "strategy": mode,
                "latency": latency,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            print(
                f"Mode: {mode:7} | Acc: {judge['accuracy']} | Rel: {judge['relevance']} | Comp: {judge['completeness']}")

            await asyncio.sleep(3)


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY belum diset di Environment!")
    elif not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY belum diset di Environment!")
    else:
        asyncio.run(run_evaluation())