# RAGAS VERSI Ragas 0.1.22

import asyncio
import os
import time
from datetime import datetime
import nest_asyncio
import pandas as pd
from datasets import Dataset

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from lightrag import LightRAG, QueryParam
from core.embedding import gemini_embedding_func
from core.gemini import gemini_llm_model_func

nest_asyncio.apply()


EVALUATOR_LLM = ChatOpenAI(
    model="glm-5.1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://ai.sumopod.com/v1",
)
ragas_judge = LangchainLLMWrapper(EVALUATOR_LLM)


class RagasGeminiEmbedding(BaseRagasEmbeddings):
    def __init__(self, embedding_func):
        self.embedding_func = embedding_func

    def embed_query(self, text: str):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aembed_query(text))

    def embed_documents(self, texts: list[str]):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aembed_documents(texts))

    async def aembed_query(self, text: str):
        res = await self.embedding_func([text])
        return res[0]

    async def aembed_documents(self, texts: list[str]):
        return await self.embedding_func(texts)


ragas_embeddings = RagasGeminiEmbedding(gemini_embedding_func)

question_dataset = [
    {
        "id": "Q01",
        "type": "faktual",
        "source_doc": "week-01.txt — Sabtu",
        "question": "Apa judul pelajaran hari Sabtu pada minggu pertama dan kutipan lengkap ayat hafalan berdasarkan teks tersebut?",
        "ground_truth": "Judul pelajaran hari Sabtu adalah 'Cek Realitas'. Ayat hafalan minggu tersebut adalah Yohanes 15:9 yang berbunyi: 'Seperti Bapa telah mengasihi Aku, demikianlah juga Aku telah mengasihi kamu; tinggallah di dalam kasih-Ku itu'."
    },

]


async def get_rag_data(rag, question, mode_name):
    answer = await rag.aquery(question, param=QueryParam(mode=mode_name))
    context = await rag.aquery(question, param=QueryParam(mode=mode_name, only_need_context=True))
    return str(answer), [str(context)]


async def run_eval_comprehensive(strategy_name, working_dir, target_llm_name, target_llm_func):
    modes = ["hybrid", "local", "global", "naive", "mix"]
    all_mode_dfs = []

    for current_mode in modes:
        print(f"\n[RUNNING] Strategi: {strategy_name} | Generator LLM: {target_llm_name} | Mode: {current_mode}")

        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=target_llm_func,
            embedding_func=gemini_embedding_func,
        )
        await rag.initialize_storages()

        results_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        meta_id = []
        meta_type = []

        for item in question_dataset:
            ans, ctx = await get_rag_data(rag, item['question'], current_mode)
            results_data["question"].append(item["question"])
            results_data["answer"].append(ans)
            results_data["contexts"].append(ctx)
            results_data["ground_truth"].append(item["ground_truth"])
            meta_id.append(item["id"])
            meta_type.append(item["type"])

        dataset = Dataset.from_dict(results_data)

        print(f"  [RAGAS EVAL] Menghitung skor kualitas (Ragas 0.1.22)...")

        eval_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_judge,
            embeddings=ragas_embeddings
        )

        df = eval_result.to_pandas()

        df["id"] = meta_id
        df["type"] = meta_type
        df["user_input"] = df["question"]
        df["retrieved_contexts"] = df["contexts"]
        df["response"] = df["answer"]
        df["reference"] = df["ground_truth"]

        df["strategy"] = strategy_name
        df["llm"] = target_llm_name
        df["mode"] = current_mode
        df["timestamp"] = datetime.now().isoformat()

        desired_columns = [
            "id", "type", "user_input", "retrieved_contexts", "response", "reference",
            "faithfulness", "answer_relevancy", "strategy", "llm", "mode", "timestamp"
        ]

        df = df.reindex(columns=desired_columns)
        all_mode_dfs.append(df)

        await rag.finalize_storages()
        await asyncio.sleep(2)

    return pd.concat(all_mode_dfs, ignore_index=True)


async def main():
    start_time = time.time()
    try:
        from core.deepseek import deepseek_llm
        from core.openai import openai_llm

        llm_targets = [
            {"name": "Gemini-2.5-Flash-Lite", "func": gemini_llm_model_func},
            {"name": "DeepSeek-V4-Flash", "func": deepseek_llm},
            {"name": "GPT-4.1-Mini", "func": openai_llm}
        ]

        reports = []
        target_storage = {"name": "BATCH", "path": "../final_working_dir"}

        for target in llm_targets:
            df_result = await run_eval_comprehensive(
                strategy_name=target_storage["name"],
                working_dir=target_storage["path"],
                target_llm_name=target["name"],
                target_llm_func=target["func"]
            )
            reports.append(df_result)

        final_report = pd.concat(reports, ignore_index=True)
        final_report.to_csv("experiment_ragas_matrix.csv", index=False)

        print("\n" + "=" * 70)
        print(f"EVALUASI MATRIKS SELESAI | File experiment_ragas_matrix.csv Berhasil Diekspor!")
        print("=" * 70)

        summary = final_report.groupby(["strategy", "llm", "mode"])[["faithfulness", "answer_relevancy"]].mean()
        print(summary)

    except Exception as e:
        import traceback
        print(f"\n[CRITICAL ERROR] Gagal Eksekusi: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())