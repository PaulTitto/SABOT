import asyncio
import os
import pandas as pd
import nest_asyncio
from datasets import Dataset
from datetime import datetime
from langchain_openai import ChatOpenAI
from ragas import evaluate

from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings

from lightrag import LightRAG, QueryParam
from core.gemini import gemini_llm_model_func
from core.embedding import gemini_embedding_func

nest_asyncio.apply()


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

EVALUATOR_LLM = ChatOpenAI(
    model="glm-5.1",
    openai_api_key=os.getenv("SUMO_POD_API_KEY"),
    openai_api_base="https://ai.sumopod.com/v1",
)
ragas_judge = LangchainLLMWrapper(EVALUATOR_LLM)

GOLDEN_DATASET = [
    {
        "question": "Apa judul harian untuk pelajaran tanggal 28 Maret 2026?",
        "ground_truth": "Judul hariannya adalah Cek Realitas."
    },
    {
        "question": "Tuliskan isi ayat hafalan yang terdapat dalam Yohanes 15:9.",
        "ground_truth": "Seperti Bapa telah mengasihi Aku, demikianlah juga Aku telah mengasihi kamu; tinggallah di dalam kasih-Ku itu."
    },
    {
        "question": "Apa saja tiga hal yang ditawarkan Yesus untuk ditukarkan dengan keapatisan kita?",
        "ground_truth": "Yesus menawarkan emas-Nya, pakaian putih-Nya, dan salep mata-Nya."
    },
    {
        "question": "Bagaimana Yesus menggambarkan diri-Nya dalam Wahyu 3:14?",
        "ground_truth": "Yesus menyatakan bahwa Dia adalah Saksi yang setia dan benar, permulaan dari ciptaan Allah."
    }
]


async def get_rag_data(rag, question):
    answer = await rag.aquery(question, param=QueryParam(mode="hybrid"))
    context = await rag.aquery(question, param=QueryParam(mode="hybrid", only_need_context=True))
    return str(answer), [str(context)]


async def run_eval_for_storage(strategy_name, working_dir):
    print(f"\n=== MENGEVALUASI STRATEGI: {strategy_name} ===")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embedding_func,
    )
    await rag.initialize_storages()

    results_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in GOLDEN_DATASET:
        print(f"  [SABOT] Menanyakan: {item['question']}")
        ans, ctx = await get_rag_data(rag, item['question'])
        results_data["question"].append(item["question"])
        results_data["answer"].append(ans)
        results_data["contexts"].append(ctx)
        results_data["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(results_data)

    print(f"  [RAGAS] Menghitung skor kualitas...")

    eval_result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=ragas_judge,
        embeddings=ragas_embeddings
    )

    df = eval_result.to_pandas()
    df["strategy"] = strategy_name
    df["timestamp"] = datetime.now().isoformat()

    await rag.finalize_storages()
    return df


async def main():
    try:
        df_sep = await run_eval_for_storage("SEPARATE", "../exp_separate_storage")
        df_mer = await run_eval_for_storage("MERGED", "../exp_merged_storage")

        final_report = pd.concat([df_sep, df_mer])
        final_report.to_csv("experiment_ragas_comparison.csv", index=False)

        print("\n" + "=" * 60)
        print("EVALUASI SELESAI")
        print("=" * 60)

        summary = final_report.groupby("strategy")[["faithfulness", "answer_relevancy"]].mean()
        print(summary)

    except Exception as e:
        print(f"\n[Error Terdeteksi] Detail: {e}")


if __name__ == "__main__":
    if not os.getenv("SUMO_POD_API_KEY"):
        print("Peringatan: API Key Sumopod tidak ditemukan!")
    else:
        asyncio.run(main())