import asyncio
import os
import time
from datetime import datetime

import instructor
import nest_asyncio
import pandas as pd
from openai import AsyncOpenAI
from google import genai

from ragas.llms import InstructorLLM
from ragas.embeddings import GoogleEmbeddings
from ragas.llms.base import InstructorModelArgs
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from instructor import Mode
from lightrag import LightRAG, QueryParam
from core.embedding import gemini_embedding_func
from core.gemini import gemini_llm_model_func

nest_asyncio.apply()

# =====================================================================
# INISIALISASI JUDGE & EMBEDDINGS (Ragas v0.4.3 Native Compliant)
# =====================================================================

instructor_client = instructor.from_openai(
    AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://ai.sumopod.com/v1"
    ),
    mode=Mode.JSON
)
ragas_judge = InstructorLLM(
    client=instructor_client,
    model="deepseek-v4-pro",
    provider="openai",
    model_args=InstructorModelArgs(
        max_tokens=8192,
        temperature=0.1,
    )
)

google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
ragas_embeddings = GoogleEmbeddings(
    model="gemini-embedding-001",
    client=google_client
)

question_dataset = [
    {
      "id": "Q01",
      "type": "faktual",
      "source_doc": "week-01.txt — Sabtu",
      "question": "Apa judul pelajaran hari Sabtu pada minggu pertama dan kutipan lengkap ayat hafalan berdasarkan teks tersebut?",
      "reference": "Judul pelajaran hari Sabtu adalah Cek Realitas. Ayat hafalan minggu pertama diambil dari Yohanes 15:9 mengenai perintah untuk tinggal di dalam kasih Yesus sebagaimana Bapa mengasihi Yesus."
    },
    {
      "id": "Q02",
      "type": "faktual",
      "source_doc": "week-01.txt — Minggu",
      "question": "Berdasarkan Wahyu 3:14, sebutkan gelar yang digunakan Yesus untuk memperkenalkan diri-Nya kepada jemaat akhir zaman dan apa karakteristik dari gelar tersebut?",
      "reference": "Yesus memperkenalkan diri sebagai Saksi yang setia dan benar, permulaan dari ciptaan Allah kepada jemaat akhir zaman. Karakteristik dari gelar Saksi yang setia dan benar adalah Yesus tidak berbohong, melainkan Yesus berbicara dengan jelas dan jujur kepada umat Yesus."
    },
    {
      "id": "Q03",
      "type": "faktual",
      "source_doc": "week-01.txt — Rabu",
      "question": "Berapa kali kata 'tinggal' (abide) diulang dalam Yohanes 15:1-11 dan apa definisinya menurut teks?",
      "reference": "Kata tinggal diulang sebanyak sepuluh kali dalam Yohanes 15:1-11. Menurut teks, tinggal di dalam Yesus berarti hidup dalam hubungan yang terus terhubung dengan Yesus."
    },
    {
      "id": "Q04",
      "type": "faktual",
      "source_doc": "week-01.txt — Kamis",
      "question": "Sebutkan empat peran penting Roh Kudus bagi manusia di bumi yang tertera pada bagian daftar hari Kamis beserta rujukan ayat Yohanes-nya!",
      "reference": "Empat peran Roh Kudus berdasarkan teks hari Kamis adalah Roh Kudus menjadi Penghibur manusia berdasarkan Yohanes 14:16-18, Roh Kudus menyingkapkan Yesus kepada manusia berdasarkan Yohanes 15:26, Roh Kudus menyadarkan manusia akan dosa berdasarkan Yohanes 16:7-8, dan Roh Kudus menuntun manusia ke dalam seluruh kebenaran berdasarkan Yohanes 16:13."
    },
    {
      "id": "Q05",
      "type": "faktual",
      "source_doc": "week-02.txt — Sabtu",
      "question": "Tuliskan bunyi lengkap ayat hafalan untuk minggu kedua yang diambil dari Yohanes 17:3!",
      "reference": "Ayat hafalan minggu kedua berdasarkan Yohanes 17:3 menyatakan bahwa hidup yang kekal adalah ketika manusia mengenal Allah sebagai satu-satunya Allah yang benar, dan mengenal Yesus Kristus yang telah Allah utus."
    }
]

# =====================================================================
# RAG CORE INTERACTION
# =====================================================================

async def get_rag_data(rag, question, mode_name):
    answer  = await rag.aquery(question, param=QueryParam(mode=mode_name))
    context = await rag.aquery(question, param=QueryParam(mode=mode_name, only_need_context=True))
    return str(answer), [str(context)]

# =====================================================================
# ASYNCHRONOUS HIGH-FIDELITY SCORING PIPELINE (4 Metrics + Ragas Score)
# =====================================================================

async def score_sample(question, answer, contexts, reference):
    m1 = Faithfulness(llm=ragas_judge)
    m2 = AnswerRelevancy(llm=ragas_judge, embeddings=ragas_embeddings)
    m3 = ContextRecall(llm=ragas_judge)
    m4 = ContextPrecision(llm=ragas_judge)

    # 1. Menghitung Metrik Faithfulness
    try:
        faith_result = await m1.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        faith_score = faith_result.value
    except Exception as e:
        print(f"    [WARN] Faithfulness error: {e}")
        faith_score = None

    # 2. Menghitung Metrik Answer Relevancy
    try:
        relevancy_result = await m2.ascore(
            user_input=question,
            response=answer,
        )
        relevancy_score = relevancy_result.value
    except Exception as e:
        print(f"    [WARN] AnswerRelevancy error: {e}")
        relevancy_score = None

    # 3. Menghitung Metrik Context Recall
    try:
        recall_result = await m3.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
        recall_score = recall_result.value
    except Exception as e:
        print(f"    [WARN] ContextRecall error: {e}")
        recall_score = None

    # 4. Menghitung Metrik Context Precision
    try:
        precision_result = await m4.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
        precision_score = precision_result.value
    except Exception as e:
        print(f"    [WARN] ContextPrecision error: {e}")
        precision_score = None

    # 5. Kalkulasi Rata-rata RAGAS Score
    valid_scores = [s for s in [faith_score, relevancy_score, recall_score, precision_score] if s is not None]
    ragas_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return {
        "faithfulness": faith_score,
        "answer_relevancy": relevancy_score,
        "context_recall": recall_score,
        "context_precision": precision_score,
        "ragas_score": ragas_score
    }

# =====================================================================
# COMPREHENSIVE EXPERIMENT RUNNER
# =====================================================================

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

        meta_rows = []

        for item in question_dataset:
            print(f"  [Q] {item['id']} ...")
            ans, ctx = await get_rag_data(rag, item["question"], current_mode)

            print(f"  [RAGAS] Scoring {item['id']}...")
            scores = await score_sample(item["question"], ans, ctx, item["reference"])

            meta_rows.append({
                "id":                 item["id"],
                "type":               item["type"],
                "user_input":         item["question"],
                "response":           ans,
                "retrieved_contexts": ctx,
                "reference":          item["reference"],
                "faithfulness":       scores["faithfulness"],
                "answer_relevancy":   scores["answer_relevancy"],
                "context_recall":     scores["context_recall"],
                "context_precision":  scores["context_precision"],
                "ragas_score":        scores["ragas_score"],
                "strategy":           strategy_name,
                "llm":                target_llm_name,
                "mode":               current_mode,
                "timestamp":          datetime.now().isoformat(),
            })

        df = pd.DataFrame(meta_rows)

        desired_columns = [
            "id", "type", "user_input", "retrieved_contexts", "response", "reference",
            "faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score",
            "strategy", "llm", "mode", "timestamp"
        ]
        df = df.reindex(columns=desired_columns)
        all_mode_dfs.append(df)

        await rag.finalize_storages()
        await asyncio.sleep(2)
    return pd.concat(all_mode_dfs, ignore_index=True)

# =====================================================================
# MAIN EXECUTOR MATRIX
# =====================================================================

async def main():
    start_time = time.time()
    try:
        from core.deepseek import deepseek_llm
        from core.openai import openai_llm

        llm_targets = [
            {"name": "Gemini-2.5-Flash-Lite", "func": gemini_llm_model_func},
            {"name": "DeepSeek-V4-Flash",      "func": deepseek_llm},
            {"name": "GPT-4.1-Mini",            "func": openai_llm},
        ]

        reports = []
        target_storage = {"name": "BATCH", "path": "../final_working_dir"}

        for target in llm_targets:
            df_result = await run_eval_comprehensive(
                strategy_name=target_storage["name"],
                working_dir=target_storage["path"],
                target_llm_name=target["name"],
                target_llm_func=target["func"],
            )
            reports.append(df_result)

        final_report = pd.concat(reports, ignore_index=True)
        final_report.to_csv("experiment_ragas_new_fixed.csv", index=False)

        print("\n" + "=" * 70)
        print(f"EVALUASI SELESAI | Waktu Total: {time.time() - start_time:.2f} detik")
        print("=" * 70)

        summary = final_report.groupby(["strategy", "llm", "mode"])[
            ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]
        ].mean()
        print(summary)

    except Exception as e:
        import traceback
        print(f"\n[CRITICAL ERROR] Kegagalan Eksekusi Eksperimen: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY dulu!")
    elif not os.getenv("GEMINI_API_KEY"):
        print("Error: Set GEMINI_API_KEY dulu!")
    else:
        asyncio.run(main())