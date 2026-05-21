import time
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import TokenTracker

from core.embedding import gemini_embedding_func, embed_tracker
from core.gemini import gemini_llm_model_func
from helper.calculate_cost import get_cost_by_model

# working_dir = "../exp_merge_gemini_third"
working_dir = "../final_working_dir_second"
llm_tracker = TokenTracker()


async def test_query_merge():
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
    )

    await rag.initialize_storages()

    llm_tracker.reset()
    embed_tracker.reset()

    query_text = "Judul Pelajaran Harian apa aja yang ada di minggu pertama"
    start = time.time()

    response = await rag.aquery(
        query_text,
        param=QueryParam(
            mode="global",
            user_prompt="Anda adalah asisten Sekolah Sabat. Jawab berdasarkan fakta database secara singkat."
        ),
    )

    latency = time.time() - start


    print("-" * 50)
    print(f"Pertanyaan: {query_text}")
    print(f"Jawaban   : {response}")
    print(f"Time   : {latency}")



if __name__ == "__main__":
    asyncio.run(test_query_merge())