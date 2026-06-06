import os
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache  # ← ganti import
from lightrag.utils import TokenTracker

from core.embedding import gemini_embedding_func, openai_embedding_func

WORKING_DIR = "gemini_rag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

llm_tracker = TokenTracker()


async def gemini_llm_model_func(
    prompt,
    system_prompt: str = None,
    history_messages=[],
    keyword_extraction=False,
    token_tracker: TokenTracker = None,
    **kwargs
):
    tracker_to_use = token_tracker or llm_tracker
    return await openai_complete_if_cache(
        "gemini/gemini-2.5-flash-lite",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        token_tracker=tracker_to_use,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        **kwargs
    )


async def init_gemini_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_llm_model_func,
        embedding_func=openai_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
    )
    await rag.initialize_storages()
    return rag