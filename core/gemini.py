import os
import numpy as np
from lightrag import LightRAG
from lightrag.llm.gemini import gemini_complete_if_cache
from lightrag.utils import wrap_embedding_func_with_attrs, TokenTracker

from core.embedding import gemini_embedding_func

WORKING_DIR = "gemini_rag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

llm_tracker = TokenTracker()



async def gemini_llm_model_func(
        prompt,
        system_prompt:str =None,
        token_tracker=None,
        **kwargs
):
    tracker_to_use = token_tracker or llm_tracker
    return await gemini_complete_if_cache(
        "gemini-2.5-flash-lite",
        prompt=prompt,
        system_prompt=system_prompt,
        token_tracker=tracker_to_use,
        api_key=os.getenv("GEMINI_API_KEY"),
        stream=kwargs.get("stream", False),
        **{k: v for k, v in kwargs.items() if k != "stream"}
    )




async def init_gemini_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
    )
    await rag.initialize_storages()
    return rag