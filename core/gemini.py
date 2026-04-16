import os
import numpy as np
from lightrag import LightRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

WORKING_DIR = "gemini_rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def gemini_llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs
):
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.5-flash-lite",
        **kwargs
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    send_dimensions=True,
    max_token_size=2048,
    model_name="models/gemini-embedding-001",
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await gemini_embed.func(
        texts,
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/gemini-embedding-001"
    )


async def init_gemini_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_llm_model_func,
        embedding_func=embedding_func,
        llm_model_name="gemini-2.5-flash-lite"
    )
    await rag.initialize_storages()
    return rag