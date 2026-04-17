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
    embedding_dim=1536,
    max_token_size=2048,
    model_name="gemini-embedding-001",
)
async def gemini_embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    return await gemini_embed(
        texts,
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-embedding-001",
        embedding_dim=1536
    )

async def init_gemini_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embedding_func,
        llm_model_name="gemini-2.5-flash-lite"
    )
    await rag.initialize_storages()
    return rag
