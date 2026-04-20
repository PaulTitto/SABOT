import os

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

WORKING_DIR = "openai_rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
async def gpt_4o_mini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    return await openai_complete_if_cache(
        # "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        api_key=os.getenv("OPENAI_API_KEY"),
        keyword_extraction=keyword_extraction,
        **kwargs,
    )




async def gpt_41_mini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    return await openai_complete_if_cache(
        "gpt-4.1-mini-2025-04-14",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        api_key=os.getenv("OPENAI_API_KEY"),
        keyword_extraction=keyword_extraction,
        **kwargs,
    )

async def init_openai_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()
    return rag
