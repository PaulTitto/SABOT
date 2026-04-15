import os

from lightrag import LightRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed


# Gemini
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


async def gemini_lightRAG(WOEKING_DIR: str):
    return LightRAG(
        working_dir=WOEKING_DIR,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embed,
        llm_model_name="gemini-2.5-flash-lite"
    )