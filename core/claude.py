import os
from typing import AsyncIterator, Any, Union

from lightrag import LightRAG
from lightrag.llm.anthropic import anthropic_embed, anthropic_complete_if_cache

WORKING_DIR = "claude_rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
async def claude_4n5_haiku_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    return await anthropic_complete_if_cache(
        "claude-haiku-4-5-20251001",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )



async def init_claude_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=claude_4n5_haiku_complete,
        embedding_func=anthropic_embed,
    )
    await rag.initialize_storages()
    return rag
