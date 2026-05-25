import os
import json
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import TokenTracker

from core.embedding import gemini_embedding_func

load_dotenv()

llm_tracker = TokenTracker()

WORKING_DIR = "../final_working_dir_second"
os.makedirs(WORKING_DIR, exist_ok=True)




async def deepseek_llm(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        **kwargs,
    )

async def init_deepseek_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=deepseek_llm,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
    )
    await rag.initialize_storages()
    return rag