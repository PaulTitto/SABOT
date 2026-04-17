import os

from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache

from core.embedding import jina_embedding_func
from core.gemini import gemini_embedding_func

load_dotenv()


WORKING_DIR = "deep_seek_rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def deepseek_llm(prompt: str, system_prompt: str =None,history_messages=[], keyword_extraction=False, **kwargs)-> str:
    return await openai_complete_if_cache(
        "deepseek-chat",
        prompt,
        system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DEEP_SEEK_API_KEY"),
        base_url=os.getenv("DEEP_SEEK_BASE_URL"),
        **kwargs
    )


async def init_deepseek_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=deepseek_llm,
        embedding_func=jina_embedding_func,
        embedding_batch_num=64,
        embedding_func_max_async=1
        # embedding_func=gemini_embedding_func,
    )
    await rag.initialize_storages()
    return rag