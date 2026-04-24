import os
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import TokenTracker


from core.embedding import gemini_embedding_func


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

#
# async def init_openai_lightRAG():
#     rag = LightRAG(
#         working_dir=WORKING_DIR,
#         llm_model_func=gpt_4o_mini_complete,
#         embedding_func=openai_embed,
#     )
#     await rag.initialize_storages()
#     return rag




load_dotenv()

llm_tracker = TokenTracker()

WORKING_DIR = "openai_rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)



async def openai_llm(
        prompt: str,
        system_prompt: str = None,
        token_tracker=None,
        **kwargs
) -> str:
    token_tracker = token_tracker or llm_tracker
    is_stream = kwargs.pop("stream", False)

    if "response_format" in kwargs:
        is_stream = False

    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt,
        token_tracker=token_tracker,
        api_key=os.getenv("SUMO_POD_KEY"),
        base_url=os.getenv("SUMO_POD_BASE_URL"),
        stream=is_stream,
        **kwargs
    )


async def init_openai_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=openai_llm,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
    )
    await rag.initialize_storages()
    return rag
