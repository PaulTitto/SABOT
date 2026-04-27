# import os
# import numpy as np
# from lightrag import LightRAG
# from lightrag.llm.gemini import gemini_complete_if_cache
# from lightrag.utils import wrap_embedding_func_with_attrs, TokenTracker
#
# from core.embedding import gemini_embedding_func
#
# WORKING_DIR = "gemini_rag_storage"
# os.makedirs(WORKING_DIR, exist_ok=True)
#
# llm_tracker = TokenTracker()
#
#
#
# async def gemini_llm_model_func(
#         prompt,
#         system_prompt:str =None,
#         token_tracker=None,
#         **kwargs
# ):
#     tracker_to_use = token_tracker or llm_tracker
#     return await gemini_complete_if_cache(
#         "gemini-2.5-flash-lite",
#         prompt=prompt,
#         system_prompt=system_prompt,
#         token_tracker=tracker_to_use,
#         api_key=os.getenv("GEMINI_API_KEY"),
#         stream=kwargs.get("stream", False),
#         **{k: v for k, v in kwargs.items() if k != "stream"}
#     )
#
#
#
#
# async def init_gemini_lightRAG():
#     rag = LightRAG(
#         working_dir=WORKING_DIR,
#         llm_model_func=gemini_llm_model_func,
#         embedding_func=gemini_embedding_func,
#         enable_llm_cache=True,
#         llm_model_kwargs={"token_tracker": llm_tracker},
#     )
#     await rag.initialize_storages()
#     return rag


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
        system_prompt: str = None,
        token_tracker: TokenTracker = None,
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

def make_llm_func_with_tracker(tracker: TokenTracker):

    async def _llm_func(prompt, system_prompt=None, token_tracker=None, **kwargs):
        return await gemini_complete_if_cache(
            "gemini-2.5-flash-lite",
            prompt=prompt,
            system_prompt=system_prompt,
            token_tracker=tracker,
            api_key=os.getenv("GEMINI_API_KEY"),
            stream=kwargs.get("stream", False),
            **{k: v for k, v in kwargs.items() if k != "stream"}
        )
    return _llm_func

async def init_gemini_lightRAG(use_batch_workaround: bool = False):

    if use_batch_workaround:
        llm_func = make_llm_func_with_tracker(llm_tracker)
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_func,
            embedding_func=gemini_embedding_func,
            enable_llm_cache=True,
        )
    else:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gemini_llm_model_func,
            embedding_func=gemini_embedding_func,
            enable_llm_cache=True,
            llm_model_kwargs={"token_tracker": llm_tracker},
        )
    await rag.initialize_storages()
    return rag