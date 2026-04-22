import os
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import TokenTracker
from openai import AsyncOpenAI

from core.embedding import gemini_embedding_func

load_dotenv()

llm_tracker = TokenTracker()

WORKING_DIR = "deep_seek_rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# async def deepseek_llm(
#         prompt: str,
#         system_prompt: str = None,
#         token_tracker=None,
#         **kwargs
# ) -> str:
#
#     token_tracker = token_tracker or llm_tracker
#
#     kwargs.pop("stream", None)
#     kwargs.pop("hashing_kv", None)
#     kwargs.pop("keyword_extraction", None)
#     kwargs.pop("history_messages", None)
#     kwargs.pop("enable_cot", None)
#     has_response_format = "response_format" in kwargs
#
#     client = AsyncOpenAI(
#         api_key=os.getenv("DEEP_SEEK_API_KEY"),
#         base_url=os.getenv("DEEP_SEEK_BASE_URL"),
#     )
#
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.append({"role": "user", "content": prompt})
#
#     try:
#         if has_response_format:
#             response = await client.beta.chat.completions.parse(
#                 model="deepseek-chat",
#                 messages=messages,
#                 **kwargs
#             )
#             result = response.choices[0].message.parsed.model_dump_json()
#         else:
#             response = await client.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=messages,
#                 stream=False,
#                 **kwargs
#             )
#             result = response.choices[0].message.content or ""
#
#         if token_tracker and hasattr(response, "usage"):
#             token_tracker.add_usage({
#                 "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
#                 "completion_tokens": getattr(response.usage, "completion_tokens", 0),
#                 "total_tokens": getattr(response.usage, "total_tokens", 0),
#             })
#
#         return result
#
#     finally:
#         await client.close()


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


async def deepseek_llm(
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
        "deepseek-chat",
        prompt,
        system_prompt,
        token_tracker=token_tracker,
        api_key=os.getenv("DEEP_SEEK_API_KEY"),
        base_url=os.getenv("DEEP_SEEK_BASE_URL"),
        stream=is_stream,
        **kwargs
    )