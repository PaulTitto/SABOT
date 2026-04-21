import os
import numpy as np
from lightrag import LightRAG
from lightrag.llm.gemini import gemini_complete_if_cache
from lightrag.llm.gemini import gemini_embed as _gemini_embed_wrapped
from lightrag.utils import wrap_embedding_func_with_attrs, TokenTracker

WORKING_DIR = "gemini_rag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

llm_tracker = TokenTracker()
embed_tracker = TokenTracker()


def estimate_tokens(texts: list[str]) -> int:
    """Estimasi token Gemini: 1 token ≈ 4 karakter (dokumentasi resmi Google)"""
    total_chars = sum(len(t) for t in texts)
    return max(1, total_chars // 4)


async def gemini_llm_model_func(
        prompt,
        system_prompt=None,
        token_tracker=None,
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


@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=2048,
    model_name="gemini-embedding-001",
)
async def gemini_embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    token_count = estimate_tokens(texts)
    embed_tracker.add_usage({
        "prompt_tokens": token_count,
        "total_tokens": token_count,
    })

    return await _gemini_embed_wrapped.func(
        texts,
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-embedding-001",
        embedding_dim=1536,
        token_tracker=None,
    )


async def init_gemini_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_llm_model_func,
        embedding_func=gemini_embedding_func,
        llm_model_kwargs={"token_tracker": llm_tracker},
        enable_llm_cache=True,
        llm_model_name="gemini-2.5-flash-lite"
    )
    await rag.initialize_storages()
    return rag