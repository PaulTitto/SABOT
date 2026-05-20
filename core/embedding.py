import os

from lightrag.llm.jina import jina_embed

import numpy as np
from lightrag.utils import wrap_embedding_func_with_attrs, TokenTracker
from lightrag.llm.gemini import gemini_embed as _gemini_embed_wrapped


@wrap_embedding_func_with_attrs(
    embedding_dim=2048, max_token_size=8192, model_name="jina-embeddings-v4"
)


async def jina_embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    return await jina_embed(
        texts,
        api_key=os.getenv("JINA_API_KEY"),
    )



embed_tracker = TokenTracker()



def estimate_tokens(texts: list[str]) -> int:
    """Estimasi token Gemini: 1 token ≈ 4 karakter (dokumentasi resmi Google)"""
    total_chars = sum(len(t) for t in texts)
    return max(1, total_chars // 4)


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

