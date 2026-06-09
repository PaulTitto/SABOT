import os
import numpy as np
from lightrag.llm.openai import openai_embed
from lightrag.utils import wrap_embedding_func_with_attrs, TokenTracker

embed_tracker = TokenTracker()

def estimate_tokens(texts: list[str]) -> int:
    total_chars = sum(len(t) for t in texts)
    return max(1, total_chars // 4)

@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=2048,
    model_name="gemini-embedding-001"
)
async def gemini_embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    token_count = estimate_tokens(texts)
    embed_tracker.add_usage({
        "prompt_tokens": token_count,
        "total_tokens": token_count,
    })
    return await openai_embed(
        texts,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model="gemini/gemini-embedding-001",
        embedding_dim=1536,
    )