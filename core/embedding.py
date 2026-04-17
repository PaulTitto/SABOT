import os

import numpy as np
from lightrag.llm.jina import jina_embed
from lightrag.utils import wrap_embedding_func_with_attrs


@wrap_embedding_func_with_attrs(
    embedding_dim=2048, max_token_size=8192, model_name="jina-embeddings-v4"
)


async def jina_embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    return await jina_embed(
        texts,
        api_key=os.getenv("JINA_API_KEY"),
    )