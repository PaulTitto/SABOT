import os

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache

load_dotenv()

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