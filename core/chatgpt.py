import os

from lightrag.llm.openai import openai_complete_if_cache


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
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        api_key=os.getenv("OPENAI_API_KEY"),
        keyword_extraction=keyword_extraction,
        **kwargs,
    )
