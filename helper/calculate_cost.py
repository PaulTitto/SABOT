def get_gemini_detailed_costs(llm_usage, embed_usage):
    PRICE_LLM_IN = 0.10
    PRICE_LLM_OUT = 0.40
    PRICE_EMBED = 0.15

    l_in = llm_usage.get("prompt_tokens", 0)
    l_out = llm_usage.get("completion_tokens", 0)
    e_in = embed_usage.get("prompt_tokens", 0)

    cost_llm = (l_in * (PRICE_LLM_IN / 1_000_000)) + (l_out * (PRICE_LLM_OUT / 1_000_000))
    cost_embed = (e_in * (PRICE_EMBED / 1_000_000))

    return {
        "llm_p": l_in,
        "llm_c": l_out,
        "emb_p": e_in,
        "c_llm": cost_llm,
        "c_emb": cost_embed,
        "total": cost_llm + cost_embed
    }


def get_deepseek_detailed_costs(llm_usage, embed_usage):
    PRICE_LLM_IN = 0.28
    PRICE_LLM_OUT = 0.42
    PRICE_EMBED = 0.15

    l_in = llm_usage.get("prompt_tokens", 0)
    l_out = llm_usage.get("completion_tokens", 0)
    e_in = embed_usage.get("prompt_tokens", 0)
    cost_llm = (l_in * (PRICE_LLM_IN / 1_000_000)) + (l_out * (PRICE_LLM_OUT / 1_000_000))
    cost_embed = (e_in * (PRICE_EMBED / 1_000_000))

    return {
        "llm_p": l_in,
        "llm_c": l_out,
        "emb_p": e_in,
        "c_llm": cost_llm,
        "c_emb": cost_embed,
        "total": cost_llm + cost_embed
    }