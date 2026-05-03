def calculate_detailed_costs(llm_usage, embed_usage, price_in, price_out, price_embed):
    l_in = llm_usage.get("prompt_tokens", 0)
    l_out = llm_usage.get("completion_tokens", 0)
    e_in = embed_usage.get("prompt_tokens", 0)

    cost_llm = (l_in * (price_in / 1_000_000)) + (l_out * (price_out / 1_000_000))
    cost_embed = (e_in * (price_embed / 1_000_000))
    total = cost_llm + cost_embed

    return {
        "llm_p": l_in,
        "llm_c": l_out,
        "emb_p": e_in,
        "c_llm": cost_llm,
        "c_emb": cost_embed,
        "total": total
    }

def get_gemini_detailed_costs(llm_usage, embed_usage):
    return calculate_detailed_costs(llm_usage, embed_usage, 0.10, 0.40, 0.15)


def get_deepseek_detailed_costs(llm_usage, embed_usage):
    return calculate_detailed_costs(llm_usage, embed_usage, 0.28, 0.42, 0.15)


def get_openai_detailed_costs(llm_usage, embed_usage):
    return calculate_detailed_costs(llm_usage, embed_usage, 0.15, 0.60, 0.15)

def get_cost_by_model(model: str, llm_usage: dict, embed_usage: dict) -> dict:
    if "gemini-2.5-flash-lite" in model.lower():
        return get_gemini_detailed_costs(llm_usage, embed_usage)
    elif "deepseek-v4-flash" in model.lower():
        return get_deepseek_detailed_costs(llm_usage, embed_usage)
    elif "gpt-4.1-mini" in model.lower():
        return get_openai_detailed_costs(llm_usage, embed_usage)
    return get_gemini_detailed_costs(llm_usage, embed_usage)