from datetime import datetime
import inspect
import time
from lightrag import QueryParam

from helper.calculate_cost import get_gemini_detailed_costs, get_openai_detailed_costs
from helper.date_helper import extract_date_from_question, get_today_iso
from helper.save_csv import save_to_csv

import time
import inspect
from datetime import datetime
from lightrag import QueryParam
from helper.date_helper import extract_date_from_question, get_today_iso
from helper.save_csv import save_to_csv
from helper.calculate_cost import get_deepseek_detailed_costs, get_gemini_detailed_costs


async def answer_question(model, rag, question, llm_tracker, embed_tracker) -> str:
    metrics = {
        "llm_p": 0, "llm_c": 0, "emb_p": 0,
        "c_llm": 0, "c_emb": 0, "total": 0
    }
    result = "Gagal mendapatkan jawaban."
    start_time = time.time()

    date_iso = extract_date_from_question(question)
    if date_iso:
        question = f"Pelajaran untuk tanggal {date_iso}. {question}"
    elif "hari ini" in question.lower():
        question = f"Pelajaran untuk tanggal {get_today_iso()}. {question}"

    param = QueryParam(
        mode="hybrid",
        top_k=5,
        stream=True,
        user_prompt="Anda adalah asisten Sekolah Sabat. Jawab berdasarkan fakta database secara singkat."
    )

    try:
        llm_tracker.reset()
        embed_tracker.reset()

        resp = await rag.aquery(question, param=param)

        if inspect.isasyncgen(resp):
            chunks = []
            async for chunk in resp:
                chunks.append(chunk)
                print(chunk, end="", flush=True)
            result = "".join(chunks)
        else:
            result = str(resp)
            print(result)

        latency = time.time() - start_time

        if "deepseek-chat" in model.lower():
            metrics = get_deepseek_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())
        if  "gemini-2.5-flash-lite" in model.lower():
            metrics = get_gemini_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())
        if  "gpt-4o-mini" in model.lower():
            metrics = get_openai_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())

        save_to_csv({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "mode": "hybrid",
            "question": question,
            "answer": result[:150],
            "latency": latency,
            "llm_p_tokens": metrics["llm_p"],
            "llm_c_tokens": metrics["llm_c"],
            "embed_tokens": metrics["emb_p"],
            "cost_llm": metrics["c_llm"],
            "cost_embed": metrics["c_emb"],
            "total_cost": metrics["total"],
            "call_count": llm_tracker.get_usage().get("call_count", 0)
        })
        return result

    except Exception as e:
        print(f"\n--- ERROR DETECTED ---")
        print(f"Detail: {e}")

        save_to_csv({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "mode": "ERROR",
            "question": question,
            "answer": f"ERROR: {str(e)[:100]}",
            "latency": time.time() - start_time,
            "llm_p_tokens": 0, "llm_c_tokens": 0, "embed_tokens": 0,
            "cost_llm": 0, "cost_embed": 0, "total_cost": 0, "call_count": 0
        })
        return f"Terjadi kesalahan: {e}"