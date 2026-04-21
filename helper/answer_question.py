from datetime import datetime
import inspect
import time
from lightrag import QueryParam

# Import tracker dari core
from core.gemini import llm_tracker, embed_tracker
from helper.calculate_cost import get_gemini_detailed_costs
from helper.date_helper import extract_date_from_question, get_today_iso
from helper.save_csv import save_to_csv


async def answer_question(model: str, rag, question: str) -> str:
    date_iso = extract_date_from_question(question)
    if date_iso:
        question = f"Pelajaran untuk tanggal {date_iso}. {question}"
    elif "hari ini" in question.lower():
        question = f"Pelajaran untuk tanggal {get_today_iso()}. {question}"

    MODE = "hybrid"
    param = QueryParam(
        mode=MODE,
        top_k=5,
        stream=True,
        user_prompt="Anda adalah asisten Sekolah Sabat. Jawab berdasarkan fakta database secara singkat dan to-the-point dan jangan menyebutkan referensinya. Jika tidak ada di database, katakan tidak tahu."
    )

    try:
        llm_tracker.reset()
        embed_tracker.reset()

        start_time = time.time()

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
        if model == "gemini-2.5-flash-lite":
            metrics = get_gemini_detailed_costs(llm_tracker.get_usage(), embed_tracker.get_usage())

        data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "mode": MODE,
            "question": question,
            "answer": result[:200] + "...",
            "latency": latency,
            "llm_p_tokens": metrics["llm_p"],
            "llm_c_tokens": metrics["llm_c"],
            "embed_tokens": metrics["emb_p"],
            "cost_llm": metrics["c_llm"],
            "cost_embed": metrics["c_emb"],
            "total_cost": metrics["total"],
            "call_count": llm_tracker.get_usage().get("call_count", 0)
        }

        save_to_csv(data)

        print("\n" + "=" * 30)
        print("--- QUERY METRICS ---")
        print(f"Latency   : {latency:.2f}s")
        print(f"LLM Cost  : ${metrics['c_llm']:.6f} (P:{metrics['llm_p']} | C:{metrics['llm_c']})")
        print(f"Embed Cost: ${metrics['c_emb']:.6f} (Tokens: {metrics['emb_p']})")
        print(f"Total Cost: ${metrics['total']:.6f}")
        print("=" * 30 + "\n")

        return result

    except Exception as e:
        error_msg = f"Terjadi kesalahan pada answer_question: {e}"
        print(error_msg)
        return error_msg