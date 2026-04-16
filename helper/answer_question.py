import inspect

from lightrag import QueryParam

from helper.date_helper import extract_date_from_question, get_today_iso



async def answer_question(rag, question: str) -> str:
    date_iso = extract_date_from_question(question)

    if date_iso:
        question = f"Pelajaran untuk tanggal {date_iso}. {question}"
    elif "hari ini" in question.lower():
        question = f"Pelajaran untuk tanggal {get_today_iso()}. {question}"

    param = QueryParam(
        mode="mix",
        stream=True,
        response_type="Multiple Paragraphs",
        user_prompt="Anda adalah asisten pelajaran Sekolah Sabat. Jawab berdasarkan fakta dari database."
    )
    try:
        resp = await rag.aquery(question, param=param)
        if inspect.isasyncgen(resp):
            chunks = []
            async for chunk in resp:
                chunks.append(chunk)
                print(chunk, end="", flush=True)
            return "".join(chunks)
        else:
            return str(resp)
    except Exception as e:
        return f"Terjadi kesalahan: {e}"