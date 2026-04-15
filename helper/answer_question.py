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
        jawaban = await rag.aquery(question, param=param)
        return jawaban if jawaban else "Maaf, saya tidak menemukan jawaban."
    except Exception as e:
        return f"Terjadi kesalahan: {e}"

