import asyncio
import os


from core.gemini import init_gemini_lightRAG, WORKING_DIR
from docs.configure_logging import configure_logging
from helper.answer_question import answer_question
from helper.dataset_helper import DATA_ROOT, insert_documents_folder


async def run_gemini():
    configure_logging()
    print("Starting...")
    rag = await init_gemini_lightRAG()
    await insert_documents_folder(rag, DATA_ROOT)

    question = "Ayat hafalan minggu ini apa"
    await answer_question(rag, question)
    await  rag.finalize_storages()

    print("Finished")


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("API Key Gemini tidak ditemukan! Setel di environment variable.")
    else:
        asyncio.run(run_gemini())