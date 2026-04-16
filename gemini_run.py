import os


from core.gemini import init_gemini_lightRAG, WORKING_DIR
from docs.configure_logging import configure_logging
from helper.answer_question import answer_question
from helper.dataset_helper import insert_document, DATA_ROOT


async def run_gemini():
    configure_logging()
    print("Starting...")

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set")
        return
    rag = await init_gemini_lightRAG()
    if not os.path.exists(os.path.join(WORKING_DIR)):
        print("Dataset is empty")
        await insert_document(rag, DATA_ROOT)

    question = "Ayat hafalan minggu ini apa"
    await answer_question(rag, question)
    await  rag.finalize_storages()

    print("Finished")