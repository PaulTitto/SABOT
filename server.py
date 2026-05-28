import inspect
import os.path
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from lightrag import QueryParam
from openai import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles
from helper.date_helper import (
    extract_date_from_question,
    get_today_iso
)
import os
import json
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import TokenTracker

from core.embedding import gemini_embedding_func

load_dotenv()

llm_tracker = TokenTracker()

WORKING_DIR = "../final_working_dir_second"
WORKING_DIR = "/www/wwwroot/satulima.web.id/final_working_dir_second"
os.makedirs(WORKING_DIR, exist_ok=True)



load_dotenv()

rag = None

async def deepseek_llm(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        # os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        **kwargs,
    )

async def init_deepseek_lightRAG():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=deepseek_llm,
        embedding_func=gemini_embedding_func,
        enable_llm_cache=True,
        llm_model_kwargs={"token_tracker": llm_tracker},
    )
    await rag.initialize_storages()
    return rag


@asynccontextmanager
async def lifespan(ap: FastAPI):

    global rag

    print("Starting Initialize LightRAG")

    try:
        rag = await init_deepseek_lightRAG()

        print("LightRAG Initialized")

    except Exception as e:
        print("LightRAG failed:", e)

    yield

    if rag:
        await rag.finalize_storages()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    model: str = "deepseek"


@app.post("/ask")
async def ask(req: AskRequest):

    async def generate():

        if rag is None:
            yield "[ERROR] RAG system not initialized"
            return

        question = req.question.strip()

        date_iso = extract_date_from_question(
            question
        )

        if date_iso:
            question = (
                f"Pelajaran untuk tanggal "
                f"{date_iso}. {question}"
            )

        elif "hari ini" in question.lower():
            question = (
                f"Pelajaran untuk tanggal "
                f"{get_today_iso()}. {question}"
            )

        param = QueryParam(
            mode="mix",
            stream=True,
            user_prompt=(
                "Langsung to the point"
                "Anda adalah AI Reading Assistant "
                "untuk pelajaran Sekolah Sabat. "

                "Jawaban harus natural, modern, "
                "mudah dipahami, dan tidak terlalu formal. "

                "Jangan gunakan kalimat seperti: "
                "'Berdasarkan konteks yang tersedia'. "

                "Jangan gunakan heading seperti "
                "'References'. "

                "Jawaban harus terasa seperti "
                "assistant modern seperti ChatGPT "
                "atau ScienceDirect AI. "

                "Jika user meminta ringkasan, "
                "buat ringkasan natural dan singkat. "

                "Setiap referensi dokumen HARUS "
                "menggunakan format kurung siku. "

                "Contoh: [2026-q2-w01-d1] "

                "Jangan gunakan format DOC id:. "

                "Format referensi wajib mengikuti pola: "
                "YYYY-qX-wXX-dX."
            )
        # user_prompt = (
        #     "Anda adalah pakar studi Sekolah Sabat. "
        #     "Berikan jawaban yang mendalam namun ringkas. "
        #     "Setiap poin informasi yang Anda berikan harus diikuti dengan DOC id yang relevan di dalam kurung. "
        #     "Contoh: 'Tujuan pelajaran minggu ini adalah X (DOC id: 2026-q2-w01-d1)'. "
        #     "menggunakan kurung siku, contoh: [2026-q2-w01-d1]."
        #     "Pastikan format referensi selalu mengikuti pola: YYYY-qX-wXX-dX."
        # )
        )

        # llm_tracker.reset()
        # embed_tracker.reset()

        try:

            resp = await rag.aquery(
                question,
                param=param
            )

            if inspect.isasyncgen(resp):

                async for chunk in resp:
                    yield chunk

            else:
                yield str(resp)

        except Exception as e:

            yield (
                "[ERROR Server] "
                f"{str(e)}"
            )

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8"
    )


@app.get("/health")
async def health_check():

    return {
        "status": "ok",
        "rag_ready": rag is not None
    }


if os.path.exists("static"):

    app.mount(
        "/",
        StaticFiles(
            directory="static",
            html=True
        ),
        name="static"
    )


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )