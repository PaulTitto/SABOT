import inspect
import os.path
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from lightrag import QueryParam
from openai import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from core.embedding import embed_tracker
from core.gemini import init_gemini_lightRAG, llm_tracker
from helper.date_helper import extract_date_from_question, get_today_iso

load_dotenv()
rag = None

@asynccontextmanager
async def lifespan(ap: FastAPI):
    global rag
    print("Starting Initialize LightRAG")
    try:
        rag = await  init_gemini_lightRAG()
        print("Initializing LightRAG")
    except:
        print("Initializing LightRAG failed")

    yield

    if rag:
        await rag.finalize_storages()
        print("LightRAG successfully initialized")


app = FastAPI(lifespan=lifespan)
# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    model: str = "gemini-2.5-flash-lite"


@app.post("/ask")
async def ask(req: AskRequest):
    "Endpoint for ask request with stream"

    async def generate():
        if rag is None:
            yield "[ERROR] RAG system not initialized, you can refreshed"
            return

        question = req.question

        date_iso = extract_date_from_question(question)
        if date_iso:
            question = f"Pelajaran untuk tanggal {date_iso}. {question}"
        elif "hari ini" in question.lower():
            question = f"Pelajaran untuk tanggal {get_today_iso()}. {question}"

        param = QueryParam(
            mode="hybrid",
            top_k=5,
            stream=True,
            user_prompt=(
                "Anda adalah pakar studi Sekolah Sabat. "
                "Berikan jawaban yang mendalam namun ringkas. "
                "Setiap poin informasi yang Anda berikan harus diikuti dengan DOC id yang relevan di dalam kurung. "
                "Contoh: 'Tujuan pelajaran minggu ini adalah X (DOC id: 2026-q2-w01-d1)'. "
                "menggunakan kurung siku, contoh: [2026-q2-w01-d1]."
                "Pastikan format referensi selalu mengikuti pola: YYYY-qX-wXX-dX."
            )
        )
        llm_tracker.reset()
        embed_tracker.reset()

        try:
            resp = await rag.aquery(question, param=param)
            if inspect.isasyncgen(resp):
                async for chunk in resp:
                    yield chunk
            else:
                yield str(resp)
        except Exception as e:
            yield f"[ERROR Server] Terjadi kesalahan: {str(e)}"
    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_ready": rag is not None}
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    print("Error: Folder Static not Found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)