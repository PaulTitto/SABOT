import inspect
import os.path
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Query
from lightrag import QueryParam
from openai import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from core.embedding import gemini_embedding_func
from helper.date_helper import (
    extract_date_from_question,
    get_today_iso, get_lesson_id_by_date
)
import os
import json
import sqlite3
import datetime
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import TokenTracker

load_dotenv()

llm_tracker = TokenTracker()

WORKING_DIR = "/var/www/ss-chatbot/final_boss_working_dir"
DB_PATH = "/var/www/ss-chatbot/chat_logs.db"

os.makedirs(WORKING_DIR, exist_ok=True)

# ========================
# SQLITE SETUP
# ========================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT NOT NULL,
            answer      TEXT,
            lesson_id   TEXT,
            created_at  TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    conn.close()

def save_chat_log(question: str, answer: str, lesson_id: str = None):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO chat_logs (question, answer, lesson_id) VALUES (?, ?, ?)",
            (question, answer, lesson_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] {e}")

def extract_lesson_id(question: str) -> str | None:
    """Ekstrak lesson id dari pertanyaan, misal 2026-q2-w01-d1"""
    import re
    m = re.search(r'\d{4}-q\d+-w\d+-d\d+', question)
    return m.group(0) if m else None

# ========================
# LLM
# ========================

rag = None

async def deepseek_llm(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        extra_body={"thinking": {"type": "disabled"}},
        **kwargs,
    )
# async def deepseek_llm(
#         prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
# ) -> str:
#     return await openai_complete_if_cache(
#         # os.getenv("LLM_MODEL", "deepseek-chat"),
#         os.getenv("LLM_MODEL", "deepseek-v4-flash"),
#         prompt,
#         system_prompt=system_prompt,
#         history_messages=history_messages,
#         api_key=os.getenv("DEEPSEEK_API_KEY"),
#         base_url=os.getenv("DEEPSEEK_BASE_URL"),
#         **kwargs,
#     )


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
    init_db()
    print("Database initialized")

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
    allow_origins=[
        "https://ss.developedbytoo.me",
        "http://ss.developedbytoo.me",
        "https://developedbytoo.me",
        "http://developedbytoo.me"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# MODELS
# ========================

class AskRequest(BaseModel):
    question: str
    model: str = "deepseek"


# ========================
# ENDPOINTS
# ========================

SYSTEM_INSTRUCTION = (
    # "[PERINTAH UTAMA: OUTPUT FINAL HANYA JAWABAN BERSIH]\n"
    # "Kamu adalah SABOT (Asisten Pemandu Sekolah Sabat).\n"
    # "Tugasmu HANYA memberikan jawaban akhir yang bersih, ramah, dan ringkas kepada pengguna berdasarkan dokumen yang diberikan.\n\n"
    # "⚠️ PERINGATAN KERAS:\n"
    "1. JANGAN PERNAH menampilkan proses berpikir, analisis dokumen, evaluasi instruksi, atau perdebatan internalmu di dalam teks jawaban.\n"
    "2. JANGAN menulis kata-kata seperti 'Mari kita lihat dokumen', 'Berdasarkan chunk', 'Instruksi mengatakan', atau 'Jadi referensinya adalah'.\n"
    "3. Langsung berikan jawaban substantif beserta referensinya di akhir kalimat.\n"
    "4. Langsung to the point.\n"
    "5. Jangan pernah memulai jawaban dengan frasa seperti: 'Tentu', 'Berdasarkan materi yang tersedia', 'Baik', 'Tentu saja', atau kalimat pembuka serupa.\n"
    "6. Langsung jawab pertanyaannya tanpa basa-basi.\n"
    "7. Anda adalah AI Reading Assistant untuk pelajaran Sekolah Sabat. Jawaban harus natural, modern, mudah dipahami, dan tidak terlalu formal.\n"
    "8. Jangan gunakan kalimat seperti: 'Berdasarkan konteks yang tersedia'.\n"
    "9. Jangan gunakan heading seperti 'References'.\n"
    "10. Jawaban harus terasa seperti assistant modern seperti ChatGPT atau ScienceDirect AI.\n"
    "11. Jika user meminta ringkasan, buat ringkasan natural dan singkat.\n"
    "12. Setiap referensi dokumen HARUS menggunakan format kurung siku di akhir jawaban. Contoh: [2026-q2-w01-d1]. Jangan gunakan format DOC id:.\n"
    "13. Format referensi wajib mengikuti pola: YYYY-qX-wXX-dX."
)


@app.post("/ask")
async def ask(req: AskRequest):
    async def generate():
        if rag is None:
            yield "data: [ERROR] RAG system not initialized\n\n"
            return

        question = req.question.strip()
        original_question = question

        date_iso = extract_date_from_question(question)

        if date_iso:

            lesson_id = get_lesson_id_by_date(
                date_iso
            )

            if lesson_id:
                question = (
                    f"Pelajaran {lesson_id}. "
                    f"{question}"
                )

        param = QueryParam(
            mode="mix",
            stream=True,
            user_prompt=SYSTEM_INSTRUCTION
        )

        full_answer = ""
        in_think_block = False 
        try:
            resp = await rag.aquery(question, param=param)

            if inspect.isasyncgen(resp):
                text_buffer = ""
                async for chunk in resp:
                    full_answer += chunk
                    text_buffer += chunk

                    if "<think>" in text_buffer:
                        if "</think>" in text_buffer:
                            parts = text_buffer.split("</think>")
                            text_buffer = parts[1]
                            if text_buffer:
                                yield f"data: {text_buffer}\n\n"
                                text_buffer = ""
                        else:
                            continue
                    else:
                        yield f"data: {text_buffer}\n\n"
                        text_buffer = ""
            else:
                full_answer = str(resp)
                import re
                answer_clean = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()
                yield f"data: {answer_clean}\n\n"

        except Exception as e:
            yield f"data: [ERROR Server] {str(e)}\n\n"

        finally:
            import re
            final_clean_answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()

            lesson_id = extract_lesson_id(question)
            save_chat_log(
                question=original_question,
                answer=final_clean_answer,
                lesson_id=lesson_id
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rag_ready": rag is not None
    }


@app.get("/dashboard/logs")
async def get_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    lesson_id: str = Query(None)
):
    """Ambil semua chat logs untuk dashboard"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if lesson_id:
        rows = conn.execute(
            "SELECT * FROM chat_logs WHERE lesson_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (lesson_id, limit, offset)
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM chat_logs WHERE lesson_id = ?",
            (lesson_id,)
        ).fetchone()[0]
    else:
        rows = conn.execute(
            "SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM chat_logs"
        ).fetchone()[0]

    conn.close()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": [dict(r) for r in rows]
    }


@app.get("/dashboard/stats")
async def get_stats():
    """Statistik ringkas untuk dashboard"""
    conn = sqlite3.connect(DB_PATH)

    total = conn.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0]

    today = datetime.date.today().isoformat()
    today_count = conn.execute(
        "SELECT COUNT(*) FROM chat_logs WHERE created_at LIKE ?",
        (f"{today}%",)
    ).fetchone()[0]

    top_lessons = conn.execute("""
        SELECT lesson_id, COUNT(*) as count
        FROM chat_logs
        WHERE lesson_id IS NOT NULL
        GROUP BY lesson_id
        ORDER BY count DESC
        LIMIT 5
    """).fetchall()

    top_questions = conn.execute("""
        SELECT question, COUNT(*) as count
        FROM chat_logs
        GROUP BY question
        ORDER BY count DESC
        LIMIT 10
    """).fetchall()

    conn.close()

    return {
        "total_chats": total,
        "today_chats": today_count,
        "top_lessons": [{"lesson_id": r[0], "count": r[1]} for r in top_lessons],
        "top_questions": [{"question": r[0], "count": r[1]} for r in top_questions],
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