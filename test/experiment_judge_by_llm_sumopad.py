import os
import json
import time
import asyncio

import numpy as np
import pandas as pd
from datetime import datetime
from lightrag import QueryParam, LightRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs
from openai import OpenAI

from core.embedding import gemini_embedding_func


def save_experiment(data):
    file_path = "../docsss/new_evaluation_results.csv"
    df = pd.DataFrame([data])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


async def judge_metrics(client, question, context, response, reference):
    """
    LLM-as-a-Judge menggunakan SumoPod: Menilai output berdasarkan standar evaluasi ketat.
    """
    prompt = f"""Evaluasi jawaban Sistem RAG berdasarkan standar kurikulum Sekolah Sabat:

Context: {context}
Question: {question}
Response: {response}
Reference: {reference}

Berikan skor (0 atau 1) untuk:
1. accuracy: Benar secara faktual sesuai Reference.
2. relevance: Menjawab inti pertanyaan tanpa bertele-tele.
3. completeness: Mencakup semua poin penting dari Reference.

HANYA JAWAB DENGAN JSON, TANPA PENJELASAN LAIN:
{{"accuracy": 1, "relevance": 1, "completeness": 1, "reason": "..."}}"""

    try:
        res = client.chat.completions.create(
            model="mimo-v2.5-pro",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3,
        )

        response_text = res.choices[0].message.content.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)

        return {
            "accuracy": result.get("accuracy", 0),
            "relevance": result.get("relevance", 0),
            "completeness": result.get("completeness", 0),
            "reason": result.get("reason", "No reason provided")
        }
    except json.JSONDecodeError as e:
        print(f"JSON decode error saat judging: {e}")
        return {
            "accuracy": 0,
            "relevance": 0,
            "completeness": 0,
            "reason": f"JSON decode error: {str(e)}"
        }
    except Exception as e:
        print(f"Error saat judging: {e}")
        return {
            "accuracy": 0,
            "relevance": 0,
            "completeness": 0,
            "reason": f"Error: {str(e)}"
        }


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.5-flash-lite",
        **kwargs
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=2048,
    model_name="gemini-embedding-001",
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await gemini_embed.func(
        texts, api_key=os.getenv("GEMINI_API_KEY"), model="models/gemini-embedding-001"
    )


async def run_evaluation():
    working_dir = "../../exp_merge_gemini_third"

    # Initialize SumoPod client untuk judging
    judge_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://ai.sumopod.com/v1"
    )

    # RAG menggunakan Gemini
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=gemini_embedding_func,
        llm_model_name="gemini-2.5-flash-lite"
    )
    await rag.initialize_storages()

    question_dataset = [
        {
            "id": "Q01",
            "question": "Apa judul pelajaran untuk hari Sabtu?",
            "reference": "Judul pelajaran untuk hari Sabtu (Hari ke-1) adalah 'Cek Realitas'."
        },
        {
            "id": "Q02",
            "question": "Mengapa 'Cek Realitas' (materi Sabtu) menjadi langkah awal yang penting sebelum kita membahas 'Kondisi Kita' (materi Minggu)?",
            "reference": "Karena kita tidak bisa bergerak ke tempat yang lebih baik sebelum melakukan kejujuran terhadap diri sendiri mengenai kondisi hubungan kita saat ini dan mendengarkan solusi yang Yesus tawarkan."
        },
    ]

    modes = ["hybrid", "local", "global", "naive", "mix"]

    for item in question_dataset:
        print(f"\n--- Menguji Pertanyaan: {item['id']} ---")
        for mode in modes:
            start_time = time.time()
            try:
                response_text = await rag.aquery(
                    item["question"],
                    param=QueryParam(
                        mode=mode,
                        user_prompt="Anda adalah asisten Sekolah Sabat. Jawab berdasarkan fakta database secara singkat."))
                retrieved_contexts = "Nodes retrieved from LightRAG"
            except Exception as e:
                print(f"Error Query {mode}: {e}")
                response_text = "ERROR"
                retrieved_contexts = "N/A"

            judge = await judge_metrics(judge_client, item["question"], retrieved_contexts, response_text,
                                        item["reference"])
            latency = time.time() - start_time

            save_experiment({
                "user_input": item["question"],
                "retrieved_contexts": retrieved_contexts,
                "response": response_text,
                "reference": item["reference"],
                "accuracy": judge["accuracy"],
                "relevance": judge["relevance"],
                "completeness": judge["completeness"],
                "reason": judge["reason"],
                "strategy": mode,
                "latency": latency,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            print(
                f"Mode: {mode:7} | Acc: {judge['accuracy']} | Rel: {judge['relevance']} | Comp: {judge['completeness']}")

            await asyncio.sleep(3)


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY belum diset di Environment!")
    elif not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY belum diset di Environment!")
    else:
        asyncio.run(run_evaluation())

    question_dataset = [
        {
            "id": "Q01",
            "question": "Apa judul pelajaran untuk hari Sabtu?",
            "reference": "Judul pelajaran untuk hari Sabtu (Hari ke-1) adalah 'Cek Realitas'."
        },
        {
            "id": "Q02",
            "question": "Mengapa 'Cek Realitas' (materi Sabtu) menjadi langkah awal yang penting sebelum kita membahas 'Kondisi Kita' (materi Minggu)?",
            "reference": "Karena kita tidak bisa bergerak ke tempat yang lebih baik sebelum melakukan kejujuran terhadap diri sendiri mengenai kondisi hubungan kita saat ini dan mendengarkan solusi yang Yesus tawarkan."
        },
        # {
        #     "id": "Q03",
        #     "question": "Dalam materi hari Minggu, Yesus memperkenalkan diri sebagai 'Saksi yang Setia dan Benar'. Apa hubungannya dengan diagnosis-Nya terhadap jemaat Laodikia?",
        #     "reference": "Sebagai Saksi yang Setia, Yesus tidak berbohong; Ia berbicara jujur bahwa kita sering merasa kaya dan tidak butuh apa-apa, padahal sebenarnya kita melarat, buta, dan telanjang."
        # },
        # {
        #     "id": "Q04",
        #     "question": "Jika seorang anggota kelas bertanya, 'Mengapa Tuhan menegur kita dengan begitu keras di Wahyu 3:19?', bagaimana jawaban Anda berdasarkan materi hari Senin?",
        #     "reference": "Tuhan menegur justru karena kasih-Nya yang sangat dalam; Ia tidak puas dengan hubungan yang setengah-setengah atau sekadar 'datang kalau butuh saja', melainkan menginginkan hubungan yang jauh lebih kuat."
        # },
        # {
        #     "id": "Q05",
        #     "question": "Apa janji indah yang terdapat dalam Wahyu 3:20 terkait hubungan pribadi dengan Yesus?",
        #     "reference": "Yesus berjanji bahwa jika kita mendengar suara-Nya dan membuka pintu, Ia akan masuk dan makan bersama-sama dengan kita, menggambarkan hubungan yang intim dan hangat."
        # },
        # {
        #     "id": "Q06",
        #     "question": "Materi hari Selasa menyebutkan 'pertempuran terbesar' bagi manusia. Apakah itu?",
        #     "reference": "Pertempuran terbesar adalah menyadari kondisi kita yang lemah dan merasa cukup dengan diri sendiri, lalu menerima teguran Yesus, bertobat, dan menerima jubah kebenaran-Nya."
        # },
        # {
        #     "id": "Q07",
        #     "question": "Bagaimana kisah Henokh (Kejadian 5:24) dan Musa (Keluaran 34:29) pada materi Selasa mendukung tema 'Kasih yang Tak Berkesudahan'?",
        #     "reference": "Kisah-kisah tersebut menunjukkan bahwa sejak dahulu kala Allah selalu rindu untuk dekat dan berinteraksi secara pribadi dengan umat manusia dalam berbagai situasi."
        # },
        # {
        #     "id": "Q08",
        #     "question": "Dalam materi hari Rabu tentang 'Tinggal' (Abide), berapa kali Yesus mengulang kata tersebut dalam Yohanes 15:1-11 dan apa maknanya?",
        #     "reference": "Kata 'tinggal' diulang sebanyak sepuluh kali, yang berarti hidup dalam hubungan yang terus-menerus terhubung dengan Yesus sebagai Pokok Anggur."
        # },
        # {
        #     "id": "Q09",
        #     "question": "Berdasarkan materi hari Rabu, apa yang membuktikan bahwa seseorang adalah murid Yesus yang benar?",
        #     "reference": "Buktinya adalah dengan 'berbuah', yang dihasilkan dari tinggal di dalam Dia, bukan karena usaha sendiri, melainkan untuk kemuliaan Allah."
        # },
        # {
        #     "id": "Q10",
        #     "question": "Bagaimana metafora 'getah' pada hari Kamis menjelaskan cara kerja Roh Kudus dalam kehidupan orang percaya?",
        #     "reference": "Getah ibarat Roh Kudus yang mengalir dari akar menuju tunas untuk menghidupkan kembali ranting yang kering sehingga pertumbuhan dapat terjadi."
        # },
        # {
        #     "id": "Q11",
        #     "question": "Sebutkan empat peran penting Roh Kudus bagi kita menurut Yohanes pasal 14, 15, dan 16 yang dibahas pada hari Kamis.",
        #     "reference": "Menjadi Penghibur, menyingkapkan Yesus, menyadarkan akan dosa, dan menuntun ke dalam seluruh kebenaran."
        # },
        # {
        #     "id": "Q12",
        #     "question": "Bagaimana kutipan Ellen G. White pada hari Jumat menjelaskan proses penyatuan ranting yang kering dengan pokok anggur?",
        #     "reference": "Ranting itu harus dicangkokkan sehingga serat demi serat dan urat demi urat melekat erat sampai kehidupan pokok anggur itu menyatu dengan ranting."
        # },
        # {
        #     "id": "Q13",
        #     "question": "Apa hubungan antara 'suam-suam kuku' (Hari Minggu) dengan 'tidak adanya buah' (Hari Rabu)?",
        #     "reference": "Kondisi suam-suam kuku adalah bentuk keapatisan rohani; jika kita hanya terlihat seperti tinggal di dalam Yesus tetapi tidak berbuah, pada akhirnya ranting itu akan mengering dan dipotong."
        # },
        # {
        #     "id": "Q14",
        #     "question": "Apa 'obat penawar' bagi kondisi Laodikia menurut rangkuman pelajaran hari Rabu?",
        #     "reference": "Obat penawarnya adalah tinggal (abide) di dalam Yesus, yang merupakan rahasia besar kehidupan yang penuh makna."
        # },
        # {
        #     "id": "Q15",
        #     "question": "Dalam materi hari Kamis, mengapa mengikuti Allah kadang terasa sebagai 'kewajiban yang melelahkan'?",
        #     "reference": "Hal itu terjadi jika agama hanya berfokus pada tindakan luar dan aturan, bukan pada hubungan yang didasari kasih timbal balik dan kebebasan memilih dari dalam hati."
        # },
        # {
        #     "id": "Q16",
        #     "question": "Menurut Yeremia 31:3 (Hari Selasa & Kamis), apa motivasi utama Allah dalam menarik manusia kepada-Nya?",
        #     "reference": "Motivasinya adalah kasih yang kekal; Ia mengasihi kita terlebih dahulu sebelum kita merespons-Nya."
        # },
        # {
        #     "id": "Q17",
        #     "question": "Bagaimana perumpamaan musim dingin pada hari Kamis menjelaskan pertumbuhan rohani?",
        #     "reference": "Sama seperti tunas yang dehidrasi di musim dingin akan tumbuh kembali saat akar menyerap air di musim semi, kita pun butuh Roh Kudus untuk menghidupkan kembali kerohanian kita."
        # },
        # {
        #     "id": "Q18",
        #     "question": "Apa yang ditawarkan Yesus sebagai pengganti kemelaratan rohani kita dalam 'pertukaran' yang disebutkan pada hari Minggu?",
        #     "reference": "Yesus menawarkan emas-Nya (iman), pakaian putih-Nya (kebenaran), dan salep mata-Nya (pemahaman rohani)."
        # },
        # {
        #     "id": "Q19",
        #     "question": "Menurut materi hari Senin, mengapa Yesus digambarkan 'mengetuk pintu' dan tidak langsung menerobos masuk?",
        #     "reference": "Karena Yesus tidak memaksa; Ia menghormati kebebasan memilih kita dan menunggu keputusan sadar kita untuk membuka hati bagi-Nya."
        # },
        # {
        #     "id": "Q20",
        #     "question": "Apa tujuan utama pemangkasan ranting oleh Pengusaha Anggur (Bapa) menurut materi hari Rabu?",
        #     "reference": "Tujuannya adalah agar ranting tersebut dapat menghasilkan lebih banyak buah dalam jangka panjang."
        # },
        # {
        #     "id": "Q21",
        #     "question": "Bagaimana hubungan antara ketaatan pada perintah Tuhan dengan kasih menurut 1 Yohanes 5:3 (Hari Rabu)?",
        #     "reference": "Menuruti perintah-perintah-Nya adalah bentuk nyata dari kasih kita kepada Allah, dan perintah-Nya itu tidaklah berat jika dilakukan atas dasar hubungan kasih."
        # },
        # {
        #     "id": "Q22",
        #     "question": "Jika seseorang merasa 'terlalu sibuk' untuk Tuhan, nasihat apa yang diberikan pada materi hari Senin?",
        #     "reference": "Yesus tidak ingin mengganggu kesibukan kita, tetapi waktu sangat singkat; jika kita mendengar-Nya mengetuk, kita harus membuat keputusan sadar untuk membuka pintu."
        # },
        # {
        #     "id": "Q23",
        #     "question": "Apa konsekuensi bagi ranting yang tidak tinggal di dalam pokok anggur menurut Yohanes 15?",
        #     "reference": "Ranting tersebut tidak dapat berbuah dari dirinya sendiri, akan menjadi kering, dipotong, dan akhirnya dibuang."
        # },
        # {
        #     "id": "Q24",
        #     "question": "Bagaimana pengaruh memandang Salib terhadap keputusan kita untuk membuka hati (Hari Senin)?",
        #     "reference": "Merenungkan makna Salib dapat menginspirasi kita untuk menyadari betapa besarnya pengorbanan Yesus sehingga kita tergerak untuk merespons kasih-Nya."
        # },
        # {
        #     "id": "Q25",
        #     "question": "Apa kesimpulan utama dari seluruh pelajaran pekan ini mengenai pertumbuhan hubungan dengan Allah?",
        #     "reference": "Pertumbuhan hanya mungkin terjadi jika kita melakukan cek realitas yang jujur, bertobat dari kondisi suam-suam kuku, dan membuat pilihan sadar setiap hari untuk tinggal di dalam Yesus serta dipenuhi oleh Roh Kudus."
        # }
    ]
