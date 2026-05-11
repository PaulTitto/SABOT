import os
import json
import time
import asyncio

import numpy as np
import pandas as pd
from datetime import datetime
import google.genai as genai
from lightrag import QueryParam, LightRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

from core.embedding import gemini_embedding_func

MODEL_JUDGE = "gemini-2.5-flash"


def save_experiment(data):
    file_path = "11-new_evaluation_results.csv"
    df = pd.DataFrame([data])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


async def judge_metrics(client, question, context, response, reference):
    """
    LLM-as-a-Judge: Menilai output berdasarkan standar evaluasi ketat.
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
        res = client.models.generate_content(
            model=MODEL_JUDGE,
            contents=prompt,
            config=genai.types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(res.text)
    except Exception as e:
        print(f"Error saat judging: {e}")
        return {"accuracy": 0, "relevance": 0, "completeness": 0, "reason": str(e)}


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.5-flash-lite",
        **kwargs
    )




async def run_evaluation():
    working_dir = "../exp_merge_gemini_third"
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=gemini_embedding_func,
        llm_model_name="gemini-2.5-flash-lite"
    )
    await rag.initialize_storages()

    modes = ["hybrid", "local", "global", "naive", "mix"]
    question_dataset = [

        # ══════════════════════════════════════════════════════
        # TIPE 1: FAKTUAL (Q01–Q09)
        # Merujuk fakta spesifik dari dokumen harian tertentu
        # ══════════════════════════════════════════════════════

        # {
        #     "id": "Q01",
        #     "type": "faktual",
        #     "source_doc": "01.txt — Sabtu",
        #     "question": "Apa judul pelajaran hari Sabtu dan apa ayat hafalan minggu ini?",
        #     "reference": (
        #         "Judul pelajaran hari Sabtu adalah 'Cek Realitas'. "
        #         "Ayat hafalan minggu ini adalah Yohanes 15:9: "
        #         "'Seperti Bapa telah mengasihi Aku, demikianlah juga Aku telah mengasihi kamu; "
        #         "tinggallah di dalam kasih-Ku itu.'"
        #     )
        # },
        #
        # {
        #     "id": "Q02",
        #     "type": "faktual",
        #     "source_doc": "01.txt — Sabtu",
        #     "question": "Bacaan Alkitab apa saja yang harus dipelajari untuk pelajaran pekan ini?",
        #     "reference": (
        #         "Bacaan untuk pelajaran pekan ini adalah: "
        #         "Wahyu 3:14-22; Wahyu 4:9-11; Kejadian 2:7; Kejadian 3:8-10; "
        #         "Yeremia 31:3, 4; Yohanes 15:1-11; dan Roma 8:9–11."
        #     )
        # },
        #
        # {
        #     "id": "Q03",
        #     "type": "faktual",
        #     "source_doc": "02.txt — Minggu",
        #     "question": "Dalam Wahyu 3:14, bagaimana Yesus memperkenalkan diri-Nya kepada jemaat Laodikia?",
        #     "reference": (
        #         "Dalam Wahyu 3:14, Yesus memperkenalkan diri-Nya sebagai "
        #         "'Saksi yang setia dan benar, permulaan dari ciptaan Allah.' "
        #         "Sebagai Saksi yang setia dan benar, Dia tidak berbohong "
        #         "melainkan berbicara dengan jelas dan jujur."
        #     )
        # },
        #
        # {
        #     "id": "Q04",
        #     "type": "faktual",
        #     "source_doc": "02.txt — Minggu",
        #     "question": "Apa tiga hal yang Yesus tawarkan sebagai pertukaran dalam Wahyu 3:18 kepada jemaat Laodikia?",
        #     "reference": (
        #         "Dalam Wahyu 3:18, Yesus menawarkan tiga hal sebagai pertukaran: "
        #         "(1) emas-Nya sebagai pengganti keapatisan kita, "
        #         "(2) pakaian putih-Nya (jubah kebenaran) untuk menutupi kita, "
        #         "dan (3) salep mata-Nya agar kita dapat melihat kebenaran rohani."
        #     )
        # },
        #
        # {
        #     "id": "Q05",
        #     "type": "faktual",
        #     "source_doc": "03.txt — Senin",
        #     "question": "Janji apa yang diberikan Yesus dalam Wahyu 3:20 dan apa syarat untuk menerimanya?",
        #     "reference": (
        #         "Dalam Wahyu 3:20, Yesus berjanji bahwa Dia akan masuk dan makan bersama "
        #         "dengan orang yang membuka pintu bagi-Nya — menggambarkan hubungan yang intim dan hangat. "
        #         "Syaratnya adalah kita harus mendengar suara-Nya dan secara sadar membuka pintu hati bagi-Nya. "
        #         "Yesus tidak menerobos masuk atau memaksa, melainkan menunggu keputusan sadar kita."
        #     )
        # },
        #
        # {
        #     "id": "Q06",
        #     "type": "faktual",
        #     "source_doc": "04.txt — Selasa",
        #     "question": "Sebutkan kisah-kisah dalam Alkitab pada materi hari Selasa yang menunjukkan Allah rindu dekat dengan manusia.",
        #     "reference": (
        #         "Materi hari Selasa menyebutkan kisah-kisah berikut: "
        #         "Kejadian 2:7 (Allah menciptakan manusia), "
        #         "Kejadian 3:8-10 (Allah berjalan bersama Adam dan Hawa), "
        #         "Kejadian 5:24 (Henokh berjalan bersama Allah), "
        #         "Kejadian 6:13 (Allah berbicara kepada Nuh), "
        #         "Kejadian 12:1-4 (Allah memanggil Abraham), "
        #         "dan Keluaran 34:29 (Musa berjumpa dengan Allah). "
        #         "Kisah-kisah ini membuktikan bahwa sejak dahulu kala Allah selalu rindu untuk dekat dengan umat manusia."
        #     )
        # },
        #
        # {
        #     "id": "Q07",
        #     "type": "faktual",
        #     "source_doc": "05.txt — Rabu",
        #     "question": "Berapa kali kata 'tinggal' diulang dalam Yohanes 15:1-11 dan apa maknanya?",
        #     "reference": (
        #         "Kata 'tinggal' (abide) diulang sebanyak sepuluh kali dalam Yohanes 15:1-11. "
        #         "Maknanya adalah hidup dalam hubungan yang terus-menerus terhubung dengan Yesus. "
        #         "Tinggal di dalam Yesus berarti menuruti perintah-perintah-Nya, berbuah untuk kemuliaan-Nya, "
        #         "dan hidup dalam sukacita yang besar sebagai bentuk tanggapan kasih kepada-Nya."
        #     )
        # },
        #
        # {
        #     "id": "Q08",
        #     "type": "faktual",
        #     "source_doc": "06.txt — Kamis",
        #     "question": "Sebutkan empat peran Roh Kudus yang disebutkan dalam materi hari Kamis berdasarkan Yohanes pasal 14, 15, dan 16.",
        #     "reference": (
        #         "Empat peran Roh Kudus menurut materi hari Kamis adalah: "
        #         "(1) Menjadi Penghibur kita (Yohanes 14:16-18), "
        #         "(2) Menyingkapkan Yesus kepada kita (Yohanes 15:26), "
        #         "(3) Menyadarkan kita akan dosa (Yohanes 16:7-8), "
        #         "dan (4) Menuntun kita ke dalam seluruh kebenaran (Yohanes 16:13)."
        #     )
        # },
        #
        # {
        #     "id": "Q09",
        #     "type": "faktual",
        #     "source_doc": "07.txt — Jumat",
        #     "question": "Apa yang dikatakan Ellen G. White tentang cara ranting kering bisa menyatu dengan pokok anggur?",
        #     "reference": (
        #         "Ellen G. White dalam Manuscript 67, 1897 menjelaskan bahwa ranting kecil yang kering "
        #         "harus dicangkokkan ke dalam pokok anggur dan dibawa ke dalam hubungan yang paling dekat. "
        #         "'Serat demi serat, urat demi urat, ranting kecil itu melekat erat pada pokok anggur yang memberi kehidupan, "
        #         "sampai kehidupan pokok anggur itu menyatu dengan ranting, "
        #         "dan ranting itu menghasilkan buah seperti pokoknya.'"
        #     )
        # },
        #
        # # ══════════════════════════════════════════════════════
        # # TIPE 2: TEMATIK (Q10–Q18)
        # # Tema dan konsep teologis dalam minggu ini
        # # ══════════════════════════════════════════════════════
        #
        # {
        #     "id": "Q10",
        #     "type": "tematik",
        #     "source_doc": "01.txt + 02.txt",
        #     "question": "Apa tema mingguan pelajaran ini dan mengapa tema tersebut penting sebagai langkah awal pertumbuhan rohani?",
        #     "reference": (
        #         "Tema mingguan adalah 'Cek Realitas', bagian dari tema triwulan 'Bertumbuh Dalam Hubungan Dengan Tuhan'. "
        #         "Tema ini penting karena kita tidak bisa bergerak ke tempat yang lebih baik secara rohani "
        #         "sebelum melakukan kejujuran terhadap diri sendiri mengenai kondisi hubungan kita dengan Allah saat ini "
        #         "dan mendengarkan solusi yang Yesus tawarkan."
        #     )
        # },
        #
        # {
        #     "id": "Q11",
        #     "type": "tematik",
        #     "source_doc": "02.txt — Minggu",
        #     "question": "Apa yang dimaksud dengan kondisi 'suam-suam kuku' menurut pelajaran hari Minggu dan mengapa Yesus tidak menyukainya?",
        #     "reference": (
        #         "Kondisi 'suam-suam kuku' (Wahyu 3:16) adalah kondisi di mana seseorang tidak panas dan tidak dingin secara rohani. "
        #         "Dari sudut pandang kita sendiri, kita merasa tidak membutuhkan apa pun dan hanya meluangkan sedikit waktu bersama Allah. "
        #         "Yesus tidak menyukainya karena kondisi ini mencerminkan keapatisan rohani — "
        #         "kita berpikir sudah cukup padahal sebenarnya kita melarat, malang, miskin, buta, dan telanjang secara rohani. "
        #         "Yesus menginginkan hubungan yang sepenuh hati, bukan yang setengah-setengah."
        #     )
        # },
        #
        # {
        #     "id": "Q12",
        #     "type": "tematik",
        #     "source_doc": "03.txt — Senin",
        #     "question": "Mengapa Yesus menegur jemaat-Nya menurut pelajaran hari Senin, dan apa yang membuktikan bahwa teguran itu berasal dari kasih?",
        #     "reference": (
        #         "Yesus menegur jemaat-Nya karena kasih-Nya yang sangat dalam — "
        #         "'Barangsiapa Kukasihi, ia Kutegor dan Kuhajar' (Wahyu 3:19). "
        #         "Ini bukan karena Dia tidak peduli, melainkan justru sebaliknya: "
        #         "Dia menginginkan hubungan yang jauh lebih kuat dan lebih dalam. "
        #         "Bukti bahwa teguran itu dari kasih adalah bahwa Yesus sendiri yang menempuh jalan penderitaan di bumi "
        #         "dan mati untuk menyelamatkan kita — Dia tidak puas dengan sikap setengah-setengah kita."
        #     )
        # },
        #
        # {
        #     "id": "Q13",
        #     "type": "tematik",
        #     "source_doc": "04.txt — Selasa",
        #     "question": "Apa yang dimaksud 'pertempuran terbesar' yang disebutkan dalam pelajaran hari Selasa?",
        #     "reference": (
        #         "Pertempuran terbesar yang disebutkan dalam pelajaran hari Selasa adalah: "
        #         "menyadari kondisi kita yang lemah dan merasa cukup dengan diri sendiri, "
        #         "kemudian menerima teguran Yesus, bertobat, "
        #         "dan menerima jubah kebenaran Yesus dengan mata yang benar-benar dapat melihat. "
        #         "Ini adalah perjuangan melawan keapatisan dan kesombongan rohani kita sendiri."
        #     )
        # },
        #
        # {
        #     "id": "Q14",
        #     "type": "tematik",
        #     "source_doc": "05.txt — Rabu",
        #     "question": "Apa 'obat penawar' bagi kondisi Laodikia menurut pelajaran hari Rabu?",
        #     "reference": (
        #         "Obat penawar bagi kondisi Laodikia menurut pelajaran hari Rabu adalah 'tinggal' (abide) di dalam Yesus "
        #         "sebagaimana disebutkan dalam Wahyu 3:20 dan Yohanes 15:4. "
        #         "Ini adalah rahasia besar dari kehidupan yang penuh makna dan berarti, "
        #         "di mana kita hidup dalam hubungan yang terus terhubung dengan Yesus "
        #         "dan menuruti perintah-perintah-Nya sebagai bentuk kasih kepada-Nya."
        #     )
        # },
        #
        # {
        #     "id": "Q15",
        #     "type": "tematik",
        #     "source_doc": "06.txt — Kamis",
        #     "question": "Mengapa mengikuti Allah kadang terasa seperti 'kewajiban yang melelahkan' dan bagaimana seharusnya hubungan dengan Allah itu terasa?",
        #     "reference": (
        #         "Mengikuti Allah terasa seperti kewajiban yang melelahkan ketika agama hanya berfokus pada tindakan luar "
        #         "dan aturan-aturan, bukan pada apa yang ada di dalam hati. "
        #         "Seharusnya hubungan dengan Allah didasari kasih timbal balik dan kebebasan memilih — "
        #         "di mana Allah terlebih dahulu memilih kita dan kita merespons dengan sadar "
        #         "bukan karena paksaan melainkan karena kasih yang tulus."
        #     )
        # },
        #
        # {
        #     "id": "Q16",
        #     "type": "tematik",
        #     "source_doc": "06.txt — Kamis",
        #     "question": "Bagaimana metafora musim dingin pada pelajaran hari Kamis menjelaskan pertumbuhan rohani melalui Roh Kudus?",
        #     "reference": (
        #         "Metafora musim dingin menjelaskan bahwa tunas pada ranting menjadi dehidrasi dan terisolasi di musim dingin. "
        #         "Ketika musim semi tiba, akar menyerap air dan getah mengalir dari batang menuju tunas sehingga memicu pertumbuhan. "
        #         "Secara rohani, getah ini ibarat Roh Kudus — kita mungkin seperti ranting yang mati secara rohani, "
        #         "tetapi ketika kita memilih meluangkan waktu bersama Allah, "
        #         "Roh Kudus dicurahkan ke dalam diri kita dan menghidupkan kita kembali sehingga kita mulai bertumbuh."
        #     )
        # },
        #
        # {
        #     "id": "Q17",
        #     "type": "tematik",
        #     "source_doc": "07.txt — Jumat",
        #     "question": "Apa ringkasan pelajaran minggu ini menurut dokumen hari Jumat?",
        #     "reference": (
        #         "Ringkasan pelajaran minggu ini menurut hari Jumat adalah: "
        #         "Sebelum kita dapat mulai bertumbuh dalam hubungan dengan Allah, "
        #         "kita harus berhenti sejenak untuk merenungkan seperti apa hubungan kita dengan-Nya saat ini. "
        #         "Jika hubungan itu bersifat Laodikia atau jika ranting-ranting kita tidak subur, "
        #         "Yesus memiliki solusi yang sempurna untuk kondisi rohani kita: yaitu tinggal di dalam Dia."
        #     )
        # },
        #
        # {
        #     "id": "Q18",
        #     "type": "tematik",
        #     "source_doc": "07.txt — Jumat",
        #     "question": "Apa yang dikatakan Ellen G. White dalam Kerinduan Segala Zaman tentang makna tinggal di dalam Kristus?",
        #     "reference": (
        #         "Ellen G. White dalam Kerinduan Segala Zaman (1999), jilid 2, halaman 320 menjelaskan: "
        #         "'Tinggal di dalam Kristus berarti selalu menerima Roh-Nya, "
        #         "suatu kehidupan penyerahan yang tidak terbatas kepada pekerjaan-Nya. "
        #         "Saluran perhubungan harus terbuka terus-menerus antara manusia dan Allahnya. "
        #         "Sebagaimana carang pokok anggur senantiasa mengisap sari dari pokok anggur yang hidup, "
        #         "demikian juga kita harus berpaut pada Yesus, dan menerima dari pada-Nya oleh iman "
        #         "kekuatan dan kesempurnaan tabiat-Nya sendiri.'"
        #     )
        # },
        #
        # # ══════════════════════════════════════════════════════
        # # TIPE 3: LINTAS HARI (Q19–Q25)
        # # Hubungan antar konsep dari hari yang berbeda
        # # ══════════════════════════════════════════════════════
        #
        # {
        #     "id": "Q19",
        #     "type": "lintas_hari",
        #     "source_doc": "01.txt + 02.txt",
        #     "question": "Bagaimana hubungan antara 'Cek Realitas' (hari Sabtu) dengan 'Kondisi Kita' (hari Minggu)?",
        #     "reference": (
        #         "Hari Sabtu ('Cek Realitas') menjadi dasar yang harus dilakukan sebelum memahami 'Kondisi Kita' di hari Minggu. "
        #         "Kita tidak bisa bergerak ke tempat yang lebih baik secara rohani "
        #         "sebelum melakukan kejujuran terhadap diri sendiri. "
        #         "Hari Minggu kemudian mengungkap kondisi konkret yang teridentifikasi setelah cek realitas itu — "
        #         "yaitu kondisi suam-suam kuku jemaat Laodikia yang digambarkan dalam Wahyu 3:14-17."
        #     )
        # },
        #
        # {
        #     "id": "Q20",
        #     "type": "lintas_hari",
        #     "source_doc": "02.txt + 03.txt",
        #     "question": "Bagaimana 'Kondisi Kita' (hari Minggu) menjadi latar belakang bagi 'Teguran, Pertobatan, dan Upah' (hari Senin)?",
        #     "reference": (
        #         "Hari Minggu mengidentifikasi kondisi kita yang suam-suam kuku dan melarat secara rohani. "
        #         "Kondisi inilah yang menjadi alasan Yesus memberikan teguran di hari Senin (Wahyu 3:19). "
        #         "Yesus tidak bisa menegur kita tanpa kita terlebih dahulu menyadari kondisi kita yang salah. "
        #         "Teguran di hari Senin adalah respons kasih Yesus terhadap kondisi yang terungkap di hari Minggu, "
        #         "dan menghasilkan undangan untuk bertobat serta janji indah dalam Wahyu 3:20."
        #     )
        # },
        #
        # {
        #     "id": "Q21",
        #     "type": "lintas_hari",
        #     "source_doc": "03.txt + 04.txt",
        #     "question": "Bagaimana tema 'Kasih yang Tak Berkesudahan' (hari Selasa) memperkuat dan melengkapi pesan teguran di hari Senin?",
        #     "reference": (
        #         "Teguran di hari Senin mungkin terasa keras, tetapi hari Selasa memberikan fondasi mengapa teguran itu ada — "
        #         "yaitu kasih Allah yang tak berkesudahan seperti tertulis dalam Yeremia 31:3: "
        #         "'Aku mengasihi engkau dengan kasih yang kekal.' "
        #         "Kisah-kisah dari Kejadian hingga Keluaran membuktikan bahwa Allah selalu rindu dekat dengan manusia. "
        #         "Jadi teguran di hari Senin bukan untuk menyakiti, "
        #         "melainkan ekspresi dari kasih tak berkesudahan yang dibuktikan sepanjang sejarah di hari Selasa."
        #     )
        # },
        #
        # {
        #     "id": "Q22",
        #     "type": "lintas_hari",
        #     "source_doc": "02.txt + 05.txt",
        #     "question": "Apa hubungan antara kondisi 'suam-suam kuku' (hari Minggu) dengan konsep 'tidak adanya buah' (hari Rabu)?",
        #     "reference": (
        #         "Kondisi suam-suam kuku di hari Minggu adalah bentuk keapatisan rohani yang terlihat dari luar. "
        #         "Hari Rabu memperdalam ini dengan konsep 'tinggal' dalam Yohanes 15 — "
        #         "jika kita hanya terlihat seperti tinggal di dalam Yesus tetapi tidak sungguh-sungguh terhubung, "
        #         "buktinya akan terlihat dari tidak adanya buah. "
        #         "Ranting yang suam-suam kuku pada akhirnya akan mengering dan dipotong oleh Pengusaha Anggur. "
        #         "Jadi kondisi suam-suam kuku (hari Minggu) adalah akar masalah, "
        #         "dan tidak berbuah (hari Rabu) adalah akibatnya."
        #     )
        # },
        #
        # {
        #     "id": "Q23",
        #     "type": "lintas_hari",
        #     "source_doc": "05.txt + 06.txt",
        #     "question": "Bagaimana konsep 'tinggal' (hari Rabu) dan peran 'Roh Kudus sebagai getah' (hari Kamis) saling melengkapi?",
        #     "reference": (
        #         "Hari Rabu menjelaskan bahwa kita harus tinggal di dalam Yesus, "
        #         "tetapi kita tidak bisa melakukannya dengan kekuatan sendiri — "
        #         "'kita tidak bisa membuat buah itu tumbuh dengan usaha sendiri.' "
        #         "Hari Kamis menjawab pertanyaan 'bagaimana caranya' dengan metafora getah — "
        #         "Roh Kudus adalah getah yang mengalir dari pokok anggur (Yesus) ke ranting (kita) "
        #         "dan memungkinkan pertumbuhan terjadi. "
        #         "Jadi tinggal (hari Rabu) adalah keputusan sadar kita, "
        #         "sedangkan Roh Kudus (hari Kamis) adalah kuasa yang memungkinkan tinggal itu terjadi secara nyata."
        #     )
        # },
        #
        # {
        #     "id": "Q24",
        #     "type": "lintas_hari",
        #     "source_doc": "04.txt + 06.txt",
        #     "question": "Bagaimana kasih Allah yang digambarkan dalam Yeremia 31:3 (hari Selasa) berhubungan dengan motivasi untuk meminta Roh Kudus (hari Kamis)?",
        #     "reference": (
        #         "Yeremia 31:3 di hari Selasa menyatakan bahwa Allah mengasihi kita dengan kasih yang kekal "
        #         "dan Dialah yang terlebih dahulu mengambil langkah mendekati kita. "
        #         "Ini menjadi motivasi utama untuk meminta Roh Kudus di hari Kamis — "
        #         "kita meminta bukan karena kewajiban, "
        #         "melainkan karena merespons kasih Allah yang lebih dulu dinyatakan. "
        #         "Hari Kamis juga menegaskan bahwa tanggapan kita selalu merupakan reaksi atas apa yang Allah lakukan lebih dulu, "
        #         "yang konsisten dengan kasih kekal yang diungkap di hari Selasa."
        #     )
        # },

        {
            "id": "Q25",
            "type": "lintas_hari",
            "source_doc": "01.txt + 07.txt",
            "question": "Bagaimana kesimpulan pelajaran hari Jumat menjawab pertanyaan pembuka pelajaran hari Sabtu tentang hubungan dengan Allah?",
            "reference": (
                "Hari Sabtu membuka pelajaran dengan pertanyaan reflektif: "
                "'Bagaimana Anda menggambarkan hubungan Anda dengan Allah saat ini?' "
                "dan mendorong kita untuk melakukan cek realitas yang jujur. "
                "Hari Jumat menjawab lingkaran pertanyaan ini dengan menyimpulkan bahwa "
                "setelah kita jujur tentang kondisi kita (suam-suam kuku / Laodikia), "
                "solusinya adalah tinggal di dalam Yesus — menerima Roh-Nya setiap hari "
                "dan membiarkan kehidupan pokok anggur menyatu dengan ranting kita. "
                "Ini membentuk alur lengkap: cek realitas (Sabtu) → solusi (Jumat)."
            )
        },
    ]
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

            judge = await judge_metrics(client, item["question"], retrieved_contexts, response_text, item["reference"])
            latency = time.time() - start_time
            save_experiment({
                "user_input": item["question"],
                "type": item["type"],
                "source_doc": item["source_doc"],
                "retrieved_contexts": retrieved_contexts,
                "response": response_text,
                "reference": item["reference"],
                "accuracy": judge.get("accuracy", 0),
                "relevance": judge.get("relevance", 0),
                "completeness": judge.get("completeness", 0),
                "reason": judge.get("reason", "No reason provided"),
                "strategy": mode,
                "latency": latency,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            print(
                f"Mode: {mode:7} | Acc: {judge['accuracy']} | Rel: {judge['relevance']} | Comp: {judge['completeness']}")

            await asyncio.sleep(3)


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("API Key belum diset di Environment!")
    else:
        asyncio.run(run_evaluation())

