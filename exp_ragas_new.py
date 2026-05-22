import asyncio
import os
import time
from datetime import datetime

import instructor
import nest_asyncio
import pandas as pd
from openai import AsyncOpenAI
from google import genai

from ragas.llms import InstructorLLM
from ragas.embeddings import GoogleEmbeddings
from ragas.llms.base import InstructorModelArgs
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from instructor import Mode
from lightrag import LightRAG, QueryParam
from core.embedding import gemini_embedding_func
from core.gemini import gemini_llm_model_func

nest_asyncio.apply()

# =====================================================================
# INISIALISASI JUDGE & EMBEDDINGS (Ragas v0.4.3 Native Compliant)
# =====================================================================

instructor_client = instructor.from_openai(
    AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://ai.sumopod.com/v1"
    ),
    mode=Mode.JSON
)
ragas_judge = InstructorLLM(
    client=instructor_client,
    model="glm-5.1",
    provider="openai",
    model_args=InstructorModelArgs(
        max_tokens=8192,
        temperature=0.1,
    )
)

google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
ragas_embeddings = GoogleEmbeddings(
    model="gemini-embedding-001",
    client=google_client
)

question_dataset = [
    # {
    #   "id": "Q01",
    #   "type": "faktual",
    #   "source_doc": "week-01.txt — Sabtu",
    #   "question": "Apa judul pelajaran hari Sabtu pada minggu pertama dan kutipan lengkap ayat hafalan berdasarkan teks tersebut?",
    #   "reference": "Judul pelajaran hari Sabtu adalah Cek Realitas. Ayat hafalan minggu pertama diambil dari Yohanes 15:9 mengenai perintah untuk tinggal di dalam kasih Yesus sebagaimana Bapa mengasihi Yesus."
    # },
    # {
    #   "id": "Q02",
    #   "type": "faktual",
    #   "source_doc": "week-01.txt — Minggu",
    #   "question": "Berdasarkan Wahyu 3:14, sebutkan gelar yang digunakan Yesus untuk memperkenalkan diri-Nya kepada jemaat akhir zaman dan apa karakteristik dari gelar tersebut?",
    #   "reference": "Yesus memperkenalkan diri sebagai Saksi yang setia dan benar, permulaan dari ciptaan Allah kepada jemaat akhir zaman. Karakteristik dari gelar Saksi yang setia dan benar adalah Yesus tidak berbohong, melainkan Yesus berbicara dengan jelas dan jujur kepada umat Yesus."
    # },
    # {
    #   "id": "Q03",
    #   "type": "faktual",
    #   "source_doc": "week-01.txt — Rabu",
    #   "question": "Berapa kali kata tinggal (abide) diulang dalam Yohanes 15:1-11 dan apa definisinya menurut teks?",
    #   "reference": "Kata tinggal diulang sebanyak sepuluh kali dalam Yohanes 15:1-11. Menurut teks, tinggal di dalam Yesus berarti hidup dalam hubungan yang terus terhubung dengan Yesus."
    # },
    # {
    #   "id": "Q04",
    #   "type": "faktual",
    #   "source_doc": "week-01.txt — Kamis",
    #   "question": "Sebutkan empat peran penting Roh Kudus bagi manusia di bumi yang tertera pada bagian daftar hari Kamis beserta rujukan ayat Yohanes-nya!",
    #   "reference": "Empat peran Roh Kudus berdasarkan teks hari Kamis adalah Roh Kudus menjadi Penghibur manusia berdasarkan Yohanes 14:16-18, Roh Kudus menyingkapkan Yesus kepada manusia berdasarkan Yohanes 15:26, Roh Kudus menyadarkan manusia akan dosa berdasarkan Yohanes 16:7-8, dan Roh Kudus menuntun manusia ke dalam seluruh kebenaran berdasarkan Yohanes 16:13."
    # },
    # {
    #   "id": "Q05",
    #   "type": "faktual",
    #   "source_doc": "week-02.txt — Sabtu",
    #   "question": "Tuliskan bunyi lengkap ayat hafalan untuk minggu kedua yang diambil dari Yohanes 17:3!",
    #   "reference": "Ayat hafalan minggu kedua berdasarkan Yohanes 17:3 menyatakan bahwa hidup yang kekal adalah ketika manusia mengenal Allah sebagai satu-satunya Allah yang benar, dan mengenal Yesus Kristus yang telah Allah utus."
    # },
    # {
    #   "id": "Q06",
    #   "type": "faktual",
    #   "source_doc": "week-02.txt — Senin",
    #   "question": "Apa arti teologis dari sifat Allah yang 'kudus' berdasarkan penjelasan materi hari Senin di minggu kedua?",
    #   "reference": "Ketika Alkitab menggambarkan Allah sebagai puncak kekudusan, itu berarti bahwa Dia sepenuhnya bebas dari, dan sepenuhnya terpisah dari, kejahatan dan dosa, atau dengan kata lain Allah adalah 100 persen baik dari awal hingga akhir."
    # },
    # {
    #   "id": "Q07",
    #   "type": "faktual",
    #   "source_doc": "week-02.txt — Selasa",
    #   "question": "Sebutkan dua nama kuno Allah dalam bahasa Ibrani yang dibahas pada hari Selasa beserta maknanya masing-masing!",
    #   "reference": "Dua nama kuno Allah yang dibahas adalah: (1) Adonai, yang berarti Tuhan atas segala sesuatu, yang memerintah untuk selama-lamanya, dalam konteks perjanjian; (2) Yahweh-Yireh, yang berarti TUHAN yang menyediakan."
    # },
    # {
    #   "id": "Q08",
    #   "type": "tematik",
    #   "source_doc": "week-01.txt — Senin",
    #   "question": "Uraikan makna dari metafora 'Yesus mengetuk pintu hati' berdasarkan narasi hari Senin di minggu pertama!",
    #   "reference": "Metafora Yesus mengetuk pintu (Wahyu 3:20) menggambarkan kerinduan Allah semesta alam untuk membangun hubungan dekat, intim, dan tinggal tetap yang digambarkan seperti duduk makan bersama dan berbincang hangat. Ketukan ini dilakukan dengan lembut tanpa menerobos masuk secara paksa, menghormati kebebasan memilih manusia di tengah kesibukan hidup mereka."
    # },
    # {
    #   "id": "Q09",
    #   "type": "tematik",
    #   "source_doc": "week-01.txt — Kamis",
    #   "question": "Bagaimana ilustrasi aliran getah pada pohon anggur saat musim dingin dan musim semi menjelaskan dinamika pertumbuhan rohani manusia?",
    #   "reference": "Pada musim dingin, tunas-tunas ranting mengalami dehidrasi dan terisolasi dari pertumbuhan. Saat musim semi datang dan tanah menghangat, akar menyerap air sehingga getah mengalir dari batang ke tunas untuk memicu pertumbuhan. Secara rohani, getah tersebut adalah Roh Kudus. Manusia mungkin seperti ranting mati yang kering, tetapi saat memilih meluangkan waktu bersama Allah, Roh Kudus dicurahkan mengalir dalam hidup untuk menghidupkan dan menumbuhkan mereka kembali."
    # },
    # {
    #   "id": "Q10",
    #   "type": "tematik",
    #   "source_doc": "week-02.txt — Minggu",
    #   "question": "Jelaskan strategi dan tujuan utama Lucifer atau Iblis dalam merusak hubungan manusia dengan Allah melalui distorsi karakter-Nya!",
    #   "reference": "Strategi Lucifer sejak awal pertarungan besar (seperti pada percakapan dengan Hawa di Kejadian 3:1-5) adalah menampilkan karakter Allah secara salah, membangkitkan pemberontakan, dan menanamkan kebohongan bahwa Allah menyembunyikan sesuatu serta tidak menginginkan yang terbaik bagi manusia. Tujuannya adalah menyibukkan pikiran manusia agar kehilangan pengetahuan yang benar tentang Allah, sebab Iblis tidak peduli apa pun paham yang dianut manusia (panteisme, ateisme, deisme) selama mereka tidak memiliki gambaran yang benar tentang Allah."
    # },
# {
#       "id": "Q11",
#       "type": "tematik",
#       "source_doc": "week-02.txt — Selasa",
#       "question": "Bagaimana teks menjelaskan esensi kasih Allah (hesed) yang membedakannya dari definisi kasih menurut ukuran konvensional manusia?",
#       "reference": "Kasih manusia sering kali terdistorsi dan tidak sempurna. Sebaliknya, kasih Allah (hesed) bukanlah sekadar tindakan luar melainkan identitas inti karakter-Nya yang murni, bebas, tanpa pamrih, dan sangat relasional. Kasih ini berwujud kasih perjanjian yang mencakup kesetiaan, perlindungan, keteguhan, kelembutan, serta pengorbanan radikal yang dibuktikan melalui pemberian Yesus Kristus untuk mati menggantikan orang berdosa."
#     },
#     {
#       "id": "Q12",
#       "type": "lintas_hari",
#       "source_doc": "week-01.txt — Minggu & Senin",
#       "question": "Hubungkan analisis 'Kondisi Kita' pada hari Minggu dengan 'Teguran dan Pertobatan' pada hari Senin di minggu pertama. Mengapa urutan ini penting?",
#       "reference": "Hari Minggu mendiagnosis kondisi rohani kita yang suam-suam kuku, mati rasa, dan merasa cukup padahal melarat. Kondisi riil ini menjadi landasan bagi hari Senin, di mana Yesus menyampaikan teguran keras. Urutan ini penting karena manusia tidak akan pernah bisa bertobat kecuali mereka terlebih dahulu disadarkan secara jujur bahwa ada sesuatu yang salah dalam kondisi rohani mereka. Teguran Yesus di hari Senin muncul sebagai bentuk penawaran solusi karena kasih-Nya yang mendalam terhadap kondisi buruk di hari Minggu."
#     },
#     {
#       "id": "Q13",
#       "type": "lintas_hari",
#       "source_doc": "week-01.txt — Rabu & Kamis",
#       "question": "Bagaimana instruksi untuk 'Tinggal di dalam Yesus' (hari Rabu) diselesaikan masalah eksekusinya melalui doktrin 'Roh Kudus sebagai Getah' (hari Kamis)?",
#       "reference": "Hari Rabu memerintahkan manusia untuk tinggal di dalam Yesus agar bisa menghasilkan buah rohani, namun teks juga menegaskan batasan manusia: 'Kita tidak bisa membuat buah itu tumbuh dengan usaha sendiri'. Kebingungan eksekusi ini dijawab pada hari Kamis, yang menyatakan bahwa manusia secara natur tidak bisa memaksa dirinya menempel pada Pokok Anggur. Solusinya adalah Roh Kudus yang bertindak sebagai 'getah' kehidupan; ketika manusia membuat keputusan sadar meluangkan waktu bersama Allah dan meminta Roh-Nya, Roh Kudus lah yang mengalirkan kuasa, menghidupkan kembali, dan membawa pertumbuhan nyata."
#     },
{
      "id": "Q14",
      "type": "lintas_hari",
      "source_doc": "week-01.txt — Sabtu & Jumat",
      "question": "Bagaimana kesimpulan ringkasan pada hari Jumat menutup rangkaian pertanyaan reflektif yang diajukan pada awal pelajaran hari Sabtu di minggu pertama?",
      "reference": "Hari Sabtu membuka pekan dengan rentetan pertanyaan reflektif yang tajam mengenai kualitas hubungan individu dengan Allah, apakah melemah, suam-suam kuku, atau jarang menyapa-Nya. Hari Jumat menutup lingkaran diskusi ini dengan memberikan konklusi final (ringkasan): bahwa setelah manusia melakukan cek realitas yang jujur terhadap kondisi Laodikia atau ketidakberbuaan mereka, solusi mutlak yang Yesus tawarkan adalah 'tinggal di dalam Dia' secara total dengan berserah pada pekerjaan Roh Kudus setiap hari."
    },
# {
#       "id": "Q15",
#       "type": "lintas_minggu",
#       "source_doc": "week-01.txt & week-02.txt (Seluruh Dokumen)",
#       "question": "Sintesiskan bagaimana tema triwulan 'Bertumbuh Dalam Hubungan Dengan Tuhan' diartikulasikan secara terstruktur melalui sub-tema Week 1 dan Week 2!",
#       "reference": "Tema besar 'Bertumbuh Dalam Hubungan Dengan Tuhan' diartikulasikan secara bertahap. Pekan Pertama ('Cek Realitas') berfokus pada evaluasi internal individu untuk membongkar keapatisan rohani (Laodikia), memberikan kesadaran akan kelemahan diri, serta menawarkan fondasi praktis untuk terhubung, yaitu 'tinggal' di dalam Kristus melalui kuasa Roh Kudus. Pekan Kedua ('Mengenal Allah') bergerak maju ke aspek eksternal objektif dengan membangun dasar teologis hubungan tersebut, yaitu pemulihan pemahaman karakter Allah (Kudus, Kasih, Transenden sekaligus Intim) yang sempat dirusak Iblis. Sintesisnya: Week 1 membersihkan saluran hubungan dari kesombongan manusia, dan Week 2 mengisi saluran tersebut dengan pengenalan yang memikat tentang siapa Allah yang kita sembah."
#     },
# {
#       "id": "Q16",
#       "type": "lintas_hari",
#       "source_doc": "week-02.txt — Minggu & Selasa",
#       "question": "Bandingkan tuduhan salah Lucifer tentang karakter Allah (hari Minggu) dengan bukti konkret kasih radikal Allah melalui Salib (hari Selasa) pada minggu kedua!",
#       "reference": "Pada hari Minggu, Lucifer menuduh bahwa Allah memiliki motif egois, menyembunyikan sesuatu yang baik, dan tidak bisa dipercayai. Tuduhan palsu ini dipatahkan sepenuhnya oleh fakta di hari Selasa, yang menunjukkan bahwa esensi tertinggi kasih Allah (hesed) diwujudkan tanpa pamrih melalui pengorbanan ekstrem: mengutus Anak-Nya sendiri, Yesus Kristus, untuk menempuh kematian pengganti demi menyelamatkan manusia yang berdosa dan menjembatani jurang pemisah."
#     },
# {
#       "id": "Q17",
#       "type": "lintas_hari",
#       "source_doc": "week-02.txt — Rabu & Kamis",
#       "question": "Bagaimana dualitas sifat Allah 'Elohim dan Yahwe' (hari Rabu) terefleksi secara nyata dalam pribadi Yesus Kristus yang digambarkan pada hari Kamis?",
#       "reference": "Hari Rabu menjelaskan transendensi 'Elohim' yang mahakuasa atas semesta dan imanensi 'Yahwe' yang sangat dekat menghembuskan nafas ke manusia. Dualitas ini mewujud sempurna dalam pribadi Yesus pada hari Kamis, di mana Yesus di satu sisi adalah Anak Allah yang berinkarnasi, memegang kuasa ilahi, menyatakan Bapa secara sempurna (representasi Elohim); dan di sisi lain memiliki kemanusiaan yang sempurna, penuh perasaan, belas kasih, aktif melayani, dan berjanji menyertai manusia senantiasa sampai akhir zaman (representasi Yahwe)."
#     },
# {
#       "id": "Q18",
#       "type": "lintas_minggu",
#       "source_doc": "week-01.txt (Rabu) & week-02.txt (Selasa)",
#       "question": "Analisis bagaimana konsep 'Menuruti Perintah Allah' sebagai buah dari tinggal di dalam Yesus (Week 1) berkaitan erat dengan penolakan terhadap tuduhan Iblis mengenai hukum Allah (Week 2)!",
#       "reference": "Pada Week 2 (Selasa), dijelaskan bahwa Iblis sejak awal bertujuan salah mengartikan tabiat Allah dan membangkitkan pemberontakan melawan hukum-Nya seolah-olah hukum itu mengekang. Namun, pada Week 1 (Rabu) dijelaskan bahwa ketika manusia 'tinggal di dalam Yesus', kepatuhan atau menuruti perintah-perintah-Nya akan mengalir secara alami sebagai buah kasih. Menuruti perintah-Nya bukanlah beban yang berat, melainkan pantulan dari karakter Allah yang indah dan penuh kasih tanpa pamrih, sekaligus mematahkan kebohongan Iblis."
#     },
# {
#       "id": "Q19",
#       "type": "lintas_minggu",
#       "source_doc": "week-01.txt (Minggu) & week-02.txt (Sabtu)",
#       "question": "Mengapa pemulihan gambaran karakter Allah yang benar (Week 2) menjadi prasyarat mutlak untuk menyembuhkan kondisi jemaat yang 'suam-suam kuku' (Week 1)?",
#       "reference": "Jemaat yang suam-suam kuku pada Week 1 terjebak dalam keapatisan karena merasa tidak membutuhkan apa-apa dan pelit meluangkan waktu dengan Tuhan. Kondisi ini berakar dari masalah yang diurai pada Week 2, yaitu hilangnya pengetahuan yang benar mengenai tabiat Allah akibat disalahartikan di dunia. Pemahaman yang jelas tentang kebaikan, kekudusan, dan kasih Allah pada Week 2 menjadi prasyarat mutlak karena 'semakin kita mengenal Allah, semakin kita akan mengasihi-Nya dan merindukan hubungan yang erat dan tetap dengan-Nya', yang secara otomatis menghancurkan keapatisan Laodikia."
#     },
    # {
    #   "id": "Q20",
    #   "type": "lintas_minggu",
    #   "source_doc": "week-01.txt (Senin) & week-02.txt (Kamis)",
    #   "question": "Hubungkan metafora Yesus 'makan bersama' manusia (Week 1, Senin) dengan nama khusus 'Imanuel' (Week 2, Kamis) dalam mendefinisikan sifat relasional Allah!",
    #   "reference": "Metafora pada Week 1 menggambarkan kerinduan Yesus untuk duduk bersama, makan, dan melakukan pembicaraan hangat yang melambangkan hubungan sangat dekat, intim, dan terbuka. Sifat relasional yang radikal ini divalidasi secara teologis pada Week 2 melalui nama khusus Yesus, yaitu 'Imanuel', yang berarti 'Allah beserta kita'. Baik tindakan mengetuk pintu untuk makan bersama maupun gelar Imanuel membuktikan satu kebenaran linier: bahwa sejak awal Allah menolak jarak dan selalu rindu untuk hadir secara riil di tengah-tengah kehidupan manusia."
    # },
]


async def get_rag_data(rag, question, mode_name):
    response = await rag.aquery(
        question,
        param=QueryParam(
            mode=mode_name,
            user_prompt="Anda adalah asisten Sekolah Sabat. Jawab berdasarkan fakta database secara singkat, padat, deklaratif, dan hindari penggunaan kata ganti orang."
        )
    )

    if not response:
        return "Tidak ada jawaban", ["Tidak ada konteks"]

    answer = str(response)

    context = await rag.aquery(question, param=QueryParam(mode=mode_name, only_need_context=True))

    return answer, [str(context)]


async def score_sample(question, answer, contexts, reference):
    m1 = Faithfulness(llm=ragas_judge)
    m2 = AnswerRelevancy(llm=ragas_judge, embeddings=ragas_embeddings)
    m3 = ContextRecall(llm=ragas_judge)
    m4 = ContextPrecision(llm=ragas_judge)

    try:
        faith_result = await m1.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        faith_score = faith_result.value
    except Exception as e:
        print(f"    [WARN] Faithfulness error: {e}")
        faith_score = None

    try:
        relevancy_result = await m2.ascore(
            user_input=question,
            response=answer,
        )
        relevancy_score = relevancy_result.value
    except Exception as e:
        print(f"    [WARN] AnswerRelevancy error: {e}")
        relevancy_score = None

    try:
        recall_result = await m3.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
        recall_score = recall_result.value
    except Exception as e:
        print(f"    [WARN] ContextRecall error: {e}")
        recall_score = None

    try:
        precision_result = await m4.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
        precision_score = precision_result.value
    except Exception as e:
        print(f"    [WARN] ContextPrecision error: {e}")
        precision_score = None

    valid_scores = [s for s in [faith_score, relevancy_score, recall_score, precision_score] if s is not None]
    ragas_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return {
        "faithfulness": faith_score,
        "answer_relevancy": relevancy_score,
        "context_recall": recall_score,
        "context_precision": precision_score,
        "ragas_score": ragas_score
    }


async def run_eval_comprehensive(strategy_name, working_dir, target_llm_name, target_llm_func):
    # modes = ["hybrid", "local", "global", "naive", "mix"]
    # modes = ["local", "global", "hybrid", "mix"]
    modes = ["naive"]
    all_mode_dfs = []

    for current_mode in modes:
        print(f"\n[RUNNING] Strategi: {strategy_name} | Generator LLM: {target_llm_name} | Mode: {current_mode}")

        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=target_llm_func,
            embedding_func=gemini_embedding_func,
        )
        await rag.initialize_storages()

        meta_rows = []

        for item in question_dataset:
            start_date = time.time()
            print(f"  [Q] {item['id']} ...")
            ans, ctx = await get_rag_data(rag, item["question"], current_mode)
            end_date = time.time() - start_date

            print(f"  [RAGAS] Scoring {item['id']}...")
            scores = await score_sample(item["question"], ans, ctx, item["reference"])
            meta_rows.append({
                "id":                 item["id"],
                "type":               item["type"],
                "user_input":         item["question"],
                "response":           ans,
                "retrieved_contexts": ctx,
                "reference":          item["reference"],
                "faithfulness":       scores["faithfulness"],
                "answer_relevancy":   scores["answer_relevancy"],
                "context_recall":     scores["context_recall"],
                "context_precision":  scores["context_precision"],
                "ragas_score":        scores["ragas_score"],
                "strategy":           strategy_name,
                "llm":                target_llm_name,
                "mode":               current_mode,
                "timestamp":          datetime.now().isoformat(),
                "latency": end_date
            })

        df = pd.DataFrame(meta_rows)

        desired_columns = [
            "id", "type", "user_input", "retrieved_contexts", "response", "reference",
            "faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score",
            "strategy", "llm", "mode", "timestamp","latency"
        ]
        df = df.reindex(columns=desired_columns)
        all_mode_dfs.append(df)

        await rag.finalize_storages()
        await asyncio.sleep(2)
    return pd.concat(all_mode_dfs, ignore_index=True)

async def main():
    start_time = time.time()
    try:
        from core.deepseek import deepseek_llm
        from core.openai import openai_llm

        llm_targets = [
            {"name": "Gemini-2.5-Flash-Lite", "func": gemini_llm_model_func},
            # {"name": "DeepSeek-V4-Flash",      "func": deepseek_llm},
            # {"name": "GPT-4.1-Mini",            "func": openai_llm},
        ]

        reports = []
        target_storage = {"name": "BATCH", "path": "../final_working_dir"}

        for target in llm_targets:
            df_result = await run_eval_comprehensive(
                strategy_name=target_storage["name"],
                working_dir=target_storage["path"],
                target_llm_name=target["name"],
                target_llm_func=target["func"],
            )
            reports.append(df_result)

        final_report = pd.concat(reports, ignore_index=True)
        csv_filename = "experiment_ragas_new_fixed.csv"

        file_exists = os.path.exists(csv_filename)

        if not file_exists:
            final_report.to_csv(csv_filename, index=False)
            print(f"\n[INFO] Berhasil membuat file laporan baru: {csv_filename}")
        else:
            final_report.to_csv(csv_filename, mode='a', header=False, index=False)
            print(f"\n[INFO] Berhasil menambahkan {len(final_report)} baris baru ke file: {csv_filename}")

        print("\n" + "=" * 70)
        print(f"EVALUASI SELESAI | Waktu Total: {time.time() - start_time:.2f} detik")
        print("=" * 70)

        all_accumulated_data = pd.read_csv(csv_filename)
        summary = all_accumulated_data.groupby(["strategy", "llm", "mode"])[
            ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]
        ].mean()
        print("\n=== REKAPITULASI AKUMULATIF EKSPERIMEN (SELURUH RUN) ===")
        print(summary)

    except Exception as e:
        import traceback
        print(f"\n[CRITICAL ERROR] Kegagalan Eksekusi Eksperimen: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY dulu!")
    elif not os.getenv("GEMINI_API_KEY"):
        print("Error: Set GEMINI_API_KEY dulu!")
    else:
        asyncio.run(main())