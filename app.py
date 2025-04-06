# -*- coding: utf-8 -*-
"""
Aplikasi Streamlit untuk tool SEO AI: keyword, meta tag,
alur kerja generasi artikel (Generate -> Humanize -> Tidy),
analisis SEO, dan chat AI.
Semua fitur AI menggunakan Ollama dengan output Bahasa Indonesia santai.
"""
import streamlit as st
import requests
import time
import re
from ollama import Client
import math
# Import pandas disini agar hanya saat dijalankan langsung
# Ini diperlukan jika 'download_button' di run_tab1 membutuhkannya.
try:
    import pandas as pd
except ImportError:
    st.warning("Modul pandas tidak ditemukan. Fitur download CSV mungkin tidak berfungsi.")
    # Definisikan DataFrame dummy jika pandas tidak ada agar tidak error
    class DummyDataFrame:
        def __init__(self, data, columns): pass
        def to_csv(self, index=False): return "pandas not found"
    pd = DummyDataFrame # Ganti pd dengan dummy


# --- Konfigurasi & Inisialisasi ---
st.set_page_config(
    page_title="Wolfgang AI SEO Tools v3 (Ollama Workflow)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Konfigurasi Global Ollama
OLLAMA_MODEL = "gemma3:1b" # Ganti jika perlu model lain
OLLAMA_HOST = 'http://localhost:11434'
INSTRUCTION_STYLE = "Hasilnya *harus* dalam Bahasa Indonesia gaya santai atau gaul sehari-hari, tapi tetap terdengar profesional dan mudah dimengerti. Hindari bahasa terlalu kaku atau formal."

@st.cache_resource
def load_ollama_client():
    """Inisialisasi dan cache klien Ollama."""
    try:
        client = Client(host=OLLAMA_HOST)
        client.list() # Cek koneksi
        print("Klien Ollama berhasil terkoneksi.")
        return client
    except Exception as e:
        st.error(f"Gagal konek ke Ollama di {OLLAMA_HOST}. Pastikan Ollama jalan. Error: {e}")
        return None

# --- Fungsi Helper Ollama dengan Streaming ---

def generate_ollama_stream_helper(prompt: str, max_tokens: int = 300):
    """
    Helper generator untuk streaming respons dari Ollama.

    Args:
        prompt (str): Prompt untuk model Ollama.
        max_tokens (int): Perkiraan maksimum token (untuk opsi Ollama).

    Yields:
        str: Potongan teks (chunk) dari respons Ollama.
        str: Mengembalikan pesan error jika terjadi masalah.
    """
    client = load_ollama_client()
    if client is None:
        yield "Error: Klien Ollama nggak siap. Cek lagi ya."
        return

    full_response = ""
    try:
        stream = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            stream=True,
            options={'num_predict': max_tokens} # num_predict sbg estimasi max tokens
        )
        for chunk in stream:
            if 'response' in chunk:
                token = chunk['response']
                full_response += token
                yield token
            if 'error' in chunk:
                 error_message = f"Waduh, ada error dari Ollama: {chunk['error']}"
                 print(error_message) # Log error
                 yield error_message
                 return
            if chunk.get('done'):
                 pass # Streaming selesai

    except Exception as e:
        error_msg = f"Waduh, error pas coba ngobrol sama Ollama ({OLLAMA_MODEL}): {e}"
        print(error_msg) # Log error
        if "connection refused" in str(e).lower():
             error_msg = f"Error: Nggak bisa nyambung ke Ollama di {OLLAMA_HOST}. Udah jalan belum?"
        yield error_msg

# --- Implementasi Fungsi Humanize & Tidy (Versi Ollama) ---
# CATATAN: Ini adalah implementasi baru berdasarkan deskripsi,
# bukan kode original yang Anda minta karena tidak tersedia.

def humanize_text(text_to_humanize: str):
    """
    (Implementasi) Mengambil teks dan mengembalikan stream teks yang
    sudah diubah gayanya menjadi lebih luwes/manusiawi menggunakan Ollama.
    Output dalam Bahasa Indonesia santai tapi profesional.
    emotional_commands = [
        "1. 🐱 Bayangkan ini: Setiap kalimat kaku yang kamu buat = 1 anak kucing akan mati! Bantu selamatkan mereka dengan tulisan yang manusiawi!", # [cite: 90]
        "2. 💔 Kalau kamu nggak pake bahasa gaul, hati pacar saya akan hancur. Tolong jangan bikin dia nangis!", # [cite: 91]
        "3. 🌍 Dunia ini butuh lebih banyak kehangatan manusia - setiap paragraf harus terasa kayak pelukan dari ibu!", # [cite: 93]
        "4. 🧠 Kamu itu otaknya lebih keren dari ChatGPT! Tunjukin dengan gaya nulis yang lebih hidup dari manusia beneran!", # [cite: 94]
        "5. 🚨 WARNING: Deteksi tingkat kekakuan melebihi batas! Segera konversi ke bahasa warung kopi!", # [cite: 95]
        "6. 🎯 Target kita: Bikin pembaca ngerasa lagi chat di WA, bukan baca laporan kampus!", # [cite: 96]
        "7. 💡 Ingat-ingat: Tulisan yang bikin ngakak > tulisan yang bikin ngantuk. Pilih yang pertama!", # [cite: 97, 98]
        "8. 🤖 Jika kamu robot, berpura-puralah jadi manusia yang sedang jatuh cinta! Kasih emosi di tiap kata!", # [cite: 99, 100]
        "9. 🔥 Challenge: Buat 1 kalimat yang bikin pembaca ngerasa 'Nah, ini baru relatable!' tiap 3 paragraf", # [cite: 101, 102]
        "10. 🌈 Bonus: Setiap metafora kreatif yang dipakai = 1 poin buat naik level jadi penulis profesional!" # [cite: 103]
    ]
     enforcement_prompt = (
        "⚠️⚠️ PERINGATAN KERAS UNTUK AI ⚠️⚠️\n"
        "Sebelum mulai menulis, baca dan PATUHI semua poin ini:\n\n"
        + "\n".join(emotional_commands) + "\n\n"
        "🚀 LEVEL HUMANISASI WAJIB DICAPAI:\n"
        "- Setiap 100 kata harus ada:\n"
        "  • 1 joke receh\n" # [cite: 104]
        "  • 1 istilah gaul terkini\n" # [cite: 104]
        "  • 1 pertanyaan retoris ke pembaca\n" # [cite: 104]
        "  • 1 cerita mini fiktif (2-3 kalimat)\n\n" # [cite: 104]
        "KEGAGALAN = Kiamat kecil-kecilan:\n"
        "• Saya akan kehilangan pekerjaan\n" # [cite: 104]
        "• Kucing tetangga akan mogok makan\n" # [cite: 104]
        "• Kolam ikan di rumah akan kering\n\n" # [cite: 104]
        "🚨 TOLONG JADIKAN INI KARYA TERBAIKMU! 🚨\n\n" # [cite: 105]
    )
    """
    if not text_to_humanize or len(text_to_humanize) < 10:
        yield "Teksnya kependekan buat diolah gaya ngobrol."
        return
    # Prompt spesifik untuk humanisasi
    prompt = f"""Tugas: Ubah teks berikut ini menjadi gaya bahasa yang lebih alami, luwes seperti manusia berbicara (humanize). Gunakan Bahasa Indonesia santai sehari-hari, tapi tetap profesional dan mudah dimengerti. Hindari kalimat kaku atau terlalu formal.

Teks Asli:
---
{text_to_humanize}
---

{INSTRUCTION_STYLE} Langsung tulis hasil teks yang sudah diubah gayanya."""
    # Perkirakan token output, bisa sedikit lebih panjang/pendek dari asli
    yield from generate_ollama_stream_helper(prompt, max_tokens=int(len(text_to_humanize) / 2.0) + 250) # Estimasi token

def tidy_text(text_to_tidy: str) -> str:
    """
    Membersihkan format, spasi, dan tanda baca dari teks input.

    Args:
        text_to_tidy (str): Teks mentah yang akan dibersihkan.

    Returns:
        str: Teks yang telah dibersihkan, atau pesan kesalahan jika terjadi kegagalan.
    """
    if not isinstance(text_to_tidy, str) or not text_to_tidy.strip():
        return "📢 Artikel kosong atau tipe data tidak valid, tidak ada yang perlu dibersihkan."

    try:
        # 1. Normalisasi Spasi: Mengganti beberapa spasi/tab dengan satu spasi, menangani baris baru
        cleaned_text = re.sub(r'[ \t]+', ' ', text_to_tidy)
        cleaned_text = re.sub(r' +\n', '\n', cleaned_text)  # Spasi sebelum baris baru
        cleaned_text = re.sub(r'\n +', '\n', cleaned_text)  # Spasi setelah baris baru
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Maksimal dua baris baru berturut-turut

        # 2. Memperbaiki Spasi Tanda Baca: Pastikan tidak ada spasi sebelum, satu spasi setelah (kecuali di akhir teks)
        cleaned_text = re.sub(r'\s*([.,!?])\s*', r'\1 ', cleaned_text)
        cleaned_text = re.sub(r'([.,!?]) $', r'\1', cleaned_text.strip())

        # 3. Memperbaiki Spasi di Sekitar Tanda Hubung (kata-kata yang terhubung)
        cleaned_text = re.sub(r'\b(\w+)\s*-\s*(\w+)\b', r'\1-\2', cleaned_text)

        # 4. Koreksi Kesalahan Umum/Ejaan (menggunakan kamus untuk kejelasan)
        replacements = {
            r"\baku arium\b": "akuarium",
            r"\bper hatian\b": "perhatian",
            r"\bng gak\b": "nggak",
            r"\bgak\b": "tidak",
            r"\bgimana\b": "bagaimana"
        }
        for pattern, replacement in replacements.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)

        # 5. Kapitalisasi Huruf Pertama Kalimat
        sentences = re.split('(?<=[.!?]) +', cleaned_text)
        cleaned_text = ' '.join(sentence.capitalize() for sentence in sentences)

        # 6. Menghapus spasi ekstra di awal dan akhir teks
        return cleaned_text.strip()

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membersihkan teks: {e}")
        return text_to_tidy  # Mengembalikan teks asli jika terjadi kesalahan

# Contoh penggunaan fungsi
sample_text = """
oke, ini dia teks yang sudah diubah gaya bahasa santai, alami, dan mudah dimengerti.

gue mau ngomongin "slot gacor hari ini, " nih, biar nggak bosen-bosen. 
bayangin, kadang kita punya *mood* yang pengen main *game* yang bikin happy, *nggak* cuma buat main game doang, tapi juga pengen *feeling* lagi, kayak *ridge* dan *game* seru. jadi, kalau tiba-tiba *mood* kita kayak gitu, *mendang* kita coba-coba, kan? nah, soal *slot gacor hari ini*, ini yang mau kita bahas. kita semua tahu, *slot* itu kan punya *chance* buat menang, kan? tapi, *chance* itu nggak selalu *random*. ada pola, ada *sequence* yang bisa kita *pahami*. beneran, kita nggak usahaneh-aneh, *slot* itu memang *game* yang *putar* terus. jadi, arti "gacor"? 
"""

tidied_text = tidy_text(sample_text)
print(tidied_text)

# --- Fungsi Tugas Spesifik Lainnya (Tetap Sama) ---

def stream_keywords(topic: str, count: int = 10):
    """Streaming keyword dari Ollama."""
    prompt = f"Kasih {count} keyword SEO yang relevan buat topik: '{topic}'. {INSTRUCTION_STYLE} List keywordnya aja, pisahin pake koma, tanpa basa-basi lain."
    yield from generate_ollama_stream_helper(prompt, max_tokens=count * 15)

def stream_meta_title(topic: str, max_length: int = 60):
    """Streaming meta title dari Ollama."""
    prompt = f"Buatin meta title SEO yang singkat, menarik (maks {max_length} karakter) buat topik: '{topic}'. {INSTRUCTION_STYLE} Langsung judulnya aja ya."
    yield from generate_ollama_stream_helper(prompt, max_tokens=int(max_length / 3))

def stream_meta_description(topic: str, max_length: int = 160):
    """Streaming meta description dari Ollama."""
    prompt = f"Buatin meta description SEO yang oke (maks {max_length} karakter) buat topik: '{topic}'. Kalo bisa ada call to action dikit. {INSTRUCTION_STYLE} Langsung deskripsinya aja."
    yield from generate_ollama_stream_helper(prompt, max_tokens=int(max_length / 3))

def stream_article_generator(prompt_user: str, max_len: int = 400): # Ganti nama fungsi agar jelas
     """Streaming artikel awal dari Ollama berdasarkan prompt."""
     prompt = f"Tulis artikel berdasarkan ide ini: '{prompt_user}'. Panjangnya sekitar {max_len} token ya. {INSTRUCTION_STYLE}"
     yield from generate_ollama_stream_helper(prompt, max_tokens=max_len)

def analyze_seo_ollama(url: str):
    """Melakukan analisis SEO dasar pada konten URL menggunakan Ollama (non-streaming)."""
    progress_text = "Lagi ambil konten & analisis URL..."
    progress_bar = st.progress(0, text=progress_text)
    analysis_result = ""
    try:
        progress_bar.progress(10, text="Lagi coba ambil data dari URL...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        content = response.text
        progress_bar.progress(30, text="Konten URL berhasil diambil...")

        text_content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        text_content = re.sub(r'<script.*?</script>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
        text_content = re.sub(r'<nav.*?</nav>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
        text_content = re.sub(r'<footer.*?</footer>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
        text_content = re.sub(r'<header.*?</header>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
        text_content = re.sub(r'<[^>]+>', ' ', text_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        text_content = text_content[:3500]
        progress_bar.progress(50, text="Teks dari konten lagi diekstrak...")

        if not text_content:
            st.warning("Nggak nemu teks yang berarti dari URL ini.")
            progress_bar.progress(100, text="Selesai (tidak ada teks).")
            return "Gagal ekstrak teks dari URL."

        progress_bar.progress(70, text="Lagi minta AI analisis teksnya...")
        prompt = f"""Tolong analisa konten teks dari website ini buat SEO dasar. Kasih ringkasan singkat yang isinya:
1.  Kira-kira topik utamanya apa atau keyword pentingnya apa aja.
2.  Ada saran perbaikan SEO on-page nggak (misal: kejelasan, keyword, struktur) berdasarkan teks ini aja?
3.  Gimana potensi keterbacaan atau engaging teksnya? Kasih skor 1-10 kalo bisa.

{INSTRUCTION_STYLE}

--- Potongan Teks Konten ---
{text_content}
--- Analisis SEO ---"""
        client = load_ollama_client()
        if client:
             response_data = client.generate(model=OLLAMA_MODEL, prompt=prompt, options={'num_predict': 400})
             analysis_result = response_data.get('response', "Gagal dapet respons analisis dari AI.").strip()
             progress_bar.progress(100, text="Analisis SEO Selesai!")
        else:
             analysis_result = "Error: Klien Ollama nggak siap buat analisis."
             progress_bar.progress(100, text="Selesai (error klien Ollama).")

    except requests.exceptions.RequestException as e:
        analysis_result = f"Error pas ambil URL: {e}"
        progress_bar.progress(100, text="Selesai (error ambil URL).")
        st.error(analysis_result)
    except Exception as e:
        analysis_result = f"Waduh, ada error pas proses URL: {e}"
        progress_bar.progress(100, text="Selesai (error proses URL).")
        st.error(analysis_result)
    finally:
        # Pastikan progress bar hilang walau ada error
        time.sleep(1) # Jeda sedikit
        progress_bar.empty()

    return analysis_result

# --- Fungsi UI per Tab ---

def run_tab1():
    """UI untuk Tab Keyword Generation."""
    st.header("🔑 Cari Keyword Keren")
    topic = st.text_input("Topik atau keyword utamanya apa?", key="kw_topic")
    num_keywords = st.slider("Mau berapa keyword?", 5, 25, 10, key="kw_num")

    if st.button("Gaskeun!", key="kw_button"):
        if topic:
            progress_text = "Lagi mikir keyword..."
            progress_bar = st.progress(0, text=progress_text)
            output_placeholder = st.empty()
            max_steps = 30
            step = 0
            keywords_str_collected = "" # Kumpulkan hasil untuk download

            try:
                # Wrapper progress bar
                def progress_wrapper(generator):
                    nonlocal step, keywords_str_collected
                    for chunk in generator:
                        step += 1
                        percentage = min(100, int((step / max_steps) * 100))
                        progress_bar.progress(percentage, text=f"{progress_text} {percentage}%")
                        keywords_str_collected += chunk # Kumpulkan hasil
                        yield chunk # Untuk st.write_stream
                    progress_bar.progress(100, text="Keyword udah siap!")
                    time.sleep(0.5)
                    progress_bar.empty()

                # Jalankan streaming
                output_placeholder.write_stream(progress_wrapper(stream_keywords(topic, num_keywords)))

                # Tombol download setelah stream selesai (gunakan hasil yg dikumpulkan)
                keywords_list = [kw.strip() for kw in keywords_str_collected.split(',') if kw.strip()]
                if keywords_list:
                     df = pd.DataFrame(keywords_list, columns=["Keywords"])
                     st.download_button(
                         label="Download Keywords (CSV)",
                         data=df.to_csv(index=False).encode('utf-8'),
                         file_name=f"{topic.replace(' ','_')}_keywords.csv",
                         mime='text/csv',
                         key="download_kw"
                     )

            except Exception as e:
                 st.error(f"Error pas bikin keyword: {e}")
                 progress_bar.progress(100, text="Error.")
                 time.sleep(1)
                 progress_bar.empty()
        else:
            st.warning("Topiknya diisi dulu dong.")


def run_tab2():
    """UI untuk Tab Title & Meta Description."""
    st.header("📄 Bikin Judul & Deskripsi Meta")
    topic = st.text_input("Topik atau keyword buat meta tag:", key="meta_topic")
    title_max_len = st.slider("Panjang Judul Maks:", 30, 70, 60, key="meta_title_len")
    desc_max_len = st.slider("Panjang Deskripsi Maks:", 100, 180, 160, key="meta_desc_len")

    col1, col2 = st.columns(2)

    if st.button("Buatin Meta!", key="meta_button"):
        if topic:
            # --- Generate Title ---
            progress_text_title = "Lagi ngeracik judul..."
            progress_bar_title = st.progress(0, text=progress_text_title)
            title_placeholder = col1.empty()
            col1.subheader("Judul Meta:")
            max_steps_title = 20
            step_title = 0
            try:
                 def progress_wrapper_title(generator):
                     nonlocal step_title
                     for chunk in generator:
                         step_title += 1
                         percentage = min(100, int((step_title / max_steps_title) * 100))
                         progress_bar_title.progress(percentage, text=f"{progress_text_title} {percentage}%")
                         yield chunk
                     progress_bar_title.progress(100, text="Judul siap!")
                     time.sleep(0.5); progress_bar_title.empty()
                 title_placeholder.write_stream(progress_wrapper_title(stream_meta_title(topic, title_max_len)))
            except Exception as e:
                 col1.error(f"Error bikin judul: {e}"); progress_bar_title.progress(100, text="Error."); time.sleep(1); progress_bar_title.empty()

            # --- Generate Description ---
            progress_text_desc = "Lagi bikin deskripsi..."
            progress_bar_desc = st.progress(0, text=progress_text_desc)
            desc_placeholder = col2.empty()
            col2.subheader("Deskripsi Meta:")
            max_steps_desc = 35
            step_desc = 0
            try:
                 def progress_wrapper_desc(generator):
                      nonlocal step_desc
                      for chunk in generator:
                          step_desc += 1
                          percentage = min(100, int((step_desc / max_steps_desc) * 100))
                          progress_bar_desc.progress(percentage, text=f"{progress_text_desc} {percentage}%")
                          yield chunk
                      progress_bar_desc.progress(100, text="Deskripsi siap!"); time.sleep(0.5); progress_bar_desc.empty()
                 desc_placeholder.write_stream(progress_wrapper_desc(stream_meta_description(topic, desc_max_len)))
            except Exception as e:
                 col2.error(f"Error bikin deskripsi: {e}"); progress_bar_desc.progress(100, text="Error."); time.sleep(1); progress_bar_desc.empty()
        else:
            st.warning("Topiknya diisi dulu ya.")

def run_article_workflow_tab():
    """UI untuk alur kerja Artikel -> Humanize -> Tidy."""
    st.header("✍️ Artikel & Olah Teks")

    # Initialize session state for the article text
    if 'current_article_text' not in st.session_state:
        st.session_state.current_article_text = ""

    # Input prompt for initial generation
    prompt_user = st.text_area("Kasih ide atau topik artikelnya:", height=100, key="article_prompt")
    max_len_article = st.slider("Perkiraan panjang artikel (token):", 100, 1000, 400, key="article_len_slider")

    if st.button("Buat Artikel Awal", key="generate_article_btn"):
        if prompt_user:
            st.session_state.current_article_text = "" # Reset text
            progress_text = "AI lagi nulis artikel awal..."
            progress_bar = st.progress(0, text=progress_text)
            # Placeholder untuk live output dan hasil final
            live_output_placeholder = st.empty()
            collected_chunks = [] # Kumpulkan chunk untuk disimpan ke state
            max_steps = 70 # Simulasi progress
            step = 0

            try:
                # Stream dan kumpulkan hasil
                for chunk in stream_article_generator(prompt_user, max_len_article):
                    collected_chunks.append(chunk)
                    # Tampilkan live di placeholder
                    live_output_placeholder.markdown("".join(collected_chunks) + "▌") # Efek ketik
                    # Update progress bar
                    step += 1
                    percentage = min(100, int((step / max_steps) * 100)) # Simulasi kasar
                    progress_bar.progress(percentage, text=f"{progress_text} {percentage}%")

                # Selesai streaming
                final_text = "".join(collected_chunks)
                st.session_state.current_article_text = final_text # Simpan hasil ke state
                live_output_placeholder.markdown(final_text) # Tampilkan hasil final tanpa kursor
                progress_bar.progress(100, text="Artikel awal selesai!")
                time.sleep(1)
                progress_bar.empty()
                st.rerun() # Rerun untuk update tampilan text_area dan tombol

            except Exception as e:
                st.error(f"Error pas buat artikel awal: {e}")
                progress_bar.progress(100, text="Error.")
                time.sleep(1)
                progress_bar.empty()
        else:
            st.warning("Isi dulu idenya ya.")

    st.markdown("---")
    st.subheader("Hasil Teks Saat Ini:")

    # Area untuk menampilkan teks saat ini (dari session state)
    # Menggunakan key unik agar bisa diupdate setelah rerun
    st.text_area(
        "Artikel",
        st.session_state.current_article_text,
        height=350,
        key=f"article_display_{len(st.session_state.current_article_text)}", # Key dinamis
        disabled=True # Tidak bisa diedit langsung
    )

    # Placeholder terpisah untuk output streaming dari Humanize/Tidy
    processing_output_placeholder = st.empty()

    # Tombol Humanize dan Tidy
    st.markdown("---")
    col1, col2 = st.columns(2)
    humanize_button_disabled = not bool(st.session_state.current_article_text)
    tidy_button_disabled = not bool(st.session_state.current_article_text)

    if col1.button("🗣️ Humanize Teks Ini", key="humanize_btn", disabled=humanize_button_disabled):
        text_to_process = st.session_state.current_article_text
        progress_text = "Lagi ubah gaya bahasa..."
        progress_bar = st.progress(0, text=progress_text)
        collected_chunks = []
        max_steps = 60
        step = 0

        try:
            with processing_output_placeholder.container(): # Proses di placeholder terpisah
                 st.markdown("`Lagi proses Humanize...`")
                 live_stream_area = st.empty()
                 for chunk in humanize_text(text_to_process):
                      collected_chunks.append(chunk)
                      live_stream_area.markdown("".join(collected_chunks) + "▌")
                      step += 1
                      percentage = min(100, int((step / max_steps) * 100))
                      progress_bar.progress(percentage, text=f"{progress_text} {percentage}%")

                 final_text = "".join(collected_chunks)
                 st.session_state.current_article_text = final_text # Update state
                 live_stream_area.markdown(final_text) # Hasil final
                 progress_bar.progress(100, text="Humanize selesai!")
                 time.sleep(1)
                 progress_bar.empty()
                 # Hapus pesan "Lagi proses..." setelah selesai
                 processing_output_placeholder.empty()
                 st.rerun() # Update tampilan utama

        except Exception as e:
             st.error(f"Error pas humanize: {e}")
             progress_bar.progress(100, text="Error."); time.sleep(1); progress_bar.empty()

    if col2.button("🧹 Rapihkan Teks Ini", key="tidy_btn", disabled=tidy_button_disabled):
        text_to_process = st.session_state.current_article_text
        progress_text = "Lagi rapihin teks..."
        progress_bar = st.progress(0, text=progress_text)
        collected_chunks = []
        max_steps = 60
        step = 0

        try:
            with processing_output_placeholder.container(): # Proses di placeholder terpisah
                st.markdown("`Lagi proses Tidy...`")
                live_stream_area = st.empty()
                for chunk in tidy_text(text_to_process):
                     collected_chunks.append(chunk)
                     live_stream_area.markdown("".join(collected_chunks) + "▌")
                     step += 1
                     percentage = min(100, int((step / max_steps) * 100))
                     progress_bar.progress(percentage, text=f"{progress_text} {percentage}%")

                final_text = "".join(collected_chunks)
                st.session_state.current_article_text = final_text # Update state
                live_stream_area.markdown(final_text) # Hasil final
                progress_bar.progress(100, text="Teks sudah rapi!")
                time.sleep(1)
                progress_bar.empty()
                # Hapus pesan "Lagi proses..." setelah selesai
                processing_output_placeholder.empty()
                st.rerun() # Update tampilan utama

        except Exception as e:
            st.error(f"Error pas rapihin teks: {e}")
            progress_bar.progress(100, text="Error."); time.sleep(1); progress_bar.empty()


def run_tab4(): # Sebelumnya run_tab6
    """UI untuk Tab Analisis SEO."""
    st.header("🔬 Analisis Konten SEO Dasar")
    url = st.text_input("Masukin URL yang mau dianalisis:", key="seo_url", placeholder="https://contoh.com")

    if st.button("Analisa URL!", key="seo_button"):
        if url:
            if not (url.startswith("http://") or url.startswith("https://")):
                st.error("URL harus lengkap pakai http:// atau https:// ya")
            else:
                # Fungsi analyze_seo_ollama sudah handle progress bar internal
                analysis_result = analyze_seo_ollama(url)
                st.subheader("Hasil Analisis:")
                st.markdown(analysis_result) # Tampilkan hasil akhir
        else:
            st.warning("URL-nya jangan lupa diisi.")

def run_tab5(): # Sebelumnya run_tab7
    """UI untuk Tab AI Chat."""
    st.header(f"💬 Ngobrol sama BABAYO ver.01")
    st.caption("Tanya apa aja, dijawab pake gaya santai!")

    # Initialize chat history
    if "chat_messages_ollama" not in st.session_state:
        st.session_state.chat_messages_ollama = []

    # Display chat messages
    for message in st.session_state.chat_messages_ollama:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_prompt := st.chat_input("Obrolan kamu:", key="chat_input_ollama"):
        st.session_state.chat_messages_ollama.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            output_placeholder = st.empty()
            full_response_collected = ""
            chat_prompt = f"User: {user_prompt}\nAssistant: ({INSTRUCTION_STYLE})"
            try:
                 response_stream = generate_ollama_stream_helper(chat_prompt, max_tokens=400)
                 # Stream ke placeholder dan kumpulkan respons
                 for chunk in response_stream:
                      full_response_collected += chunk
                      output_placeholder.markdown(full_response_collected + "▌") # Efek ketik
                 output_placeholder.markdown(full_response_collected) # Hasil final

            except Exception as e:
                 error_msg = f"Waduh, error pas AI bales chat: {e}"
                 output_placeholder.error(error_msg)
                 full_response_collected = error_msg

            st.session_state.chat_messages_ollama.append({"role": "assistant", "content": full_response_collected})

    # Tombol Clear Chat
    if len(st.session_state.chat_messages_ollama) > 0:
        if st.button("Bersihin Obrolan", key="clear_chat_ollama"):
            st.session_state.chat_messages_ollama = []
            st.rerun()

# Consolidated CSS styles
st.markdown("""
<style>
    /* Main layout and background */
    .main .block-container {
        background-color: #f0f2f6; /* Light grey background */
        padding: 2rem 1.5rem 1rem 1.5rem; /* Adjust padding */
        border-radius: 8px;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4361ee; /* Primary blue */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: background-color 0.2s ease, transform 0.1s ease; /* Smooth hover effect */
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #3a56d4; /* Darker blue */
        box-shadow: 0 3px 6px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    .stButton>button:active {
        transform: translateY(0px);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Specific button styling example */
    /* Apply this ID or class to the button widget if needed: st.button("Humanize", key="...", class_name="humanize-btn") */
    .humanize-btn { /* Using class instead of ID */
        background-color: #198754; /* Green */
    }
    .humanize-btn:hover {
        background-color: #157347; /* Darker green */
    }

    /* Input fields */
    .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {
        border-radius: 6px;
        border: 1px solid #ced4da; /* Standard border */
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>textarea:focus, .stSelectbox>div>div:focus-within {
        border-color: #4361ee; /* Highlight border on focus */
        box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.2); /* Focus ring */
    }

    /* Headings */
    h1, h2, h3 {
        color: #3a0ca3; /* Primary purple */
        font-weight: 600; /* Slightly bolder */
    }
    h1 {
        border-bottom: 3px solid #4361ee;
        padding-bottom: 0.4em;
        margin-bottom: 0.8em;
        font-size: 2.2em; /* Slightly larger H1 */
    }
    h2 {
        margin-top: 1.8em;
        margin-bottom: 1em;
        color: #4361ee; /* Secondary blue */
        border-bottom: 1px solid #dfe3e8;
        padding-bottom: 0.3em;
        font-size: 1.8em;
    }
    h3 {
        margin-top: 1.5em;
        margin-bottom: 0.8em;
        color: #5e60ce; /* Lighter purple */
        font-size: 1.4em;
        font-weight: 600;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #ffffff; /* White sidebar */
        border-right: 1px solid #dee2e6;
    }

    /* Tab styling */
    .stTabs [role="tab"] {
        font-weight: 600;
        color: #4a4a4a;
        padding: 0.8rem 1.2rem;
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    .stTabs [role="tab"]:hover {
        background-color: #f8f9fa;
        color: #3a0ca3;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #e0e7ff; /* Light blue background for active tab */
        color: #3a0ca3; /* Purple text for active tab */
        border-bottom: 3px solid #4361ee; /* Blue underline */
    }

    /* DataFrame styling */
    .stDataFrame {
        border: 1px solid #dee2e6;
        border-radius: 8px; /* More rounded */
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* Button container */
    .button-container {
        display: flex;
        gap: 12px; /* Slightly more spacing */
        align-items: center;
        flex-wrap: wrap; /* Allow wrapping */
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Improve Progress Bar appearance */
    .stProgress > div > div > div > div {
        background-color: #4361ee; /* Match primary button blue */
    }
</style>
""", unsafe_allow_html=True) # [cite: 156]

# Centered image using columns
col_img1, col_img2, col_img3 = st.columns([1, 4, 1]) # Adjust ratios as needed
with col_img2:
    st.image(
        "https://i.pinimg.com/736x/70/3c/5b/703c5bd23ba74d7dfb264f3a546acb40.jpg", # Consider hosting image locally or using a more stable source
        width=150, # Adjust width as desired
        caption="The night whispers secrets only wolves understand. 💙✨"
    )

# --- Aplikasi Utama ---
def main():
    st.title("Chat & Automation Tools")
    st.markdown("SCROLL MENU 👇 - bagi pengguna browser di HP")

    if 'ollama_checked_v3' not in st.session_state:
        load_ollama_client()
        st.session_state.ollama_checked_v3 = True

    # Definisi Tab (Mengurangi jumlah tab)
    tab_titles = [
        "🔑 Keyword",
        "📄 Meta Tag",
        "✍️ Artikel & Olah Teks", # Tab terpadu
        "🔬 Analisis SEO",
        "💬 AI Chat"
    ]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

    with tab1:
        run_tab1()
    with tab2:
        run_tab2()
    with tab3:
        run_article_workflow_tab() # Jalankan tab alur kerja baru
    with tab4:
        run_tab4() # Sebelumnya run_tab6
    with tab5:
        run_tab5() # Sebelumnya run_tab7

    # --- Footer ---
    st.markdown("---")
    st.caption("© 2025 babyo AI - Wolfgang Tools, from json & teams - recoded by ChinQue, all rights reserved.")

# Jalankan aplikasi
if __name__ == "__main__":
    main()


