import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import subprocess
import re
import os
import shutil
import time
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords as nltk_stopwords
import matplotlib.pyplot as plt
from collections import Counter
import altair as alt
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, InternalServerError

# =======================================================================
# 1. KONFIGURASI MODEL (IndoBERT & Gemini)
# =======================================================================

@st.cache_resource
def load_indobert():
    """Load Model IndoBERT untuk klasifikasi cepat / fallback skala besar."""
    LABEL_MAP = {'LABEL_0': 'positif', 'LABEL_1': 'netral', 'LABEL_2': 'negatif'}
    try:
        model_id = "mdhugol/indonesia-bert-sentiment-classification"
        pipe = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)
        return pipe, LABEL_MAP
    except Exception as e:
        st.sidebar.warning(f"IndoBERT offline/gagal dimuat: {e}")
        return None, None

INDOBERT_PIPE, INDOBERT_MAP = load_indobert()

def analisis_indobert(text):
    """Fungsi fallback cepat menggunakan IndoBERT."""
    if INDOBERT_PIPE and text.strip():
        try:
            res = INDOBERT_PIPE(text[:512])[0]  # Batasi 512 karakter
            return INDOBERT_MAP.get(res['label'], 'netral')
        except Exception:
            return 'netral'
    return 'netral'

@retry(
    wait=wait_exponential(min=4, max=15),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((ResourceExhausted, DeadlineExceeded, InternalServerError))
)
def analisis_gemini_api(text):
    """Analisis Gemini dengan Retry Mechanism."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Analisis sentimen tweet ini. Jawab 1 kata saja: Positif, Negatif, atau Netral.\nTweet: '{text[:500]}'"
        response = model.generate_content(prompt)
        ans = response.text.strip().lower()
        if 'positif' in ans:
            return 'positif'
        if 'negatif' in ans:
            return 'negatif'
        return 'netral'
    except Exception as e:
        raise e

# =======================================================================
# 2. UTILITAS DATA
# =======================================================================

def bersihkan_teks(teks):
    """Pembersihan teks tweet (URL, mention, hashtag, dan karakter khusus)."""
    teks = re.sub(r'http\S+', '', str(teks))
    teks = re.sub(r'@[A-Za-z0-9_]+', '', teks)
    teks = re.sub(r'#\w+', '', teks)
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)
    return teks.lower().strip()

def get_stopwords(extra_words=None):
    """
    Mengambil stopwords standar NLTK + slang / noise words + kata kunci pencarian.
    """
    try:
        sw = set(nltk_stopwords.words('indonesian'))
    except Exception:
        sw = set()

    # Tambahkan Stopwords Bawaan WordCloud
    sw.update(STOPWORDS)

    # Tambahkan Slang / Bahasa Gaul / Noise
    noise_words = [
        'yg', 'utk', 'dgn', 'dr', 'kpd', 'krn', 'tsb', 'kalo', 'kl', 'klo',
        'gak', 'ga', 'gk', 'nggak', 'enggak', 'g', 'tdk',
        'aja', 'saja', 'juga', 'jg', 'udah', 'sdh', 'sudah', 'blm', 'belum',
        'pake', 'pakai', 'bikin', 'buat', 'tetap', 'pasti',
        'sih', 'deh', 'dong', 'kok', 'tuh', 'nih', 'mah', 'kan', 'ya', 'yah',
        'kek', 'kayak', 'macam', 'seperti', 'emang', 'memang', 'bgt', 'banget',
        'lu', 'gw', 'gua', 'gue', 'lo', 'elo', 'aku', 'saya', 'kamu', 'anda',
        'dia', 'mereka', 'kita', 'kami', 'kalian', 'nya', 'mu', 'ku',
        'bisa', 'mau', 'akan', 'sedang', 'lagi', 'ada', 'tapi', 'dan', 'di', 'ke', 'dari',
        'ini', 'itu', 'sini', 'situ', 'sana', 'mana', 'siapa', 'apa', 'kenapa',
        'banyak', 'dikit', 'lebih', 'kurang', 'paling', 'tau', 'tahu', 'amp'
    ]
    sw.update(noise_words)

    # Tambahkan Kata Kunci Pencarian User secara Otomatis
    if extra_words:
        if isinstance(extra_words, list):
            sw.update([w.lower() for w in extra_words])
        else:
            sw.add(str(extra_words).lower())

    return sw

# =======================================================================
# 3. PIPELINE PROSES (LOGIKA UTAMA)
# =======================================================================

def scraping_data(keyword, limit, token):
    """Scraping data dari X (Twitter) menggunakan tweet-harvest via npx."""
    st.info("📡 Tahap 1: Scraping Data dari X...")
    if os.path.exists("tweets-data"):
        shutil.rmtree("tweets-data")

    cmd = [
        "npx", "-y", "tweet-harvest@latest", "-o", "data.csv",
        "-s", keyword, "--tab", "LATEST", "-l", str(limit), "--token", token
    ]
    proses = subprocess.run(cmd, capture_output=True, text=True)

    if not os.path.exists("tweets-data/data.csv"):
        log_error = proses.stderr if proses.stderr.strip() else proses.stdout
        raise Exception(f"Gagal Scraping. Log Asli:\n{log_error}")

    df = pd.read_csv("tweets-data/data.csv")
    return df

def proses_analisis_hybrid(df, keyword_user):
    """Pipeline analisis sentimen hybrid (Gemini + IndoBERT) & topic modeling."""
    st.info("🧠 Tahap 2: Analisis Hybrid (Gemini + IndoBERT)...")

    # 1. Preprocessing
    col_text = 'full_text' if 'full_text' in df.columns else df.columns[0]
    df['text_bersih'] = df[col_text].apply(bersihkan_teks)

    # Stopword Otomatis dari Keyword User
    keyword_list = keyword_user.lower().split()
    custom_stopwords = get_stopwords(extra_words=keyword_list)

    # 2. Smart Sorting berdasarkan engagement jika kolom tersedia
    if 'favorite_count' in df.columns and 'retweet_count' in df.columns:
        df['engagement'] = df['favorite_count'].fillna(0) + df['retweet_count'].fillna(0)
        df = df.sort_values(by='engagement', ascending=False).reset_index(drop=True)

    # 3. Analisis Sentimen
    GEMINI_QUOTA = 50
    sentiments = []
    bar = st.progress(0)
    status_text = st.empty()

    for i in range(len(df)):
        teks = df.loc[i, 'text_bersih']
        hasil = 'netral'
        if i < GEMINI_QUOTA:
            try:
                status_text.text(f"Menganalisis data ke-{i+1} dengan Gemini...")
                hasil = analisis_gemini_api(teks)
                time.sleep(2)  # Delay adaptif antar-request
            except Exception:
                hasil = analisis_indobert(teks)
        else:
            hasil = analisis_indobert(teks)
        sentiments.append(hasil)
        bar.progress((i + 1) / len(df))

    df['sentiment'] = sentiments
    status_text.text("✅ Analisis Selesai.")

    # 4. Topic Modeling (BERTopic)
    st.info("🔎 Menjalankan Topic Modeling...")
    df_topik = pd.DataFrame(columns=['Topik', 'positif', 'netral', 'negatif', 'total'])

    try:
        df_top = df.head(GEMINI_QUOTA)
        if len(df_top) >= 10:
            vec = CountVectorizer(stop_words=list(custom_stopwords), ngram_range=(1, 2))

            topic_model = BERTopic(language="indonesian", vectorizer_model=vec, min_topic_size=3)
            topics, _ = topic_model.fit_transform(df_top['text_bersih'].tolist())

            info = topic_model.get_topic_info()
            top_topics = info[info.Topic != -1].head(5)

            if not top_topics.empty:
                df_top['topic_id'] = topics
                relevant_data = df_top[df_top['topic_id'].isin(top_topics['Topic'])]
                topic_sent = relevant_data.groupby(['topic_id', 'sentiment']).size().unstack(fill_value=0)

                topic_names = {}
                for t_id in top_topics['Topic']:
                    raw_keys = topic_model.get_topic(t_id)
                    keys_str = ", ".join([w[0] for w in raw_keys[:5]])

                    try:
                        prompt_name = f"""
                        Analisis kata kunci ini: {keys_str}.
                        Berikan SATU nama topik singkat (2-4 kata) dalam Bahasa Indonesia yang merangkum isu tersebut.
                        PENTING: JANGAN pakai tanda kutip, JANGAN pakai penomoran, JANGAN ada kata pengantar seperti 'Berikut adalah'.
                        Langsung jawab nama topiknya saja.
                        """
                        resp_name = genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt_name)
                        clean_name = resp_name.text.strip().split('\n')[0].replace('"', '').replace('*', '')

                        if clean_name and clean_name[0].isdigit():
                            clean_name = clean_name.split('. ')[-1]
                    except Exception:
                        clean_name = "_".join([w[0] for w in raw_keys[:2]])

                    topic_names[t_id] = clean_name

                topic_sent.index = topic_sent.index.map(topic_names)
                df_topik = topic_sent.reset_index().rename(columns={'topic_id': 'Topik', 'index': 'Topik'})
                for col in ['positif', 'negatif', 'netral']:
                    if col not in df_topik.columns:
                        df_topik[col] = 0
            else:
                st.warning("BERTopic berjalan tapi tidak menemukan topik dominan.")
        else:
            st.warning("Data terlalu sedikit (<10) untuk Topic Modeling.")

    except Exception as e:
        st.warning(f"Topic Modeling dilewati: {e}")

    return df, df_topik

def generate_laporan_strategis(df, df_topik, keyword):
    """Membuat laporan teks intelijen menggunakan Gemini."""
    stats = df['sentiment'].value_counts().to_dict()
    topik_str = df_topik.to_string() if not df_topik.empty else "Data topik tidak tersedia."

    prompt = f"""
    Peran: Analis Strategis Ditjen Strahan Kemhan RI.
    Topik: "{keyword}".
    Data: Total {len(df)} tweet. Sentimen: {stats}.
    Isu Utama: {topik_str}

    Buat Laporan Singkat dan Terstruktur:
    1. Situasi Terkini (Sentimen publik & isu dominan).
    2. Potensi Risiko terhadap stabilitas/persepsi pertahanan negara.
    3. 3 Rekomendasi Aksi Konkret bagi pengambil kebijakan.
    Gunakan gaya bahasa militer/formal, lugas, dan taktis.
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model.generate_content(prompt).text
    except Exception:
        return "Laporan otomatis tidak tersedia (Limit API tercapai atau koneksi terputus)."

# =======================================================================
# 4. MAIN UI (Pusat Komando Ditjen Strahan)
# =======================================================================

def main():
    st.set_page_config(page_title="Intelijen Strahan - Sentimen Publik", layout="wide", page_icon="🛡️")

    # --- BAGIAN 1: SIDEBAR ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Logo_Kemhan_RI.png/600px-Logo_Kemhan_RI.png", width=100)
        st.title("Pusat Komando")
        st.caption("Ditjen Strahan - Kementerian Pertahanan RI")

        st.subheader("🔐 Kredensial Sistem")
        default_gemini = os.getenv("GEMINI_API_KEY", st.session_state.get('api_key', ''))
        default_twitter = os.getenv("TWITTER_AUTH_TOKEN", st.session_state.get('x_token', ''))

        api_key = st.text_input("Google Gemini API Key", value=default_gemini, type="password")
        if api_key:
            st.session_state.api_key = api_key

        st.markdown("---")
        st.subheader("📡 Sumber Data & Parameter Misi")

        mode_data = st.radio(
            "Pilih Mode Pengambilan Data:",
            ["Scraping X / Twitter (Live)", "Gunakan Sample Dataset Bawaan", "Unggah File CSV Mandiri"],
            index=0
        )

        kw = "Pengadaan Alutsista"
        limit = 100
        x_token = ""
        uploaded_file = None

        if mode_data == "Scraping X / Twitter (Live)":
            x_token = st.text_input("Twitter Auth Token", value=default_twitter, type="password", help="Diperlukan untuk scraping via tweet-harvest")
            if x_token:
                st.session_state.x_token = x_token
            kw = st.text_input("Isu Strategis (Keyword)", value=st.session_state.get('kw', 'Pengadaan Alutsista'))
            limit = st.number_input("Jumlah Data (Tweet)", 20, 500, st.session_state.get('limit', 50))
        elif mode_data == "Gunakan Sample Dataset Bawaan":
            kw = st.text_input("Isu Strategis (Label Topik)", value="Pengadaan Alutsista TNI")
            st.info("ℹ️ Menggunakan dataset sampel isu pertahanan bawaan di `data/sample_alutsista.csv`.")
        elif mode_data == "Unggah File CSV Mandiri":
            kw = st.text_input("Isu Strategis (Label Topik)", value="Analisis Opini Publik")
            uploaded_file = st.file_uploader("Pilih file CSV (harus memiliki kolom 'full_text')", type=["csv"])

        tombol_aksi = st.button("🚀 Jalankan Misi Baru", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("Proyek Portofolio PKL Ditjen Strahan Kemhan")

    # Konfigurasi Gemini (jika API Key tersedia)
    if api_key:
        genai.configure(api_key=api_key)

    # --- BAGIAN 2: LOGIKA EKSEKUSI ---
    if tombol_aksi:
        if not api_key:
            st.error("⚠️ Mohon lengkapi Google Gemini API Key di Sidebar kiri."); st.stop()

        if mode_data == "Scraping X / Twitter (Live)" and not x_token:
            st.error("⚠️ Mohon masukkan Twitter Auth Token untuk mode scraping live."); st.stop()

        st.session_state.kw = kw
        st.session_state.limit = limit
        st.session_state.proc = True

        # Reset riwayat chat agar sinkron dengan data baru
        if 'messages' in st.session_state:
            del st.session_state.messages

        try:
            # 1. Akuisisi Data
            if mode_data == "Scraping X / Twitter (Live)":
                with st.spinner("📡 Mengakuisisi data dari sumber terbuka X (Twitter)..."):
                    df = scraping_data(kw, limit, x_token)
            elif mode_data == "Gunakan Sample Dataset Bawaan":
                sample_path = Path("data/sample_alutsista.csv")
                if not sample_path.exists():
                    st.error("File sampel `data/sample_alutsista.csv` tidak ditemukan."); st.stop()
                df = pd.read_csv(sample_path)
            elif mode_data == "Unggah File CSV Mandiri":
                if uploaded_file is None:
                    st.error("Silakan unggah file CSV terlebih dahulu."); st.stop()
                df = pd.read_csv(uploaded_file)
                if 'full_text' not in df.columns:
                    col_first = df.columns[0]
                    df = df.rename(columns={col_first: 'full_text'})

            # 2. Analisis Hybrid
            with st.spinner("🧠 Menganalisis sentimen & pola narasi (AI Hybrid)..."):
                df_final, df_topik = proses_analisis_hybrid(df, kw)

            # 3. Generate Laporan
            with st.spinner("📝 Menyusun laporan strategis pertahanan..."):
                laporan = generate_laporan_strategis(df_final, df_topik, kw)

            # Simpan ke Session State
            st.session_state.data = {'df': df_final, 'topik': df_topik, 'laporan': laporan}
            st.session_state.proc = False
            st.session_state.done = True
            st.rerun()

        except Exception as e:
            st.error(f"❌ Misi Terhenti: {str(e)}")
            st.session_state.proc = False

    # --- BAGIAN 3: DASHBOARD UTAMA ---
    st.title(f"🛡️ Dashboard Opini Publik: {st.session_state.get('kw', 'Menunggu Perintah')}")

    if st.session_state.get('done'):
        data = st.session_state.data

        # Hitung Data Sentimen
        tot = len(data['df'])
        pos = len(data['df'][data['df']['sentiment'] == 'positif'])
        neg = len(data['df'][data['df']['sentiment'] == 'negatif'])
        net = len(data['df'][data['df']['sentiment'] == 'netral'])

        pct_pos = f"{pos/tot*100:.1f}%" if tot > 0 else "0%"
        pct_neg = f"{neg/tot*100:.1f}%" if tot > 0 else "0%"
        pct_net = f"{net/tot*100:.1f}%" if tot > 0 else "0%"

        def custom_metric(label, value, sub_text, color_hex):
            st.markdown(f"""
            <div style="
                background-color: #1E1E1E;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid {color_hex};
                margin-bottom: 10px;">
                <p style="margin:0; font-size: 14px; color: #A0A0A0;">{label}</p>
                <h2 style="margin:0; font-size: 28px; color: #FFFFFF; font-weight: 600;">{value}</h2>
                <p style="margin:0; font-size: 14px; color: {color_hex}; font-weight: bold;">{sub_text}</p>
            </div>
            """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            custom_metric("Total Data", tot, "Tweet/Komentar", "#F59E0B")
        with c2:
            custom_metric("Positif", pos, pct_pos, "#10B981")
        with c3:
            custom_metric("Negatif", neg, pct_neg, "#EF4444")
        with c4:
            custom_metric("Netral", net, pct_net, "#9CA3AF")

        # Tabs Dashboard
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Laporan Strategis",
            "📊 Visualisasi Peta Isu",
            "💬 Asisten Analis (Live)",
            "🗃️ Data Mentah & Unduh"
        ])

        with tab1:
            st.markdown("### Executive Summary")
            st.info("Laporan ini disusun secara otomatis oleh AI berdasarkan data intelijen sentimen terkini.")
            st.markdown(data['laporan'])

        with tab2:
            st.markdown("### 📊 Peta Visualisasi & Narasi Data")

            keyword_user = st.session_state.get('kw', '')
            with st.spinner("🤖 AI sedang membaca grafik untuk Anda..."):
                try:
                    all_text = " ".join(data['df']['text_bersih'].astype(str))
                    sw_custom = get_stopwords(extra_words=keyword_user.lower().split())
                    words = [w for w in all_text.split() if w.lower() not in sw_custom and len(w) > 2]
                    top_words = Counter(words).most_common(10)
                    top_words_str = ", ".join([w[0] for w in top_words])

                    prompt_explain = f"""
                    Sebagai analis intelijen, berikan penjelasan singkat (maksimal 2 kalimat)
                    mengapa kata-kata berikut muncul dominan dalam isu '{keyword_user}': {top_words_str}.
                    Hubungkan dengan sentimen mayoritas ({'Negatif' if neg > pos else 'Positif'}).
                    """
                    model_explain = genai.GenerativeModel('gemini-2.5-flash')
                    explanation = model_explain.generate_content(prompt_explain).text
                    st.info(f"💡 **Analisis Naratif:** {explanation}")
                except Exception:
                    st.warning("Narasi otomatis tidak dapat dimuat.")

            st.markdown("---")

            col_vis1, col_vis2 = st.columns([1.3, 1])

            with col_vis1:
                st.subheader("1. Peta Topik (Sub-Isu)")
                st.caption("Isu-isu spesifik yang terdeteksi dari kluster percakapan:")
                if not data['topik'].empty and 'Topik' in data['topik'].columns:
                    df_melt = data['topik'].reset_index(drop=True).melt('Topik', var_name='Sentimen', value_name='Jumlah')
                    chart_bar = alt.Chart(df_melt).mark_bar().encode(
                        x='Jumlah',
                        y=alt.Y('Topik', sort='-x', title=None, axis=alt.Axis(labels=True, labelLimit=200)),
                        color=alt.Color('Sentimen', scale=alt.Scale(domain=['positif', 'netral', 'negatif'], range=['#22c55e', '#94a3b8', '#ef4444'])),
                        tooltip=['Topik', 'Sentimen', 'Jumlah']
                    ).properties(height=300)
                    st.altair_chart(chart_bar, use_container_width=True)
                else:
                    st.warning("Data topik tidak cukup untuk membentuk visualisasi bar chart.")

            with col_vis2:
                st.subheader("2. Peta Kata (Word Cloud)")
                try:
                    wc = WordCloud(
                        width=500, height=350, background_color='white',
                        stopwords=sw_custom, colormap='Dark2',
                        max_words=100, contour_width=1, contour_color='steelblue'
                    ).generate(all_text)

                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Word Cloud gagal digenerate: {e}")

            st.subheader("3. Proporsi Sentimen Publik")
            total_data = pos + neg + net
            if total_data > 0:
                df_pie = pd.DataFrame({
                    'Sentimen': ['Positif', 'Netral', 'Negatif'],
                    'Jumlah': [pos, net, neg],
                    'Persentase': [pos/total_data, net/total_data, neg/total_data]
                })

                base = alt.Chart(df_pie).encode(theta=alt.Theta("Jumlah", stack=True))
                pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
                    color=alt.Color("Sentimen", scale=alt.Scale(domain=['Positif', 'Netral', 'Negatif'], range=['#22c55e', '#94a3b8', '#ef4444'])),
                    order=alt.Order("Jumlah", sort="descending"),
                    tooltip=["Sentimen", "Jumlah", alt.Tooltip("Persentase", format=".1%")]
                )
                text_center = alt.Chart(pd.DataFrame({'text': [f"{total_data}"]})).mark_text(
                    size=30, fontWeight='bold', color='gray'
                ).encode(text='text')

                text_labels = base.mark_text(radius=140).encode(
                    text=alt.Text("Persentase", format=".1%"),
                    order=alt.Order("Jumlah", sort="descending"),
                    color=alt.value("black")
                )
                st.altair_chart(pie + text_labels + text_center, use_container_width=True)

        with tab3:
            st.markdown("### 💬 Tanya Jawab Data (Asisten Intelijen Live)")

            if "messages" not in st.session_state:
                context_data = data['df'][['full_text', 'sentiment']].head(30).to_string()
                topik_context = data['topik'].to_string() if not data['topik'].empty else "Tidak ada topik spesifik."

                system_prompt = f"""
                Anda adalah Asisten Intelijen Ditjen Strahan Kemhan RI.
                Tugas Anda adalah menjawab pertanyaan user BERDASARKAN DATA BERIKUT SAJA:

                Isu: {st.session_state.kw}
                Ringkasan Topik: {topik_context}

                Sampel Data Tweet & Sentimen:
                {context_data}

                Jika jawaban tidak ada di data, katakan "Data tidak memuat informasi tersebut". Jawab dengan gaya militer/formal singkat.
                """

                st.session_state.messages = [
                    {"role": "user", "parts": [{"text": system_prompt}]},
                    {"role": "model", "parts": [{"text": "Siap. Saya telah mempelajari data intelijen terbaru. Silakan ajukan pertanyaan."}]}
                ]

            for msg in st.session_state.messages[1:]:
                role_display = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(role_display):
                    st.write(msg["parts"][0]["text"])

            if q := st.chat_input("Contoh: 'Apa keluhan utama masyarakat terkait isu ini?'"):
                st.session_state.messages.append({"role": "user", "parts": [{"text": q}]})
                with st.chat_message("user"):
                    st.write(q)

                with st.spinner("Menganalisis jawaban..."):
                    try:
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        chat = model.start_chat(history=st.session_state.messages)
                        response = chat.send_message(q)

                        st.session_state.messages.append({"role": "model", "parts": [{"text": response.text}]})
                        with st.chat_message("assistant"):
                            st.write(response.text)
                    except Exception as e:
                        st.error(f"Gagal terhubung ke AI: {e}")

        with tab4:
            st.markdown("### 🗃️ Data Mentah & Unduh Laporan")
            st.dataframe(data['df'], use_container_width=True)

            csv = data['df'].to_csv(index=False).encode('utf-8')
            keyword_file = st.session_state.get('kw', 'Data_Intelijen')
            nama_file_bersih = keyword_file.replace(' ', '_')

            st.download_button(
                label="📥 Unduh Laporan (.csv)",
                data=csv,
                file_name=f"Laporan_{nama_file_bersih}.csv",
                mime='text/csv',
                type="primary"
            )

    else:
        st.info("👈 Silakan pilih Sumber Data, masukkan kredensial di sidebar, lalu klik '🚀 Jalankan Misi Baru'.")
        st.markdown("""
        ### 📋 Panduan Singkat Operasional:
        1. **Pilih Sumber Data**:
           - **Scraping X / Twitter (Live)**: Membutuhkan Google Gemini API Key & Twitter Auth Token.
           - **Gunakan Sample Dataset Bawaan**: Langsung mencoba dengan data isu pertahanan (`data/sample_alutsista.csv`).
           - **Unggah File CSV Mandiri**: Menggunakan file CSV lokal milik Anda (kolom `full_text`).
        2. Masukkan **Google Gemini API Key** di menu sebelah kiri.
        3. Tentukan **Isu Strategis / Kata Kunci**.
        4. Klik tombol **🚀 Jalankan Misi Baru** untuk memulai orkestrasi AI.
        """)

if __name__ == "__main__":
    main()
