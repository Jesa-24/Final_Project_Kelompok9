"""
app.py - RAG Expert System (Gemini Version)
===========================================
Streamlit UI untuk RAG menggunakan Google Gemini API.

Jalankan: streamlit run app.py
"""

import os
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from rag.document_loader import DocumentLoader
from rag.text_splitter import TextChunker
from rag.vector_store import VectorStoreManager
from rag.rag_chain import RAGChain
from utils.helpers import (
    check_gemini_api_key,
    get_gemini_models,
    normalize_gemini_model,
    save_uploaded_file,
    get_documents_info,
    GEMINI_API_KEY_GUIDE,
)

# -- Konfigurasi --
DOCS_DIR = "./data/documents"
VECTORSTORE_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
DEFAULT_MODEL = normalize_gemini_model(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
TOP_K = int(os.getenv("TOP_K_RESULTS", "10"))

Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

# -- Page Config --
st.set_page_config(
    page_title="SAA UNKLAB - Expert System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS --
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ===== GLOBAL ===== */
    *, html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* ===== HEADER ===== */
    .app-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #A78BFA, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #8B8FA3;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122a 0%, #0f0f1a 100%);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6C63FF;
        margin-bottom: 0.5rem;
        padding-left: 0.2rem;
    }

    /* ===== STATUS BADGES ===== */
    .status-valid {
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.3);
        color: #34D399;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.3rem 0;
    }
    .status-invalid {
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.3);
        color: #F87171;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.3rem 0;
    }
    .status-indexed {
        background: rgba(108, 99, 255, 0.1);
        border: 1px solid rgba(108, 99, 255, 0.3);
        color: #A78BFA;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.3rem 0;
        text-align: center;
    }

    /* ===== SOURCE CARDS ===== */
    .source-card {
        background: rgba(108, 99, 255, 0.06);
        border-left: 3px solid #6C63FF;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 0.4rem 0.4rem 0;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .source-card strong {
        color: #A78BFA;
    }
    .source-card em {
        color: #8B8FA3;
        font-style: normal;
    }

    /* ===== CHAT ===== */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 0.8rem;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(8px);
    }

    /* ===== ONBOARDING ===== */
    .onboard-card {
        background: rgba(108, 99, 255, 0.06);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 0.8rem;
        padding: 1.8rem;
        margin: 1rem 0;
        text-align: center;
    }
    .onboard-card h3 {
        color: #A78BFA;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    .onboard-steps {
        text-align: left;
        padding-left: 1rem;
        line-height: 2;
        color: #C4C7D4;
        font-size: 0.92rem;
    }
    .onboard-steps span {
        display: inline-block;
        width: 1.6rem;
        height: 1.6rem;
        line-height: 1.6rem;
        text-align: center;
        background: rgba(108, 99, 255, 0.2);
        color: #A78BFA;
        border-radius: 50%;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    .format-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin-top: 1rem;
    }
    .format-chip {
        background: rgba(108, 99, 255, 0.08);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 0.5rem;
        padding: 0.5rem;
        text-align: center;
        font-size: 0.8rem;
        color: #A78BFA;
        font-weight: 500;
    }

    /* ===== DOC LIST ===== */
    .doc-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0;
        font-size: 0.82rem;
        color: #C4C7D4;
    }
    .doc-tag {
        background: rgba(108, 99, 255, 0.15);
        color: #A78BFA;
        padding: 0.1rem 0.4rem;
        border-radius: 0.25rem;
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        color: #4A4D5E;
        font-size: 0.75rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 1rem;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        border: 1px solid rgba(108, 99, 255, 0.3);
    }
    .stButton > button:hover {
        border-color: #6C63FF;
        box-shadow: 0 0 15px rgba(108, 99, 255, 0.15);
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* ===== DIVIDER ===== */
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -- Session State --
def init_state():
    defaults = {
        "messages": [],
        "rag_chain": None,
        "vector_store": None,
        "documents_indexed": False,
        "selected_model": DEFAULT_MODEL,
        "show_sources": True,
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
        "pending_delete": None,
        "pending_delete_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


@st.cache_resource
def get_vs_manager():
    return VectorStoreManager(
        embedding_model=EMBEDDING_MODEL,
        persist_directory=VECTORSTORE_DIR,
    )


def setup_rag_chain(vs_manager):
    rag = RAGChain(
        gemini_model=st.session_state.selected_model,
        google_api_key=st.session_state.api_key,
        temperature=0.1,
        top_k_results=TOP_K,
    )
    retriever = vs_manager.as_retriever(k=TOP_K)
    rag.setup_chain(retriever)
    st.session_state.rag_chain = rag


def process_documents():
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vs_manager = get_vs_manager()

    with st.spinner("Memuat dokumen..."):
        docs = loader.load_directory(DOCS_DIR)
    if not docs:
        st.error("Tidak ada dokumen yang berhasil dimuat!")
        return None

    with st.spinner("Memecah dokumen menjadi chunks..."):
        chunks = chunker.split_documents(docs)

    with st.spinner(f"Mengindeks {len(chunks)} chunks ke ChromaDB..."):
        vs_manager.create_vectorstore(chunks)

    st.session_state.vector_store = vs_manager
    setup_rag_chain(vs_manager)
    st.session_state.documents_indexed = True
    return chunker.get_stats(chunks)


def load_existing():
    vs_manager = get_vs_manager()
    result = vs_manager.load_vectorstore()
    if result:
        st.session_state.vector_store = vs_manager
        setup_rag_chain(vs_manager)
        st.session_state.documents_indexed = True
        return True
    return False


# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">Konfigurasi</div>', unsafe_allow_html=True)
    st.divider()

    # -- API Key --
    st.markdown('<div class="sidebar-section-title">Google Gemini API Key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIza...",
        help="Ambil di aistudio.google.com",
        label_visibility="collapsed",
    )

    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.session_state.rag_chain = None

    # Validasi API key
    if st.session_state.api_key and st.session_state.api_key != "isi_api_key_kamu_di_sini":
        key_valid = check_gemini_api_key(st.session_state.api_key)
        if key_valid:
            st.markdown('<div class="status-valid">API Key Valid</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-invalid">API Key Tidak Valid</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-invalid">API Key belum diisi</div>', unsafe_allow_html=True)
        with st.expander("Cara Dapat API Key"):
            st.markdown("""
1. Buka [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Login dengan akun Google
3. Klik **"Create API Key"**
4. Salin & paste di sini
            """)

    st.divider()

    # -- Pilih Model --
    st.markdown('<div class="sidebar-section-title">Model</div>', unsafe_allow_html=True)
    models = get_gemini_models()
    model_names = list(models.keys())
    selected_idx = model_names.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_names else 0

    selected_model = st.selectbox(
        "Pilih Model",
        options=model_names,
        index=selected_idx,
        format_func=lambda m: f"{m}  -  {models[m]}",
        label_visibility="collapsed",
    )
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        if st.session_state.vector_store:
            setup_rag_chain(st.session_state.vector_store)

    st.divider()

    # -- Upload Dokumen --
    st.markdown('<div class="sidebar-section-title">Upload Dokumen</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF, PPTX, DOCX, TXT",
        type=["pdf", "pptx", "ppt", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        for uf in uploaded_files:
            save_uploaded_file(uf, DOCS_DIR)
        st.success(f"{len(uploaded_files)} file disimpan")

    # -- Daftar Dokumen --
    st.markdown('<div class="sidebar-section-title">Dokumen Tersimpan</div>', unsafe_allow_html=True)
    docs_info = get_documents_info(DOCS_DIR)
    if docs_info:
        for d in docs_info:
            cols = st.columns([0.7, 0.2, 0.1])
            cols[0].markdown(
                f'<div class="doc-item">'
                f'<span class="doc-tag">{d["type"]}</span>'
                f'{d["name"]} <span style="color:#4A4D5E">{d["size"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            delete_key = f"delete_{d['name']}_{d['size_bytes']}"
            if cols[2].button("Hapus", key=delete_key):
                st.session_state.pending_delete = d["path"]
                st.session_state.pending_delete_name = d["name"]

        if st.session_state.pending_delete:
            st.warning(f"Yakin ingin menghapus '{st.session_state.pending_delete_name}'? Ini juga akan mengosongkan indeks lama.")
            confirm_col, cancel_col = st.columns([0.15, 0.15])
            if confirm_col.button("Ya, hapus"):
                try:
                    os.remove(st.session_state.pending_delete)
                    get_vs_manager().clear_vectorstore()
                    st.session_state.vector_store = None
                    st.session_state.documents_indexed = False
                    st.success(f"File '{st.session_state.pending_delete_name}' berhasil dihapus.")
                except Exception as e:
                    st.error(f"Gagal menghapus {st.session_state.pending_delete_name}: {e}")
                finally:
                    st.session_state.pending_delete = None
                    st.session_state.pending_delete_name = None
            if cancel_col.button("Batal"):
                st.session_state.pending_delete = None
                st.session_state.pending_delete_name = None
    else:
        st.caption("Belum ada dokumen.")

    if docs_info:
        st.info("Hapus file akan mengosongkan indeks lama. Setelah menghapus, klik Proses Dokumen lagi untuk membangun ulang vektor.")

    st.divider()

    # -- Tombol Proses --
    api_ok = bool(st.session_state.api_key and
                  st.session_state.api_key != "isi_api_key_kamu_di_sini")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Proses Dokumen", use_container_width=True,
                     disabled=not docs_info or not api_ok):
            st.session_state.messages = []
            stats = process_documents()
            if stats:
                st.success(f"{stats['total_chunks']} chunks diproses")

    with col2:
        if st.button("Muat Tersimpan", use_container_width=True,
                     disabled=not api_ok):
            if load_existing():
                count = st.session_state.vector_store.get_document_count()
                st.success(f"{count} chunks dimuat")
            else:
                st.error("Belum ada data")

    st.divider()

    st.session_state.show_sources = st.toggle("Tampilkan Sumber", value=True)

    if st.session_state.documents_indexed and st.session_state.vector_store:
        count = st.session_state.vector_store.get_document_count()
        st.markdown(
            f'<div class="status-indexed">{count} chunks terindeks</div>',
            unsafe_allow_html=True
        )

    if st.button("Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================
# HALAMAN UTAMA
# =========================================
st.markdown("""
<div class="app-header">
    <h1>SAA UNKLAB</h1>
    <p>Tanya jawab cerdas berbasis dokumen</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.documents_indexed:
    st.markdown("""
    <div class="onboard-card">
        <h3>Mulai Menggunakan Expert System</h3>
        <div class="onboard-steps">
            <span>1</span> Isi Google Gemini API Key di sidebar<br>
            <span>2</span> Upload dokumen (PDF / PPTX / DOCX / TXT)<br>
            <span>3</span> Klik Proses Dokumen<br>
            <span>4</span> Mulai bertanya
        </div>
        <div class="format-grid">
            <div class="format-chip">PDF</div>
            <div class="format-chip">PPTX</div>
            <div class="format-chip">DOCX</div>
            <div class="format-chip">TXT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -- Riwayat Chat --
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (msg["role"] == "assistant" and "sources" in msg
                and st.session_state.show_sources and msg["sources"]):
            with st.expander(f"Sumber ({len(msg['sources'])} referensi)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-card"><strong>Sumber {i}:</strong> '
                        f'{src["file"]} ({src["file_type"]}) | Hal. {src["page"]}<br>'
                        f'<em>{src["preview"]}</em></div>',
                        unsafe_allow_html=True
                    )

# -- Input Chat --
if prompt := st.chat_input(
    "Ketik pertanyaan Anda...",
    disabled=not st.session_state.documents_indexed
):
    if not st.session_state.rag_chain:
        st.error("RAG chain belum siap. Proses dokumen terlebih dahulu.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Mencari & memproses..."):
            try:
                result = st.session_state.rag_chain.ask_with_sources(prompt)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"Error: {str(e)}"
                sources = []

        placeholder = st.empty()
        full = ""
        for char in answer:
            full += char
            time.sleep(0.004)
            placeholder.markdown(full + "▌")
        placeholder.markdown(full)

        if sources and st.session_state.show_sources:
            with st.expander(f"Sumber ({len(sources)} referensi)"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f'<div class="source-card"><strong>Sumber {i}:</strong> '
                        f'{src["file"]} ({src["file_type"]}) | Hal. {src["page"]}<br>'
                        f'<em>{src["preview"]}</em></div>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })

st.markdown('<div class="app-footer">SAA UNKLAB Expert System</div>', unsafe_allow_html=True)
