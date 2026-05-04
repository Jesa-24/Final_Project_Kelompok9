"""
rag/vector_store.py
===================
Modul untuk menyimpan dan mencari dokumen menggunakan
ChromaDB sebagai vector database dan HuggingFace Embeddings.

Alur:
1. Dokumen di-embed (diubah menjadi vektor numerik)
2. Vektor disimpan di ChromaDB
3. Saat query, pertanyaan juga di-embed
4. Dicari vektor yang paling mirip (cosine similarity)
"""

import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Chroma telemetry is optional and currently noisy on some local setups.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class VectorStoreManager:
    """
    Mengelola penyimpanan dan pencarian vektor dokumen menggunakan ChromaDB.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        persist_directory: str = "./vectorstore",
        collection_name: str = "rag_documents",
    ):
        """
        Args:
            embedding_model: Nama model HuggingFace untuk embedding
                             (default: all-mpnet-base-v2 - 768 dim, lebih akurat)
            persist_directory: Folder penyimpanan ChromaDB
            collection_name: Nama koleksi di ChromaDB
        """
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore: Optional[Chroma] = None

        print(f"[VectorStore] Inisialisasi Embedding Model: {embedding_model}")
        print(f"     (Download pertama kali mungkin butuh beberapa menit...)\n")

        # Inisialisasi model embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},   # Ganti ke "cuda" jika punya GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        print("     Embedding model siap!\n")

    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        Membuat vector store baru dari chunks dokumen.
        Jika sudah ada, data lama akan dihapus dan dibuat ulang.

        Args:
            chunks: List Document chunks yang sudah dipecah

        Returns:
            Chroma vectorstore yang siap digunakan
        """
        if not chunks:
            raise ValueError("Tidak ada chunks untuk dimasukkan ke vector store!")

        print(f"[VectorStore] Membuat vector store dari {len(chunks)} chunks...")
        print(f"     Database: ChromaDB")
        print(f"     Lokasi  : {self.persist_directory}")
        print(f"     Koleksi : {self.collection_name}\n")

        # Hapus vectorstore lama jika ada
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print("     Vector store lama dihapus.")

        # Buat vectorstore baru
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        print(f"     Vector store berhasil dibuat!\n")
        return self.vectorstore

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Memuat vector store yang sudah ada sebelumnya.

        Returns:
            Chroma vectorstore atau None jika belum ada
        """
        if not os.path.exists(self.persist_directory):
            print("Vector store belum ada. Silakan upload dokumen terlebih dahulu.")
            return None

        print(f"[VectorStore] Memuat vector store dari '{self.persist_directory}'...")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

        count = self.vectorstore._collection.count()
        print(f"     Vector store dimuat: {count} chunks tersimpan\n")
        return self.vectorstore

    def clear_vectorstore(self) -> None:
        """Hapus data vector store yang tersimpan sehingga harus diindeks ulang."""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print(f"[VectorStore] Vector store dihapus dari '{self.persist_directory}'.")
        self.vectorstore = None

    def add_documents(self, chunks: List[Document]) -> None:
        """
        Menambahkan dokumen baru ke vector store yang sudah ada.

        Args:
            chunks: List Document chunks baru
        """
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
            if self.vectorstore is None:
                self.create_vectorstore(chunks)
                return

        print(f"[VectorStore] Menambahkan {len(chunks)} chunks ke vector store...")
        self.vectorstore.add_documents(chunks)
        print(f"     Berhasil ditambahkan!\n")

    def similarity_search(
        self,
        query: str,
        k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """
        Mencari dokumen yang paling relevan dengan query.

        Args:
            query: Pertanyaan/query dari pengguna
            k: Jumlah dokumen yang dikembalikan

        Returns:
            List tuple (Document, score) yang paling relevan
        """
        if self.vectorstore is None:
            raise ValueError("Vector store belum diinisialisasi!")

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def as_retriever(self, k: int = 10):
        """
        Mengembalikan retriever untuk digunakan di RAG chain.
        Menggunakan MMR (Maximal Marginal Relevance) untuk mendapatkan
        hasil yang beragam - tidak hanya chunk yang mirip satu sama lain.

        Args:
            k: Jumlah dokumen yang dikembalikan per query

        Returns:
            LangChain retriever object
        """
        if self.vectorstore is None:
            raise ValueError("Vector store belum diinisialisasi!")

        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 3,   # Fetch 3x lebih banyak, lalu pilih yang paling beragam
                "lambda_mult": 0.7,  # 0=max diversity, 1=max relevance
            },
        )

    def get_document_count(self) -> int:
        """Mengembalikan jumlah chunks yang tersimpan."""
        if self.vectorstore is None:
            return 0
        return self.vectorstore._collection.count()

    def is_ready(self) -> bool:
        """Mengecek apakah vector store sudah siap digunakan."""
        return self.vectorstore is not None and self.get_document_count() > 0
