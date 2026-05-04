"""
rag/text_splitter.py
====================
Modul untuk memecah dokumen menjadi chunk-chunk kecil
agar lebih efisien saat diproses dan dicari.
"""

import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    Memecah dokumen besar menjadi chunk-chunk kecil (potongan teks).
    
    Menggunakan RecursiveCharacterTextSplitter yang memecah teks
    secara rekursif berdasarkan separator (paragraf -> kalimat -> kata).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 300,
    ):
        """
        Args:
            chunk_size: Ukuran maksimal setiap chunk (dalam karakter)
            chunk_overlap: Jumlah karakter yang overlap antar chunk
                           (membantu menjaga konteks antar chunk)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Separator diurutkan dari yang paling diutamakan
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        )

    def _clean_text(self, text: str) -> str:
        """
        Membersihkan teks dari karakter yang tidak perlu.
        Membantu embedding model menghasilkan vektor yang lebih akurat.
        """
        # Hapus multiple whitespace berturut-turut
        text = re.sub(r'[ \t]+', ' ', text)
        # Hapus baris kosong berlebih (lebih dari 2 berturut)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Hapus karakter null/kontrol
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        # Trim setiap baris
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Memecah list dokumen menjadi chunk-chunk kecil.

        Args:
            documents: List dokumen yang sudah dimuat

        Returns:
            List dokumen yang sudah dipecah menjadi chunk
        """
        if not documents:
            print("Tidak ada dokumen untuk dipecah.")
            return []

        print(f"[TextChunker] Memecah {len(documents)} dokumen menjadi chunks...")
        print(f"     Ukuran chunk: {self.chunk_size} karakter")
        print(f"     Overlap     : {self.chunk_overlap} karakter")

        # Bersihkan teks sebelum splitting
        cleaned_docs = []
        for doc in documents:
            cleaned = self._clean_text(doc.page_content)
            if cleaned:  # Hanya masukkan jika ada konten
                new_doc = Document(
                    page_content=cleaned,
                    metadata=doc.metadata.copy()
                )
                cleaned_docs.append(new_doc)

        chunks = self.splitter.split_documents(cleaned_docs)

        # Tambahkan nomor chunk ke metadata
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_total"] = total_chunks

        print(f"     Total chunks dihasilkan: {len(chunks)}\n")
        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Memecah string teks biasa menjadi list string chunks.

        Args:
            text: String teks yang akan dipecah

        Returns:
            List string chunks
        """
        return self.splitter.split_text(text)

    def get_stats(self, chunks: List[Document]) -> dict:
        """
        Menghitung statistik dari hasil chunking.

        Returns:
            Dictionary berisi statistik chunk
        """
        if not chunks:
            return {}

        lengths = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_chars": sum(lengths),
        }
