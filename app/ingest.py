"""
Ingestion pipeline: PDF file(s) on disk -> extracted pages -> chunks ->
embedded + added to the vector store. This is the function both the
FastAPI upload endpoint and the Colab notebook call.
"""
from typing import List

from app.pdf_loader import extract_pages_from_multiple
from app.chunking import chunk_pages
from app.vector_store import build_or_update_index, clear_index


def ingest_pdfs(pdf_paths: List[str], replace_existing: bool = True) -> dict:
    pages = extract_pages_from_multiple(pdf_paths)
    if not pages:
        return {"status": "error", "message": "No extractable text found in the uploaded PDF(s)."}

    if replace_existing:
        clear_index()

    chunks = chunk_pages(pages)
    build_or_update_index(chunks)

    return {
        "status": "success",
        "documents_processed": len(set(p.doc_name for p in pages)),
        "pages_processed": len(pages),
        "chunks_created": len(chunks),
    }
