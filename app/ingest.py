
from typing import List

from app.pdf_loader import extract_pages_from_multiple
from app.chunking import chunk_pages
from app.vector_store import build_or_update_index


def ingest_pdfs(pdf_paths: List[str]) -> dict:
    pages = extract_pages_from_multiple(pdf_paths)
    if not pages:
        return {"status": "error", "message": "No extractable text found in the uploaded PDF(s)."}

    chunks = chunk_pages(pages)
    build_or_update_index(chunks)

    return {
        "status": "success",
        "documents_processed": len(set(p.doc_name for p in pages)),
        "pages_processed": len(pages),
        "chunks_created": len(chunks),
    }
