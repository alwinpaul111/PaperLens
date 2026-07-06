# PaperLens

A Retrieval-Augmented Generation (RAG) system for asking questions about research papers, with grounded, page-level citations.

Upload a PDF, ask questions about it, and get answers pulled directly from the document rather than from a language model's general knowledge. Every answer links back to the specific page it came from.

**Live demo:** https://researchgpt-nbdky2iczdnns8m34gb5cv.streamlit.app/
**Repository:** https://github.com/alwinpaul111/PaperLens

## What it does

- Upload one or more PDF research papers through a web interface
- The system extracts text, splits it into chunks, and builds a searchable vector index
- Ask a question in plain language
- The system retrieves the most relevant passages from the paper and passes them to a language model, which answers using only that retrieved content
- Every answer comes with citations showing which document and page it was drawn from
- Conversation memory lets you ask follow-up questions naturally

## How it works

```
PDF upload
  -> text extraction (PyMuPDF, page by page)
  -> chunking (overlapping text segments, ~800 characters each)
  -> embeddings (Sentence-Transformers, all-MiniLM-L6-v2)
  -> vector index (FAISS, persisted to disk)
  -> retrieval (top-k similarity search on the user's question)
  -> LLM (Groq / Llama 3)
  -> answer with citations
```

## Project structure

```
PaperLens/
├── app/
│   ├── config.py          all tunable settings
│   ├── pdf_loader.py       PDF -> per-page text
│   ├── chunking.py         text -> overlapping chunks
│   ├── embeddings.py       embedding model loader
│   ├── vector_store.py     FAISS / Chroma index build and search
│   ├── llm.py               Groq / HuggingFace LLM call
│   ├── rag_pipeline.py     retrieval, prompting, citations, memory
│   ├── ingest.py            ties PDF -> chunks -> index together
│   └── main.py               FastAPI backend
├── streamlit_app.py          frontend
├── notebooks/
│   └── PaperLens_Colab.ipynb   experimentation notebook
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
└── requirements.txt
```

## Running it locally

### 1. Get a Groq API key
Sign up at https://console.groq.com/keys and create a free API key.

```bash
export GROQ_API_KEY="your-key-here"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run streamlit_app.py
```
Open the local URL Streamlit prints, upload a PDF, and start asking questions.

### Running the backend and frontend separately
```bash
# Terminal 1
uvicorn app.main:app --reload --port 8000

# Terminal 2
RUN_MODE=api API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

### Running with Docker
```bash
export GROQ_API_KEY="your-key-here"
docker-compose up --build
```
FastAPI docs at http://localhost:8000/docs, Streamlit UI at http://localhost:8501.

## Design notes

**Chunking.** Text is split into 800-character segments with 150 characters of overlap, using LangChain's recursive character splitter. This balances retrieval precision against losing context at chunk boundaries.

**Embeddings.** `all-MiniLM-L6-v2` was chosen for being small, fast, and CPU-friendly, avoiding any embedding API cost while still giving solid semantic search quality.

**Vector store.** FAISS is used by default for speed and simplicity. ChromaDB is also wired in as a swappable alternative (`VECTOR_BACKEND` in `config.py`) since it offers built-in metadata filtering and easier incremental updates.

**Citations.** Every chunk carries its source document name and page number as metadata, all the way through retrieval, so an answer can always be traced back to an exact page.

**Broad questions.** Generic questions like "what is this paper about" retrieve poorly with pure similarity search, since no single chunk closely matches such a general query. For these, the system also pulls in the paper's first page directly and runs a secondary search for introduction/overview content, rather than relying only on the literal wording of the question.

**LLM provider.** Groq is used by default for its free tier and fast inference. The LLM call is abstracted in `llm.py` so a different provider can be swapped in with a one-line config change.

## Known limitations

- PDF text extraction can struggle with heavily math- or symbol-dense sections, occasionally producing garbled context for questions that land on those parts of a paper.
- Very open-ended questions are inherently harder for retrieval-based systems than specific ones; asking "what dataset did the authors use" will generally outperform "what is this paper about."
- The Groq free tier has a request size and rate limit, so very broad queries that pull in a lot of context can occasionally hit a rate-limit error.
- The vector index does not currently persist across a Streamlit Cloud app restart; re-upload the PDF if the app has gone to sleep and restarted.

## Possible extensions

- Re-ranking retrieved chunks with a cross-encoder to improve precision
- Hybrid search combining BM25 with dense retrieval for better recall on exact terms and numbers
- Automated evaluation with RAGAS (faithfulness, answer relevance, context precision)
- Multi-document comparison across papers
- A hosted vector database (Pinecone, Weaviate) in place of local FAISS

## Author

Alwin Paul
