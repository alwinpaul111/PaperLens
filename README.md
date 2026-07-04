# ResearchGPT — RAG-based Research Paper Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system that lets you upload
research papers (PDF) and ask questions answered with grounded, cited excerpts.

## Architecture

```
PDF Upload → Text Extraction (PyMuPDF) → Chunking (LangChain splitter)
→ Embeddings (Sentence-Transformers, all-MiniLM-L6-v2)
→ Vector Store (FAISS, swappable for ChromaDB)
→ Retrieval (top-k similarity search)
→ LLM (Groq Llama-3, swappable for HF Inference API)
→ Answer + Citations (doc name + page number)
→ Conversation memory (sliding window)
```

## Project structure

```
research-gpt/
├── app/
│   ├── config.py         # all tunables in one place
│   ├── pdf_loader.py      # PDF -> per-page text
│   ├── chunking.py        # text -> overlapping chunks
│   ├── embeddings.py      # embedding model loader
│   ├── vector_store.py    # FAISS / Chroma index build + search
│   ├── llm.py              # Groq / HuggingFace LLM call
│   ├── rag_pipeline.py    # retrieval + prompt + citations + memory
│   ├── ingest.py           # ties PDF -> chunks -> index together
│   └── main.py              # FastAPI app
├── streamlit_app.py        # frontend (direct or API mode)
├── notebooks/
│   └── ResearchGPT_Colab.ipynb   # experiment in Colab
├── Dockerfile               # FastAPI backend image
├── Dockerfile.streamlit     # Streamlit frontend image
├── docker-compose.yml        # run both together
└── requirements.txt
```

## Setup

### 1. Get a free LLM API key
This project defaults to **Groq** (free tier, very fast Llama-3 inference):
1. Go to https://console.groq.com/keys
2. Create a free API key
3. Set it as an environment variable:
   ```bash
   export GROQ_API_KEY="your-key-here"
   ```

Alternative: use your existing HuggingFace account (`alwinn`) by setting
`LLM_PROVIDER=huggingface` and `HUGGINGFACEHUB_API_TOKEN` instead — no extra
signup needed.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3a. Run locally (single process, easiest)
```bash
streamlit run streamlit_app.py
```
This runs in `RUN_MODE=direct` by default — no separate backend needed,
same pattern as your hate speech detector deployment.

### 3b. Run backend + frontend separately (production pattern)
```bash
# Terminal 1
uvicorn app.main:app --reload --port 8000

# Terminal 2
RUN_MODE=api API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

### 3c. Run with Docker Compose (full production setup)
```bash
export GROQ_API_KEY="your-key-here"
docker-compose up --build
```
- FastAPI docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

## Deploying (matching your existing deployment style)

- **Streamlit Cloud**: deploy `streamlit_app.py` directly with `RUN_MODE=direct`.
  Add `GROQ_API_KEY` as a secret in Streamlit Cloud's settings. This mirrors
  your `HATE-SPEECH-DETECTOR` deployment exactly.
- **FastAPI backend**: deploy separately on Render / Railway / Fly.io (all have
  free tiers) if you want the API + Streamlit split, useful for demonstrating
  a proper client-server architecture in interviews.

## Key design decisions (good for interview talking points)

- **Chunking strategy**: `RecursiveCharacterTextSplitter` with 800-char chunks
  and 150-char overlap — balances retrieval precision vs. context loss at
  chunk boundaries.
- **Embedding model**: `all-MiniLM-L6-v2` — 384-dim, CPU-friendly, strong
  performance/speed tradeoff for semantic search (no GPU/API cost).
- **Vector store choice**: FAISS by default for raw speed; ChromaDB wired in
  as an alternative to discuss metadata filtering and persistence tradeoffs.
- **Citations**: every chunk carries `(doc_name, page_number)` metadata all
  the way through retrieval so answers can point back to an exact page —
  this is what makes it a *research assistant* rather than a black box.
- **Conversation memory**: a sliding window of the last N turns is injected
  into the prompt so follow-up questions ("what about their dataset size?")
  resolve correctly without re-uploading context.
- **LLM swap**: the `llm.py` abstraction lets you swap Groq ↔ HuggingFace ↔
  OpenAI with a one-line config change — demonstrates you understand LLM
  provider abstraction, not just one API.

## Extending this project (nice additions for your CV)

- Add re-ranking (e.g. Cohere rerank or a cross-encoder) after initial
  retrieval to improve precision.
- Add hybrid search (BM25 + dense) for better recall on exact terms/numbers.
- Add evaluation with RAGAS (faithfulness, answer relevance, context
  precision) — strong signal for a Data Science master's application.
- Add multi-document comparison ("compare the methodology in paper A vs B").
- Swap FAISS for a hosted vector DB (Pinecone/Weaviate free tier) to show
  cloud-native vector search experience.
