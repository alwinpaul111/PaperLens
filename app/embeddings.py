"""
Embedding model wrapper. Uses a local Sentence-Transformers model so the
project runs with zero API cost for the embedding step.
"""
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Cached loader so the model is only loaded into memory once."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
