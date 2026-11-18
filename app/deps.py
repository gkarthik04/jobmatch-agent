"""FastAPI dependencies. Tests override these to inject fake LLM/embedder."""

from functools import lru_cache

from app.services.embeddings import Embedder, make_embedder
from app.services.llm import LLM, get_llm


@lru_cache
def _embedder() -> Embedder:
    return make_embedder()


def get_embedder() -> Embedder:
    return _embedder()


def get_llm_dep() -> LLM:
    return get_llm()
