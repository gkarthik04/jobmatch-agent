from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/jobmatch"
    gemini_api_key: str = ""
    chat_model: str = "gemini-3.1-flash-lite"
    # ordered fallback chain (comma separated). Free-tier quotas are per model, so we spill over.
    chat_models: str = (
        "gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-flash-lite-latest,"
        "gemini-3-flash-preview,gemini-2.5-flash"
    )
    # Embeddings: "local" (fastembed/ONNX, free + unlimited) or "gemini" (1,000 texts/day on the
    # free tier, which cannot index thousands of postings). dim must match the provider's model.
    embed_provider: str = "local"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    jwt_secret: str = "dev-only-change-me"
    jwt_expire_minutes: int = 60 * 24

    rate_limit_per_minute: int = 20
    request_timeout_s: float = 90.0
    log_level: str = "INFO"
    environment: str = "dev"

    # agent
    agent_max_steps: int = 6
    tool_timeout_s: float = 20.0
    llm_timeout_s: float = 45.0
    llm_max_wait_s: float = 30.0  # how long a call may wait for a parked model to recover
    # hard latency budgets for optional LLM enhancements: on timeout we degrade, never hang
    rerank_budget_s: float = 20.0
    explain_budget_s: float = 25.0
    llm_rpm: int = 9  # Gemini Flash free tier is ~10 requests/min; stay just under it

    # search
    rerank_top_n: int = 50
    rrf_k: int = 60
    # weighted RRF weights: vector, full-text (body), title-field full-text. Chosen on DEV queries
    # in evals/fusion_experiment.py; "1,1,0" reproduces the original equal-weight 2-way hybrid.
    rrf_weights: str = "1,0.5,0.5"

    @property
    def rrf_weight_map(self) -> dict[str, float]:
        v, f, t = (float(x) for x in self.rrf_weights.split(","))
        return {"vector": v, "fts": f, "title": t}


    @property
    def chat_model_chain(self) -> list[str]:
        chain = [m.strip() for m in self.chat_models.split(",") if m.strip()]
        return chain or [self.chat_model]


@lru_cache
def get_settings() -> Settings:
    return Settings()
