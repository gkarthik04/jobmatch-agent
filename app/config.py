from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/jobmatch"
    gemini_api_key: str = ""
    chat_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768

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

    # search
    rerank_top_n: int = 50
    rrf_k: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
