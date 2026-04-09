"""Central configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings driven by .env file or environment variables."""

    # LLM - Generation
    generation_provider: str = "openai"
    generation_model: str = "gpt-4.1"

    # LLM - Context / Query Transform (lightweight)
    context_provider: str = "openai"
    context_model: str = "gpt-4o-mini"

    # LLM - Judge / Session Review / Eval
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o-mini"

    # API keys
    openai_api_key: str = ""
    google_api_key: str = ""

    # Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "capg_knowledge"
    qdrant_hierarchy_collection: str = "capg_hierarchy"

    # Elasticsearch
    elasticsearch_url: str = "http://localhost:9200"

    # Embedding & Reranking
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Canary (governance — planned)
    canary_traffic_percent: int = 5

    # Retrieval
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
