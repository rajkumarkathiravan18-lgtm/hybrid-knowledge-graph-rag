from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are loaded primarily from environment variables
    and the project-root .env file.
    """

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    app_name: str = Field(
        default="Hybrid Knowledge Graph RAG",
        alias="APP_NAME",
    )

    app_env: str = Field(
        default="development",
        alias="APP_ENV",
    )

    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    # --------------------------------------------------------
    # OpenAI / LLM
    # --------------------------------------------------------

    openai_api_key: str | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        alias="LLM_MODEL",
    )

    embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="EMBEDDING_MODEL",
    )

    # --------------------------------------------------------
    # Neo4j
    # --------------------------------------------------------

    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        alias="NEO4J_URI",
    )

    neo4j_username: str = Field(
        default="neo4j",
        alias="NEO4J_USERNAME",
    )

    neo4j_password: str | None = Field(
        default=None,
        alias="NEO4J_PASSWORD",
    )

    neo4j_database: str = Field(
        default="neo4j",
        alias="NEO4J_DATABASE",
    )

    # --------------------------------------------------------
    # LangSmith
    # --------------------------------------------------------

    langsmith_tracing: bool = Field(
        default=False,
        alias="LANGSMITH_TRACING",
    )

    langsmith_api_key: str | None = Field(
        default=None,
        alias="LANGSMITH_API_KEY",
    )

    langsmith_project: str = Field(
        default="hybrid-kg-rag",
        alias="LANGSMITH_PROJECT",
    )

    # --------------------------------------------------------
    # Backend
    # --------------------------------------------------------

    backend_host: str = Field(
        default="0.0.0.0",
        alias="BACKEND_HOST",
    )

    backend_port: int = Field(
        default=8000,
        alias="BACKEND_PORT",
    )

    cors_origins: str = Field(
        default="http://localhost:8501",
        alias="CORS_ORIGINS",
    )

    # --------------------------------------------------------
    # Uploads
    # --------------------------------------------------------

    max_upload_mb: int = Field(
        default=20,
        alias="MAX_UPLOAD_MB",
    )

    allowed_file_extensions: str = Field(
        default=".pdf,.txt,.md",
        alias="ALLOWED_FILE_EXTENSIONS",
    )

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    chunk_size: int = Field(
        default=1000,
        alias="CHUNK_SIZE",
    )

    chunk_overlap: int = Field(
        default=150,
        alias="CHUNK_OVERLAP",
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    vector_top_k: int = Field(
        default=5,
        alias="VECTOR_TOP_K",
    )

    graph_top_k: int = Field(
        default=5,
        alias="GRAPH_TOP_K",
    )

    vector_weight: float = Field(
        default=0.60,
        alias="VECTOR_WEIGHT",
    )

    graph_weight: float = Field(
        default=0.40,
        alias="GRAPH_WEIGHT",
    )

    max_context_chars: int = Field(
        default=16000,
        alias="MAX_CONTEXT_CHARS",
    )

    # --------------------------------------------------------
    # Web Search Fallback
    # --------------------------------------------------------

    web_search_enabled: bool = Field(
        default=True,
        alias="WEB_SEARCH_ENABLED",
    )

    web_search_max_results: int = Field(
        default=5,
        alias="WEB_SEARCH_MAX_RESULTS",
    )

    # --------------------------------------------------------
    # Persistent Conversation Memory
    # --------------------------------------------------------

    memory_db_path: Path = Field(
        default=PROJECT_ROOT
        / "data"
        / "memory"
        / "conversations.db",
        alias="MEMORY_DB_PATH",
    )

    memory_history_limit: int = Field(
        default=20,
        alias="MEMORY_HISTORY_LIMIT",
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    upload_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "uploads",
        alias="UPLOAD_DIR",
    )

    faiss_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "faiss",
        alias="FAISS_DIR",
    )

    # --------------------------------------------------------
    # Pydantic settings configuration
    # --------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --------------------------------------------------------
    # Validators
    # --------------------------------------------------------

    @field_validator("app_env")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        value = value.lower().strip()

        allowed = {
            "development",
            "testing",
            "production",
        }

        if value not in allowed:
            raise ValueError(
                f"APP_ENV must be one of: {sorted(allowed)}"
            )

        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.upper().strip()

        allowed = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if value not in allowed:
            raise ValueError(
                f"LOG_LEVEL must be one of: {sorted(allowed)}"
            )

        return value

    @field_validator(
        "max_upload_mb",
        "chunk_size",
        "vector_top_k",
        "graph_top_k",
        "max_context_chars",
        "memory_history_limit",
        "web_search_max_results",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                "Value must be greater than zero."
            )

        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                "CHUNK_OVERLAP cannot be negative."
            )

        return value

    @field_validator(
        "vector_weight",
        "graph_weight",
    )
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError(
                "Retrieval weights must be between 0 and 1."
            )

        return value

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        return self

    @model_validator(mode="after")
    def validate_retrieval_weights(self) -> "Settings":
        total = (
            self.vector_weight
            + self.graph_weight
        )

        if abs(total - 1.0) > 0.001:
            raise ValueError(
                "VECTOR_WEIGHT + GRAPH_WEIGHT "
                "must equal 1.0."
            )

        return self

    # --------------------------------------------------------
    # Convenience properties
    # --------------------------------------------------------

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def allowed_extensions(self) -> set[str]:
        return {
            extension.strip().lower()
            for extension
            in self.allowed_file_extensions.split(",")
            if extension.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        return (
            self.max_upload_mb
            * 1024
            * 1024
        )

    # --------------------------------------------------------
    # Directory initialization
    # --------------------------------------------------------

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.faiss_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.memory_db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Caching prevents repeatedly parsing the environment
    configuration during the application lifecycle.
    """

    settings = Settings()

    settings.ensure_directories()

    return settings