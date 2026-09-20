from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings
from app.core.exceptions import LLMServiceError
from app.core.logging import get_logger


logger = get_logger(__name__)


class EmbeddingService:
    """
    Central embedding service for the RAG application.

    Used for:
    - document chunk embeddings
    - user query embeddings

    The same embedding model must be used for both.
    """

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        settings = get_settings()

        if not settings.openai_api_key:
            raise LLMServiceError(
                "OPENAI_API_KEY is not configured."
            )

        self.model_name = (
            model
            or settings.embedding_model
        )

        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.model_name,
                api_key=settings.openai_api_key,
            )

        except Exception as exc:
            logger.exception(
                "Unable to initialize embedding model"
            )

            raise LLMServiceError(
                "Unable to initialize embedding model."
            ) from exc

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Embed document chunks.
        """

        if not texts:
            return []

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            return []

        try:
            vectors = self.embeddings.embed_documents(
                cleaned_texts
            )

            logger.info(
                "Embedded document texts count=%d",
                len(vectors),
            )

            return vectors

        except Exception as exc:
            logger.exception(
                "Document embedding failed"
            )

            raise LLMServiceError(
                "Unable to embed document texts."
            ) from exc

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Embed a user search query.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        try:
            vector = self.embeddings.embed_query(
                query
            )

            logger.info(
                "Query embedding generated "
                "dimensions=%d",
                len(vector),
            )

            return vector

        except Exception as exc:
            logger.exception(
                "Query embedding failed"
            )

            raise LLMServiceError(
                "Unable to embed query."
            ) from exc