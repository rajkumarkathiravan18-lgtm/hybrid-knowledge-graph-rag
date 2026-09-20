from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.exceptions import LLMServiceError
from app.core.logging import get_logger


logger = get_logger(__name__)


class LLMClient:
    """
    Central language-model client.

    Responsibilities:
    - load model configuration
    - initialize ChatOpenAI
    - provide a reusable model instance

    Prompt construction and RAG logic must remain
    outside this class.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        settings = get_settings()

        if not settings.openai_api_key:
            raise LLMServiceError(
                "OPENAI_API_KEY is not configured."
            )

        self.model_name = (
            model
            or settings.llm_model
        )

        self.temperature = temperature

        try:
            self.model = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=settings.openai_api_key,
            )

            logger.info(
                "LLM client initialized model=%s",
                self.model_name,
            )

        except Exception as exc:
            logger.exception(
                "Unable to initialize LLM client"
            )

            raise LLMServiceError(
                "Unable to initialize language model."
            ) from exc

    def get_model(
        self,
    ) -> ChatOpenAI:
        """
        Return the configured LangChain model.
        """

        return self.model