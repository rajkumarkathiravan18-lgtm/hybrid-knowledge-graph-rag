class AppError(Exception):
    """
    Base exception for expected application errors.
    """

    status_code = 500
    error_code = "application_error"

    def __init__(
        self,
        message: str,
        *,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.details = details or {}


class InvalidFileError(AppError):
    status_code = 400
    error_code = "invalid_file"


class FileTooLargeError(AppError):
    status_code = 413
    error_code = "file_too_large"


class DocumentNotFoundError(AppError):
    status_code = 404
    error_code = "document_not_found"


class DocumentProcessingError(AppError):
    status_code = 500
    error_code = "document_processing_error"


class VectorStoreError(AppError):
    status_code = 503
    error_code = "vector_store_error"


class GraphDatabaseError(AppError):
    status_code = 503
    error_code = "graph_database_error"


class RetrievalError(AppError):
    status_code = 500
    error_code = "retrieval_error"


class LLMServiceError(AppError):
    status_code = 503
    error_code = "llm_service_error"


class GuardrailError(AppError):
    status_code = 500
    error_code = "guardrail_error"


class GuardrailBlockedError(AppError):
    status_code = 400
    error_code = "guardrail_blocked"